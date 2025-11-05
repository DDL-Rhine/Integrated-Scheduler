import collections
import copy
import math


class LucidPolicy(object):
    def __init__(self, time_fn):
        self._time_fn = time_fn
        self._status = {}
        self._pack_pairs = {}

    def optimize(self, jobs, nodes, prev_allocations, node_template):
        from simulator import args 
        allocations = {} # {k: v for k, v in prev_allocations.items() if k in jobs}
        for job in jobs.values():
            completion_epoch = job.application.get_completion_epoch(
                    job.target_batch_size)
            placement = tuple()
            while sum(placement) < job.replica_lower_bound:
                placement = (*placement, min(job.replica_lower_bound - sum(placement), 4))
            
            
            atomic_bsz = job.target_batch_size // sum(placement)
            # import pdb; pdb.set_trace()
            if completion_epoch <= job.epoch:
                job.remaining = 1
            else:
                job.remaining = job.application.get_iteration(job.target_batch_size, completion_epoch)
                
        min_replicas = {}
        for key, job in jobs.items():
            min_replicas[key] = job.replica_lower_bound  
        
        num_gpus = sum(node.resources["nvidia.com/gpu"] for node in nodes.values())
        num_replicas = {}
        
        self._status = {key: val for key, val in self._status.items() if key in jobs}
        for key, job in jobs.items():
            if key not in self._status:
                self._status[key] = 'PENDING'
        
        
        # for gpu sharing 
        pack_pairs = list() 
        inSharingList = list() 
        inSharingKeys = list() 
        # update prev_allocations 
        if args.GPUSharing: 
            for job_pair in self._pack_pairs:
                if (job_pair[0] not in self._status) and (job_pair[1] not in self._status): 
                    continue 
                if job_pair[0] in self._status and job_pair[1] in self._status: # all not finished 
                    pack_pairs.append(job_pair)
                    inSharingKeys.append(job_pair[0])
                    inSharingKeys.append(job_pair[1])
                    if self._status[job_pair[0]] == "PENDING": 
                        import pdb; pdb.set_trace() 
                    if self._status[job_pair[1]] == "PENDING": 
                        import pdb; pdb.set_trace() 
                        
                    found = False 
                    for key, job in jobs.items(): 
                        if key == job_pair[0]: 
                            found = True
                            inSharingList.append(job)
                    assert found == True 
                    for key, job in jobs.items(): 
                        if key == job_pair[1]: 
                            inSharingList.append(job)
                    continue 
                
                if job_pair[0] not in self._status: # update resource allocation 
                    alloc = prev_allocations[job_pair[1]]
                    prev_allocations[job_pair[1]] = [alloc[0]] 
                
                if job_pair[1] not in self._status: 
                    alloc = prev_allocations[job_pair[0]]
                    prev_allocations[job_pair[0]] = [alloc[0]]
        
        # execute before sharing 
        # for key, job in jobs.items(): # sorted(jobs.items(), key=lambda item: item[1].deserved_service):
        for key, job in sorted(jobs.items(), key=lambda item: item[1].creation_timestamp):
            if self._status[key] == "RUNNING":
                if isinstance(prev_allocations[key], list): 
                    num_gpus -= len(prev_allocations[key])
                    num_replicas[key] = len(prev_allocations[key])
                    allocations[key] = prev_allocations[key]
                elif isinstance(prev_allocations[key], tuple): 
                    num_gpus -= 0.5
                    num_replicas[key] = 0.5
                    allocations[key] = prev_allocations[key]
                    # not update num_replicas and allocations
                
        sharingKeys = list() 
        sharingList = list() 
        
        if args.GPUSharing: 
            for key, job in jobs.items():
                sharingKeys.append(key)
                sharingList.append(job) 
            
            gpu_util_list = [self.predidct_gpu_util(job) for job in sharingList]
            for i in range(len(sharingList)):
                if sharingList[i] in inSharingList: 
                    continue
                
                for j in range(i+1, len(sharingList)):
                    if sharingList[j] in inSharingList: continue 
                    if sharingList[i].replica_lower_bound > 1: continue 
                    if sharingList[j].replica_lower_bound > 1: continue 
                    
                    if sharingKeys[i] in prev_allocations and isinstance(prev_allocations[sharingKeys[i]], tuple): continue
                    if sharingKeys[j] in prev_allocations and isinstance(prev_allocations[sharingKeys[j]], tuple): continue 
                    if sharingKeys[i] in prev_allocations and sharingKeys[j] in prev_allocations: continue 
                    
                    if gpu_util_list[i] <= 50 and gpu_util_list[j] <= 50: 
                        if sharingKeys[i] not in prev_allocations and sharingKeys[j] not in prev_allocations:
                            if num_gpus == 0: 
                                continue 
                            else:  
                                num_gpus -= 1
                        
                        # we have more gpus to allocate 
                        pack_pairs.append((sharingKeys[i], sharingKeys[j]))
                        inSharingList.append(sharingList[i])
                        inSharingList.append(sharingList[j])
                        inSharingKeys.append(sharingKeys[i])
                        inSharingKeys.append(sharingKeys[j])
                            
                        jobs.pop(sharingKeys[i])
                        jobs.pop(sharingKeys[j])
                        break 
            
            
        for key, job in sorted(jobs.items(), key=lambda item: item[1].remaining):
            if self._status[key] == "RUNNING":
                continue 
            if job in inSharingList: continue 
            if min_replicas[key] > num_gpus:
                num_replicas[key] = 0
                continue
            num_replicas[key] = min_replicas[key]
            num_gpus -= min_replicas[key]
            self._status[key] = 'RUNNING'
            
        
        # Placements.
        # allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas[k]}
        job_keys = sorted(jobs, key=lambda k: num_replicas[k] if k in num_replicas else 0)
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        no_sharing_allocations = copy.deepcopy(allocations)
        remove_keys = list()    
        for key, alloc in no_sharing_allocations.items():
            if isinstance(alloc, tuple): 
                remove_keys.append(key)
        for key in remove_keys: 
            no_sharing_allocations.pop(key)
        
        free_gpus = collections.Counter(total_gpus) - collections.Counter(sum(no_sharing_allocations.values(), []))
        if args.GPUSharing: 
            for key, alloc in allocations.items(): 
                if isinstance(alloc, tuple): 
                    free_gpus[alloc[0]] -= 0.5
        
        for key in job_keys:
            if key in inSharingKeys: continue
            if num_replicas[key] > 0 and not allocations.get(key):
                # Allocate resources.
                allocations[key] = []
                while len(allocations[key]) < num_replicas[key]:
                    node_idx, count = free_gpus.most_common(1)[0]
                    num = int(min(count, num_replicas[key] - len(allocations[key])))
                    allocations[key].extend([node_idx] * num)
                    free_gpus[node_idx] -= num
        
        for i in range(len(pack_pairs)):
            key1, key2 = pack_pairs[i][0], pack_pairs[i][1]
            if key1 in allocations or key2 in allocations:
                alloc = allocations[key1] if key1 in allocations else allocations[key2]
                new_alloc = (alloc[0], 0.5)
                allocations[key1] = new_alloc
                allocations[key2] = copy.deepcopy(new_alloc)
                self._status[key1] = 'RUNNING'
                self._status[key2] = 'RUNNING'
                continue 
            
            found = False 
            for node_idx in free_gpus:
                if free_gpus[node_idx] > 0:  
                    allocations[key1] = (node_idx, 0.5)
                    allocations[key2] = (node_idx, 0.5)
                    free_gpus[node_idx] -= 1
                    found = True 
                    self._status[key1] = 'RUNNING'
                    self._status[key2] = 'RUNNING'
                    break 
            if not found: 
                import pdb; pdb.set_trace()
            
        self._pack_pairs = pack_pairs
        return allocations, -1 # len(nodes)

    def predidct_gpu_util(self, job): 
        local_bsz = math.ceil(job.target_batch_size / 1 - 1e-8)
        # gpu_util = job.application.get_gpu_util(local_bsz, job.frozen_layer)
        gpu_util = job.application.get_gpu_util(local_bsz, 0)
        return gpu_util
        
    def predict_step_time(self, job, num_replicas):
        placement = ()
        while sum(placement) < num_replicas:
            placement = (*placement, min(num_replicas - sum(placement), 4))
        local_bsz = math.ceil(job.target_batch_size / num_replicas - 1e-8)
        accum_steps = math.ceil(local_bsz / job.application.max_local_bsz - 1e-8) - 1
        if num_replicas == 1:
            accum_steps = max(1, accum_steps)
        atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
        count = num_replicas * (accum_steps + 1)
        atomic_bsz = min(atomic_bsz, int(job.application.max_batch_size / count))
        step_time, sync_time = job.application.get_throughput(placement, atomic_bsz, job.frozen_layer)
        return step_time + (step_time - sync_time) * accum_steps
