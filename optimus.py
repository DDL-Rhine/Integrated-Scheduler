import collections
import copy
import math
import numpy as np 

class OptimusPolicy(object):
    def __init__(self):
        pass

    def optimize(self, jobs, nodes, prev_allocations, node_template):
        from simulator import args 
        allocations = {} # {k: v for k, v in prev_allocations.items() if k in jobs}
        for job in jobs.values():
            completion_epoch = job.application.get_completion_epoch(
                    job.target_batch_size)
            if completion_epoch <= job.epoch:
                job.remaining = 1
                job.priority = self.predict_step_time(job, 1) - self.predict_step_time(job, 2) 
            else:
                # job.remaining = job.replica_lower_bound  # 
                job.remaining = job.application.get_iteration(job.target_batch_size, completion_epoch)
                job.priority = self.predict_step_time(job, 1) - self.predict_step_time(job, 2) 
                
        min_replicas = {}
        for key, job in jobs.items():
            min_replicas[key] = job.replica_lower_bound  # math.ceil(job.target_batch_size / job.application.max_local_bsz)
            assert job.replica_lower_bound
            # min_replicas[key] = 1
        
        num_gpus = sum(node.resources["nvidia.com/gpu"] for node in nodes.values())
        num_replicas = {}
        gain = {}
        
        sharingList = list() 
        inSharingList = list() 
        sharingKeys = list() 
        inSharingKeys = list() 
        
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        skipSharing = sum(min_replicas.values()) <= num_gpus
        
        if args.GPUSharing and not skipSharing:
            for key, job in sorted(jobs.items(), key=lambda item: item[1].priority):
            # for key, job in sorted(jobs.items(), key=lambda item: item[1].remaining):
                sharingKeys.append(key)
                sharingList.append(job) 
            
            pack_pairs = list() 
            gpu_util_list = [self.predidct_gpu_util(job) for job in sharingList]
            skip_between_sharing = False 
            
            for i in range(len(sharingList)):
                if sharingList[i] in inSharingList: 
                    continue
                if i % 5 != 0: continue 
                for j in range(i+1, len(sharingList)):
                    if sharingList[i].replica_lower_bound > 1: continue 
                    if sharingList[j].replica_lower_bound > 1: continue 
                    
                    
                    if gpu_util_list[i] <= 50 and gpu_util_list[j] <= 50 and sharingList[j] not in inSharingList: 
                        pack_pairs.append((sharingList[i], sharingList[j]))
                        inSharingList.append(sharingList[i])
                        inSharingList.append(sharingList[j])
                        inSharingKeys.append(sharingKeys[i])
                        inSharingKeys.append(sharingKeys[j])
                        jobs.pop(sharingKeys[i])
                        jobs.pop(sharingKeys[j])
                        min_replicas.pop(sharingKeys[i])
                        min_replicas.pop(sharingKeys[j])
                        if len(pack_pairs) + sum(min_replicas.values()) == num_gpus: 
                            skip_between_sharing = True
                        break
                    if skip_between_sharing: 
                        break 
            
            if len(pack_pairs) > 0: 
                num_gpus = max(0, num_gpus - len(pack_pairs))
                assert num_gpus >= 0 

        # # prioritize the GPU sharing operation 
        for key, job in sorted(jobs.items(), key=lambda item: item[1].remaining):
            if job in inSharingList: continue 
            if min_replicas[key] > num_gpus:
                num_replicas[key] = 0
                gain[key] = 0
                continue
            num_replicas[key] = min_replicas[key]
            num_gpus -= min_replicas[key]
        
        allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas[k]}
        job_keys = sorted(jobs, key=lambda k: num_replicas[k])
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        free_gpus = collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
        
        gain = {}        
        for key, job in jobs.items():
            if num_replicas[key] > 0: 
                delta = 1 if 'llama' not in key else 4 
                if num_replicas[key] + delta <= job.replica_upper_bound: 
                    gain[key] = (self.predict_step_time(job, num_replicas[key]) - \
                        self.predict_step_time(job, num_replicas[key]+delta)) / delta
        
        
        while num_gpus > 0 and len(gain) > 0 and max(gain.values()) > 0: 
            key = max(gain, key=lambda k: gain[k])
            job = jobs[key]
            delta = 1 if 'llama' not in key else 4 
            if num_gpus < delta: 
                gain.pop(key)
                continue 
            num_gpus -= delta 
            num_replicas[key] += delta 
            if num_replicas[key] + delta  > job.replica_upper_bound or num_gpus < delta: 
                gain[key] = 0 
            else: 
                gain[key] = (self.predict_step_time(job, num_replicas[key]) - \
                        self.predict_step_time(job, num_replicas[key]+delta)) / delta
            

        for key in job_keys:
            if key in inSharingKeys: continue
            if num_replicas[key] > 0 and not allocations.get(key):
                # Allocate resources.
                allocations[key] = []
                while len(allocations[key]) < num_replicas[key]:
                    node_idx, count = free_gpus.most_common(1)[0]
                    num = min(count, num_replicas[key] - len(allocations[key]))
                    allocations[key].extend([node_idx] * num)
                    free_gpus[node_idx] -= num
        
        import pprint 
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(free_gpus)
        
        for key, job in zip(inSharingKeys, inSharingList): 
            flag = False 
            for node_idx in free_gpus:
                if free_gpus[node_idx] > 0:  
                    allocations[key] = (node_idx, 0.5)
                    free_gpus[node_idx] -= 0.5
                    flag = True 
                    break 
                
            if not flag: 
                import pdb; pdb.set_trace() 
                    
        return allocations, -1 # len(nodes)

    def predidct_gpu_util(self, job): 
        local_bsz = math.ceil(job.target_batch_size / 1 - 1e-8)
        # gpu_util = job.application.get_gpu_util(local_bsz, job.frozen_layer)
        gpu_util = job.application.get_gpu_util(local_bsz, job.frozen_layer)
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
        
        step_time, sync_time = job.application.get_throughput(placement, atomic_bsz, 0)
        return (step_time + (step_time - sync_time) * accum_steps) # * ((job.max_progress - job.progress ) / job.target_batch_size)
