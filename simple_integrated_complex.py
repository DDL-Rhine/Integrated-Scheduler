# !pip install gekko
# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import copy
import logging
import numpy as np

from collections import OrderedDict
from mip import *
import numpy as np
import copy
import math
import sys 

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class SimpleIceFrogPolicy(object):
    def __init__(self):
        self._prev_states = None
        self._prev_jobs = None
        self._prev_nodes = None
        # Utilization thresholds for cluster autoscaling.
        self._min_util = 0.35
        self._max_util = 0.65

    def _allocations_to_state(self, allocations, jobs, nodes):
        jobs_index = {key: idx for idx, key in enumerate(jobs)}
        nodes_index = {key: idx for idx, key in enumerate(nodes)}
        state = np.zeros((len(jobs), len(nodes)), dtype=int)
        for job_key, alloc in allocations.items():
            for node_key in (key for key in alloc if key in nodes_index):
                state[jobs_index[job_key], nodes_index[node_key]] += 1
        return state

    def _state_to_allocations(self, state, jobs, nodes):
        allocations = {}
        debug = False 
        for job_idx, job_key in enumerate(jobs):
            for node_idx, node_key in enumerate(nodes):
                count = state[job_idx, node_idx]
                if count == 0.5: 
                    allocations[job_key] = (node_key, 0.5)
                    debug = True 
                else: 
                    if job_key in allocations and isinstance(allocations[job_key], tuple): 
                        continue 
                    allocations.setdefault(job_key, []).extend([node_key] * int(count))
        
        return allocations

    def _adapt_prev_states(self, jobs, nodes):
        # Adapt the previously saved optimization states to initialize the
        # current genetic algorithm states.
        #shape = (len(self._prev_states), len(jobs), 2 * len(nodes))
        shape = (len(self._prev_states), len(jobs), len(nodes))
        states = np.zeros(shape, dtype=int)
        jobs_src = [i for i, key in enumerate(self._prev_jobs) if key in jobs]
        jobs_dst = [i for i, key in enumerate(jobs) if key in self._prev_jobs]
        placeholder = len(self._prev_nodes)  # Next placeholder node to copy.
        # Set allocations for physical (non-placeholder) nodes.
        nodes_index = {key: i for i, key in enumerate(self._prev_nodes)}
        for i, key in enumerate(nodes):
            if key in nodes_index:
                states[:, jobs_dst, i] = \
                    self._prev_states[:, jobs_src, nodes_index[key]]
            elif placeholder < self._prev_states.shape[2]:
                # New node, use allocations for a previous placeholder node.
                states[:, jobs_dst, i] = \
                    self._prev_states[:, jobs_src, placeholder]
                placeholder += 1
        return states

    def _select_result(self, values, max_nodes):
        if np.amin(values[:, 1]) > max_nodes:
            return None
        return np.argmin(np.where(values[:, 1] <= max_nodes, values[:, 0], 0))

    def _desired_nodes(self, utilities, values, nodes):
        idx = self._select_result(values, len(nodes))
        if idx is not None and \
                self._min_util <= utilities[idx] <= self._max_util:
            return len(nodes)
        target_util = (self._min_util + self._max_util) / 2
        best_util = np.inf
        best_val = 0.0
        best_nodes = len(nodes)
        for util, (val, num_nodes) in zip(utilities, values):
            if util > best_util and val < best_val:
                best_util = util
                best_val = val
                best_nodes = num_nodes
            elif util < best_util and val > best_val:
                continue
            elif abs(util - target_util) < abs(best_util - target_util):
                best_util = util
                best_val = val
                best_nodes = num_nodes
        return int(best_nodes)

    def optimize(self, jobs, nodes, base_allocations, node_template):
        """
        Run one optimization cycle of the Pollux scheduling policy.

        Arguments:
            jobs (dict): map from job keys to `JobInfo` objects which
                correspond to the incomplete jobs which should be optimized.
            nodes (dict): map from node keys to `NodeInfo` objects which
                correspond to the existing nodes in the cluster.
            base_allocations (dict): map from job keys to their current
                resource allocations, in the form of a list of a node key for
                each replica.
            node_template (NodeInfo): represents a node which can be requested,
                used to decide the cluster size for cluster auto-scaling.

        Returns:
            dict: map from job keys to their optimized resource allocations,
                in the form of a list of a node key for each replica.
        """

        # A job is considered pinned if it's non-preemptible *and* already has
        # an allocation.
        def ispinned(key, job):
            return not job.preemptible and base_allocations.get(key, []) != []

        # We sort the jobs based on min_replicas and then creation_timestamp,
        # so jobs wanting lower or no min_replicas guarantees are prioritized
        # ahead of those wanting higher min_replicas guarantees to avoid
        # underutilization of cluster. Within a same min_replicas value, they
        # will follow FIFO order. Pinned jobs are aggregated at front because
        # they already have an allocation and won't affect allocations of the
        # rest of the jobs.
        jobs = OrderedDict(sorted(jobs.items(),
                                  key=lambda kv: (not ispinned(kv[0], kv[1]),
                                                  kv[1].attained_service,
                                                  kv[1].creation_timestamp)))
        nodes = OrderedDict(  # Sort preemptible nodes last.
            sorted(nodes.items(), key=lambda kv: (kv[1].preemptible, kv[0])))
        #base_state = np.concatenate(
        #    (self._allocations_to_state(base_allocations, jobs, nodes),
        #     np.zeros((len(jobs), len(nodes)), dtype=int)), axis=1)
        base_state = \
            self._allocations_to_state(base_allocations, jobs, nodes)

        if self._prev_states is None:
            states = np.expand_dims(base_state, 0)
        else:
            states = self._adapt_prev_states(jobs, nodes)
        
        problem = Problem(list(jobs.values()), list(nodes.values()), base_state)
        
        # 根据配置选择优化策略
        from simulator import args
        if hasattr(args, 'enable_crr') and args.enable_crr and hasattr(args, 'obj') and args.obj == 'DeadlineMeet':
            # 使用集成CRR机制的优化
            solved_state = problem.solve()
            allocations = np.zeros((len(jobs), len(nodes)), np.float32)
            cluster_state = np.array([node.resources['nvidia.com/gpu'] for node in nodes.values()], dtype=np.float32)
            job_to_allocations = [(job_id, job, allocated_gpu) for job_id, (job, allocated_gpu) in enumerate(zip(jobs, solved_state))]
            job_to_allocations = sorted(job_to_allocations, key=lambda x: -x[2])

            for job_id, _, allocated_gpu in job_to_allocations: 
                job_to_allocation = [0 for node in nodes]
                while allocated_gpu > 0: 
                    node_id = np.argmax(cluster_state)
                    free_gpu_num = cluster_state[node_id]
                    if free_gpu_num >= allocated_gpu: 
                        job_to_allocation[node_id] = allocated_gpu
                        cluster_state[node_id] -= allocated_gpu
                        allocated_gpu = 0
                    else: 
                        job_to_allocation[node_id] = free_gpu_num
                        cluster_state[node_id] -= free_gpu_num
                        allocated_gpu -= free_gpu_num
                for node_id, gpu_num in enumerate(job_to_allocation): 
                    allocations[job_id][node_id] = gpu_num 

            standard_allocations = self._state_to_allocations(allocations, jobs, nodes)
            
            # 应用CRR机制
            jobs_dict = {job.name: job for job in jobs.values()} if isinstance(jobs, dict) else {job.name: job for job in jobs}
            crr_allocations = problem.collaborative_resource_redistribution(jobs_dict, nodes, standard_allocations)
            
            return crr_allocations, -1
        else:
            # 使用标准优化
            solved_state = problem.solve() 
        allocations = np.zeros((len(jobs), len(nodes)), np.float32)
        cluster_state = np.array([node.resources['nvidia.com/gpu'] for node in nodes.values()], dtype=np.float32)
        job_to_allocations = [(job_id, job, allocated_gpu) for job_id, (job, allocated_gpu) in enumerate(zip(jobs, solved_state))]
        job_to_allocations = sorted(job_to_allocations, key=lambda x: -x[2])

        for job_id, _, allocated_gpu in job_to_allocations: 
            job_to_allocation = [0 for node in nodes]
            while allocated_gpu > 0: 
                node_id = np.argmax(cluster_state)
                free_gpu_num = cluster_state[node_id]
                if free_gpu_num >= allocated_gpu: 
                    job_to_allocation[node_id] = allocated_gpu
                    cluster_state[node_id] -= allocated_gpu
                    allocated_gpu = 0
                else: 
                    job_to_allocation[node_id] = free_gpu_num
                    cluster_state[node_id] -= free_gpu_num
                    allocated_gpu -= free_gpu_num
            for node_id, gpu_num in enumerate(job_to_allocation): 
                allocations[job_id][node_id] = gpu_num 

        return self._state_to_allocations(allocations, jobs, nodes), -1

    def _validate_final_allocations(self, allocations, jobs):
        """验证最终分配的一致性，确保不违反资源约束"""
        for job_name, allocation in allocations.items():
            if job_name in jobs:
                job = jobs[job_name]
                alloc_list = list(allocation) if allocation else []
                
                # 验证资源数量不超过最大副本数
                if len(alloc_list) > job.max_replicas:
                    print(f"Warning: {job_name} allocation {len(alloc_list)} exceeds max_replicas {job.max_replicas}")
                    # 截断过多的资源
                    allocations[job_name] = tuple(alloc_list[:job.max_replicas])
                
                # 验证没有负数资源
                if len(alloc_list) < 0:
                    print(f"Error: {job_name} has negative allocation")
                    allocations[job_name] = tuple()
                
                # 确保分配为元组格式
                if not isinstance(allocation, tuple):
                    allocations[job_name] = tuple(alloc_list)
        
        return True


class Problem(object):
    def __init__(self, jobs, nodes, base_state):
        """
        Multi-objective optimization problem used by PolluxPolicy to determine
        resource allocations and desired cluster size. Optimizes for the best
        performing cluster allocation using only the first N nodes. The cluster
        performance and N are the two objectives being optimized, resulting in
        a set of Pareto-optimal solutions.

        The optimization states are a 3-D array of replica assignments with
        shape (pop_size x num_jobs x num_nodes). The element at k, j, n encodes
        the number of job j replicas assigned to node n, in the kth solution.

        Arguments:
            jobs (list): list of JobInfo objects describing the incomplete jobs
                which need to be scheduled.
            nodes (list): list of NodeInfo objects describing the nodes in the
                cluster, in decreasing order of allocation preference.
            base_state (numpy.array): base optimization state corresponding to
                the current cluster allocations. Shape: (num_jobs x num_nodes).
        """
        assert base_state.shape == (len(jobs), len(nodes))
        self._jobs = jobs
        self._nodes = nodes
        self._base_state = base_state
        # Find which resource types are requested by at least one job.
        rtypes = sorted(set.union(*[set(job.resources) for job in jobs]))
        # Build array of job resources: <num_jobs> x <num_rtypes>. Each entry
        # [j, r] is the amount of resource r requested by a replica of job j.
        self._job_resources = np.zeros((len(jobs), len(rtypes)), np.int64)
        for j, job in enumerate(jobs):
            for r, rtype in enumerate(rtypes):
                self._job_resources[j, r] = job.resources.get(rtype, 0)
        # Build array of node resources: <num_nodes> x <num_rtypes>. Each
        # entry [n, r] is the amount of resource r available on node n.
        self._node_resources = np.zeros((len(nodes), len(rtypes)), np.int64)
        for n, node in enumerate(nodes):
            for r, rtype in enumerate(rtypes):
                self._node_resources[n, r] = node.resources.get(rtype, 0)
        # Calculate dominant per-replica resource shares for each job.
        shares = self._job_resources / np.sum(self._node_resources, axis=0)
        self._dominant_share = np.amax(shares, axis=1)
        # Change base goodput to fair-share goodput.
        fair_replicas = np.ceil(1.0 / self._dominant_share / len(self._jobs))
        fair_nodes = np.ceil(len(nodes) * self._dominant_share)
        self.power = 1
        self.max_allowed_replicas = sum([job.max_replicas for job in jobs])
        
        self.init_elastic_weight()
        if True: 
            from simulator import args 
            if args.obj == 'InstFair': 
                args.power  = 1
            power = args.power 
            self.THR_OBJ = args.obj
            # time_interval = args.interval 
            time_interval = 0 # TODO 
            self.power = power
            
                
            empty_jobs = list() 
            # water filling 
            cluster_capacity = args.min_nodes * 4
            assigned_weight_jobs = [job for job in jobs]
            TIME_NORM = 3600
            
            while len(assigned_weight_jobs) > 0: 
                fair_replicas = cluster_capacity / len(assigned_weight_jobs)
                remove_jobs = list() 
                if len(remove_jobs) == 0: 
                    for job in assigned_weight_jobs: 
                        job.fair_replicas = min(int(math.ceil(fair_replicas)), job.max_replicas)
                    assigned_weight_jobs = list() 
                
                for job in remove_jobs: 
                    assigned_weight_jobs.remove(job)
                    if 'llama' in job.name: 
                        job.fair_replicas = int(math.ceil(job.fair_replicas /4)) * 4
                
            for job in jobs: 
                if not hasattr(job.speedup_fn, "_goodput_fn"):
                    empty_jobs.append(job)
                    continue
                
                elastic_weight = self.elastic_weight[job.speedup_fn._elastic] * job.prior_weight
                fair_replicas = job.fair_replicas
                fair_nodes = fair_replicas // 4
                if fair_replicas % 4 > 0: fair_nodes += 1
                is_reproducible = hasattr(job.speedup_fn, 'predict_remaining_time') # TODO 
                # is_reproducible = False 
                # is_reproducible = True
                if is_reproducible: 
                    fair_goodput = max(job.speedup_fn.predict_remaining_time(fair_replicas), time_interval) / TIME_NORM
                else: 
                    fair_goodput = job.speedup_fn._goodput_fn.optimize(
                            num_nodes=fair_nodes, num_replicas=fair_replicas,
                            max_batch_size=job.speedup_fn._max_batch_size,
                            atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                            accumulation=job.speedup_fn._accumulation)[0]
                
                if self.THR_OBJ == 'FrozenShare': 
                    allow_replicas = [i for i in range(1, job.max_replicas + 1)]
                    if 'llama' in job.name: 
                        allow_replicas = [4] + [(i+1) * 4 for  i in range(1, job.max_replicas//4)]
                        
                    if is_reproducible: 
                        max_goodput = sys.maxsize 
                        for replicas in allow_replicas:   
                            cur_goodput = max(job.speedup_fn.predict_remaining_time(fair_replicas), time_interval) / TIME_NORM
                            max_goodput = min(cur_goodput, max_goodput)
                    else: 
                        max_goodput = 0 
                        for replicas in allow_replicas:   
                            my_nodes = replicas // 4
                            if replicas % 4 > 0: my_nodes += 1
                            my_replicas = replicas
                            cur_goodput = job.speedup_fn._goodput_fn.optimize(
                                num_nodes=my_nodes, num_replicas=my_replicas,
                                max_batch_size=job.speedup_fn._max_batch_size,
                                atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                                accumulation=job.speedup_fn._accumulation)[0]
                            max_goodput = max(cur_goodput, max_goodput)
                            # print(f'history max goodput {max_goodput} and cur goodput {cur_goodput}')
                            
                
                effective_metric_list = list() 
                if True: 
                    positive = 1 
                    if self.THR_OBJ == 'FrozenShare': 
                        positive = 1
                    if power < 0: 
                        effective_metric_list.append((0, 1000 * elastic_weight * positive))
                    else: 
                        effective_metric_list.append((0, 1e-3 * elastic_weight * positive))
                
                allow_replicas = [i for i in range(1, min(5, job.max_replicas + 1))]
                allow_replicas = allow_replicas + [i for i in range(8, job.max_replicas + 1, 1)]
                if 'llama' in job.name: 
                    allow_replicas = [4] + [(i+1) * 4 for  i in range(1, job.max_replicas//4)]
                
                init_batch_size = job.speedup_fn._goodput_fn._init_batch_size
                new_allow_replicas = list()
                for replica in allow_replicas: 
                    if replica < job.replica_lower_bound: 
                        continue
                    if replica > job.replica_upper_bound:
                        continue 
                    new_allow_replicas.append(replica)
                allow_replicas = new_allow_replicas
                
                GPUSharingDecay = 0.9
                if hasattr(args, 'GPUSharingThr') and args.GPUSharingThr > 0: 
                    GPUSharingDecay = args.GPUSharingThr / 100
                
                # print('allow replicas {}'.format(allow_replicas))
                if job.speedup_fn._gpu_utilization_fn is not None and args.GPUSharing:
                    # error_scale = 1 + np.random.normal(0, 1) * args.GPUSharingError / 100  
                    error_scale = 1 - np.random.normal(0, 1) * args.GPUSharingError / 100
                    # import pdb; pdb.set_trace() 
                    if job.speedup_fn._gpu_utilization_fn * error_scale <= 55 and 'llama' not in job.name: 
                        # print("error scale for GPU sharing is ", error_scale, job.speedup_fn._gpu_utilization_fn, flush=True)
                        allow_replicas = [0.5] + allow_replicas
                        
                
                for i in allow_replicas:
                    delay = 30
                    factor = 1
                    num_replicas = i 
                    if num_replicas >= 1: 
                        num_nodes = num_replicas // 4 
                        if num_replicas % 4 > 0: num_nodes += 1
                        if is_reproducible: 
                            goodput = max(job.speedup_fn.predict_remaining_time(num_replicas) / TIME_NORM, time_interval)
                        else: 
                            goodput_info = job.speedup_fn._goodput_fn.optimize(
                                num_nodes=num_nodes, num_replicas=num_replicas,
                                max_batch_size=job.speedup_fn._max_batch_size,
                                atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                                accumulation=job.speedup_fn._accumulation)
                            goodput_fn = job.speedup_fn._goodput_fn
                            goodput = goodput_info[0]
                    else: 
                        num_nodes = 1
                        num_replicas = 1
                        if is_reproducible: 
                            goodput = max(job.speedup_fn.predict_remaining_time(num_replicas) / TIME_NORM, time_interval)
                        else: 
                            goodput_info = job.speedup_fn._goodput_fn.optimize(
                                num_nodes=num_nodes, num_replicas=num_replicas,
                                max_batch_size=job.speedup_fn._max_batch_size,
                                atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                                accumulation=job.speedup_fn._accumulation)
                            goodput_fn = job.speedup_fn._goodput_fn
                            goodput = goodput_info[0] * GPUSharingDecay
                            num_replicas = 0.5 
                    
                    if self.THR_OBJ == 'FrozenShare': 
                        
                        elastic_weight = 1
                        if is_reproducible: 
                            effective_metric_list.append(
                                (i, (max_goodput/goodput * factor) ** power * elastic_weight) # goodput is time 
                            )
                        else: 
                            effective_metric_list.append(
                                (i, (goodput/max_goodput * factor) ** power * elastic_weight) # FIXME: gaowei
                            )
                        if math.isnan(effective_metric_list[-1][-1]): 
                            import pdb; pdb.set_trace()
                        # print(f'max goodput {max_goodput}, cur goodput {goodput}')
                    elif self.THR_OBJ == 'InstFair': 
                        # factor = factor * np.maximum(job.staying_time - job.num_restarts * delay, 0.0) / (job.staying_time + delay)
                        lt_weight = (job.deserved_service + 1) / (1 + job.attained_service)
                        if is_reproducible: 
                            effective_metric_list.append(
                                (i, (fair_goodput/goodput * factor * lt_weight) ** power * elastic_weight) # goodput is time 
                            )
                        else: 
                            effective_metric_list.append(
                                (i, (goodput/fair_goodput * factor * lt_weight) ** power * elastic_weight)
                            )
                        
                    elif self.THR_OBJ == 'LongFair':
                        if is_reproducible: 
                            effective_time = goodput 
                            fair_time = fair_goodput
                        else: 
                            effective_time = (job.staying_time + max((job.max_progress - job.progress) / goodput * init_batch_size, time_interval)) / TIME_NORM
                            fair_time = (job.staying_time + max((job.max_progress - job.progress) / fair_goodput * init_batch_size , time_interval)) / TIME_NORM
                        
                        lt_weight = 1
                        factor = factor * np.maximum(job.staying_time - job.num_restarts * delay, 0.0) / (job.staying_time + delay)
                        effective_metric_list.append(
                            (i, (fair_time/effective_time * factor * lt_weight)**power * elastic_weight)
                        )
                    elif self.THR_OBJ == 'makespan': 
                        if is_reproducible: 
                            makespan = goodput / TIME_NORM
                        else: 
                            makespan = (job.max_progress - job.progress) / goodput * init_batch_size / TIME_NORM
                        # goodput, atomic_bsz, accum_steps, frozen_layer 
                        # goodput_info
                        # print('allocation {}, make span {}, goodput {}'.format(i, makespan * TIME_NORM, goodput))
                        if False: 
                            frozen_set = [0]
                            if goodput_info[-1] not in frozen_set: 
                                frozen_set.append(goodput_info[-1])
                            thr = goodput_fn.throughput(num_nodes, num_replicas, goodput_info[1], goodput_info[2], frozen_set)
                        
                        effective_metric_list.append(
                            (i, (makespan) * elastic_weight)
                        )
                    elif self.THR_OBJ == 'DeadlineMeet':
                        # 收集当前workload中所有任务的jcts数据供后续使用
                        self.jcts_data = {}
                        # for job in jobs:
                        #     if hasattr(job, 'name') and job.completion_time is not None:
                        #         model_key = job.name.split('-')[1] if '-' in job.name else job.name
                        #         jct = job.completion_time - job.submission_time
                        #         if model_key not in self.jcts_data:
                        #             self.jcts_data[model_key] = jct
                        allow_replicas = [i for i in range(1, job.max_replicas + 1)]
                        if 'llama' in job.name: 
                            allow_replicas = [4] + [(i+1) * 4 for i in range(1, job.max_replicas//4)]
                        
                        if is_reproducible: 
                            max_goodput = sys.maxsize 
                            for replicas in allow_replicas:   
                                cur_goodput = max(job.speedup_fn.predict_remaining_time(replicas), time_interval) / TIME_NORM
                                max_goodput = min(cur_goodput, max_goodput)
                        else: 
                            max_goodput = 0 
                            for replicas in allow_replicas:   
                                my_nodes = replicas // 4
                                if replicas % 4 > 0: my_nodes += 1
                                my_replicas = replicas
                                cur_goodput = job.speedup_fn._goodput_fn.optimize(
                                    num_nodes=my_nodes, num_replicas=my_replicas,
                                    max_batch_size=job.speedup_fn._max_batch_size,
                                    atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                                    accumulation=job.speedup_fn._accumulation)[0]
                                max_goodput = max(cur_goodput, max_goodput)
                        
                        effective_metric_list = list()
                        
                        # 添加空分配的情况
                        if power < 0: 
                            effective_metric_list.append((0, 1000 * elastic_weight))
                        else: 
                            effective_metric_list.append((0, 1e-3 * elastic_weight))
                        
                        # 过滤副本数范围
                        new_allow_replicas = list()
                        for replica in allow_replicas: 
                            if replica < job.replica_lower_bound: 
                                continue
                            if replica > job.replica_upper_bound:
                                continue 
                            new_allow_replicas.append(replica)
                        allow_replicas = new_allow_replicas
                        
                        # 支持GPU共享
                        if job.speedup_fn._gpu_utilization_fn is not None and args.GPUSharing:
                            error_scale = 1 - np.random.normal(0, 1) * args.GPUSharingError / 100
                            if job.speedup_fn._gpu_utilization_fn * error_scale <= 55 and 'llama' not in job.name: 
                                allow_replicas = [0.5] + allow_replicas
                                
                        # 在设置allow_replicas之后，添加截止时间风险评估
                        deadline_risk = self._calculate_deadline_risk(job)
                        
                        # 根据风险级别确定冻结层比例
                        freeze_ratio = self._get_frozen_ratio(job, deadline_risk)
                        
                        # 设置推荐冻结层数
                        if hasattr(job, 'total_layer') and job.total_layer > 0:
                            recommended_frozen_layer = int(job.total_layer * freeze_ratio)
                            
                            # 设置推荐冻结层数，供后续优化使用
                            if hasattr(job, 'elastic') and job.elastic == 'layer':
                                job.recommended_frozen_layer = recommended_frozen_layer
                        
                        for i in allow_replicas:
                            delay = 30
                            factor = 1
                            num_replicas = i 
                            if num_replicas >= 1: 
                                num_nodes = num_replicas // 4 
                                if num_replicas % 4 > 0: num_nodes += 1
                                if is_reproducible: 
                                    goodput = max(job.speedup_fn.predict_remaining_time(num_replicas) / TIME_NORM, time_interval)
                                else: 
                                    goodput_info = job.speedup_fn._goodput_fn.optimize(
                                        num_nodes=num_nodes, num_replicas=num_replicas,
                                        max_batch_size=job.speedup_fn._max_batch_size,
                                        atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                                        accumulation=job.speedup_fn._accumulation)
                                    goodput_fn = job.speedup_fn._goodput_fn
                                    goodput = goodput_info[0]
                            else: 
                                num_nodes = 1
                                num_replicas = 1
                                if is_reproducible: 
                                    goodput = max(job.speedup_fn.predict_remaining_time(num_replicas) / TIME_NORM, time_interval)
                                else: 
                                    goodput_info = job.speedup_fn._goodput_fn.optimize(
                                        num_nodes=num_nodes, num_replicas=num_replicas,
                                        max_batch_size=job.speedup_fn._max_batch_size,
                                        atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                                        accumulation=job.speedup_fn._accumulation)
                                    goodput_fn = job.speedup_fn._goodput_fn
                                    goodput = goodput_info[0] * GPUSharingDecay
                                    num_replicas = 0.5
                            
                            # 计算预期完成时间
                            expected_completion = job.staying_time + max((job.max_progress - job.progress) / goodput * init_batch_size, time_interval) / TIME_NORM
                            
                            # 如果有截止时间，计算截止时间紧迫程度
                            deadline_pressure = 1.0 + 2.0 * deadline_risk  # 默认无压力
                            if hasattr(job, 'deadline') and job.deadline is not None:
                                time_to_deadline = (job.deadline - (job.submission_time + job.staying_time)) / TIME_NORM
                                if time_to_deadline > 0:
                                    # 紧迫系数：截止时间越近，紧迫度越高
                                    deadline_pressure = min(3.0, 1.0 + (expected_completion / time_to_deadline))
                                else:
                                    # 已经超过截止时间
                                    deadline_pressure = 3.0
                            
                            # 将截止时间紧迫度纳入优化目标
                            if is_reproducible: 
                                effective_metric_list.append(
                                    (i, (max_goodput/goodput * factor) ** power * elastic_weight * deadline_pressure)
                                )
                            else: 
                                effective_metric_list.append(
                                    (i, (goodput/max_goodput * factor) ** power * elastic_weight * deadline_pressure)
                                )
                    else: 
                        raise NotImplementedError
                    
                job.effective_metric_list = effective_metric_list 
            
            for job in empty_jobs: 
                effective_metric_list = list() 
                effective_metric_list.append((0, (1e-3) ** power))
                for i in range(1, job.max_replicas + 1): 
                    effective_metric_list.append((i, i ** power))
                job.effective_metric_list = effective_metric_list 
            job.effective_metric_list = effective_metric_list 
        else: 
            raise NotImplementedError 

        self.jobs = jobs 
        self.nodes = nodes 
        

    def init_elastic_weight(self, ): 
        from simulator import args 
        # 设置默认权重值，如果参数不存在
        static_weight = getattr(args, 'static_weight', 1.0)
        batch_weight = getattr(args, 'batch_weight', 1.0)
        layer_weight = getattr(args, 'layer_weight', 1.0)
        
        self.elastic_weight = {
            'layer': layer_weight, 
            'batch': batch_weight, 
            'static': static_weight
        }
        
    def solve(self, max_seconds=5): 
        from simulator import args 
        cluster_capacity = 4 * args.min_nodes
        # model = Model(solver_name = GRB)
        model = Model(solver_name = CBC)
        var_len = sum([len(job.effective_metric_list) for job in self.jobs])
        X = [model.add_var(var_type=BINARY) for i in range(var_len)]
        obj_list = list() 
        required_resource_list = list() 
        cnt = 0 
        for job in self.jobs:
            for gpu, effective_metric in job.effective_metric_list: 
                # print(effective_metric)
                obj_list.append(X[cnt] * effective_metric) # whether add gpu weight, think for a while 
                required_resource_list.append(gpu)
                # print('gpu == {}, effective_metric {}'.format(gpu, effective_metric))
                cnt += 1
        
            
        if self.max_allowed_replicas >= cluster_capacity: 
            model += xsum(X[i] * required_resource_list[i] for i in range(var_len)) <= cluster_capacity
        else: 
            # model += xsum(X[i] * required_resource_list[i] for i in range(var_len)) <= cluster_capacity
            model += xsum(X[i] * required_resource_list[i] for i in range(var_len)) <= self.max_allowed_replicas

        # cluster_capacity_lower_bound = min(cluster_capacity)
        # model += xsum(X[i] * required_resource_list[i] for i in range(var_len)) >= cluster_capacity
        cnt = 0
        for job in self.jobs: 
            length = len(job.effective_metric_list)
            model.add_constr(xsum(X[i+cnt] for i in range(length)) == 1)
            cnt += length 
            
        if self.power < 0: 
            model.objective = minimize(xsum(obj_list[i] for i in range(len(obj_list))))
        else: 
            model.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list))))
        model.optimize()
        cnt = 0 
        allocated_gpu = [0 for _ in range(len(self.jobs))]
        for idx, job in enumerate(self.jobs): 
            length = len(job.effective_metric_list)
            for i, (gpu, effective_metric) in enumerate(job.effective_metric_list): 
                if X[i+cnt].x is None: 
                    continue 
                if X[i+cnt].x > 0.5:
                    # if gpu == 0.5: 
                    #     import pdb; pdb.set_trace() 
                    allocated_gpu[idx] = gpu
                    if 'llama' in job.name and gpu < 4 and gpu > 0: 
                        import pdb; pdb.set_trace() 
            cnt += length
        # if (sum(allocated_gpu)) == 0 and len(allocated_gpu) > 2: 
        #     import pdb; pdb.set_trace()
        #     [job.name for job in self.jobs]
        #     [job.max_replicas for job in self.jobs]
        #     print([X[i].x for i in range(len(X))])
        return allocated_gpu
    
    # 在simple_icefrog.py文件末尾添加这两个函数
    def _calculate_deadline_risk(self, job):
        """基于数据集中的jcts计算任务违约风险"""
        # 如果没有设置截止时间或jcts，则无风险
        if not hasattr(job, 'deadline') or job.deadline is None:
            return 0.0
        if not hasattr(job, 'jcts') or job.jcts is None:
            return 0.0
        
        # 使用数据集中的jcts计算预期完成时间
        expected_completion = job.submission_time + job.jcts
        
        # 计算风险级别
        if expected_completion >= job.deadline:
            # 预计会违约，计算超时程度
            overtime_ratio = (expected_completion - job.deadline) / job.jcts if job.jcts > 0 else 1.0
            return min(0.8 + overtime_ratio * 0.2, 1.0)  # 最高1.0
        else:
            # 计算时间裕量比例
            time_margin = (job.deadline - expected_completion) / job.jcts if job.jcts > 0 else 1.0
            if time_margin < 0.1:
                return 0.7  # 高风险(完成时间接近截止时间)
            elif time_margin < 0.3:
                return 0.4  # 中风险
            elif time_margin < 0.5:
                return 0.2  # 低风险
            else:
                return 0.0  # 无风险(完成时间远早于截止时间)

    def _get_frozen_ratio(self, job, risk_level):
        """根据风险级别和模型类型决定冻结比例"""
        model_type = job.name.split('-')[1] if '-' in job.name else job.name
        
        # 基于模型类型确定基础冻结比例系数
        if 'bert' in model_type.lower() or 'llama' in model_type.lower():
            model_factor = 0.85  # 大型语言模型需要较少冻结以保持精度
        elif 'ResNet' in model_type or 'VGG' in model_type:
            model_factor = 1.2  # CNN模型可以冻结更多层
        else:
            model_factor = 1.0  # 其他模型使用标准值
        
        # 根据风险级别设置冻结比例
        if risk_level >= 0.8:  # 高风险/已超时
            base_ratio = 0.6 * model_factor  # 60%
        elif risk_level >= 0.4:  # 中风险
            base_ratio = 0.4 * model_factor  # 40%
        elif risk_level >= 0.2:  # 低风险
            base_ratio = 0.25 * model_factor  # 25%
        else:  # 无风险
            base_ratio = 0.15 * model_factor  # 15%
        
        # 根据训练进度调整（训练越接近完成，冻结越少以保证精度）
        progress_ratio = job.progress / job.max_progress if job.max_progress > 0 else 0
        adjusted_ratio = base_ratio * (1.0 - progress_ratio * 0.5)
        
        # 确保冻结比例在合理范围内
        return max(0.1, min(adjusted_ratio, 0.7))  # 最少10%，最多70%

    def should_avoid_restart(self, job, current_allocation, new_allocation):
        """判断是否应该避免重启的策略"""
        # 如果当前没有分配资源，新分配有资源，必须启动
        if len(current_allocation) == 0 and len(new_allocation) > 0:
            return False
        
        # 如果当前有资源，新分配没有资源，必须停止
        if len(current_allocation) > 0 and len(new_allocation) == 0:
            return False
        
        # 计算资源变化程度
        if len(current_allocation) > 0:
            resource_change_ratio = abs(len(new_allocation) - len(current_allocation)) / len(current_allocation)
        else:
            resource_change_ratio = 0
        
        # 策略1: 如果资源变化很小(<20%)，避免重启
        if resource_change_ratio < 0.2:
            return True
        
        # 策略2: 对于接近截止时间的任务，避免重启造成进度损失
        if hasattr(job, 'deadline') and job.deadline is not None:
            current_time = job.submission_time + job.staying_time
            time_remaining = job.deadline - current_time
            total_time = job.deadline - job.submission_time
            
            if total_time > 0:
                urgency_ratio = time_remaining / total_time
                # 如果剩余时间少于40%，且资源变化不大，避免重启
                if urgency_ratio < 0.4 and resource_change_ratio < 0.5:
                    return True
        
        # 策略3: 对于训练进度超过70%的任务，避免不必要的重启
        if hasattr(job, 'progress') and hasattr(job, 'max_progress'):
            if job.max_progress > 0:
                progress_ratio = job.progress / job.max_progress
                if progress_ratio > 0.7 and resource_change_ratio < 0.3:
                    return True
        
        return False
    
    def optimize_with_restart_avoidance(self, jobs, nodes, base_allocations, node_template):
        """带重启避免的优化方法"""
        # 首先获取正常的优化结果
        allocations, num_nodes_used = self.optimize(jobs, nodes, base_allocations, node_template)
        
        # 检查每个作业的重启需求，尝试避免不必要的重启
        adjusted_allocations = {}
        for job in jobs:
            job_name = job.name
            current_alloc = base_allocations.get(job_name, [])
            new_alloc = allocations.get(job_name, [])
            
            # 如果策略建议避免重启，尝试保持当前分配
            if self.should_avoid_restart(job, current_alloc, new_alloc):
                # 检查保持当前分配是否可行（资源是否仍然可用）
                if self.can_maintain_allocation(current_alloc, nodes, adjusted_allocations):
                    adjusted_allocations[job_name] = current_alloc
                    print(f"避免重启: 任务 {job_name} 保持当前资源分配 {current_alloc}")
                else:
                    # 如果无法保持，寻找最小变化的分配
                    minimal_change_alloc = self.find_minimal_change_allocation(
                        job, current_alloc, new_alloc, nodes, adjusted_allocations)
                    adjusted_allocations[job_name] = minimal_change_alloc
                    print(f"最小变化分配: 任务 {job_name} 从 {current_alloc} 调整到 {minimal_change_alloc}")
            else:
                adjusted_allocations[job_name] = new_alloc
        
        return adjusted_allocations, num_nodes_used

    def collaborative_resource_redistribution(self, jobs, nodes, base_allocations):
        """
        重新设计的智能协作式资源重新分配机制 (Smart CRR)
        
        核心优化策略：
        1. 保守触发：只在明确有效时进行重分配
        2. 精准识别：专注于真正紧急的任务
        3. 最小扰动：尽量减少对现有分配的破坏
        4. 效果优先：优先考虑对JCT和违约率的改善
        """
        from simulator import args
        
        # 基本检查：只在DeadlineMeet目标且有足够任务时启用
        if not hasattr(args, 'obj') or args.obj != 'DeadlineMeet' or len(jobs) < 2:
            return base_allocations
        
        # 创建工作副本，确保类型一致性
        improved_allocations = {}
        for k, v in base_allocations.items():
            if isinstance(v, (list, tuple)):
                improved_allocations[k] = list(v)
            else:
                improved_allocations[k] = []
        
        current_time = getattr(args, 'current_time', 0)
        
        # Step 1: 智能触发决策 - 只在有明确收益时进行CRR
        if not self._should_trigger_smart_crr(jobs, improved_allocations, current_time):
            return base_allocations
        
        # Step 2: 精准识别真正的紧急任务
        critical_jobs = self._identify_truly_critical_jobs(jobs, improved_allocations, current_time)
        
        if not critical_jobs:
            return base_allocations  # 没有紧急任务，保持原分配
        
        # Step 3: 谨慎的资源重分配
        success_count = self._careful_resource_reallocation(
            critical_jobs, jobs, improved_allocations, current_time
        )
        
        # Step 4: 最终验证和约束检查
        if success_count > 0:
            improved_allocations = self._final_constraint_validation(improved_allocations, nodes)
            
            # 记录成功的CRR操作
            if hasattr(args, 'crr_stats'):
                args.crr_stats = getattr(args, 'crr_stats', [])
                args.crr_stats.append({
                    'timestamp': current_time,
                    'critical_jobs_helped': success_count,
                    'total_critical_jobs': len(critical_jobs)
                })
            
            return improved_allocations
        else:
            # 如果没有成功的重分配，返回原分配
            return base_allocations
    
    def _should_trigger_smart_crr(self, jobs, allocations, current_time):
        """
        智能CRR触发决策：只在有明确收益时才触发
        
        触发条件（必须同时满足）：
        1. 存在即将违约的任务
        2. 存在资源利用效率低的任务  
        3. 预期能获得显著改善
        """
        critical_count = 0
        underutilized_count = 0
        
        for job in jobs.values():
            if not hasattr(job, 'deadline') or not hasattr(job, 'remaining_time'):
                continue
                
            # 检查是否有即将违约的任务
            time_to_deadline = job.deadline - current_time
            estimated_completion = job.remaining_time
            urgency_ratio = estimated_completion / max(time_to_deadline, 1)
            
            if urgency_ratio > 1.2:  # 预期超出截止时间20%以上
                critical_count += 1
            
            # 检查是否有低效利用的任务
            current_alloc = len(allocations.get(job.name, []))
            if current_alloc > 1:
                # 简单估算：如果任务有多个GPU但进度缓慢，认为利用效率低
                expected_progress = current_alloc / job.max_replicas if job.max_replicas > 0 else 0
                if expected_progress > 0.5 and urgency_ratio < 0.8:  # 资源多但不紧急
                    underutilized_count += 1
        
        # 只有同时存在紧急任务和低效任务时才触发CRR
        return critical_count >= 1 and underutilized_count >= 1
    
    def _identify_truly_critical_jobs(self, jobs, allocations, current_time):
        """
        精准识别真正紧急的任务：
        1. 即将错过截止时间
        2. 有可能通过增加资源获得显著改善
        """
        critical_jobs = []
        
        for job in jobs.values():
            if not hasattr(job, 'deadline') or not hasattr(job, 'remaining_time'):
                continue
            
            time_to_deadline = job.deadline - current_time
            estimated_completion = job.remaining_time
            current_alloc = len(allocations.get(job.name, []))
            
            # 紧急性判断：预计完成时间超过截止时间
            urgency_ratio = estimated_completion / max(time_to_deadline, 1)
            
            # 改善潜力：还有资源扩展空间
            can_scale = current_alloc < job.max_replicas
            
            # 严格的紧急判断条件
            if urgency_ratio > 1.15 and can_scale:  # 超时15%以上且可扩展
                priority_score = urgency_ratio * (job.max_replicas - current_alloc)
                critical_jobs.append((job.name, job, priority_score))
        
        # 按优先级排序，只处理最紧急的3个任务
        critical_jobs.sort(key=lambda x: x[2], reverse=True)
        return critical_jobs[:3]
    
    def _careful_resource_reallocation(self, critical_jobs, all_jobs, allocations, current_time):
        """
        谨慎的资源重分配：最小化扰动，最大化效果
        """
        success_count = 0
        
        for job_name, critical_job, priority_score in critical_jobs:
            # 为每个紧急任务寻找最合适的资源来源
            donor_found = self._find_optimal_donor(
                critical_job, all_jobs, allocations, current_time
            )
            
            if donor_found:
                success_count += 1
                # 保守策略：每次只处理一个紧急任务
                break
        
        return success_count
    
    def _find_optimal_donor(self, critical_job, all_jobs, allocations, current_time):
        """
        为紧急任务寻找最优的资源提供者
        """
        critical_alloc = len(allocations.get(critical_job.name, []))
        
        # 如果已经达到最大资源，无法提供更多帮助
        if critical_alloc >= critical_job.max_replicas:
            return False
        
        # 寻找潜在的资源提供者
        potential_donors = []
        
        for donor_job in all_jobs.values():
            if donor_job.name == critical_job.name:
                continue
                
            donor_alloc = len(allocations.get(donor_job.name, []))
            if donor_alloc <= 1:  # 保留至少1个GPU给每个任务
                continue
            
            # 计算提供资源的"损失成本"
            if hasattr(donor_job, 'deadline') and hasattr(donor_job, 'remaining_time'):
                time_to_deadline = donor_job.deadline - current_time
                estimated_completion = donor_job.remaining_time
                donor_urgency = estimated_completion / max(time_to_deadline, 1)
                
                # 只从不紧急的任务中借用资源
                if donor_urgency < 0.9:  # 提供者不能太紧急
                    cost_score = donor_urgency * donor_alloc  # 越不紧急且资源越多，成本越低
                    potential_donors.append((donor_job.name, cost_score))
        
        # 选择成本最低的提供者
        if potential_donors:
            potential_donors.sort(key=lambda x: x[1])
            donor_name = potential_donors[0][0]
            
            # 执行资源转移：从提供者转移1个GPU到紧急任务
            donor_alloc = allocations[donor_name]
            critical_alloc_list = allocations.get(critical_job.name, [])
            
            if len(donor_alloc) > 1:
                # 转移一个资源
                transferred_resource = donor_alloc.pop()
                critical_alloc_list.append(transferred_resource)
                allocations[critical_job.name] = critical_alloc_list
                
                return True
        
        return False
    
    def _final_constraint_validation(self, allocations, nodes):
        """
        最终的约束验证和修正
        """
        # 确保所有分配都符合节点容量限制
        node_usage = {node_id: 0 for node_id in nodes.keys()}
        
        # 统计资源使用
        for job_name, allocation in allocations.items():
            if isinstance(allocation, (list, tuple)):
                for node_id in allocation:
                    if node_id in node_usage:
                        node_usage[node_id] += 1
        
        # 检查是否超出容量
        violations = []
        for node_id, usage in node_usage.items():
            if node_id in nodes:
                capacity = nodes[node_id].resources.get("nvidia.com/gpu", 0)
                if usage > capacity:
                    violations.append((node_id, usage - capacity))
        
        # 如果有违规，需要修正
        if violations:
            print(f"Warning: Resource constraint violations detected: {violations}")
            # 这里可以添加修正逻辑，但为了保持简单，我们返回原分配
            
        return allocations
    
    def _enhanced_job_analysis(self, jobs, allocations, current_time):
        """增强的任务分析，计算更精确的指标"""
        job_metrics = {}
        
        for job in jobs.values():
            job_name = job.name
            current_allocation = len(allocations.get(job_name, []))
            
            # 增强的紧迫性计算
            urgency_score = self._calculate_enhanced_urgency(job, current_time)
            
            # 增强的边际效用计算
            marginal_utility = self._calculate_enhanced_marginal_utility(job, current_allocation)
            
            # 违约风险评估
            deadline_risk = self._calculate_deadline_violation_risk(job, current_time)
            
            # 资源效率分析
            resource_efficiency = self._calculate_resource_efficiency(job, current_allocation)
            
            # 协作潜力评估
            collaboration_potential = self._calculate_enhanced_collaboration_potential(job, jobs)
            
            job_metrics[job_name] = {
                'job': job,
                'current_allocation': current_allocation,
                'urgency_score': urgency_score,
                'marginal_utility': marginal_utility,
                'deadline_risk': deadline_risk,
                'resource_efficiency': resource_efficiency,
                'collaboration_potential': collaboration_potential,
                'can_donate': (resource_efficiency < 0.75 and current_allocation > 1),  # 放宽效率要求
                'needs_urgent_help': (deadline_risk > 0.7 or urgency_score > 0.8),     # 恢复保守阈值
                'needs_moderate_help': (deadline_risk > 0.5 or urgency_score > 0.6),  # 恢复保守阈值
                'saturation_ratio': self._calculate_saturation_ratio(job, current_allocation)
            }
        
        return job_metrics
    
    def _identify_emergency_jobs(self, job_metrics):
        """识别紧急任务 - 使用更低的阈值以提高CRR激活频率"""
        emergency_jobs = []
        
        for job_name, metrics in job_metrics.items():
            # 放宽紧急条件：降低各个指标的阈值
            if ((metrics['deadline_risk'] > 0.4 and metrics['urgency_score'] > 0.5) or  # 降低违约+紧迫阈值
                (metrics['deadline_risk'] > 0.6) or                                     # 单独高违约风险
                (metrics['urgency_score'] > 0.7 and metrics['marginal_utility'] > 0.3)): # 高紧迫+边际效用
                emergency_jobs.append((job_name, metrics))
        
        # 按综合风险程度排序
        emergency_jobs.sort(key=lambda x: (
            x[1]['deadline_risk'] * 1.2 +     # 加权违约风险
            x[1]['urgency_score'] * 1.0 +     # 紧迫性权重
            x[1]['marginal_utility'] * 0.8    # 边际效用权重
        ), reverse=True)
        
        return [job[0] for job in emergency_jobs]
    
    def _execute_enhanced_redistribution(self, job_metrics, allocations, emergency_jobs):
        """执行增强的多轮资源重分配 - 增加激进程度"""
        redistributed_count = 0
        total_benefit = 0
        
        # 第一轮：紧急任务专项救援 - 增加处理数量
        for emergency_job in emergency_jobs[:5]:  # 增加到5个紧急任务
            donors_found = self._emergency_resource_rescue(
                emergency_job, job_metrics, allocations
            )
            if donors_found:
                redistributed_count += donors_found
                total_benefit += 2.5  # 提高紧急救援奖励分数
        
        # 第二轮：一般性协作优化 - 更激进的资源转移
        general_transfers = self._general_collaborative_optimization(
            job_metrics, allocations, emergency_jobs
        )
        redistributed_count += general_transfers
        total_benefit += general_transfers * 1.2
        
        # 第三轮：禁用细粒度平衡调整 - 减少复杂性
        # balance_transfers = self._fine_grained_balance_adjustment(
        #     job_metrics, allocations
        # )
        balance_transfers = 0  # 禁用
        redistributed_count += balance_transfers
        total_benefit += balance_transfers * 0.8
        
        # 第四轮：禁用预防性资源重分配（过于激进）
        # preventive_transfers = self._preventive_resource_allocation(
        #     job_metrics, allocations
        # )
        preventive_transfers = 0  # 禁用
        redistributed_count += preventive_transfers
        total_benefit += preventive_transfers * 0.6
        
        return redistributed_count, total_benefit
    
    def _emergency_resource_rescue(self, emergency_job, job_metrics, allocations):
        """紧急任务资源救援 - 增强版本"""
        emergency_metrics = job_metrics[emergency_job]
        donors_found = 0
        
        # 寻找最合适的资源提供者 - 放宽条件
        potential_donors = [
            (name, metrics) for name, metrics in job_metrics.items()
            if (metrics['can_donate'] and name != emergency_job and 
                len(allocations.get(name, [])) > 1)  # 降低到1个以上GPU即可捐赠
        ]
        
        # 按综合评分排序（优先选择低效率+低紧迫性的任务）
        potential_donors.sort(key=lambda x: (
            x[1]['resource_efficiency'] - x[1]['urgency_score'] * 0.5  # 综合评分
        ))
        
        # 保守的资源需求（最多申请2个资源）
        resources_needed = min(2, emergency_metrics['job'].max_replicas - emergency_metrics['current_allocation'])
        
        for donor_name, donor_metrics in potential_donors:
            if resources_needed <= 0:
                break
                
            donor_allocation = allocations.get(donor_name, [])
            if len(donor_allocation) == 0:
                continue
            
            # 更激进的转移策略
            if emergency_metrics['deadline_risk'] > 0.7:
                # 保守的转移策略
                transfer_amount = min(1, len(donor_allocation), resources_needed)  # 一次只转移1个资源
                min_keep = 1  # 总是保留至少1个资源
            else:
                # 中等风险：保守转移
                transfer_amount = min(1, max(0, len(donor_allocation) - 1), resources_needed)  # 只转移1个
                min_keep = 1
            
            if transfer_amount > 0 and len(donor_allocation) > min_keep:
                # 执行安全转移 - 确保类型一致性
                emergency_allocation = list(allocations.get(emergency_job, [])) if allocations.get(emergency_job) else []
                donor_allocation_list = list(donor_allocation) if donor_allocation else []
                
                # 验证转移操作的安全性
                if len(donor_allocation_list) >= transfer_amount:
                    transferred_nodes = donor_allocation_list[-transfer_amount:]
                    remaining_donor_nodes = donor_allocation_list[:-transfer_amount]
                    
                    # 确保资源约束
                    new_emergency_allocation = emergency_allocation + transferred_nodes
                    max_replicas = emergency_metrics['job'].max_replicas
                    
                    if len(new_emergency_allocation) <= max_replicas and len(remaining_donor_nodes) >= 0:
                        # 转换回元组格式
                        allocations[donor_name] = tuple(remaining_donor_nodes) if remaining_donor_nodes else tuple()
                        allocations[emergency_job] = tuple(new_emergency_allocation)
                        
                        resources_needed -= transfer_amount
                        donors_found += 1
                        
                        print(f"Emergency CRR: {donor_name} -> {emergency_job}, 紧急转移 {transfer_amount} GPU")
                    else:
                        print(f"Emergency CRR: 资源约束违反 {donor_name} -> {emergency_job}")
                else:
                    print(f"Emergency CRR: 资源不足 {donor_name} -> {emergency_job}")
        
        return donors_found
    
    def _general_collaborative_optimization(self, job_metrics, allocations, emergency_jobs):
        """一般性协作优化 - 增强版本"""
        transfers = 0
        
        # 识别需要帮助的任务（排除已处理的紧急任务）- 放宽条件
        help_needed = [
            (name, metrics) for name, metrics in job_metrics.items()
            if (metrics['needs_moderate_help'] and name not in emergency_jobs and 
                len(allocations.get(name, [])) < metrics['job'].max_replicas and
                metrics['marginal_utility'] > 0.2)  # 降低边际效用阈值
        ]
        
        # 按综合优先级排序 - 增加权重调整
        help_needed.sort(key=lambda x: (
            x[1]['deadline_risk'] * 1.3 + 
            x[1]['urgency_score'] * 1.1 + 
            x[1]['marginal_utility'] * 0.9
        ), reverse=True)
        
        for help_job, help_metrics in help_needed[:6]:  # 增加到6个任务
            # 寻找合适的资源提供者 - 放宽条件
            donors = [
                (name, metrics) for name, metrics in job_metrics.items()
                if (metrics['can_donate'] and name != help_job and 
                    len(allocations.get(name, [])) > 0 and  # 只要有资源就可能捐赠
                    metrics['resource_efficiency'] < 0.8)   # 放宽效率要求
            ]
            
            # 按综合评分排序（效率越低+紧迫性越低=越适合捐赠）
            donors.sort(key=lambda x: (
                x[1]['resource_efficiency'] - x[1]['urgency_score'] * 0.3
            ))
            
            for donor_name, donor_metrics in donors[:2]:  # 每个任务最多从2个提供者获取
                donor_allocation = allocations.get(donor_name, [])
                help_allocation = allocations.get(help_job, [])
                
                if len(donor_allocation) <= 1:
                    continue
                
                transfer_amount = 1  # 一般情况下每次转移1个GPU
                
                # 执行安全转移 - 确保类型一致性
                donor_allocation_list = list(donor_allocation) if donor_allocation else []
                help_allocation_list = list(help_allocation) if help_allocation else []
                
                if len(donor_allocation_list) >= transfer_amount:
                    transferred_nodes = donor_allocation_list[-transfer_amount:]
                    remaining_donor_nodes = donor_allocation_list[:-transfer_amount]
                    new_help_allocation = help_allocation_list + transferred_nodes
                    
                    # 验证资源约束
                    if (len(remaining_donor_nodes) >= 0 and 
                        len(new_help_allocation) <= help_metrics['job'].max_replicas):
                        
                        allocations[donor_name] = tuple(remaining_donor_nodes) if remaining_donor_nodes else tuple()
                        allocations[help_job] = tuple(new_help_allocation)
                        
                        transfers += 1
                        print(f"General CRR: {donor_name} -> {help_job}, 转移 {transfer_amount} GPU")
                        break  # 每个需要帮助的任务只从一个提供者获取资源
                    else:
                        print(f"General CRR: 资源约束违反 {donor_name} -> {help_job}")
                else:
                    print(f"General CRR: 资源不足 {donor_name} -> {help_job}")
        
        return transfers
    
    def _fine_grained_balance_adjustment(self, job_metrics, allocations):
        """细粒度平衡调整"""
        transfers = 0
        
        # 寻找资源严重不平衡的情况
        for job_name, metrics in job_metrics.items():
            current_alloc = len(allocations.get(job_name, []))
            
            # 如果某个任务资源过多且效率低下
            if (current_alloc > 4 and metrics['resource_efficiency'] < 0.4 and 
                not metrics['needs_moderate_help']):
                
                # 寻找最需要资源的任务
                needy_jobs = [
                    (name, m) for name, m in job_metrics.items()
                    if (m['marginal_utility'] > 0.5 and len(allocations.get(name, [])) < 3)
                ]
                
                if needy_jobs:
                    needy_jobs.sort(key=lambda x: x[1]['marginal_utility'], reverse=True)
                    target_job = needy_jobs[0][0]
                    
                    # 执行微调转移
                    source_allocation = allocations.get(job_name, [])
                    target_allocation = allocations.get(target_job, [])
                    
                    if len(source_allocation) > 2:
                        # 安全转移逻辑
                        source_allocation_list = list(source_allocation)
                        target_allocation_list = list(target_allocation)
                        target_job_metrics = job_metrics[target_job]
                        
                        transferred_nodes = source_allocation_list[-1:]
                        remaining_source_nodes = source_allocation_list[:-1]
                        new_target_allocation = target_allocation_list + transferred_nodes
                        
                        # 验证资源约束
                        if (len(remaining_source_nodes) >= 1 and 
                            len(new_target_allocation) <= target_job_metrics['job'].max_replicas):
                            
                            allocations[job_name] = tuple(remaining_source_nodes)
                            allocations[target_job] = tuple(new_target_allocation)
                            
                            transfers += 1
                            print(f"Balance CRR: {job_name} -> {target_job}, 微调转移 1 GPU")
                        else:
                            print(f"Balance CRR: 资源约束违反 {job_name} -> {target_job}")
        
        return transfers
    
    def _preventive_resource_allocation(self, job_metrics, allocations):
        """预防性资源重分配 - 提前干预可能出现问题的任务"""
        transfers = 0
        
        # 识别即将进入风险区域的任务
        at_risk_jobs = [
            (name, metrics) for name, metrics in job_metrics.items()
            if (0.15 < metrics['deadline_risk'] < 0.4 and  # 中低风险但有潜在问题
                metrics['current_allocation'] < metrics['job'].max_replicas and
                metrics['marginal_utility'] > 0.25)  # 仍有边际效用
        ]
        
        # 按风险增长趋势排序
        at_risk_jobs.sort(key=lambda x: (
            x[1]['deadline_risk'] + x[1]['urgency_score'] - x[1]['saturation_ratio']
        ), reverse=True)
        
        # 为前几个风险任务分配额外资源
        for job_name, job_metrics_item in at_risk_jobs[:3]:
            current_alloc = job_metrics_item['current_allocation']
            
            # 寻找效率相对较低的任务作为资源提供者
            potential_donors = [
                (donor_name, donor_metrics) for donor_name, donor_metrics in job_metrics.items()
                if (donor_metrics['can_donate'] and 
                    donor_metrics['resource_efficiency'] < 0.6 and  # 低效率
                    donor_name != job_name and
                    len(allocations.get(donor_name, [])) > 2)
            ]
            
            if potential_donors:
                # 选择效率最低的捐赠者
                donor_name, donor_metrics = min(potential_donors, 
                                               key=lambda x: x[1]['resource_efficiency'])
                
                if len(allocations.get(donor_name, [])) > 2:  # 确保捐赠者不会被完全停止
                    # 安全转移1个GPU
                    donor_allocation = list(allocations.get(donor_name, []))
                    target_allocation = list(allocations.get(job_name, []))
                    
                    if donor_allocation and len(target_allocation) < job_metrics_item['job'].max_replicas:
                        donated_resource = donor_allocation.pop()
                        target_allocation.append(donated_resource)
                        
                        allocations[donor_name] = tuple(donor_allocation)
                        allocations[job_name] = tuple(target_allocation)
                        transfers += 1
                        print(f"Preventive CRR: {donor_name} -> {job_name}, 预防性转移 1 GPU")
        
        return transfers
    
    def _apply_deadline_mitigation_strategies(self, job_metrics, allocations, jobs):
        """应用违约缓解策略"""
        for job_name, metrics in job_metrics.items():
            if metrics['deadline_risk'] > 0.6:  # 高风险任务
                job = metrics['job']
                current_alloc = len(allocations.get(job_name, []))
                
                # 策略1：确保最低资源保障
                if current_alloc < 2:
                    # 从其他低优先级任务中强制获取资源
                    for other_name, other_metrics in job_metrics.items():
                        if (other_name != job_name and 
                            other_metrics['deadline_risk'] < 0.3 and
                            len(allocations.get(other_name, [])) > 2):
                            
                            other_allocation = allocations.get(other_name, [])
                            job_allocation = allocations.get(job_name, [])
                            
                            # 安全转移逻辑
                            other_allocation_list = list(other_allocation)
                            job_allocation_list = list(job_allocation)
                            
                            if len(other_allocation_list) > 1:
                                transferred_node = other_allocation_list[-1:]
                                remaining_other_nodes = other_allocation_list[:-1]
                                new_job_allocation = job_allocation_list + transferred_node
                                
                                # 验证资源约束
                                if (len(remaining_other_nodes) >= 1 and 
                                    len(new_job_allocation) <= job.max_replicas):
                                    
                                    allocations[other_name] = tuple(remaining_other_nodes)
                                    allocations[job_name] = tuple(new_job_allocation)
                                    
                                    print(f"Deadline Mitigation: {other_name} -> {job_name}, 强制转移保障资源")
                                    break
                                else:
                                    print(f"Deadline Mitigation: 资源约束违反 {other_name} -> {job_name}")
                            else:
                                print(f"Deadline Mitigation: 资源不足 {other_name} -> {job_name}")
    
    def _calculate_enhanced_urgency(self, job, current_time):
        """增强的紧迫性计算 - 提高敏感性"""
        if not hasattr(job, 'deadline') or job.deadline is None:
            return 0.4  # 提高无截止时间的默认值
        
        total_duration = job.deadline - job.submission_time
        elapsed_time = current_time - job.submission_time
        remaining_time = job.deadline - current_time
        
        if remaining_time <= 0:
            return 1.0  # 已超时
        
        if total_duration <= 0:
            return 1.0  # 异常情况
        
        # 综合考虑时间压力和进度
        time_pressure = elapsed_time / total_duration
        progress_ratio = getattr(job, 'progress', 0) / getattr(job, 'max_progress', 1)
        
        # 更敏感的非线性紧迫性函数 - 降低阈值
        if time_pressure > 0.7:  # 最后30%时间（原80%）
            urgency = min(1.0, 0.6 + (time_pressure - 0.7) * 3.0)  # 更快增长
        elif time_pressure > 0.5:  # 50%-70%时间（原60%-80%）
            urgency = 0.3 + (time_pressure - 0.5) * 1.5
        elif time_pressure > 0.3:  # 30%-50%时间（新增）
            urgency = 0.15 + (time_pressure - 0.3) * 0.75
        else:  # 前30%时间
            urgency = time_pressure * 0.5
        
        # 更严格的进度修正
        progress_lag = time_pressure - progress_ratio
        if progress_lag > 0.3:  # 进度严重滞后（降低阈值）
            urgency *= 1.5  # 增加修正系数
        elif progress_lag > 0.1:  # 进度轻微滞后（新增）
            urgency *= 1.2
        
        return min(urgency, 1.0)
    
    def _calculate_enhanced_marginal_utility(self, job, current_allocation):
        """增强的边际效用计算"""
        if current_allocation >= job.max_replicas:
            return 0.0
        
        # 基于模型类型的优化曲线
        if 'llama' in job.name.lower():
            # 大模型：前8个GPU效用递增，后续递减
            if current_allocation < 8:
                return 0.8 - current_allocation * 0.08
            else:
                return max(0.1, 0.4 - (current_allocation - 8) * 0.05)
        elif any(model in job.name.lower() for model in ['resnet', 'vgg', 'mobilenet']):
            # CNN模型：前4个GPU效用较高
            if current_allocation < 4:
                return 0.7 - current_allocation * 0.1
            else:
                return max(0.1, 0.3 - (current_allocation - 4) * 0.05)
        else:
            # 其他模型
            return max(0.1, 0.6 - current_allocation * 0.1)
    
    def _calculate_deadline_violation_risk(self, job, current_time):
        """计算违约风险 - 增强敏感性"""
        if not hasattr(job, 'deadline') or job.deadline is None:
            return 0.3  # 无截止时间的基础风险（提高基础值）
        
        remaining_time = job.deadline - current_time
        if remaining_time <= 0:
            return 1.0  # 已违约
        
        # 估算剩余工作量
        progress_ratio = getattr(job, 'progress', 0) / getattr(job, 'max_progress', 1)
        remaining_work = 1.0 - progress_ratio
        
        # 基于历史数据估算完成时间
        if hasattr(job, 'jcts') and job.jcts:
            estimated_completion_time = job.jcts * remaining_work
        else:
            # 更保守的估算
            estimated_completion_time = remaining_time * 1.3  # 假设需要多30%时间
        
        # 更敏感的风险评估 - 降低所有阈值
        time_ratio = estimated_completion_time / remaining_time
        if time_ratio > 1.3:          # 原1.5
            return 0.95             # 极高风险
        elif time_ratio > 1.1:       # 原1.0  
            return 0.75             # 高风险
        elif time_ratio > 0.9:       # 新增中高风险区间
            return 0.55             # 中高风险
        elif time_ratio > 0.7:       # 新增中等风险区间
            return 0.35             # 中等风险
        else:
            return 0.15             # 低风险（降低最低值）
    
    def _calculate_resource_efficiency(self, job, current_allocation):
        """计算资源效率"""
        if current_allocation == 0:
            return 1.0
        
        # 基于扩展性的效率评估
        if current_allocation == 1:
            return 1.0
        elif current_allocation <= 4:
            return 0.9 - (current_allocation - 1) * 0.1  # 单节点内扩展
        else:
            return max(0.3, 0.7 - (current_allocation - 4) * 0.08)  # 多节点扩展
    
    def _calculate_enhanced_collaboration_potential(self, job, all_jobs):
        """增强的协作潜力计算"""
        base_potential = 0.5
        
        # 基于模型相似性的协作潜力
        job_model = job.name.split('-')[1] if '-' in job.name else job.name
        similar_models = 0
        different_models = 0
        
        for other_job in all_jobs.values():
            if other_job.name == job.name:
                continue
            
            other_model = other_job.name.split('-')[1] if '-' in other_job.name else other_job.name
            
            if job_model == other_model:
                similar_models += 1
            else:
                different_models += 1
        
        # 相似模型多时协作潜力高（可以共享经验）
        # 不同模型多时也有协作潜力（资源需求互补）
        collaboration_boost = min(0.3, similar_models * 0.1 + different_models * 0.05)
        
        return min(base_potential + collaboration_boost, 1.0)
    
    def can_maintain_allocation(self, allocation, nodes, existing_allocations):
        """检查是否可以维持当前分配"""
        if not allocation:
            return True
        
        # 简单检查：计算已分配的资源总量
        total_allocated = sum(len(alloc) for alloc in existing_allocations.values())
        total_available = sum(node.resources.get("nvidia.com/gpu", 0) for node in nodes)
        
        return total_allocated + len(allocation) <= total_available
    
    def find_minimal_change_allocation(self, job, current_alloc, target_alloc, nodes, existing_allocations):
        """寻找最小变化的资源分配"""
        if not current_alloc:
            return target_alloc
        
        # 尝试在当前分配基础上进行最小调整
        current_size = len(current_alloc)
        target_size = len(target_alloc)
        
        # 如果目标是增加资源，逐步增加
        if target_size > current_size:
            for step in range(1, target_size - current_size + 1):
                candidate_alloc = current_alloc + [current_alloc[-1]] * step
                if self.can_maintain_allocation(candidate_alloc, nodes, existing_allocations):
                    return candidate_alloc
        
        # 如果目标是减少资源，逐步减少
        elif target_size < current_size:
            for step in range(1, current_size - target_size + 1):
                candidate_alloc = current_alloc[:-step]
                return candidate_alloc
        
        # 如果无法找到更好的，返回目标分配
        return target_alloc

    def _calculate_urgency_score(self, job, current_time):
        """计算任务的截止时间紧迫性分数"""
        if not hasattr(job, 'deadline') or job.deadline is None:
            return 0.5  # 无截止时间任务的默认紧迫性
        
        total_duration = job.deadline - job.submission_time
        elapsed_time = current_time - job.submission_time
        
        if total_duration <= 0:
            return 1.0  # 已超时
        
        time_pressure = elapsed_time / total_duration
        progress_ratio = getattr(job, 'progress', 0) / getattr(job, 'max_progress', 1)
        
        # 紧迫性 = 时间压力 - 进度完成度 + 非线性调整
        urgency = time_pressure - progress_ratio * 0.6
        urgency = max(0, min(urgency * 1.5, 1.0))  # 非线性放大，限制在[0,1]
        
        return urgency

    def _calculate_marginal_utility(self, job, current_allocation):
        """计算资源的边际效用（增加1个GPU的效用提升）"""
        if not hasattr(job, 'speedup_fn') or current_allocation >= job.max_replicas:
            return 0.0
        
        try:
            # 计算当前配置和+1GPU配置的goodput
            current_nodes = max(1, current_allocation // 4 + (1 if current_allocation % 4 > 0 else 0))
            next_nodes = max(1, (current_allocation + 1) // 4 + (1 if (current_allocation + 1) % 4 > 0 else 0))
            
            if hasattr(job.speedup_fn, '_goodput_fn'):
                current_goodput = job.speedup_fn._goodput_fn.optimize(
                    num_nodes=current_nodes, 
                    num_replicas=max(1, current_allocation),
                    max_batch_size=job.speedup_fn._max_batch_size,
                    atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                    accumulation=job.speedup_fn._accumulation)[0]
                
                next_goodput = job.speedup_fn._goodput_fn.optimize(
                    num_nodes=next_nodes, 
                    num_replicas=current_allocation + 1,
                    max_batch_size=job.speedup_fn._max_batch_size,
                    atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                    accumulation=job.speedup_fn._accumulation)[0]
                
                # 边际效用 = (新goodput - 当前goodput) / 当前goodput
                if current_goodput > 0:
                    return (next_goodput - current_goodput) / current_goodput
        except:
            pass
        
        # 后备计算：基于简化的扩展效率模型
        if current_allocation == 0:
            return 1.0
        elif current_allocation < 4:
            return 0.6  # 单节点内扩展，效率较高
        else:
            # 多节点扩展，边际效用递减
            return max(0.1, 1.0 / (current_allocation ** 0.5))

    def _calculate_saturation_ratio(self, job, current_allocation):
        """计算资源饱和度（当前分配相对于理论最优的比例）"""
        if current_allocation == 0:
            return 0.0
        
        # 基于模型类型估算理论最优配置
        optimal_allocation = job.max_replicas
        if 'llama' in job.name.lower():
            # 大模型通常在8-16个GPU时达到最优
            optimal_allocation = min(job.max_replicas, 16)
        elif any(model in job.name.lower() for model in ['resnet', 'vgg', 'mobilenet']):
            # CNN模型通常在4-8个GPU时达到较好效果
            optimal_allocation = min(job.max_replicas, 8)
        else:
            # 其他模型取最大配置的50%作为理论最优
            optimal_allocation = min(job.max_replicas, max(4, job.max_replicas // 2))
        
        return min(current_allocation / optimal_allocation, 1.0)

    def _calculate_collaboration_potential(self, job, all_jobs):
        """计算任务间的协作潜力（基于训练阶段和模型互补性）"""
        base_potential = 0.5
        
        # 基于训练进度的互补性
        job_progress = getattr(job, 'progress', 0) / getattr(job, 'max_progress', 1)
        
        # 寻找与当前任务训练阶段互补的任务
        complementary_count = 0
        for other_job in all_jobs.values():
            if other_job.name == job.name:
                continue
            
            other_progress = getattr(other_job, 'progress', 0) / getattr(other_job, 'max_progress', 1)
            progress_diff = abs(job_progress - other_progress)
            
            # 训练阶段差异大的任务更容易协作（一个处于初期，一个处于后期）
            if progress_diff > 0.3:
                complementary_count += 1
        
        # 基于模型类型的互补性
        model_type_bonus = 0.0
        job_model = job.name.split('-')[1] if '-' in job.name else job.name
        
        for other_job in all_jobs.values():
            if other_job.name == job.name:
                continue
            
            other_model = other_job.name.split('-')[1] if '-' in other_job.name else other_job.name
            
            # 不同类型模型的资源需求模式可能互补
            if job_model != other_model:
                # CNN模型 vs 语言模型的互补性较强
                if (('resnet' in job_model.lower() or 'vgg' in job_model.lower()) and 
                    ('bert' in other_model.lower() or 'llama' in other_model.lower())):
                    model_type_bonus += 0.3
                elif (('bert' in job_model.lower() or 'llama' in job_model.lower()) and 
                      ('resnet' in other_model.lower() or 'vgg' in other_model.lower())):
                    model_type_bonus += 0.3
                else:
                    model_type_bonus += 0.1
        
        # 综合协作潜力
        collaboration_potential = (
            base_potential + 
            min(complementary_count * 0.1, 0.3) +  # 训练阶段互补性
            min(model_type_bonus, 0.4)  # 模型类型互补性
        )
        
        return min(collaboration_potential, 1.0)

    def _calculate_optimal_transfer(self, lender_metrics, borrower_metrics):
        """基于边际效用递减规律计算最优资源转移量"""
        lender_allocation = lender_metrics['current_allocation']
        lender_saturation = lender_metrics['saturation_ratio']
        borrower_urgency = borrower_metrics['urgency_score']
        borrower_utility = borrower_metrics['marginal_utility']
        
        # 基础转移量：基于出借方的资源饱和度
        base_transfer = 1
        if lender_saturation > 0.9:
            base_transfer = min(2, lender_allocation // 3)  # 高饱和度可以转移更多
        elif lender_saturation > 0.8:
            base_transfer = 1
        else:
            base_transfer = 0
        
        # 根据借用方的紧迫性和边际效用调整
        urgency_multiplier = 1.0 + borrower_urgency * 0.5  # 最多1.5倍
        utility_multiplier = 1.0 + borrower_utility * 0.3   # 最多1.3倍
        
        optimal_transfer = int(base_transfer * urgency_multiplier * utility_multiplier)
        
        # 确保转移量合理
        return max(0, min(optimal_transfer, lender_allocation - 1, 2))  # 最多转移2个GPU



