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
import sys

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
        state = np.zeros((len(jobs), len(nodes)), dtype=np.int)
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
        states = np.zeros(shape, dtype=np.int)
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
        #     np.zeros((len(jobs), len(nodes)), dtype=np.int)), axis=1)
        base_state = \
            self._allocations_to_state(base_allocations, jobs, nodes)

        if self._prev_states is None:
            states = np.expand_dims(base_state, 0)
        else:
            states = self._adapt_prev_states(jobs, nodes)
        
        # 检查是否启用DeadlineMeet模式的新机制
        from simulator import args
        if hasattr(args, 'obj') and args.obj == 'DeadlineMeet':
            # 应用资源预留与渐进式解冻机制
            jobs_list = list(jobs.values())
            nodes_list = list(nodes.values())
            
            enhanced_allocations = self.resource_reservation_progressive_unfreezing(
                jobs_list, nodes_list, base_allocations)
            
            # 如果新机制产生了有效分配，使用它；否则回退到原有机制
            if enhanced_allocations:
                print("应用资源预留与渐进式解冻机制")
                return enhanced_allocations, -1
        
        problem = Problem(list(jobs.values()), list(nodes.values()), base_state)
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

    def resource_reservation_progressive_unfreezing(self, jobs, nodes, base_allocations):
        """
        资源预留与渐进式解冻机制 (Resource Reservation with Progressive Unfreezing)
        
        核心思想：
        1. 时间感知资源预留：为接近截止时间的任务预留额外资源
        2. 渐进式层解冻：根据时间紧迫度动态调整冻结层数
        3. 双阶段优化：先保证截止时间，再优化整体性能
        
        理论依据：
        - 控制论中的前馈控制：预测未来需求并提前调整
        - 博弈论中的Nash均衡：平衡个体最优和全局最优
        - 深度学习中的curriculum learning：渐进式难度调整
        """
        urgent_jobs = []
        normal_jobs = []
        current_time = max(job.submission_time + getattr(job, 'staying_time', 0) for job in jobs)
        
        # 第一阶段：识别紧急任务并计算预留需求
        for job in jobs:
            urgency_score = self._calculate_urgency_score(job, current_time)
            if urgency_score > 0.6:  # 高紧急度阈值
                urgent_jobs.append((job, urgency_score))
            else:
                normal_jobs.append(job)
        
        # 按紧急度排序
        urgent_jobs.sort(key=lambda x: x[1], reverse=True)
        
        # 第二阶段：为紧急任务预留资源
        reserved_allocations = {}
        remaining_resources = {node_id: node.resources.get("nvidia.com/gpu", 0) 
                             for node_id, node in enumerate(nodes)}
        
        for job, urgency_score in urgent_jobs:
            # 计算预留资源量（基于紧急度和预期加速比）
            base_replicas = len(base_allocations.get(job.name, []))
            reserved_replicas = self._calculate_reserved_replicas(job, urgency_score, base_replicas)
            
            # 动态调整冻结层数
            optimal_freeze_ratio = self._calculate_progressive_freeze_ratio(job, urgency_score)
            
            # 分配预留资源
            allocation = self._allocate_reserved_resources(
                job, reserved_replicas, remaining_resources, nodes)
            
            if allocation:
                reserved_allocations[job.name] = allocation
                # 更新剩余资源
                for node_id in allocation:
                    remaining_resources[node_id] -= 1
                
                # 设置动态冻结参数
                self._apply_progressive_unfreezing(job, optimal_freeze_ratio, urgency_score)
        
        # 第三阶段：为普通任务分配剩余资源
        normal_allocations = self._allocate_normal_jobs(normal_jobs, remaining_resources, nodes)
        
        # 合并分配结果
        final_allocations = {**reserved_allocations, **normal_allocations}
        
        return final_allocations

    def _calculate_urgency_score(self, job, current_time):
        """计算任务紧急度分数 (0-1, 1为最紧急)"""
        if not hasattr(job, 'deadline') or job.deadline is None:
            return 0.0
        
        # 基于时间的紧急度
        total_time = job.deadline - job.submission_time
        elapsed_time = current_time - job.submission_time
        remaining_time = job.deadline - current_time
        
        if total_time <= 0 or remaining_time <= 0:
            return 1.0  # 已超时或即将超时
        
        time_pressure = 1.0 - (remaining_time / total_time)
        
        # 基于进度的紧急度
        progress_ratio = job.progress / job.max_progress if job.max_progress > 0 else 0
        progress_pressure = max(0, time_pressure - progress_ratio)  # 进度滞后压力
        
        # 基于预期完成时间的紧急度
        if hasattr(job, 'jcts') and job.jcts is not None:
            expected_completion = current_time + job.jcts * (1 - progress_ratio)
            completion_pressure = max(0, (expected_completion - job.deadline) / total_time)
        else:
            completion_pressure = 0
        
        # 综合紧急度分数 (加权平均)
        urgency_score = 0.4 * time_pressure + 0.3 * progress_pressure + 0.3 * completion_pressure
        return min(1.0, max(0.0, urgency_score))

    def _calculate_reserved_replicas(self, job, urgency_score, base_replicas):
        """计算需要预留的资源量"""
        # 基础预留策略：紧急度越高，预留资源越多
        max_boost = min(job.max_replicas - base_replicas, 4)  # 最多增加4个副本
        
        # 根据模型类型调整预留策略
        if 'llama' in job.name.lower():
            # 大模型需要4的倍数
            boost_replicas = int(urgency_score * max_boost / 4) * 4
        else:
            boost_replicas = int(urgency_score * max_boost)
        
        return min(base_replicas + boost_replicas, job.max_replicas)

    def _calculate_progressive_freeze_ratio(self, job, urgency_score):
        """计算渐进式冻结比例"""
        # 基础冻结比例（基于模型类型）
        model_type = job.name.split('-')[1] if '-' in job.name else job.name
        
        if 'bert' in model_type.lower() or 'llama' in model_type.lower():
            base_freeze_ratio = 0.5  # 语言模型基础冻结50%
        elif any(x in model_type for x in ['ResNet', 'VGG', 'MobileNet']):
            base_freeze_ratio = 0.6  # CNN模型可以冻结更多
        else:
            base_freeze_ratio = 0.4
        
        # 根据紧急度动态调整
        # 紧急度越高，冻结比例越高（优先速度）
        urgency_adjustment = urgency_score * 0.3
        progressive_ratio = min(0.8, base_freeze_ratio + urgency_adjustment)
        
        # 根据训练进度微调
        progress_ratio = job.progress / job.max_progress if job.max_progress > 0 else 0
        if progress_ratio > 0.8:  # 训练后期减少冻结以保证精度
            progressive_ratio *= 0.8
        
        return progressive_ratio

    def _allocate_reserved_resources(self, job, target_replicas, remaining_resources, nodes):
        """为紧急任务分配预留资源"""
        allocation = []
        needed_gpus = target_replicas
        
        # 优先分配到资源充足的节点
        sorted_nodes = sorted(remaining_resources.items(), 
                            key=lambda x: x[1], reverse=True)
        
        for node_id, available_gpus in sorted_nodes:
            if needed_gpus <= 0:
                break
            
            # 计算在此节点可分配的GPU数量
            can_allocate = min(needed_gpus, available_gpus)
            allocation.extend([node_id] * can_allocate)
            needed_gpus -= can_allocate
        
        return allocation if needed_gpus == 0 else []

    def _allocate_normal_jobs(self, normal_jobs, remaining_resources, nodes):
        """为普通任务分配剩余资源"""
        allocations = {}
        total_remaining = sum(remaining_resources.values())
        
        if total_remaining == 0 or not normal_jobs:
            return allocations
        
        # 简单的公平分配策略
        fair_share = total_remaining // len(normal_jobs)
        
        for job in normal_jobs:
            target_replicas = min(fair_share, job.max_replicas)
            allocation = self._allocate_reserved_resources(
                job, target_replicas, remaining_resources, nodes)
            
            if allocation:
                allocations[job.name] = allocation
                # 更新剩余资源
                for node_id in allocation:
                    remaining_resources[node_id] -= 1
        
        return allocations

    def _apply_progressive_unfreezing(self, job, freeze_ratio, urgency_score):
        """应用渐进式解冻策略"""
        # 计算需要冻结的层数
        if hasattr(job, 'total_layer') and job.total_layer > 0:
            frozen_layers = int(job.total_layer * freeze_ratio)
            
            # 设置渐进式解冻调度
            if urgency_score > 0.8:
                # 高紧急度：快速解冻策略
                job.unfreezing_schedule = 'aggressive'
                job.unfreezing_interval = 50  # 每50个epoch解冻一层
            elif urgency_score > 0.6:
                # 中紧急度：平衡解冻策略
                job.unfreezing_schedule = 'balanced'
                job.unfreezing_interval = 100  # 每100个epoch解冻一层
            else:
                # 低紧急度：保守解冻策略
                job.unfreezing_schedule = 'conservative'
                job.unfreezing_interval = 200  # 每200个epoch解冻一层
            
            # 更新任务的冻结层参数
            job.frozen_layer = frozen_layers
            job.dynamic_freeze_ratio = freeze_ratio


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
        assert hasattr(args, 'static_weight')
        assert hasattr(args, 'batch_weight')
        assert hasattr(args, 'layer_weight')
        self.elastic_weight = {
            'layer': args.layer_weight, 
            'batch': args.batch_weight, 
            'static': args.static_weight
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

    def resource_reservation_progressive_unfreezing(self, jobs, nodes, base_allocations):
        """
        资源预留与渐进式解冻机制 (Resource Reservation with Progressive Unfreezing)
        
        核心思想：
        1. 时间感知资源预留：为接近截止时间的任务预留额外资源
        2. 渐进式层解冻：根据时间紧迫度动态调整冻结层数
        3. 双阶段优化：先保证截止时间，再优化整体性能
        
        理论依据：
        - 控制论中的前馈控制：预测未来需求并提前调整
        - 博弈论中的Nash均衡：平衡个体最优和全局最优
        - 深度学习中的curriculum learning：渐进式难度调整
        """
        urgent_jobs = []
        normal_jobs = []
        current_time = max(job.submission_time + getattr(job, 'staying_time', 0) for job in jobs)
        
        # 第一阶段：识别紧急任务并计算预留需求
        for job in jobs:
            urgency_score = self._calculate_urgency_score(job, current_time)
            if urgency_score > 0.6:  # 高紧急度阈值
                urgent_jobs.append((job, urgency_score))
            else:
                normal_jobs.append(job)
        
        # 按紧急度排序
        urgent_jobs.sort(key=lambda x: x[1], reverse=True)
        
        # 第二阶段：为紧急任务预留资源
        reserved_allocations = {}
        remaining_resources = {node_id: node.resources.get("nvidia.com/gpu", 0) 
                             for node_id, node in enumerate(nodes)}
        
        for job, urgency_score in urgent_jobs:
            # 计算预留资源量（基于紧急度和预期加速比）
            base_replicas = len(base_allocations.get(job.name, []))
            reserved_replicas = self._calculate_reserved_replicas(job, urgency_score, base_replicas)
            
            # 动态调整冻结层数
            optimal_freeze_ratio = self._calculate_progressive_freeze_ratio(job, urgency_score)
            
            # 分配预留资源
            allocation = self._allocate_reserved_resources(
                job, reserved_replicas, remaining_resources, nodes)
            
            if allocation:
                reserved_allocations[job.name] = allocation
                # 更新剩余资源
                for node_id in allocation:
                    remaining_resources[node_id] -= 1
                
                # 设置动态冻结参数
                self._apply_progressive_unfreezing(job, optimal_freeze_ratio, urgency_score)
        
        # 第三阶段：为普通任务分配剩余资源
        normal_allocations = self._allocate_normal_jobs(normal_jobs, remaining_resources, nodes)
        
        # 合并分配结果
        final_allocations = {**reserved_allocations, **normal_allocations}
        
        return final_allocations

    def _calculate_urgency_score(self, job, current_time):
        """计算任务紧急度分数 (0-1, 1为最紧急)"""
        if not hasattr(job, 'deadline') or job.deadline is None:
            return 0.0
        
        # 基于时间的紧急度
        total_time = job.deadline - job.submission_time
        elapsed_time = current_time - job.submission_time
        remaining_time = job.deadline - current_time
        
        if total_time <= 0 or remaining_time <= 0:
            return 1.0  # 已超时或即将超时
        
        time_pressure = 1.0 - (remaining_time / total_time)
        
        # 基于进度的紧急度
        progress_ratio = job.progress / job.max_progress if job.max_progress > 0 else 0
        progress_pressure = max(0, time_pressure - progress_ratio)  # 进度滞后压力
        
        # 基于预期完成时间的紧急度
        if hasattr(job, 'jcts') and job.jcts is not None:
            expected_completion = current_time + job.jcts * (1 - progress_ratio)
            completion_pressure = max(0, (expected_completion - job.deadline) / total_time)
        else:
            completion_pressure = 0
        
        # 综合紧急度分数 (加权平均)
        urgency_score = 0.4 * time_pressure + 0.3 * progress_pressure + 0.3 * completion_pressure
        return min(1.0, max(0.0, urgency_score))

    def _calculate_reserved_replicas(self, job, urgency_score, base_replicas):
        """计算需要预留的资源量"""
        # 基础预留策略：紧急度越高，预留资源越多
        max_boost = min(job.max_replicas - base_replicas, 4)  # 最多增加4个副本
        
        # 根据模型类型调整预留策略
        if 'llama' in job.name.lower():
            # 大模型需要4的倍数
            boost_replicas = int(urgency_score * max_boost / 4) * 4
        else:
            boost_replicas = int(urgency_score * max_boost)
        
        return min(base_replicas + boost_replicas, job.max_replicas)

    def _calculate_progressive_freeze_ratio(self, job, urgency_score):
        """计算渐进式冻结比例"""
        # 基础冻结比例（基于模型类型）
        model_type = job.name.split('-')[1] if '-' in job.name else job.name
        
        if 'bert' in model_type.lower() or 'llama' in model_type.lower():
            base_freeze_ratio = 0.5  # 语言模型基础冻结50%
        elif any(x in model_type for x in ['ResNet', 'VGG', 'MobileNet']):
            base_freeze_ratio = 0.6  # CNN模型可以冻结更多
        else:
            base_freeze_ratio = 0.4
        
        # 根据紧急度动态调整
        # 紧急度越高，冻结比例越高（优先速度）
        urgency_adjustment = urgency_score * 0.3
        progressive_ratio = min(0.8, base_freeze_ratio + urgency_adjustment)
        
        # 根据训练进度微调
        progress_ratio = job.progress / job.max_progress if job.max_progress > 0 else 0
        if progress_ratio > 0.8:  # 训练后期减少冻结以保证精度
            progressive_ratio *= 0.8
        
        return progressive_ratio

    def _allocate_reserved_resources(self, job, target_replicas, remaining_resources, nodes):
        """为紧急任务分配预留资源"""
        allocation = []
        needed_gpus = target_replicas
        
        # 优先分配到资源充足的节点
        sorted_nodes = sorted(remaining_resources.items(), 
                            key=lambda x: x[1], reverse=True)
        
        for node_id, available_gpus in sorted_nodes:
            if needed_gpus <= 0:
                break
            
            # 计算在此节点可分配的GPU数量
            can_allocate = min(needed_gpus, available_gpus)
            allocation.extend([node_id] * can_allocate)
            needed_gpus -= can_allocate
        
        return allocation if needed_gpus == 0 else []

    def _allocate_normal_jobs(self, normal_jobs, remaining_resources, nodes):
        """为普通任务分配剩余资源"""
        allocations = {}
        total_remaining = sum(remaining_resources.values())
        
        if total_remaining == 0 or not normal_jobs:
            return allocations
        
        # 简单的公平分配策略
        fair_share = total_remaining // len(normal_jobs)
        
        for job in normal_jobs:
            target_replicas = min(fair_share, job.max_replicas)
            allocation = self._allocate_reserved_resources(
                job, target_replicas, remaining_resources, nodes)
            
            if allocation:
                allocations[job.name] = allocation
                # 更新剩余资源
                for node_id in allocation:
                    remaining_resources[node_id] -= 1
        
        return allocations

    def _apply_progressive_unfreezing(self, job, freeze_ratio, urgency_score):
        """应用渐进式解冻策略"""
        # 计算需要冻结的层数
        if hasattr(job, 'total_layer') and job.total_layer > 0:
            frozen_layers = int(job.total_layer * freeze_ratio)
            
            # 设置渐进式解冻调度
            if urgency_score > 0.8:
                # 高紧急度：快速解冻策略
                job.unfreezing_schedule = 'aggressive'
                job.unfreezing_interval = 50  # 每50个epoch解冻一层
            elif urgency_score > 0.6:
                # 中紧急度：平衡解冻策略
                job.unfreezing_schedule = 'balanced'
                job.unfreezing_interval = 100  # 每100个epoch解冻一层
            else:
                # 低紧急度：保守解冻策略
                job.unfreezing_schedule = 'conservative'
                job.unfreezing_interval = 200  # 每200个epoch解冻一层
            
            # 更新任务的冻结层参数
            job.frozen_layer = frozen_layers
            job.dynamic_freeze_ratio = freeze_ratio



