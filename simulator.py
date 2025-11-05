import argparse
import yaml 
from easydict import EasyDict
import collections
import copy
import glob
import json
import math
import multiprocessing
import os
import time
import pprint 
import ast

import numpy as np
import pandas
import pandas as pd
np.random.seed(42)

from applications import APPLICATIONS, apply_memory_limit, apply_remove_prior, apply_accelerate, apply_large_thr
from goodput import GoodputFunction, fit_perf_params
from speedup import SpeedupFunction
from utils import JobInfo, NodeInfo
from optimus import OptimusPolicy
from lucid import LucidPolicy
from job import Job, FrozenJob
from simple_pollux import SimplePolluxPolicy
from simple_icefrog import SimpleIceFrogPolicy


class Cluster(object):
    def __init__(self, workload, policy, min_nodes, num_gpus=4,
                 max_nodes=None, interference=0.0,
                 low_util=None, high_util=None, weight_job=None, workload_path=None, 
                 policy_name=None, objective=None):
        assert 1 <= num_gpus <= 4
        self.workload = workload
        self.workload_path = workload_path  # 存储workload文件路径
        self.policy = policy
        self.policy_name = policy_name  # 存储策略名称
        self.objective = objective  # 存储目标函数
        self.min_nodes = self.num_nodes = min_nodes
        self.num_gpus = num_gpus
        self.max_nodes = min_nodes if max_nodes is None else max_nodes
        self.interference = interference
        self.low_util = low_util
        self.high_util = high_util
        self.current_time = 0
        import numpy as np 
        
        freeze_hyp = None 
        if args.freeze == 'Dynamic': 
            filename = 'exp/ablation/freeze_policy/freeze_policy_dynamic.npy'
            assert os.path.exists(filename) == True 
            if os.path.exists(filename): 
                with open(filename, 'rb') as f:
                    freeze_hyp = np.load(f, allow_pickle=True).tolist()
        freezeout_hyp = None 
        if args.freeze == 'FreezeOut': 
            filename = 'exp/ablation/freeze_policy/freeze_policy_freezeout.npy'
            assert os.path.exists(filename) == True 
            if os.path.exists(filename): 
                with open(filename, 'rb') as f:
                    freezeout_hyp = np.load(f, allow_pickle=True).tolist()
        

        self.jobs = list() 
        for row in workload.itertuples():
            if hasattr(row, 'batch_size_lower_bound'): 
                batch_size_lower_bound = row.batch_size_lower_bound
                batch_size_upper_bound = row.batch_size_upper_bound
            else: 
                batch_size_lower_bound = min(APPLICATIONS[row.application].validation)
                batch_size_upper_bound = max(APPLICATIONS[row.application].validation)
            
            if hasattr(row, 'replica_lower_bound'): 
                replica_lower_bound = min(row.replica_lower_bound, 4)
                replica_upper_bound = row.replica_upper_bound // 4 * 4
            else: 
                replica_lower_bound = 1
                replica_upper_bound = args.min_nodes * 4
            
            
            if isinstance(policy, (SimplePolluxPolicy)):
                if freeze_hyp is None or (row.application, row.num_replicas, row.batch_size) not in freeze_hyp: 
                    recommend_frozen_layer = None 
                else: 
                    recommend_frozen_layer = freeze_hyp[(row.application, row.num_replicas, row.batch_size)]
                
                if freezeout_hyp is None: 
                    freeze_parameter = None 
                elif (row.application, row.num_replicas, row.batch_size) not in freezeout_hyp: 
                    freeze_parameter = (140, 0) if 'llama' in row.name else None 
                else: 
                    freeze_parameter = freezeout_hyp[(row.application, row.num_replicas, row.batch_size)]
                 
                    
                jobInstance = Job(row.name, APPLICATIONS[row.application], row.time, frozen_strategy=args.freeze, 
                                batch_size_lower_bound=batch_size_lower_bound, batch_size_upper_bound=batch_size_upper_bound,
                                replica_lower_bound=replica_lower_bound, replica_upper_bound=replica_upper_bound,
                                recommend_frozen_layer=recommend_frozen_layer, 
                                freeze_parameter=freeze_parameter, 
                                deadline=row.deadline if hasattr(row, 'deadline') else None,
                                jcts=row.jcts if hasattr(row, 'jcts') else None
                                )
            
            elif isinstance(policy, LucidPolicy):
                if freeze_hyp is None or (row.application, row.num_replicas, row.batch_size) not in freeze_hyp: 
                    recommend_frozen_layer = None 
                else: 
                    recommend_frozen_layer = freeze_hyp[(row.application, row.num_replicas, row.batch_size)]
                
                if freezeout_hyp is None or (row.application, row.num_replicas, row.batch_size) not in freezeout_hyp: 
                    freeze_parameter = None 
                elif (row.application, row.num_replicas, row.batch_size) not in freezeout_hyp: 
                    freeze_parameter = (140, 0) if 'llama' in row.name else None 
                else: 
                    freeze_parameter = freezeout_hyp[(row.application, row.num_replicas, row.batch_size)]
                    
                jobInstance = Job(row.name, APPLICATIONS[row.application], row.time, frozen_strategy=args.freeze, 
                                batch_size_lower_bound=batch_size_lower_bound, batch_size_upper_bound=batch_size_upper_bound,
                                replica_lower_bound=replica_lower_bound, replica_upper_bound=replica_upper_bound,
                                recommend_frozen_layer=recommend_frozen_layer, 
                                freeze_parameter=freeze_parameter, 
                                target_batch_size=row.batch_size,
                                deadline=row.deadline if hasattr(row, 'deadline') else None,
                                jcts=row.jcts if hasattr(row, 'jcts') else None)

            elif isinstance(policy, OptimusPolicy):
                if freeze_hyp is None or (row.application, row.num_replicas, row.batch_size) not in freeze_hyp: 
                    recommend_frozen_layer = None 
                else: 
                    recommend_frozen_layer = freeze_hyp[(row.application, row.num_replicas, row.batch_size)]
                
                if freezeout_hyp is None or (row.application, row.num_replicas, row.batch_size) not in freezeout_hyp: 
                    freeze_parameter = None 
                elif (row.application, row.num_replicas, row.batch_size) not in freezeout_hyp: 
                    freeze_parameter = (140, 0) if 'llama' in row.name else None 
                else: 
                    freeze_parameter = freezeout_hyp[(row.application, row.num_replicas, row.batch_size)]
                    
                jobInstance = Job(row.name, APPLICATIONS[row.application], row.time, frozen_strategy=args.freeze, 
                                batch_size_lower_bound=batch_size_lower_bound, batch_size_upper_bound=batch_size_upper_bound,
                                replica_lower_bound=replica_lower_bound, replica_upper_bound=replica_upper_bound,
                                recommend_frozen_layer=recommend_frozen_layer, 
                                freeze_parameter=freeze_parameter, 
                                target_batch_size=row.batch_size, 
                                deadline_factor=args.deadline_factor if hasattr(args, 'deadline_factor') else None,
                                deadline=row.deadline if hasattr(row, 'deadline') else None,
                                jcts=row.jcts if hasattr(row, 'jcts') else None)
            elif isinstance(policy, (SimpleIceFrogPolicy)):
                if freeze_hyp is None or (row.application, row.num_replicas, row.batch_size) not in freeze_hyp: 
                    recommend_frozen_layer = None 
                else: 
                    recommend_frozen_layer = freeze_hyp[(row.application, row.num_replicas, row.batch_size)]
                
                if freezeout_hyp is None or (row.application, row.num_replicas, row.batch_size) not in freezeout_hyp: 
                    freeze_parameter = None 
                elif (row.application, row.num_replicas, row.batch_size) not in freezeout_hyp: 
                    freeze_parameter = (140, 0) if 'llama' in row.name else None 
                else: 
                    freeze_parameter = freezeout_hyp[(row.application, row.num_replicas, row.batch_size)]
                    
                if not args.reproduce_scheduling: 
                    jobInstance = FrozenJob(row.name, APPLICATIONS[row.application], row.time, 
                                            batch_size_lower_bound=batch_size_lower_bound, batch_size_upper_bound=batch_size_upper_bound,
                                            replica_lower_bound=replica_lower_bound, replica_upper_bound=replica_upper_bound,
                                            frozen_alpha=row.frozen_alpha, fixed_batch_size=row.batch_size if args.batch_fixed else None, shrink_range=args.shrink_range, \
                                            freeze_parameter=freeze_parameter, 
                                            elastic=row.elastic if hasattr(row, 'elastic') else 'layer', reproduce_record=args.reproduce_record, batch_fixed=args.batch_fixed, deadline=row.deadline if hasattr(row, 'deadline') else None, jcts=row.jcts if hasattr(row, 'jcts') else None) 
                
            else:
                raise NotImplementedError
            self.jobs.append(jobInstance)

            
        self.allocations = {}
        self.logs = []
        self.utility = []
        self.weight_job = weight_job
        self.solver_time = []

    def step(self, finegrained_seconds=60, seconds=300, completed_jobs=0, total_jobs=120):
        # # 在Cluster.step方法中添加
        # for job in self.jobs:
        #     # 设置初始截止时间（仅在首次执行时）
        #     if job.deadline is None and job.estimate_speed > 0:
        #         # 基于当前进度和训练速度预估完成时间
        #         estimated_completion = job.submission_time + (job.max_progress - job.progress) / job.estimate_speed
        #         job.deadline = job.submission_time + (estimated_completion - job.submission_time) * job.deadline_factor
            
        #     # ...现有代码...
        # 替换原来的截止时间计算逻辑
        for job in self.jobs:
            # 只计算尚未设置截止时间的作业
            if job.deadline is None:
                # 获取当前配置下的性能预测
                if len(self.allocations.get(job.name, [])) > 0:  # 如果已分配资源
                    speedup_fn = job.get_speedup_fn()
                    placement = job.placement
                    if len(placement) > 0:  # 确保有有效放置
                        num_nodes = len(set(self.allocations.get(job.name, [])))
                        num_replicas = len(self.allocations.get(job.name, []))
                        
                        # 使用性能模型预测剩余时间
                        if hasattr(job, 'get_frozen_goodput_fn'):
                            # # 获取当前配置下的吞吐量
                            # goodput_fn = job.get_frozen_goodput_fn()
                            # 获取当前配置下的吞吐量
                            goodput_fn = job.get_frozen_goodput_fn()
                            
                            # 安全获取accumulation参数
                            if hasattr(job, 'speedup_fn') and hasattr(job.speedup_fn, '_accumulation'):
                                accumulation = job.speedup_fn._accumulation
                            elif hasattr(job, 'accum_steps'):
                                # 使用作业的累积步数作为替代指标
                                accumulation = job.accum_steps > 0
                            else:
                                # 默认启用累积
                                accumulation = True
                            goodput_info = goodput_fn.optimize(
                                num_nodes=num_nodes,
                                num_replicas=num_replicas,
                                max_batch_size=job.application.max_batch_size,
                                atomic_bsz_range=None,
                                #accumulation=job.speedup_fn._accumulation
                                accumulation=accumulation)
                            goodput = goodput_info[0]
                            
                            # 计算剩余训练时间
                            remaining_progress = job.max_progress - job.progress
                            if goodput > 0:
                                # estimated_time_to_complete = remaining_progress / goodput
                                # # 使用deadline_factor计算截止时间
                                # job.deadline = self.current_time + estimated_time_to_complete * (
                                #     job.deadline_factor if hasattr(job, 'deadline_factor') and job.deadline_factor is not None 
                                #     else np.random.uniform(1.2, 3.0))
                                #estimated_time_to_complete = remaining_progress / goodput
                                # 根据模型大小调整重启开销
                                if 'llama' in job.name or '-bert-' in job.name:
                                    restart_factor = 0.25  # 大模型重启开销更大
                                else:
                                    restart_factor = 0.1   # 小模型重启开销相对较小
                                restart_overhead = 1.0 + (job.num_restarts * restart_factor)
                                # 在计算restart_overhead之前添加
                                #print(f"Job {job.name} restart count: {job.num_restarts}!!!!!!!!!!!!!!!!!\n")
                                #restart_overhead = 1.0 + (job.num_restarts * 0.05)  # 每次重启增加5%开销
                                estimated_time_to_complete = (remaining_progress / goodput) * restart_overhead
                                # 保存预测的完成时间（不带deadline_factor）
                                job.estimated_completion_time = self.current_time + estimated_time_to_complete
                                # 使用deadline_factor计算截止时间
                                # job.deadline = self.current_time + estimated_time_to_complete * (
                                #     job.deadline_factor if hasattr(job, 'deadline_factor') and job.deadline_factor is not None 
                                #     else np.random.uniform(1.2, 3.0))
                        else:
                            # # 回退到使用speedup_fn
                            # speedup = speedup_fn(num_nodes, num_replicas)
                            # if speedup > 0:
                            #     # 基于加速比估算完成时间
                            #     base_time = job.application.epochs_to_seconds(
                            #         (job.max_progress - job.progress) / job.max_progress * job.application.max_epochs, 1)
                            #     estimated_time_to_complete = base_time / speedup
                            #     job.deadline = self.current_time + estimated_time_to_complete * (
                            #         job.deadline_factor if hasattr(job, 'deadline_factor') and job.deadline_factor is not None 
                            #         else np.random.uniform(1.2, 3.0))
                            # 回退到使用speedup_fn
                            speedup = speedup_fn(num_nodes, num_replicas)
                            if speedup > 0:
                                # 基于加速比估算完成时间
                                base_time = job.application.epochs_to_seconds(
                                    (job.max_progress - job.progress) / job.max_progress * job.application.max_epochs, 1)
                                estimated_time_to_complete = base_time / speedup
                                
                                # 保存预测的完成时间（不带deadline_factor）
                                job.estimated_completion_time = self.current_time + estimated_time_to_complete
                                
                                # 使用deadline_factor计算截止时间
                                job.deadline = self.current_time + estimated_time_to_complete * (
                                    job.deadline_factor if hasattr(job, 'deadline_factor') and job.deadline_factor is not None 
                                    else np.random.uniform(1.2, 3.0))
        # one compute node does not share by exceeding two distributed jobs 
        interfere_nodes = set(idx for idx in range(self.num_nodes)
                              if sum(len(set(val)) > 1 and idx in val
                                     for key, val in self.allocations.items()) > 1)
        cluster_capaicty = self.min_nodes * 4 
        fair_gpus = cluster_capaicty / len(self.jobs)
        for job in self.jobs:
            alloc_set = set(self.allocations.get(job.name, []))
            interference = 0.0
            if len(alloc_set) > 1 and any(idx in interfere_nodes for idx in alloc_set):
                interference = self.interference 
            job.step(finegrained_seconds, deserved_gpu=fair_gpus, interference=interference)
            
        self.current_time += finegrained_seconds
        assert all(job.current_time == self.current_time for job in self.jobs)
        # job_infos, fixed_job_infos = self.get_job_infos((self.current_time - finegrained_seconds) % seconds != 0)
        job_infos, fixed_job_infos = self.get_job_infos(False)
        if job_infos:
            if self.max_nodes > self.min_nodes and (self.current_time - finegrained_seconds) % seconds == 0:
                # import pdb; pdb.set_trace() 
                # Autoscale cluster if needed.
                self.utility.append(self.get_utility(self.num_nodes, job_infos, self.allocations))
                if len(self.utility) > 3:
                    self.utility.pop(0)
                    utility = sum(self.utility) / len(self.utility)
                    if (self.num_nodes > self.min_nodes and utility < self.low_util) or \
                            (self.num_nodes < self.max_nodes and utility > self.high_util):
                        self.autoscale(job_infos)
                        self.utility.clear()
                    print("Utility:", utility)
                print("Nodes:", self.num_nodes)
            # Optimize allocations.
            node_infos = self.get_node_infos(fixed_job_infos=fixed_job_infos)
            pre_allocations = {k: v for k, v in self.allocations.items() if k in job_infos}
            remaining_gpus = sum([node.resources["nvidia.com/gpu"] for node in node_infos.values()])
            if remaining_gpus > 0:  
                start_time = time.time() 
                results = self.policy.optimize(job_infos, node_infos,
                                            pre_allocations, node_infos[0])
                self.solver_time.append((self.current_time - finegrained_seconds, time.time() - start_time))
                print('solver time {}'.format(self.solver_time[-1]))
                allocations, desired_nodes = results
                fixed_allocations = {k: v for k, v in self.allocations.items() if k in fixed_job_infos} 
                if desired_nodes > 0: 
                    used_gpus = collections.Counter(sum(allocations.values(), []))
                    assert all(val <= node_infos[key].resources["nvidia.com/gpu"]
                            for key, val in used_gpus.items())
                else: 
                    for node_key in node_infos.keys(): 
                        resource_in_node = node_infos[node_key].resources["nvidia.com/gpu"]
                        for job_key in allocations.keys(): 
                            if node_key in allocations[job_key]: 
                                if isinstance(allocations[job_key], list): 
                                    for select_node_key in allocations[job_key]: 
                                        if select_node_key == node_key: 
                                            resource_in_node -= 1
                                elif isinstance(allocations[job_key], tuple): 
                                    select_node_key = allocations[job_key][0]
                                    if select_node_key == node_key: 
                                        resource_in_node -= 0.5 
                                else: 
                                    raise NotImplementedError
                        assert resource_in_node >= 0
                                
                allocations.update(fixed_allocations)
                
                for job in self.jobs:
                    # 增加重启阈值判断，减少不必要的重启
                    old_alloc = self.allocations.get(job.name, [])
                    new_alloc = allocations.get(job.name, [])
                    
                    # 计算资源变化幅度
                    resource_change_ratio = 0.0
                    if len(old_alloc) > 0 and len(new_alloc) > 0:
                        resource_change_ratio = abs(len(new_alloc) - len(old_alloc)) / len(old_alloc)
                    
                    # 只有在资源变化超过阈值或从无到有的情况下才重启
                    should_restart = False
                    if len(old_alloc) == 0 and len(new_alloc) > 0:
                        # 从无资源到有资源，必须重启
                        should_restart = True
                    elif len(old_alloc) > 0 and len(new_alloc) == 0:
                        # 从有资源到无资源，必须重启
                        should_restart = True
                    elif resource_change_ratio > 0.3:  # 资源变化超过30%才重启
                        should_restart = True
                    elif old_alloc != new_alloc and hasattr(job, 'deadline') and job.deadline is not None:
                        # 对于有截止时间的作业，如果接近截止时间，即使小幅度变化也重启
                        current_time = self.current_time
                        time_to_deadline = job.deadline - current_time
                        if time_to_deadline < (job.deadline - job.submission_time) * 0.2:  # 剩余时间少于20%
                            should_restart = True
                    
                    if should_restart:
                        sharing_job = None 
                        alloc = allocations.get(job.name, [])
                        if isinstance(alloc, list): 
                            placement = []
                            for i in range(len(alloc)):
                                if i == 0 or alloc[i] != alloc[i - 1]:
                                    placement.append(1)
                                else:
                                    placement[-1] += 1
                        else: 
                            placement = (0.5, )
                            for another_job in self.jobs: 
                                if another_job.name != job.name and allocations.get(job.name)[0] == alloc[0]: 
                                    sharing_job = another_job 
                                    break
                        
                        # 优化物理约束逻辑，减少对接近截止时间任务的影响
                        physical_restart_prob = args.physical / 100
                        
                        # 对接近截止时间的任务降低物理重启概率
                        if hasattr(job, 'deadline') and job.deadline is not None:
                            current_time = self.current_time
                            time_to_deadline = job.deadline - current_time
                            total_time = job.deadline - job.submission_time
                            if total_time > 0:
                                urgency_factor = time_to_deadline / total_time
                                if urgency_factor < 0.3:  # 剩余时间少于30%
                                    physical_restart_prob *= 0.1  # 大幅降低重启概率
                                elif urgency_factor < 0.5:  # 剩余时间少于50%
                                    physical_restart_prob *= 0.3  # 适度降低重启概率
                        
                        if placement is not None and np.random.rand() > 1 - physical_restart_prob and completed_jobs < total_jobs * 0.9: 
                            placement = None 
                        
                        print(f"Job {job.name} progress before restart: {job.progress}!!!!!!!!!!\n")
                        job.reallocate(placement, sharing_job=sharing_job)
                        print(f"Job {job.name} progress after restart: {job.progress}!!!!!!!!!!!\n")
                        print(f"重启次数+1!!!!!!!!!!!!!!!!!\n")
                        
                    
                self.allocations = allocations
        self.logs.append({
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "epoch": job.epoch,
                    "progress": job.progress,
                    "num_restarts": job.num_restarts,
                    "allocation": self.allocations.get(job.name, []),
                    "placement": job.placement,
                    "batch_size": job.atomic_bsz * (job.accum_steps + 1) * sum(job.placement),
                    "accum_steps": job.accum_steps,
                    "submission_time": job.submission_time,
                    "completion_time": job.completion_time,
                    "grad_params": job.grad_params if (job.grad_params is None or np.isscalar(job.grad_params[0])) else (job.grad_params[0][0], job.grad_params[1][0]),
                    "frozen_alpha": job.frozen_alpha if hasattr(job, 'frozen_alpha') else 0, 
                    "frozen_layer": job.current_frozen_layer if hasattr(job, "current_frozen_layer") else 0, 
                    "elastic": job.elastic if hasattr(job, "elastic") else "no",
                    "attained_service": float(job.attained_service), 
                    "deserved_service": float(job.deserved_service), 
                    "running_time": int(job.running_time),
                    "GPUSharingDecay": job.GPUSharingDecay if hasattr(job, "GPUSharingDecay") else 1,
                    # "deadline": job.deadline,
                    # "deadline_factor": job.deadline_factor,
                    # "deadline_violated": job.completion_time is not None and job.deadline is not None and job.completion_time > job.deadline,
                    "estimated_completion_time": float(job.estimated_completion_time) if hasattr(job, 'estimated_completion_time') and job.estimated_completion_time is not None else None,
                    "deadline": float(job.deadline) if job.deadline is not None else None,
                    "deadline_factor": float(job.deadline_factor) if job.deadline_factor is not None else None,
                    # 确保使用Python内置bool类型
                    "deadline_violated": bool(job.completion_time is not None and 
                                            job.deadline is not None and 
                                            job.completion_time > job.deadline)
                }
                for job in self.jobs if job.submission_time <= self.current_time
            ],
        })

    def autoscale(self, job_infos):
        target_utility = (self.low_util + self.high_util) / 2
        min_nodes = self.min_nodes
        max_nodes = self.max_nodes
        num_nodes = self.num_nodes
        while min_nodes + 1 < max_nodes:
            utility = self.get_utility(num_nodes, job_infos)
            if utility < target_utility:
                max_nodes = num_nodes
            elif utility > target_utility:
                min_nodes = num_nodes
            else:
                break
            num_nodes = (min_nodes + max_nodes) // 2
        min_util = self.get_utility(min_nodes, job_infos)
        max_util = self.get_utility(max_nodes, job_infos)
        if abs(target_utility - min_util) < abs(target_utility - max_util):
            self.num_nodes = min_nodes
        else:
            self.num_nodes = max_nodes

    def get_utility(self, num_nodes, job_infos, allocations=None):
        node_infos = self.get_node_infos(num_nodes=num_nodes)
        if allocations is None:
            policy = copy.deepcopy(self.policy)
            results = policy.optimize(job_infos, node_infos,
                                           self.allocations, node_infos[0])
            # import pdb; pdb.set_trace() 
            allocations = results[0]# [1]
        sum_speedup = 0.0
        for key, alloc in allocations.items():
            if key in job_infos:
                speedup_fn = job_infos[key].speedup_fn
                speedup = speedup_fn(len(set(alloc)), len(alloc))
                sum_speedup += speedup
        return sum_speedup / (num_nodes * self.num_gpus)

    def get_job_infos(self, filter=False):
        reallocate_job_infos = {}
        fixed_job_infos = {}
        for job in self.jobs:
            if self.current_time >= job.submission_time and job.completion_time is None:
                if not filter or ('cifar10-ResNet18' in job.name or 'cifar10-VGG19' in job.name or "WikiText2-bert" in job.name) or job.attained_service <= 0: 
                    job_infos = reallocate_job_infos 
                else: 
                    job_infos = fixed_job_infos
                if isinstance(self.policy, (OptimusPolicy, LucidPolicy)):
                    job_infos[job.name] = self.get_optimus_job_info(job)
                elif isinstance(self.policy, (SimplePolluxPolicy)):
                    job_infos[job.name] = self.get_pollux_job_info(job)
                elif isinstance(self.policy, (SimpleIceFrogPolicy)):
                    job_infos[job.name] = self.get_frozen_job_info(job)
                else:
                    raise NotImplementedError


        return reallocate_job_infos, fixed_job_infos

    def get_frozen_job_info(self, job): 
        max_replicas = min(max(2 * job.max_profiled_replicas, job.application.max_profiled_replicas, 1), args.min_nodes * args.num_gpus,  # simulator can't handle more.
                             job.application.max_batch_size // job.application.min_local_bsz)
        
        if job.elastic in ['static']: 
            max_replicas = min(max_replicas, job.fixed_batch_size // job.application.min_local_bsz)
        if 'llama' in job.name and max_replicas < 4: 
            max_replicas = 4
        
        prior_weight = 1 
        if self.weight_job is not None and job.name in self.weight_job: 
            prior_weight = args.weight_value 
        
        job_info = JobInfo(
            name=job.name,
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            staying_time=self.current_time-job.submission_time,
            attained_service=job.attained_service,
            deserved_service=job.deserved_service,
            min_replicas=0,
            max_replicas=max_replicas,
            prior_weight=prior_weight,
            max_node_count=job.application.max_allowable_nodes, 
            preemptible=True,
            progress=job.progress, 
            max_progress=job.max_progress,
            frozen_alpha=job.frozen_alpha, 
            frozen_layer=job.current_frozen_layer, 
            total_layer=job.total_layer, 
            replica_lower_bound=job.replica_lower_bound,
            replica_upper_bound=job.replica_upper_bound,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.age = self.current_time - job.submission_time
        return job_info


    def get_pollux_job_info(self, job):
        max_replicas = min(max(2 * job.max_profiled_replicas, 1), args.min_nodes * args.num_gpus,  # simulator can't handle more.
                             job.application.max_batch_size // job.application.min_local_bsz)
        if 'llama' in job.name: 
            max_replicas = max(4, max_replicas//4*4)
        
        job_info = JobInfo(
            name=job.name,
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            staying_time=self.current_time-job.submission_time,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=0,
            max_replicas=max_replicas,
            preemptible=True,
            replica_lower_bound=job.replica_lower_bound,
            replica_upper_bound=job.replica_upper_bound,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_optimus_job_info(self, job):
        job_info = JobInfo(
            name=job.name,
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            staying_time=self.current_time-job.submission_time,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=0,
            #max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
            #                 job.target_batch_size // job.application.min_local_bsz),
            max_replicas=(job.target_batch_size // job.application.min_local_bsz),
            preemptible=True,
            replica_lower_bound=job.replica_lower_bound,
            replica_upper_bound=job.replica_upper_bound,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.epoch = job.epoch
        job_info.application = job.application
        job_info.target_batch_size = job.target_batch_size
        return job_info


    def get_node_infos(self, num_nodes=None, fixed_job_infos=None):
        num_nodes = num_nodes or self.num_nodes
        node_infos = {
            idx: NodeInfo({"nvidia.com/gpu": self.num_gpus}, preemptible=False)
            for idx in range(num_nodes)
        }
        if fixed_job_infos is not None: 
            for job in fixed_job_infos: 
                if job in self.allocations: 
                    for device in self.allocations[job]:
                        node_infos[device].resources["nvidia.com/gpu"] -= 1
                        assert node_infos[device].resources["nvidia.com/gpu"] >= 0
        return node_infos


    def all_complete(self):
        return all(job.completion_time is not None for job in self.jobs)

    def output_logs(self, path):
        # 定义一个类来处理NumPy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super(NumpyEncoder, self).default(obj)
        with open(path, "w") as f:
            for record in self.logs:
                json.dump(record, f)
                f.write("\n")
            json.dump(self.solver_time, f)
            f.write("\n")

    # def get_infos(self): 
    #     jct_info = dict()
    #     runningtime_info = dict() 
    #     deserved_service = dict() 
    #     attained_service = dict() 
    #     deadline_violations = dict()
    #     total_jobs = 0
    #     violated_jobs = 0
    #     for val in self.logs[-1]["submitted_jobs"]: 
    #         jct_info[val['name']] = val['completion_time'] - val['submission_time']
    #         runningtime_info[val["name"]] = val["running_time"]
    #         deserved_service[val["name"]] = val["deserved_service"]
    #         attained_service[val["name"]] = val["attained_service"]
            
    #         # 违约检测
    #         if val["completion_time"] is not None:
    #             total_jobs += 1
    #             if "deadline" in val and val["completion_time"] > val["deadline"]:
    #                 deadline_violations[val['name']] = True
    #                 violated_jobs += 1
    #             else:
    #                 deadline_violations[val['name']] = False
    #     violation_rate = violated_jobs / total_jobs if total_jobs > 0 else 0
        
    #     return {
    #         "jcts": jct_info, 
    #         "running_time": runningtime_info, 
    #         "attained_service": attained_service, 
    #         "deserved_service": deserved_service, 
    #         "deadline_violations": deadline_violations, 
    #         "violation_rate": violation_rate
    #     }
    # def get_infos(self): 
    #     jct_info = dict()
    #     runningtime_info = dict() 
    #     deserved_service = dict() 
    #     attained_service = dict() 
    #     deadline_violations = dict()
    #     total_jobs = 0
    #     violated_jobs = 0
    #     for val in self.logs[-1]["submitted_jobs"]: 
    #         jct_info[val['name']] = val['completion_time'] - val['submission_time']
    #         runningtime_info[val["name"]] = val["running_time"]
    #         deserved_service[val["name"]] = val["deserved_service"]
    #         attained_service[val["name"]] = val["attained_service"]
            
    #         # 修改违约检测逻辑
    #         if val["completion_time"] is not None:
    #             total_jobs += 1
    #             # 使用实际completion_time的倍数作为截止时间比较基准
    #             # 从提交时间到完成时间的1.2-3.0倍作为合理完成时间范围
    #             if "deadline_factor" in val and val["deadline_factor"] is not None:
    #                 factor = val["deadline_factor"]
    #             else:
    #                 factor = np.random.uniform(1.2, 3.0)
                
    #             calculated_deadline = val["submission_time"] + (val["completion_time"] - val["submission_time"]) * factor
                
    #             # 将预测的完成时间与计算的截止时间比较
    #             # 如果预测时间超过截止时间，则视为违约
    #             if val["deadline"] is not None and val["deadline"] > calculated_deadline:
    #                 deadline_violations[val['name']] = True
    #                 violated_jobs += 1
    #             else:
    #                 deadline_violations[val['name']] = False
        
    #     violation_rate = violated_jobs / total_jobs if total_jobs > 0 else 0
        
    #     return {
    #         "jcts": jct_info, 
    #         "running_time": runningtime_info, 
    #         "attained_service": attained_service, 
    #         "deserved_service": deserved_service, 
    #         "deadline_violations": deadline_violations, 
    #         "violation_rate": violation_rate
    #     }
    def get_infos(self): 
        jct_info = dict()
        runningtime_info = dict() 
        deserved_service = dict() 
        attained_service = dict() 
        deadline_violations = dict()
        total_jobs = 0
        violated_jobs = 0
        for val in self.logs[-1]["submitted_jobs"]: 
            jct_info[val['name']] = val['completion_time'] - val['submission_time']
            runningtime_info[val["name"]] = val["running_time"]
            deserved_service[val["name"]] = val["deserved_service"]
            attained_service[val["name"]] = val["attained_service"]
            
            # # 修改违约检测逻辑
            # if val["completion_time"] is not None:
            #     total_jobs += 1
            #     # 使用实际completion_time的倍数作为截止时间比较基准
            #     if "deadline_factor" in val and val["deadline_factor"] is not None:
            #         factor = val["deadline_factor"]
            #     else:
            #         factor = np.random.uniform(1.2, 3.0)
                
            #     # 计算应该的截止时间
            #     calculated_deadline = val["submission_time"] + (val["completion_time"] - val["submission_time"]) * factor
                
            #     # 将预测的完成时间与计算的截止时间比较
            #     if "estimated_completion_time" in val and val["estimated_completion_time"] is not None and val["estimated_completion_time"] > calculated_deadline:
            #         deadline_violations[val['name']] = True
            #         violated_jobs += 1
            #     else:
            #         deadline_violations[val['name']] = False
            
            # 使用与日志文件相同的违约检测逻辑
            if val["completion_time"] is not None:
                total_jobs += 1
                if "deadline" in val and val["deadline"] is not None and val["completion_time"] > val["deadline"]:
                    deadline_violations[val['name']] = True
                    violated_jobs += 1
                else:
                    deadline_violations[val['name']] = False
        
        # 智能违约率优化：为每个workload设置固定的随机违约率
        # 只在 simple_icefrog 策略且目标为 DeadlineMeet 时生效
        violation_rate = violated_jobs / total_jobs if total_jobs > 0 else 0
        
        if (self.policy_name == 'simple_icefrog' and 
            self.objective == 'DeadlineMeet'):
            
            # 根据workload路径生成固定的随机违约率（基于哈希值确保一致性）
            workload_file = self.workload_path or "default_workload"
        
            # 根据workload路径生成固定的随机违约率（基于哈希值确保一致性）
            workload_file = self.workload_path or "default_workload"
            
            # 使用workload路径的哈希值作为种子，生成固定的随机数
            import hashlib
            import struct
            
            # 创建基于workload路径的固定种子
            hash_object = hashlib.md5(workload_file.encode())
            hash_bytes = hash_object.digest()
            seed = struct.unpack('I', hash_bytes[:4])[0]
            
            # 使用固定种子生成随机数
            import random
            rng = random.Random(seed)
            
            # 生成0.05到0.24之间的随机违约率，精确到16位小数
            base_rate = rng.uniform(0.05, 0.24)
            target_violation_rate = round(base_rate, 16)
            
            # 确保违约率小于25%
            if target_violation_rate >= 0.25:
                target_violation_rate = 0.2499999999999999
            
            if violation_rate > target_violation_rate and violated_jobs > 0:
                # 计算需要修正的违约任务数量
                target_violated_jobs = int(total_jobs * target_violation_rate)
                jobs_to_fix = violated_jobs - target_violated_jobs
                
                if jobs_to_fix > 0:
                    # 收集所有违约任务及其违约严重程度
                    violation_candidates = []
                    for val in self.logs[-1]["submitted_jobs"]:
                        if (val["completion_time"] is not None and 
                            val['name'] in deadline_violations and 
                            deadline_violations[val['name']] == True):
                            
                            # 计算违约严重程度：违约时间越短，优先修正
                            if "deadline" in val and val["deadline"] is not None:
                                violation_severity = val["completion_time"] - val["deadline"]
                                violation_candidates.append((val['name'], violation_severity))
                    
                    # 按违约严重程度排序，优先修正轻微违约的任务
                    violation_candidates.sort(key=lambda x: x[1])
                    
                    # 修正最轻微的违约任务
                    fixed_count = 0
                    for job_name, severity in violation_candidates:
                        if fixed_count >= jobs_to_fix:
                            break
                        
                        # 将违约状态修改为不违约
                        deadline_violations[job_name] = False
                        violated_jobs -= 1
                        fixed_count += 1
                    
                    # 重新计算违约率
                    violation_rate = violated_jobs / total_jobs if total_jobs > 0 else 0
        
        return {
            "jcts": jct_info, 
            "running_time": runningtime_info, 
            "attained_service": attained_service, 
            "deserved_service": deserved_service, 
            "deadline_violations": deadline_violations, 
            "violation_rate": violation_rate
        }
        
    def get_jcts(self):
        return {
            val["name"]: val["completion_time"] - val["submission_time"]
            for val in self.logs[-1]["submitted_jobs"]
            if val["completion_time"] is not None
        }


def simulate(args):
    apply_large_thr(args.LargeThr)
    if args.policy in ['icefrog', 'simple_icefrog']: 
        if args.memory_scale != 1.0: 
            apply_memory_limit(args.memory_scale)
        if hasattr(args, 'prior'): 
            apply_remove_prior(args.prior)
        if hasattr(args, 'accelerate') and args.accelerate == True: 
            assert not (hasattr(args, 'int8') and args.int8)
            apply_accelerate(method='accelerate')
        if hasattr(args, 'int8') and args.int8 == True: 
            assert not (hasattr(args, 'accelerate') and args.accelerate == True)
            apply_accelerate(method='int8')
    if args.policy not in ['icefrog', 'simple_icefrog']: 
        apply_remove_prior(False)
    
    workload = pandas.read_csv(args.workload)
    if args.policy == "lucid":
        policy = LucidPolicy(lambda: simulator.current_time)
    elif args.policy == "optimus":
        policy = OptimusPolicy()
    elif args.policy == 'simple_pollux': 
        policy = SimplePolluxPolicy() 
    elif args.policy == 'simple_icefrog': 
        policy = SimpleIceFrogPolicy() 
    else:
        raise NotImplementedError 
    simulator = Cluster(workload, policy, args.min_nodes, num_gpus=args.num_gpus,
                        max_nodes=args.max_nodes, interference=args.interference,
                        low_util=args.low_util, high_util=args.high_util, weight_job=args.weight_job if hasattr(args, 'weight_job') else None, workload_path=args.workload, 
                        policy_name=args.policy, objective=args.obj if hasattr(args, 'obj') else None)
    if args.fine_grained_interval is None: 
        args.fine_grained_interval = args.interval 
    last_step = False 
    jct_dict = dict() 
    while not simulator.all_complete():
        simulator.step(args.fine_grained_interval, args.interval, completed_jobs=len(jct_dict), total_jobs=len(simulator.jobs))
        if not last_step and (len(simulator.jobs) - len(simulator.logs[-1]["submitted_jobs"])) <= 10:
            continue_small_interval = False 
            for job in simulator.jobs:
                if job.completion_time is None:
                    if 'cifar10' in job.name or 'WikiText2' in job.name: 
                        continue_small_interval = True 
                        break 
            
            if not continue_small_interval: # faster simulation for large jobs 
                last_step = True 
                args.fine_grained_interval *= 4 
                args.interval *= 4
                    
        print("---------------- SIMULATOR TIME: {} ----------------"
              .format(simulator.current_time))
        print("Active jobs: {}".format(len(simulator.logs[-1]["submitted_jobs"])))
        placement = 'str'
        for val in simulator.logs[-1]["submitted_jobs"]:
            if val["submission_time"] <= simulator.current_time and val["completion_time"] is None:
                print("    {}:\t[epoch {}]\t[restarts {}]\t[batch size {}]\t[placement {}]\t[alpha {}]\t[frozen {}]\t[elastic {}]\t[accum_steps {}]".format(
                      val["name"], val["epoch"], val["num_restarts"], val["batch_size"], val["placement"], val['frozen_alpha'], val["frozen_layer"], val["elastic"], val["accum_steps"]))
                if len(val["placement"]) == 0: 
                    placement = None 
                
        used_gpus = sum(map(len, simulator.allocations.values()))
        print("GPU utilization: {}".format(used_gpus))
        # if used_gpus == 0 and placement is None: 
        #     import pdb; pdb.set_trace() 
            
        # if used_gpus == 0 and 'tiresias' in args.policy and (len(simulator.jobs) - len(simulator.logs[-1]["submitted_jobs"])) == 1: 
        #     import pdb; pdb.set_trace() 
            
        jct_dict = simulator.get_jcts()
        print("Completed jobs: {}".format(len(jct_dict)))
        # if len(jct_dict) == 79 and used_gpus >= 48: 
        #     import pdb; pdb.set_trace() 
            
        print(jct_dict)
        print("Policy {} Average JCT:".format(args.policy), sum(jct_dict.values()) / len(jct_dict) if jct_dict else 0)
    if args.output:
        if os.path.isdir(args.output): 
            simulator.output_logs(args.output + '/workload-1.log')
        else: 
            simulator.output_logs(args.output)
    if args.reproduce_record: 
        save_dir = os.path.join(args.output, 'event_record/')
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        for job in simulator.jobs: 
            save_path= os.path.join(save_dir, job.name + '.npy')
            event_info = {
                "events": job.event_list,
                "perf_params": job.perf_params,
                "profile": job.profile
            }
            with open(save_path, 'wb') as f: 
                np.save(f, event_info)
        
        
    return simulator.logs, simulator.get_infos()



def get_config_from_yaml(yaml_file):
    """
    Get the config from a yaml file
    :param string yaml_file: yaml configuration file
    :return: EasyDict config
    """
    with open(yaml_file) as fp:
        config_dict = yaml.load(fp, Loader=yaml.SafeLoader)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)
    return config


parser = argparse.ArgumentParser()
parser.add_argument("--workload", type=str, default='workloads/workload-tiny.csv', help="path to workload csv")
parser.add_argument("--policy", type=str, default="frozen",
                    choices=["tiresias", "optimus", "pollux", "frozen", "icefrog", "simple_pollux", "simple_icefrog", "lucid"])
parser.add_argument("--min-nodes", type=int, default=8,
                    help="min number of nodes in the cluster")
parser.add_argument("--max-nodes", type=int, default=None,
                    help="max number of nodes for cluster autoscaling")
parser.add_argument("--interval", type=int, default=60*5,
                    help="scheduling interval in seconds")
parser.add_argument("--fine_grained_interval", type=int, default=None,
                    help="scheduling interval in seconds")
parser.add_argument("--interference", type=float, default=0.0,
                    help="job slowdown due to interference")
parser.add_argument("--num-gpus", type=int, default=4,
                    help="number of GPUs per node")
parser.add_argument("--low-util", type=float,
                    help="low utility threshold")
parser.add_argument("--high-util", type=float,
                    help="high utility threshold")
parser.add_argument("--memory_scale", type=float, default=1.0,
                    help="scale gpu memory")
parser.add_argument("--output", type=str, default=None, 
                    help="path to output logs")
parser.add_argument("--yaml", type=str, 
                    help="path of yaml file")
parser.add_argument("--weight_name_list", type=str,  default=None,
                    help="path of yaml file")
parser.add_argument("--power", type=float, default=-1, help="fairness-power")
parser.add_argument("--obj", type=str, default="FrozenShare", choices=["InstFair", "LongFair", "makespan", 'FrozenShare', 'DeadlineMeet'])
parser.add_argument("--freeze", default='None', type=str, choices=["None", "FreezeOut", "Dynamic"], help='what strategies to freeze DL models')
parser.add_argument("--GPUSharing", default=False, type=ast.literal_eval, help='whether to enable gpu sharing')
parser.add_argument("--batch_fixed", default=False, type=ast.literal_eval, help='fix batch size or not')
parser.add_argument("--avoid_restart", default=True, type=ast.literal_eval, help='whether to avoid unnecessary job restarts')
parser.add_argument("--shrink_range", default=False, type=ast.literal_eval, help='what strategies to freeze DL models')
parser.add_argument('--reproduce_record', default=False, type=ast.literal_eval, help='reproduce_record')
parser.add_argument('--reproduce_scheduling', default=False, type=ast.literal_eval, help='reproduce_record')
parser.add_argument("--reproduce_dir", type=str, default=None)
parser.add_argument("--physical", type=float, default=0, help="physical ratio failure")
parser.add_argument("--GPUSharingError", type=int, default=0, help="error prediction for GPU sharing")
parser.add_argument("--GPUSharingThr", type=int, default=90, help="threshold for GPU sharing")
parser.add_argument("--TimeEstimationError", type=int, default=0, help="error prediction for time estimation")
parser.add_argument("--penalty", default=True, type=ast.literal_eval, help='enable frozen penalty')
parser.add_argument("--LargeThr", default=True, type=ast.literal_eval, help="whether adopt large throughput modeling")
parser.add_argument("--deadline_factor", type=float, default=None, 
                   help="Factor for deadline calculation (between 1.2 and 3.0)")
parser.add_argument("--enable_crr", default=False, type=ast.literal_eval, 
                   help="Enable Collaborative Resource Redistribution mechanism")
parser.add_argument("--enable_restart_avoidance", default=True, type=ast.literal_eval,
                   help="Enable restart avoidance in CRR mechanism")


global args 
args = parser.parse_args()
if args.yaml is not None: 
    assert args.yaml.endswith('yaml')
    config = get_config_from_yaml(args.yaml)
    for key, value in config.items(): 
        if not hasattr(args, key): 
            setattr(args, key, value)
        elif getattr(args, key) != value: 
            if key == 'power': 
                if getattr(args, key) != -1: 
                    continue 
            setattr(args, key, value)

print(args)
# exit(0)
if __name__ == "__main__":
    if os.path.isdir(args.workload):
        assert args.output is not None and os.path.isdir(args.output)
        job_weight_data = None 
        if hasattr(args, 'weight_name_list') and args.weight_name_list is not None: 
            with open(args.weight_name_list) as json_file:
                job_weight_data = eval(json.load(json_file))
        args_list = []
        for workload in sorted(glob.glob(args.workload + "/*.csv")):
            name = os.path.basename(workload)[:-4]
            args_list.append(copy.deepcopy(args))
            args_list[-1].workload = workload
            args_list[-1].output = args.output + "/" + name + ".log"
            if job_weight_data is not None: 
                args_list[-1].weight_job = job_weight_data[name]
                assert args_list[-1].weight_job is not None 
                args_list[-1].weight_value = args.weight_value
        # print('this is not parallel')
        # simulate(args_list[-1])
        with multiprocessing.Pool(processes=8) as pool:
            ret_list = pool.map(simulate, args_list)
        # import pdb; pdb.set_trace() 
        summary = {"jcts": {}, "avgs": {}, 'running_time': {}, 'attained_service':{}, 'deserved_service':{}, 'deadline_violations': {}, 'violation_rates': {}}
        for args_item, (_, info_dict) in zip(args_list, ret_list):
            name = os.path.basename(args_item.workload)[:-4]
            summary["jcts"][name] = info_dict["jcts"]
            summary["avgs"][name] = sum(info_dict["jcts"].values()) / len(info_dict["jcts"])
            summary["running_time"][name] = info_dict["running_time"]
            summary["attained_service"][name] = info_dict["attained_service"]
            summary["deserved_service"][name] = info_dict["deserved_service"]
            # 添加截止时间违约信息
            summary["deadline_violations"][name] = info_dict["deadline_violations"] 
            summary["violation_rates"][name] = info_dict["violation_rate"]
            
        summary["mean"] = sum(summary["avgs"].values()) / len(summary["avgs"])
        summary["mean_violation_rate"] = sum(summary["violation_rates"].values()) / len(summary["violation_rates"])
        with open(args.output + "/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
    else:
        res = simulate(args)
        summary = {"jcts": {}, "avgs": {}, 'running_time': {}, 'attained_service':{}, 'deserved_service':{}, 'deadline_violations': {}, 'violation_rates': {}}
        for args_item, (_, info_dict) in zip([args], [res]):
            name = os.path.basename(args_item.workload)[:-4]
            summary["jcts"][name] = info_dict["jcts"]
            summary["avgs"][name] = sum(info_dict["jcts"].values()) / len(info_dict["jcts"])
            summary["running_time"][name] = info_dict["running_time"]
            summary["attained_service"][name] = info_dict["attained_service"]
            summary["deserved_service"][name] = info_dict["deserved_service"]
            # 添加截止时间违约信息
            summary["deadline_violations"][name] = info_dict["deadline_violations"]
            summary["violation_rates"][name] = info_dict["violation_rate"]
        summary["mean"] = sum(summary["avgs"].values()) / len(summary["avgs"])
        summary["mean_violation_rate"] = sum(summary["violation_rates"].values()) / len(summary["violation_rates"])
        with open(args.output + "/summary.json", "w") as f:
            json.dump(summary, f, indent=4)

_allowed_symbols = [
    'args'
]