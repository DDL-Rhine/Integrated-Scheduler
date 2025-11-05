#!/usr/bin/env python3

"""
Simple IceFrog with Smart CRR (Collaborative Resource Redistribution)
智能协作式资源重新分配调度器

核心设计理念：
1. 保守而精准：只在明确有收益时才进行资源重分配
2. 最小扰动：尽量减少对现有任务的影响  
3. 专注效果：优先改善任务完成时间和违约率
"""

import os
import sys
import numpy as np
import time
import math
import copy
import cvxpy as cp
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from applications import APPLICATIONS
from speedup import SpeedupFunction
from utils import *
from job import Job
import utils

class SimpleIceFrogSmartCRR:
    def __init__(self, trace_handler, args, environment=None):
        self._name = "simple_icefrog_smart_crr"
        self._trace_handler = trace_handler
        self._environment = environment
        self.allow_update = args.allow_update
        self.goodput_fn = args.goodput_fn
        self.num_nodes = args.num_nodes
        self.num_gpus_per_node = args.num_gpus_per_node
        self.enable_crr = getattr(args, 'enable_crr', True)  # 默认启用CRR
        
        # 简化的参数配置
        self.CRR_TRIGGER_THRESHOLD = 0.15  # CRR触发的紧急程度阈值
        self.CRR_MAX_CRITICAL_JOBS = 2     # 每次处理的最大紧急任务数
        self.MIN_DONOR_RESOURCES = 2       # 资源提供者的最小资源数
        
        # 统计信息
        self.crr_stats = []
        
    def schedule(self, jobs_dict, nodes, prev_allocations, event):
        """主调度方法"""
        if len(jobs_dict) == 0:
            return {}
        
        # 使用标准ElasticFlow调度算法
        standard_allocations = self._elastic_flow_schedule(jobs_dict, nodes)
        
        # 如果启用了CRR，尝试优化分配
        if self.enable_crr and len(jobs_dict) >= 2:
            optimized_allocations = self._smart_crr_optimization(
                jobs_dict, nodes, standard_allocations
            )
            return optimized_allocations
        
        return standard_allocations
    
    def _elastic_flow_schedule(self, jobs_dict, nodes):
        """标准ElasticFlow调度算法（简化版本）"""
        allocations = {}
        jobs = list(jobs_dict.values())
        
        # 按优先级排序任务
        jobs.sort(key=lambda job: (
            getattr(job, 'deadline', float('inf')),  # 按截止时间排序
            -getattr(job, 'prior_weight', 1)         # 按权重排序
        ))
        
        # 计算可用资源
        total_gpus = sum(node.resources.get("nvidia.com/gpu", 0) for node in nodes.values())
        allocated_gpus = 0
        
        # 为每个任务分配资源
        for job in jobs:
            if allocated_gpus >= total_gpus:
                allocations[job.name] = []
                continue
            
            # 计算此任务可获得的资源
            remaining_gpus = total_gpus - allocated_gpus
            job_allocation = min(job.max_replicas, remaining_gpus)
            
            # 分配到具体节点
            allocation_list = []
            nodes_list = list(nodes.keys())
            
            for i in range(job_allocation):
                node_id = nodes_list[i % len(nodes_list)]
                allocation_list.append(node_id)
            
            allocations[job.name] = allocation_list
            allocated_gpus += job_allocation
        
        return allocations
    
    def _smart_crr_optimization(self, jobs_dict, nodes, base_allocations):
        """智能CRR优化"""
        current_time = getattr(jobs_dict[list(jobs_dict.keys())[0]], 'current_time', 0)
        
        # Step 1: 检查是否需要CRR
        if not self._should_trigger_crr(jobs_dict, base_allocations, current_time):
            return base_allocations
        
        # Step 2: 识别紧急任务
        critical_jobs = self._identify_critical_jobs(jobs_dict, base_allocations, current_time)
        
        if not critical_jobs:
            return base_allocations
        
        # Step 3: 执行资源重分配
        optimized_allocations = copy.deepcopy(base_allocations)
        
        helped_count = 0
        for critical_job_name in critical_jobs[:self.CRR_MAX_CRITICAL_JOBS]:
            if self._help_critical_job(critical_job_name, jobs_dict, optimized_allocations, current_time):
                helped_count += 1
        
        # Step 4: 记录统计信息
        if helped_count > 0:
            self.crr_stats.append({
                'timestamp': current_time,
                'critical_jobs_identified': len(critical_jobs),
                'critical_jobs_helped': helped_count
            })
        
        return optimized_allocations if helped_count > 0 else base_allocations
    
    def _should_trigger_crr(self, jobs_dict, allocations, current_time):
        """判断是否应该触发CRR"""
        critical_count = 0
        idle_resource_count = 0
        
        for job in jobs_dict.values():
            if not hasattr(job, 'deadline') or not hasattr(job, 'remaining_time'):
                continue
            
            # 检查紧急程度
            time_to_deadline = max(job.deadline - current_time, 1)
            urgency_ratio = job.remaining_time / time_to_deadline
            
            if urgency_ratio > (1 + self.CRR_TRIGGER_THRESHOLD):
                critical_count += 1
            
            # 检查是否有闲置资源可以重分配
            current_alloc = len(allocations.get(job.name, []))
            if current_alloc >= self.MIN_DONOR_RESOURCES and urgency_ratio < 0.8:
                idle_resource_count += 1
        
        # 只有同时有紧急任务和闲置资源时才触发
        return critical_count >= 1 and idle_resource_count >= 1
    
    def _identify_critical_jobs(self, jobs_dict, allocations, current_time):
        """识别紧急任务"""
        critical_jobs = []
        
        for job in jobs_dict.values():
            if not hasattr(job, 'deadline') or not hasattr(job, 'remaining_time'):
                continue
            
            time_to_deadline = max(job.deadline - current_time, 1)
            urgency_ratio = job.remaining_time / time_to_deadline
            current_alloc = len(allocations.get(job.name, []))
            
            # 紧急条件：预期完成时间超过截止时间，且还有扩展空间
            if urgency_ratio > (1 + self.CRR_TRIGGER_THRESHOLD) and current_alloc < job.max_replicas:
                critical_jobs.append((job.name, urgency_ratio))
        
        # 按紧急程度排序
        critical_jobs.sort(key=lambda x: x[1], reverse=True)
        return [job_name for job_name, _ in critical_jobs]
    
    def _help_critical_job(self, critical_job_name, jobs_dict, allocations, current_time):
        """为紧急任务寻找并分配额外资源"""
        critical_job = jobs_dict[critical_job_name]
        current_alloc = len(allocations.get(critical_job_name, []))
        
        if current_alloc >= critical_job.max_replicas:
            return False  # 已达到最大资源
        
        # 寻找最佳的资源提供者
        best_donor = self._find_best_donor(critical_job_name, jobs_dict, allocations, current_time)
        
        if best_donor:
            # 执行资源转移
            donor_alloc = allocations[best_donor]
            critical_alloc = allocations.get(critical_job_name, [])
            
            if len(donor_alloc) >= self.MIN_DONOR_RESOURCES:
                # 转移一个GPU
                transferred_gpu = donor_alloc.pop()
                critical_alloc.append(transferred_gpu)
                allocations[critical_job_name] = critical_alloc
                return True
        
        return False
    
    def _find_best_donor(self, critical_job_name, jobs_dict, allocations, current_time):
        """寻找最佳的资源提供者"""
        potential_donors = []
        
        for job in jobs_dict.values():
            if job.name == critical_job_name:
                continue
            
            current_alloc = len(allocations.get(job.name, []))
            if current_alloc < self.MIN_DONOR_RESOURCES:
                continue
            
            # 计算提供资源的成本
            if hasattr(job, 'deadline') and hasattr(job, 'remaining_time'):
                time_to_deadline = max(job.deadline - current_time, 1)
                urgency_ratio = job.remaining_time / time_to_deadline
                
                # 优先从不紧急的任务中借用资源
                if urgency_ratio < 0.9:  # 不紧急
                    cost = urgency_ratio  # 越不紧急，成本越低
                    potential_donors.append((job.name, cost))
        
        # 选择成本最低的提供者
        if potential_donors:
            potential_donors.sort(key=lambda x: x[1])
            return potential_donors[0][0]
        
        return None

    def get_name(self):
        return self._name
    
    def get_running_jobs(self):
        """获取当前运行的任务"""
        if hasattr(self, '_current_jobs'):
            return self._current_jobs
        return {}

# 为了兼容性，保持原有的类名和接口
SimpleIceFrog = SimpleIceFrogSmartCRR
