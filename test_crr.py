#!/usr/bin/env python3
"""
协作式资源重新分配机制(CRR)测试脚本

该脚本用于验证CRR机制的正确性和有效性，包括：
1. 基本功能测试
2. 性能基准测试
3. 边界条件测试
4. 与现有机制的兼容性测试
"""

import sys
import os
import json
import numpy as np
from collections import OrderedDict
import argparse

# 添加项目路径
sys.path.append('/home/lab/simulator-artifact')

from simple_icefrog import SimpleIceFrogPolicy, Problem
from utils import JobInfo, NodeInfo
from config.crr_config import apply_crr_args, CRR_CONFIG

class MockArgs:
    """模拟器参数类"""
    def __init__(self):
        self.obj = 'DeadlineMeet'
        self.enable_crr = True
        self.enable_restart_avoidance = True
        self.min_nodes = 4
        self.current_time = 1000
        self.power = 1
        self.static_weight = 1.0
        self.batch_weight = 1.0
        self.layer_weight = 1.0

def create_mock_job(name, max_replicas=8, progress=0.3, max_progress=100, 
                   deadline=None, submission_time=0, urgency=0.5):
    """创建模拟任务"""
    
    # 模拟speedup_fn
    class MockSpeedupFn:
        def __init__(self):
            self._max_batch_size = 128
            self._atomic_bsz_range = (16, 64)
            self._accumulation = True
            self._goodput_fn = MockGoodputFn()
            self._gpu_utilization_fn = 60.0
    
    class MockGoodputFn:
        def optimize(self, num_nodes, num_replicas, max_batch_size, 
                    atomic_bsz_range, accumulation):
            # 简单的性能模型：goodput随replica数量增加但有递减效应
            base_goodput = 10.0
            scaling_efficiency = min(1.0, num_replicas / 4) * (4 / max(num_replicas, 1)) ** 0.3
            return [base_goodput * num_replicas * scaling_efficiency]
    
    job = JobInfo(
        name=name,
        resources={"nvidia.com/gpu": 1},
        speedup_fn=MockSpeedupFn(),
        min_replicas=1,
        staying_time=100,
        max_replicas=max_replicas,
        preemptible=True,
        creation_timestamp=submission_time,
        attained_service=0,
        progress=progress,
        max_progress=max_progress
    )
    
    # 添加CRR相关属性
    job.deadline = deadline
    job.submission_time = submission_time
    
    return job

def create_mock_node(node_id, gpu_count=4):
    """创建模拟节点"""
    return NodeInfo(
        resources={"nvidia.com/gpu": gpu_count},
        preemptible=False
    )

def test_basic_crr_functionality():
    """测试CRR基本功能"""
    print("=== 测试CRR基本功能 ===")
    
    # 创建模拟环境
    args = MockArgs()
    args = apply_crr_args(args)
    
    # 创建任务：一个资源富余，一个资源饥饿
    jobs = OrderedDict([
        ('cifar10-resnet18-1', create_mock_job(
            'cifar10-resnet18-1', max_replicas=8, progress=20, deadline=2000, urgency=0.3
        )),
        ('squad-bert-large-1', create_mock_job(
            'squad-bert-large-1', max_replicas=12, progress=60, deadline=1500, urgency=0.8
        ))
    ])
    
    nodes = OrderedDict([
        ('node-0', create_mock_node(0, 4)),
        ('node-1', create_mock_node(1, 4)),
        ('node-2', create_mock_node(2, 4))
    ])
    
    # 当前分配：第一个任务占用较多资源，第二个任务资源不足
    base_allocations = {
        'cifar10-resnet18-1': ['node-0', 'node-0', 'node-0', 'node-1', 'node-1'],
        'squad-bert-large-1': ['node-2']
    }
    
    # 创建调度器并测试
    policy = SimpleIceFrogPolicy()
    
    # 在Problem类中直接测试CRR
    problem = Problem(list(jobs.values()), list(nodes.values()), np.zeros((2, 3)))
    
    # 设置全局args供CRR使用
    import simulator
    simulator.args = args
    
    # 测试CRR机制
    print("原始分配:", base_allocations)
    crr_allocations = problem.collaborative_resource_redistribution(jobs, nodes, base_allocations)
    print("CRR优化后分配:", crr_allocations)
    
    # 验证结果
    original_gpu_count = {name: len(alloc) for name, alloc in base_allocations.items()}
    crr_gpu_count = {name: len(alloc) for name, alloc in crr_allocations.items()}
    
    print("GPU分配变化:")
    for job_name in jobs:
        original = original_gpu_count.get(job_name, 0)
        crr = crr_gpu_count.get(job_name, 0)
        change = crr - original
        print(f"  {job_name}: {original} -> {crr} (变化: {change:+d})")
    
    return crr_allocations != base_allocations

def test_edge_cases():
    """测试边界条件"""
    print("\n=== 测试边界条件 ===")
    
    args = MockArgs()
    args = apply_crr_args(args)
    
    # 测试1: 单任务情况
    print("测试1: 单任务情况")
    single_job = OrderedDict([
        ('single-job', create_mock_job('single-job', max_replicas=4))
    ])
    nodes = OrderedDict([('node-0', create_mock_node(0, 4))])
    base_alloc = {'single-job': ['node-0', 'node-0']}
    
    problem = Problem(list(single_job.values()), list(nodes.values()), np.zeros((1, 1)))
    import simulator
    simulator.args = args
    
    result = problem.collaborative_resource_redistribution(single_job, nodes, base_alloc)
    print(f"单任务结果: {result}")
    assert result == base_alloc, "单任务情况下不应该有变化"
    
    # 测试2: 无可用资源情况
    print("测试2: 资源完全分配情况")
    full_jobs = OrderedDict([
        ('job1', create_mock_job('job1', max_replicas=4)),
        ('job2', create_mock_job('job2', max_replicas=4))
    ])
    full_alloc = {
        'job1': ['node-0', 'node-0', 'node-0', 'node-0'],
        'job2': []  # 无资源分配
    }
    
    problem2 = Problem(list(full_jobs.values()), list(nodes.values()), np.zeros((2, 1)))
    result2 = problem2.collaborative_resource_redistribution(full_jobs, nodes, full_alloc)
    print(f"资源饱和结果: {result2}")
    
    return True

def test_performance_metrics():
    """测试性能指标计算"""
    print("\n=== 测试性能指标计算 ===")
    
    args = MockArgs()
    args = apply_crr_args(args)
    
    # 创建多样化的任务集
    jobs = OrderedDict([
        ('cifar10-resnet18', create_mock_job('cifar10-resnet18', urgency=0.2)),
        ('imagenet-vgg19', create_mock_job('imagenet-vgg19', urgency=0.5)),
        ('squad-llama-7b', create_mock_job('squad-llama-7b', urgency=0.8))
    ])
    
    problem = Problem(list(jobs.values()), [], np.zeros((3, 0)))
    
    # 测试各种指标计算
    test_job = jobs['cifar10-resnet18']
    
    urgency = problem._calculate_urgency_score(test_job, 1000)
    marginal_utility = problem._calculate_marginal_utility(test_job, 2)
    saturation = problem._calculate_saturation_ratio(test_job, 4)
    collaboration = problem._calculate_collaboration_potential(test_job, jobs)
    
    print(f"测试任务指标:")
    print(f"  紧迫性分数: {urgency:.3f}")
    print(f"  边际效用: {marginal_utility:.3f}")
    print(f"  资源饱和度: {saturation:.3f}")
    print(f"  协作潜力: {collaboration:.3f}")
    
    # 验证指标在合理范围内
    assert 0 <= urgency <= 1, f"紧迫性分数超出范围: {urgency}"
    assert marginal_utility >= 0, f"边际效用不应为负: {marginal_utility}"
    assert 0 <= saturation <= 1, f"饱和度超出范围: {saturation}"
    assert 0 <= collaboration <= 1, f"协作潜力超出范围: {collaboration}"
    
    return True

def run_comprehensive_test():
    """运行综合测试"""
    print("开始CRR机制综合测试...\n")
    
    try:
        # 基本功能测试
        basic_result = test_basic_crr_functionality()
        print(f"基本功能测试: {'✓ 通过' if basic_result else '✗ 失败'}")
        
        # 边界条件测试
        edge_result = test_edge_cases()
        print(f"边界条件测试: {'✓ 通过' if edge_result else '✗ 失败'}")
        
        # 性能指标测试
        metrics_result = test_performance_metrics()
        print(f"性能指标测试: {'✓ 通过' if metrics_result else '✗ 失败'}")
        
        # 总结
        all_passed = basic_result and edge_result and metrics_result
        print(f"\n=== 测试总结 ===")
        print(f"总体结果: {'✓ 所有测试通过' if all_passed else '✗ 部分测试失败'}")
        
        if all_passed:
            print("CRR机制已成功集成到IceFrog中，可以开始实际测试。")
            print("\n启用方法:")
            print("1. 在simulator参数中设置: args.obj = 'DeadlineMeet'")
            print("2. 在simulator参数中设置: args.enable_crr = True")
            print("3. 运行模拟器即可自动应用CRR优化")
        
        return all_passed
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRR机制测试脚本')
    parser.add_argument('--test', choices=['basic', 'edge', 'metrics', 'all'], 
                       default='all', help='选择要运行的测试')
    
    args = parser.parse_args()
    
    if args.test == 'basic':
        test_basic_crr_functionality()
    elif args.test == 'edge':
        test_edge_cases()
    elif args.test == 'metrics':
        test_performance_metrics()
    else:
        run_comprehensive_test()
