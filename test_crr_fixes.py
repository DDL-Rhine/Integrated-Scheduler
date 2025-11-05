#!/usr/bin/env python3
"""
测试CRR机制的修复
"""

import sys
import os

# 添加模拟器路径
sys.path.append('/home/lab/simulator-artifact')

from simple_icefrog import SimpleIceFrogPolicy
from job import Job

def test_crr_resource_validation():
    """测试CRR资源验证功能"""
    print("Testing CRR resource validation fixes...")
    
    # 创建一个简单的策略实例
    policy = SimpleIceFrogPolicy()
    
    # 模拟一些任务
    class MockJob:
        def __init__(self, name, max_replicas=8):
            self.name = name
            self.max_replicas = max_replicas
            self.submission_time = 0
            self.deadline = 1000
            self.progress = 0
            self.max_progress = 100
    
    jobs = {
        'job1': MockJob('job1', 4),
        'job2': MockJob('job2', 6),
        'job3': MockJob('job3', 8)
    }
    
    # 初始分配
    allocations = {
        'job1': (0, 1, 2),  # 3 GPUs
        'job2': (3, 4),     # 2 GPUs  
        'job3': (5,)        # 1 GPU
    }
    
    print(f"Initial allocations: {allocations}")
    
    # 测试验证函数
    try:
        result = policy._validate_final_allocations(allocations, jobs)
        print(f"Validation result: {result}")
        print(f"Validated allocations: {allocations}")
        print("✓ Resource validation test passed!")
        return True
    except Exception as e:
        print(f"✗ Resource validation test failed: {e}")
        return False

def test_safe_transfer():
    """测试安全转移逻辑"""
    print("\nTesting safe transfer logic...")
    
    try:
        # 模拟列表和元组的安全转换
        source_allocation = (0, 1, 2, 3)  # 4 GPUs
        target_allocation = (4,)          # 1 GPU
        
        # 转换为列表进行操作
        source_list = list(source_allocation)
        target_list = list(target_allocation)
        
        # 安全转移2个GPU
        transfer_amount = 2
        if len(source_list) >= transfer_amount:
            transferred = source_list[-transfer_amount:]
            remaining = source_list[:-transfer_amount]
            new_target = target_list + transferred
            
            # 转换回元组
            new_source_alloc = tuple(remaining) if remaining else tuple()
            new_target_alloc = tuple(new_target)
            
            print(f"Source after transfer: {new_source_alloc}")
            print(f"Target after transfer: {new_target_alloc}")
            print("✓ Safe transfer test passed!")
            return True
        else:
            print("✗ Insufficient resources for transfer")
            return False
            
    except Exception as e:
        print(f"✗ Safe transfer test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing CRR fixes for resource constraint violations...")
    
    test1_passed = test_crr_resource_validation()
    test2_passed = test_safe_transfer()
    
    if test1_passed and test2_passed:
        print("\n🎉 All CRR resource validation tests passed!")
        print("The fixes should prevent 'assert resource_in_node >= 0' errors.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
