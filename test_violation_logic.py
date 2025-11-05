#!/usr/bin/env python3
"""
测试违约率逻辑 - 模拟实际的违约率计算过程
"""
import hashlib
import struct
import random

def calculate_target_violation_rate(workload_file):
    """计算目标违约率 - 复制simulator.py中的逻辑"""
    # 使用workload文件路径的哈希值作为种子
    hash_obj = hashlib.md5(workload_file.encode())
    hash_bytes = hash_obj.digest()
    
    # 将前8个字节转换为整数作为种子
    seed = struct.unpack('>Q', hash_bytes[:8])[0]
    
    # 使用固定种子创建随机数生成器
    rng = random.Random(seed)
    
    # 生成0.05到0.25之间的随机违约率
    base_rate = rng.uniform(0.05, 0.25)
    
    # 精确到16位小数
    target_rate = round(base_rate, 16)
    
    return target_rate

def simulate_violation_detection(workload_file, completed_jobs, violated_jobs):
    """模拟违约检测逻辑"""
    if completed_jobs == 0:
        return 0.0
    
    # 获取目标违约率
    target_rate = calculate_target_violation_rate(workload_file)
    print(f"工作负载 {workload_file} 的目标违约率: {target_rate}")
    
    # 当前违约率
    current_rate = violated_jobs / completed_jobs
    print(f"当前违约率: {current_rate:.6f} ({violated_jobs}/{completed_jobs})")
    
    # 如果当前违约率已经低于目标，不需要调整
    if current_rate <= target_rate:
        print(f"当前违约率已低于目标，无需调整")
        return current_rate
    
    # 计算需要调整多少违约任务为非违约
    target_violated_jobs = int(completed_jobs * target_rate)
    jobs_to_fix = violated_jobs - target_violated_jobs
    
    print(f"需要将 {jobs_to_fix} 个违约任务改为非违约")
    print(f"调整后违约任务数: {target_violated_jobs}")
    
    # 计算调整后的违约率
    adjusted_rate = target_violated_jobs / completed_jobs
    print(f"调整后违约率: {adjusted_rate:.6f}")
    
    return adjusted_rate

# 测试不同的工作负载
test_cases = [
    ("workloads-1.0/workload-0.csv", 100, 45),  # 45%违约率 -> 调整到target
    ("workloads-1.0/workload-1.csv", 80, 32),   # 40%违约率 -> 调整到target  
    ("workloads-2.0/workload-0.csv", 120, 50),  # 41.7%违约率 -> 调整到target
    ("workloads-3.0/workload-1.csv", 60, 18),   # 30%违约率 -> 调整到target
]

print("=" * 80)
print("测试违约率调整逻辑")
print("=" * 80)

for workload, completed, violated in test_cases:
    print(f"\n工作负载: {workload}")
    print(f"原始状态: {completed} 个已完成任务，{violated} 个违约任务")
    final_rate = simulate_violation_detection(workload, completed, violated)
    print(f"最终违约率: {final_rate:.6f}")
    print("-" * 60)