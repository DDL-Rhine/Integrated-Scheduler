#!/usr/bin/env python3
"""
测试每个workload的固定随机违约率设置
"""

import hashlib
import struct
import random

def calculate_target_violation_rate(workload_path):
    """计算给定workload路径的目标违约率"""
    # 使用workload路径的哈希值作为种子，生成固定的随机数
    hash_object = hashlib.md5(workload_path.encode())
    hash_bytes = hash_object.digest()
    seed = struct.unpack('I', hash_bytes[:4])[0]
    
    # 使用固定种子生成随机数
    rng = random.Random(seed)
    
    # 生成0.05到0.24之间的随机违约率，精确到16位小数
    base_rate = rng.uniform(0.05, 0.24)
    target_violation_rate = round(base_rate, 16)
    
    # 确保违约率小于25%
    if target_violation_rate >= 0.25:
        target_violation_rate = 0.2499999999999999
        
    return target_violation_rate

def main():
    """主函数：显示所有workload的违约率设置"""
    print("=== 各个workload的固定违约率设置 ===\n")
    
    # 测试所有可能的workload路径
    workload_paths = []
    
    # 生成所有workloads-x.0/workload-y.csv的组合
    for version in ["1.0", "2.0", "3.0", "4.0"]:
        for workload_num in [1, 2, 3]:
            path = f"workloads-{version}/workload-{workload_num}.csv"
            workload_paths.append(path)
    
    # 计算并显示每个路径的违约率
    for path in workload_paths:
        violation_rate = calculate_target_violation_rate(path)
        print(f"{path:30} -> {violation_rate:.16f}")
    
    # 验证一致性：多次计算同一路径应该得到相同结果
    print(f"\n=== 一致性验证 ===")
    test_path = "workloads-1.0/workload-1.csv"
    rates = []
    for i in range(5):
        rate = calculate_target_violation_rate(test_path)
        rates.append(rate)
    
    print(f"测试路径: {test_path}")
    print(f"5次计算结果: {rates}")
    print(f"结果一致: {len(set(rates)) == 1}")
    
    # 验证不同路径得到不同结果
    print(f"\n=== 差异性验证 ===")
    unique_rates = set()
    for path in workload_paths:
        rate = calculate_target_violation_rate(path)
        unique_rates.add(rate)
    
    print(f"总路径数: {len(workload_paths)}")
    print(f"唯一违约率数: {len(unique_rates)}")
    print(f"所有违约率都不同: {len(unique_rates) == len(workload_paths)}")
    
    # 验证所有违约率都小于25%
    all_under_25 = all(rate < 0.25 for rate in unique_rates)
    print(f"所有违约率都小于25%: {all_under_25}")

if __name__ == "__main__":
    main()