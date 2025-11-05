#!/usr/bin/env python3
"""
测试不同workload的违约率设置
"""

import os
import subprocess
import json
import time

def run_single_workload(workload_dir, output_suffix):
    """运行单个workload测试"""
    output_dir = f"test_violations_{output_suffix}"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "/home/lab/anaconda3/bin/conda", "run", "-n", "icefrog",
        "python", "-u", "simulator.py",
        "--memory_scale=1.0",
        "--policy=simple_icefrog", 
        "--obj=DeadlineMeet",
        "--avoid_restart=True",
        "--enable_crr=True",
        f"--workload={workload_dir}/",
        "--GPUSharing=True",
        "--yaml=config/macro/fairness-1.yaml",
        f"--output={output_dir}/",
        "--min-nodes=12",
        "--interval=300", 
        "--num-gpus=4"
    ]
    
    print(f"运行 {workload_dir} 测试...")
    try:
        # 设置60秒超时
        result = subprocess.run(cmd, timeout=60, capture_output=True, text=True, cwd="/home/lab/simulator-artifact")
        print(f"{workload_dir} 完成，返回码: {result.returncode}")
        
        # 检查结果
        summary_file = f"/home/lab/simulator-artifact/{output_dir}/summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                data = json.load(f)
                violation_rate = data.get('mean_violation_rate', 'N/A')
                print(f"{workload_dir} 违约率: {violation_rate}")
                return violation_rate
        else:
            print(f"{workload_dir} 没有生成summary.json")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"{workload_dir} 超时")
        return None
    except Exception as e:
        print(f"{workload_dir} 错误: {e}")
        return None

def main():
    """主函数"""
    print("开始测试不同workload的违约率设置...")
    
    workloads = [
        ("workloads-1.0", "1_0"),
        ("workloads-2.0", "2_0"), 
        ("workloads-3.0", "3_0"),
        ("workloads-4.0", "4_0")
    ]
    
    results = {}
    
    for workload_dir, suffix in workloads:
        violation_rate = run_single_workload(workload_dir, suffix)
        results[workload_dir] = violation_rate
        time.sleep(2)  # 短暂暂停
    
    print("\n=== 最终结果汇总 ===")
    for workload_dir, violation_rate in results.items():
        print(f"{workload_dir}: {violation_rate}")

if __name__ == "__main__":
    main()