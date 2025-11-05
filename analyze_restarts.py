#!/usr/bin/env python3
"""
分析各个策略中每个任务的重启次数
"""

import json
import os
import pandas as pd
from collections import defaultdict
import numpy as np

def analyze_restarts_from_logs(results_dir):
    """从日志文件中分析重启次数（细分到 workload 组合）"""
    # 结构：restart_stats[workload][strategy][combo][job_name] = restarts
    restart_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # 遍历所有工作负载和策略
    for workload in ['workload-1.0', 'workload-2.0', 'workload-3.0', 'workload-4.0']:
        workload_dir = os.path.join(results_dir, workload)
        if not os.path.exists(workload_dir):
            continue

        for strategy in os.listdir(workload_dir):
            strategy_dir = os.path.join(workload_dir, strategy)
            if not os.path.isdir(strategy_dir):
                continue

            # 分析每个工作负载文件（组合）
            for workload_file in ['workload-1.log', 'workload-2.log', 'workload-3.log']:
                log_file = os.path.join(strategy_dir, workload_file)
                if not os.path.exists(log_file):
                    continue

                combo = os.path.splitext(workload_file)[0]  # e.g., 'workload-1'
                print(f"分析 {workload}/{strategy}/{workload_file}")

                # 读取日志文件并分析重启次数（对同一 job 取该组合内的最大值）
                max_restarts = {}
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'submitted_jobs' in data:
                                for job in data['submitted_jobs']:
                                    job_name = job['name']
                                    restarts = job.get('num_restarts', 0)
                                    if restarts is not None:  # 有些任务在开始时restarts为null
                                        max_restarts[job_name] = max(max_restarts.get(job_name, 0), restarts)
                        except (json.JSONDecodeError, KeyError):
                            continue

                # 存储每个任务在该组合下的最大重启次数
                for job_name, restarts in max_restarts.items():
                    restart_stats[workload][strategy][combo][job_name] = restarts

    return restart_stats

def calculate_statistics(restart_stats):
    """计算统计信息（细分到 workload 组合）"""
    results = []

    for workload in restart_stats:
        for strategy in restart_stats[workload]:
            for combo in restart_stats[workload][strategy]:
                job_restarts = list(restart_stats[workload][strategy][combo].values())
                if job_restarts:
                    stats = {
                        'workload': workload,
                        'strategy': strategy,
                        'combo': combo,  # e.g., 'workload-1'
                        'total_jobs': len(job_restarts),
                        'total_restarts': sum(job_restarts),
                        'avg_restarts': np.mean(job_restarts),
                        'median_restarts': np.median(job_restarts),
                        'max_restarts': max(job_restarts),
                        'min_restarts': min(job_restarts),
                        'std_restarts': np.std(job_restarts),
                        'jobs_with_restarts': sum(1 for r in job_restarts if r > 0),
                        'restart_rate': sum(1 for r in job_restarts if r > 0) / len(job_restarts) * 100,
                    }
                    results.append(stats)

    return pd.DataFrame(results)

def save_detailed_results(restart_stats, output_file):
    """保存详细的重启信息（细分到 workload 组合）"""
    detailed_results = []

    for workload in restart_stats:
        for strategy in restart_stats[workload]:
            for combo in restart_stats[workload][strategy]:
                for job_name, restarts in restart_stats[workload][strategy][combo].items():
                    detailed_results.append({
                        'workload': workload,
                        'strategy': strategy,
                        'combo': combo,  # e.g., 'workload-1'
                        'job_name': job_name,
                        'num_restarts': restarts,
                    })

    df = pd.DataFrame(detailed_results)
    df.to_csv(output_file, index=False)
    print(f"详细重启信息已保存到: {output_file}")
    return df

def print_summary_by_strategy(stats_df):
    """按策略打印摘要"""
    print("\n=== 按策略统计的重启次数摘要 ===")
    print("=" * 100)
    
    # 按策略分组并计算平均值
    strategy_summary = stats_df.groupby('strategy').agg({
        'total_jobs': 'sum',
        'total_restarts': 'sum',
        'avg_restarts': 'mean',
        'restart_rate': 'mean'
    }).round(2)
    
    strategy_summary['overall_avg_restarts'] = (strategy_summary['total_restarts'] / 
                                              strategy_summary['total_jobs']).round(2)
    
    print(strategy_summary)
    
    print("\n=== 详细统计 (按工作负载/组合 和 策略) ===")
    print("=" * 100)
    
    for strategy in sorted(stats_df['strategy'].unique()):
        print(f"\n策略: {strategy}")
        print("-" * 50)
        strategy_data = stats_df[stats_df['strategy'] == strategy]
        
        for _, row in strategy_data.iterrows():
            combo_str = row['combo'] if 'combo' in row and pd.notna(row['combo']) else 'N/A'
            print(f"  {row['workload']} ({combo_str}): "
                  f"平均重启 {row['avg_restarts']:.2f}, "
                  f"总重启 {row['total_restarts']}, "
                  f"重启率 {row['restart_rate']:.1f}% "
                  f"({row['jobs_with_restarts']}/{row['total_jobs']} 任务)")

def analyze_restart_patterns(detailed_df):
    """分析重启模式"""
    print("\n=== 重启模式分析 ===")
    print("=" * 100)
    
    # 按应用类型分析
    print("\n按应用类型统计:")
    detailed_df['app_type'] = detailed_df['job_name'].str.extract(r'^([^-]+)')
    app_stats = detailed_df.groupby(['strategy', 'app_type']).agg({
        'num_restarts': ['mean', 'sum', 'count']
    }).round(2)
    
    print(app_stats)
    
    # 重启次数分布
    print("\n重启次数分布:")
    for strategy in sorted(detailed_df['strategy'].unique()):
        strategy_data = detailed_df[detailed_df['strategy'] == strategy]
        restart_counts = strategy_data['num_restarts'].value_counts().sort_index()
        print(f"\n{strategy}:")
        for restarts, count in restart_counts.items():
            percentage = count / len(strategy_data) * 100
            print(f"  {restarts} 次重启: {count} 个任务 ({percentage:.1f}%)")

def main():
    """主函数"""
    # 使用脚本所在目录作为基准，定位结果目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "8_results", "main")
    
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return
    
    print("开始分析重启统计信息...")
    
    # 分析重启次数
    restart_stats = analyze_restarts_from_logs(results_dir)
    
    # 计算统计信息
    stats_df = calculate_statistics(restart_stats)
    
    # 保存详细结果
    detailed_df = save_detailed_results(restart_stats, "restart_analysis_detailed_8results.csv")
    
    # 保存摘要统计
    summary_path = os.path.join(base_dir, "restart_analysis_summary_8results.csv")
    stats_df.to_csv(summary_path, index=False)
    print("摘要统计信息已保存到:", summary_path)
    
    # 打印摘要
    print_summary_by_strategy(stats_df)
    
    # 分析重启模式
    analyze_restart_patterns(detailed_df)
    
    print(f"\n分析完成! 共分析了 {len(stats_df)} 个策略-工作负载组合")

if __name__ == "__main__":
    main()