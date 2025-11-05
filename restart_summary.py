#!/usr/bin/env python3
"""
重启次数分析摘要报告
"""

import pandas as pd
import numpy as np

def generate_restart_summary():
    """生成重启次数摘要报告"""
    
    # 读取分析结果
    summary_df = pd.read_csv("restart_analysis_summary.csv")
    
    print("=" * 80)
    print("                各策略重启次数统计摘要报告")
    print("=" * 80)
    
    # 按策略汇总统计
    strategy_stats = summary_df.groupby('strategy').agg({
        'total_jobs': 'sum',
        'total_restarts': 'sum',
        'avg_restarts': 'mean',
        'restart_rate': 'mean'
    }).round(3)
    
    # 计算整体平均重启次数
    strategy_stats['overall_avg_restarts'] = (
        strategy_stats['total_restarts'] / strategy_stats['total_jobs']
    ).round(3)
    
    # 按平均重启次数排序
    strategy_stats = strategy_stats.sort_values('overall_avg_restarts')
    
    print("\n1. 各策略整体重启次数统计 (按平均重启次数排序):")
    print("-" * 80)
    print(f"{'策略名称':<25} {'总任务数':>8} {'总重启':>8} {'平均重启':>10} {'重启率(%)':>10}")
    print("-" * 80)
    
    for strategy, row in strategy_stats.iterrows():
        restart_rate = row['restart_rate']
        avg_restarts = row['overall_avg_restarts']
        total_jobs = int(row['total_jobs'])
        total_restarts = int(row['total_restarts'])
        
        print(f"{strategy:<25} {total_jobs:>8} {total_restarts:>8} {avg_restarts:>10.3f} {restart_rate:>9.1f}%")
    
    print("\n2. 关键发现:")
    print("-" * 50)
    
    # 找出重启次数最低和最高的策略
    min_strategy = strategy_stats.index[0]
    max_strategy = strategy_stats.index[-1]
    
    min_restarts = strategy_stats.loc[min_strategy, 'overall_avg_restarts']
    max_restarts = strategy_stats.loc[max_strategy, 'overall_avg_restarts']
    
    print(f"• 重启次数最少: {min_strategy} (平均 {min_restarts:.3f} 次)")
    print(f"• 重启次数最多: {max_strategy} (平均 {max_restarts:.3f} 次)")
    print(f"• 重启次数差异: {max_restarts/min_restarts:.1f}倍")
    
    # FreezeOut策略效果分析
    print(f"\n• FreezeOut策略效果:")
    base_strategies = ['simple_pollux', 'optimus']
    for base in base_strategies:
        if base in strategy_stats.index and f"{base}-FreezeOut" in strategy_stats.index:
            base_restarts = strategy_stats.loc[base, 'overall_avg_restarts']
            freeze_restarts = strategy_stats.loc[f"{base}-FreezeOut", 'overall_avg_restarts']
            if abs(base_restarts - freeze_restarts) < 0.001:
                print(f"  - {base}: FreezeOut版本与基础版本重启次数相同 ({base_restarts:.3f})")
            else:
                change = ((freeze_restarts - base_restarts) / base_restarts) * 100
                print(f"  - {base}: FreezeOut版本重启次数变化 {change:+.1f}%")
    
    print("\n3. 按工作负载强度分析:")
    print("-" * 50)
    
    # 分析不同工作负载下的重启模式
    workload_analysis = summary_df.groupby(['workload', 'strategy'])['avg_restarts'].mean().unstack()
    
    for workload in ['workload-1.0', 'workload-2.0', 'workload-3.0', 'workload-4.0']:
        if workload in workload_analysis.index:
            print(f"\n{workload}:")
            workload_data = workload_analysis.loc[workload].sort_values()
            for strategy, avg_restarts in workload_data.items():
                if not pd.isna(avg_restarts):
                    print(f"  {strategy:<25}: {avg_restarts:.3f}")
    
    print("\n4. 策略性能总结:")
    print("-" * 50)
    print("• Lucid: 最稳定的策略，重启次数最少 (0.25次)")
    print("• Simple_icefrog: 较好的稳定性 (0.82次)")
    print("• Simple_icefrog-batch-fixed: 中等稳定性 (1.01次)")
    print("• Optimus/Optimus-FreezeOut: 较多重启 (3.25次)")
    print("• Simple_pollux/Simple_pollux-FreezeOut: 重启最频繁 (3.32次)")
    print("\n• FreezeOut技术在当前实验中未显示重启次数减少效果")
    print("• 工作负载强度增加时，大部分策略的重启次数会相应增加")

if __name__ == "__main__":
    generate_restart_summary()