#!/usr/bin/env python3
import csv
import json
import random
import os

# 加载JCTs数据
json_path = '/home/lab/simulator-artifact/0_results/main/workload-4.0/simple_icefrog-batch-fixed/summary.json'
csv_path = '/home/lab/simulator-artifact/workloads-4.0/workload-3.csv'

# 读取JSON数据
with open(json_path, 'r') as f:
    data = json.load(f)
    jcts_values = data['jcts']['workload-3']

# 处理CSV文件
temp_csv_path = csv_path + '.temp'

with open(csv_path, 'r') as infile, open(temp_csv_path, 'w', newline='') as outfile:
    # 读取现有CSV
    reader = csv.DictReader(infile)
    
    # 创建新CSV的字段名（包括新增列）
    fieldnames = reader.fieldnames.copy()
    fieldnames.extend(['jcts', 'deadline'])
    
    # 创建带有更新字段名的写入器
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # 处理每一行
    for row in reader:
        # 直接使用'name'列
        job_name = row['name']
        
        # 获取此作业的JCT值
        jct = jcts_values.get(job_name)
        
        if jct:
            # 添加JCT值
            row['jcts'] = jct
            
            # 计算截止时间(time + jcts * 随机因子)
            time = int(row['time'])
            random_factor = random.uniform(0.5, 1.5)
            deadline = time + int(jct) * random_factor
            row['deadline'] = int(deadline)
        else:
            # 如果在JCTs数据中找不到作业
            row['jcts'] = 0
            row['deadline'] = int(row['time'])
        
        # 写入更新后的行
        writer.writerow(row)

# 用更新后的文件替换原始文件
os.replace(temp_csv_path, csv_path)

print(f"已更新 {csv_path}，添加了jcts和deadline列")