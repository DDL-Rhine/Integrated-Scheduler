#!/bin/bash
set -e 

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate icefrog

# 切换到simulator-artifact根目录
cd /home/lab/simulator-artifact

echo "Running missing results: optimus-FreezeOut, simple_pollux, simple_pollux-FreezeOut"

prefix=""

# 只运行缺失的方法
for method in simple_pollux optimus
do
    for ident in 1.0 2.0 3.0 4.0 
    do 
        interval=120
        num_node=12
        yaml=fairness-1
        workload="workloads-$ident"
        GPUSharing=False
        
        echo "Processing $method for workload-$ident"
        
        # 跳过基础版本，只运行FreezeOut版本
        echo "Skipping $method base version for workload-$ident"
        
        # FreezeOut 版本
        freeze=FreezeOut
        save_dir=12_results/main/workload-$ident/$method-$freeze
        if [[ ! -f "$save_dir/summary.json" ]]; then
            echo "Running $method-$freeze for workload-$ident"
            rm -rf $save_dir/*  # 清空目录
            mkdir -p $save_dir
            $prefix python -u simulator.py --memory_scale=1.0 --policy=$method  \
                --freeze=$freeze \
                --workload=$workload/ \
                --yaml=config/macro/$yaml.yaml \
                --output=$save_dir \
                --GPUSharing=$GPUSharing \
                --min-nodes=$num_node --interval=$interval --num-gpus=4
        else
            echo "$method-$freeze already exists for workload-$ident"
        fi
    done
done

echo "All missing results have been generated!"