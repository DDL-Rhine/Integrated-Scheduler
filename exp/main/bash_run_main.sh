set -e 
#prefix="srun -p llm_s --gres=gpu:0 --cpus-per-task=4 "

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate icefrog

prefix=""

for method in simple_icefrog simple_pollux optimus lucid 
# for method in simple_icefrog 
do
    for ident in 1.0 2.0 3.0 4.0 
    # for ident in 1.0 
    do 
        # interval=300
        interval=120
        num_node=12
        for yaml in fairness-1
        do 
            save_dir=../../9_results_origin/main/workload-$ident/$method
            mkdir -p $save_dir
            workload="../../workloads-$ident"
            GPUSharing=False
            if [[ $method == "simple_icefrog" ]]; then 
                GPUSharing=True
                interval=300
            fi 
            if [[ $method == "lucid" ]]; then 
                GPUSharing=True
                interval=30
            fi 
            
            # # 基础版本 - 所有方法都运行
            # if [[ $method == "simple_icefrog" ]]; then 
            #     save_dir=../../12_results_restart/main/workload-$ident/$method
            #     mkdir -p $save_dir
            #     $prefix python -u ../../simulator.py --memory_scale=1.0 --policy=$method \
            #         --workload=$workload/ \
            #         --GPUSharing=$GPUSharing \
            #         --yaml=../../config/macro/$yaml.yaml \
            #         --output=$save_dir \
            #         --min-nodes=$num_node --interval=$interval --num-gpus=4 &
            # fi 
            
            # # simple_icefrog 的 batch-fixed 版本
            # if [[ $method == "simple_icefrog" ]]; then 
            #     save_dir=../../12_results_restart/main/workload-$ident/$method-batch-fixed
            #     mkdir -p $save_dir
            #     $prefix python -u ../../simulator.py --memory_scale=1.0 --policy=$method \
            #         --batch_fixed=True \
            #         --workload=$workload/ \
            #         --GPUSharing=$GPUSharing \
            #         --yaml=../../config/macro/$yaml.yaml \
            #         --output=$save_dir \
            #         --min-nodes=$num_node --interval=$interval --num-gpus=4 &
            # fi 
            # 添加对DeadlineMeet目标的支持
            if [[ $method == "simple_icefrog" ]]; then 
                save_dir=../../9_results_origin/main/workload-$ident/$method-deadline
                mkdir -p $save_dir
                $prefix python -u ../../simulator.py --memory_scale=1.0 --policy=$method \
                    --obj=DeadlineMeet \
                    --avoid_restart=False \
                    --enable_crr=False \
                    --workload=$workload/ \
                    --GPUSharing=$GPUSharing \
                    --yaml=../../config/macro/$yaml.yaml \
                    --output=$save_dir \
                    --min-nodes=$num_node --interval=$interval --num-gpus=4 &
            fi

            # if [[ $method == "lucid" ]]; then 
            #     save_dir=../../12_results_restart/main/workload-$ident/$method
            #     mkdir -p $save_dir
            #     $prefix python -u ../../simulator.py --memory_scale=1.0 --policy=$method  \
            #         --GPUSharing=$GPUSharing \
            #         --workload=$workload/ \
            #         --yaml=../../config/macro/$yaml.yaml \
            #         --output=$save_dir \
            #         --min-nodes=$num_node --interval=$interval --num-gpus=4 &
            # fi 

            # if [[ $method == "optimus" || $method == "simple_pollux" ]]; then 
            #     # 基础版本
            #     save_dir=../../12_results_restart/main/workload-$ident/$method
            #     mkdir -p $save_dir
            #     $prefix python -u ../../simulator.py --memory_scale=1.0 --policy=$method  \
            #         --workload=$workload/ \
            #         --yaml=../../config/macro/$yaml.yaml \
            #         --output=$save_dir \
            #         --GPUSharing=$GPUSharing \
            #         --min-nodes=$num_node --interval=$interval --num-gpus=4 &
                
            #     # FreezeOut 版本
            #     freeze=FreezeOut
            #     save_dir=../../12_results_restart/main/workload-$ident/$method-$freeze
            #     mkdir -p $save_dir
            #     $prefix python -u ../../simulator.py --memory_scale=1.0 --policy=$method  \
            #         --freeze=$freeze \
            #         --workload=$workload/ \
            #         --yaml=../../config/macro/$yaml.yaml \
            #         --output=$save_dir \
            #         --GPUSharing=$GPUSharing \
            #         --min-nodes=$num_node --interval=$interval --num-gpus=4 &
            # fi 
            sleep 5
        done 
    
    done
    
done
wait 
# python exp/main/plot_workload_simple.py True