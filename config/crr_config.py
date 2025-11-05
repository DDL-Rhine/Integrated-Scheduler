"""
协作式资源重新分配机制(CRR)配置文件

该配置文件定义了CRR机制的各项参数，用于优化DeadlineMeet目标下的任务调度性能。
"""

# CRR机制启用开关
ENABLE_CRR = True

# CRR核心参数
CRR_CONFIG = {
    # 资源饱和度阈值 - 超过此值的任务被视为资源富余
    'saturation_threshold': 0.8,
    
    # 边际效用阈值 - 低于此值时资源增加收益有限
    'marginal_utility_threshold': 0.1,
    
    # 紧迫性阈值 - 超过此值的任务被视为需要优先资源
    'urgency_threshold': 0.6,
    
    # 资源需求阈值 - 超过此值时任务被视为资源饥饿
    'resource_need_threshold': 0.3,
    
    # 协作收益阈值 - 超过此值才执行资源重新分配
    'collaboration_benefit_threshold': 0.3,
    
    # 最大资源转移量 - 单次协作中最多转移的GPU数量
    'max_transfer_amount': 2,
    
    # 最大重分配次数比例 - 相对于集群总容量的比例
    'max_redistributions_ratio': 0.25,
    
    # 重启避免机制
    'enable_restart_avoidance': True,
    
    # 资源变化容忍度 - 低于此比例的变化尝试避免重启
    'resource_change_tolerance': 0.2,
    
    # 紧迫任务重启避免阈值 - 剩余时间比例
    'urgent_task_restart_threshold': 0.4,
    
    # 高进度任务重启避免阈值 - 训练进度比例
    'high_progress_restart_threshold': 0.7
}

# CRR性能监控参数
CRR_MONITORING = {
    # 启用性能统计
    'enable_stats': True,
    
    # 统计信息保存路径
    'stats_file': '/home/lab/simulator-artifact/logs/crr_stats.json',
    
    # 详细日志级别
    'log_level': 'INFO',
    
    # 性能指标报告间隔
    'report_interval': 100  # 每100个调度周期报告一次
}

# 模型特定的优化参数
MODEL_SPECIFIC_CONFIG = {
    # 大语言模型配置
    'llama': {
        'optimal_gpu_range': (8, 16),
        'collaboration_bonus': 0.3,
        'frozen_layer_compatibility': True
    },
    
    # CNN模型配置
    'resnet': {
        'optimal_gpu_range': (4, 8),
        'collaboration_bonus': 0.2,
        'frozen_layer_compatibility': True
    },
    
    'vgg': {
        'optimal_gpu_range': (4, 8),
        'collaboration_bonus': 0.2,
        'frozen_layer_compatibility': True
    },
    
    # BERT模型配置
    'bert': {
        'optimal_gpu_range': (4, 12),
        'collaboration_bonus': 0.25,
        'frozen_layer_compatibility': True
    },
    
    # 默认配置
    'default': {
        'optimal_gpu_range': (4, 8),
        'collaboration_bonus': 0.1,
        'frozen_layer_compatibility': False
    }
}

def get_model_config(job_name):
    """根据任务名称获取对应的模型配置"""
    job_name_lower = job_name.lower()
    
    for model_type, config in MODEL_SPECIFIC_CONFIG.items():
        if model_type in job_name_lower:
            return config
    
    return MODEL_SPECIFIC_CONFIG['default']

def apply_crr_args(args):
    """将CRR配置应用到模拟器参数中"""
    args.enable_crr = ENABLE_CRR
    args.enable_restart_avoidance = CRR_CONFIG['enable_restart_avoidance']
    
    # 设置CRR相关参数
    for key, value in CRR_CONFIG.items():
        setattr(args, f'crr_{key}', value)
    
    # 设置监控参数
    for key, value in CRR_MONITORING.items():
        setattr(args, f'crr_monitor_{key}', value)
    
    return args
