"""
配置文件
"""

# 应用配置
APP_CONFIG = {
    'title': '医学数据分析平台',
    'icon': '🏥',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# 数据配置
DATA_CONFIG = {
    'max_file_size': 200,  # MB
    'supported_formats': ['csv', 'xlsx', 'xls'],
    'encoding': 'utf-8'
}

# 模型配置
MODEL_CONFIG = {
    'random_state': 42,
    'test_size_range': (0.1, 0.5),
    'default_test_size': 0.2,
    'cv_folds': 5
}

# 可视化配置
PLOT_CONFIG = {
    'color_palette': 'Set3',
    'figure_height': 400,
    'figure_width': 600,
    'dpi': 100
}

# SHAP配置
SHAP_CONFIG = {
    'max_samples_for_kernel': 100,
    'max_display_features': 20
}

# 统计检验配置
STATS_CONFIG = {
    'alpha': 0.05,
    'confidence_level': 0.95
}
