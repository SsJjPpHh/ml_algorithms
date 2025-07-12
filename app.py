# -*- coding: utf-8 -*-
"""
医学数据分析平台
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import traceback
from datetime import datetime
import io
import sys
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="医学数据分析平台",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/medical-data-analysis-platform',
        'Report a bug': "https://github.com/yourusername/medical-data-analysis-platform/issues",
        'About': "医学数据分析平台 v1.0.0"
    }
)

# 导入自定义模块
@st.cache_resource
def load_modules():
    """缓存模块导入"""
    modules = {}
    try:
        from data_utils import (
            load_data, 
            preprocess_data, 
            get_data_info, 
            handle_missing_values,
            detect_outliers,
            encode_categorical_variables
        )
        modules['data_utils'] = {
            'load_data': load_data,
            'preprocess_data': preprocess_data,
            'get_data_info': get_data_info,
            'handle_missing_values': handle_missing_values,
            'detect_outliers': detect_outliers,
            'encode_categorical_variables': encode_categorical_variables
        }
        
        from ml_algorithms import (
            get_ml_algorithms,
            train_model,
            evaluate_model,
            cross_validate_model,
            hyperparameter_tuning
        )
        modules['ml_algorithms'] = {
            'get_ml_algorithms': get_ml_algorithms,
            'train_model': train_model,
            'evaluate_model': evaluate_model,
            'cross_validate_model': cross_validate_model,
            'hyperparameter_tuning': hyperparameter_tuning
        }
        
        from plotting import (
            create_plots,
            plot_correlation_matrix,
            plot_distribution,
            plot_boxplot,
            plot_scatter,
            plot_model_performance,
            plot_feature_importance
        )
        modules['plotting'] = {
            'create_plots': create_plots,
            'plot_correlation_matrix': plot_correlation_matrix,
            'plot_distribution': plot_distribution,
            'plot_boxplot': plot_boxplot,
            'plot_scatter': plot_scatter,
            'plot_model_performance': plot_model_performance,
            'plot_feature_importance': plot_feature_importance
        }
        
        from stats_utils import (
            perform_statistical_tests,
            descriptive_statistics,
            correlation_analysis,
            hypothesis_testing,
            survival_analysis
        )
        modules['stats_utils'] = {
            'perform_statistical_tests': perform_statistical_tests,
            'descriptive_statistics': descriptive_statistics,
            'correlation_analysis': correlation_analysis,
            'hypothesis_testing': hypothesis_testing,
            'survival_analysis': survival_analysis
        }
        
        from interpretability import (
            explain_model,
            plot_shap_values,
            plot_feature_importance_shap,
            generate_model_report
        )
        modules['interpretability'] = {
            'explain_model': explain_model,
            'plot_shap_values': plot_shap_values,
            'plot_feature_importance_shap': plot_feature_importance_shap,
            'generate_model_report': generate_model_report
        }
        
        return modules, True
        
    except Exception as e:
        st.error(f"❌ 模块导入失败: {str(e)}")
        st.error("请检查所有依赖是否正确安装")
        return {}, False

# 加载模块
modules, modules_loaded = load_modules()

# 应用主标题
st.title("🏥 医学数据分析平台")
st.markdown("### 专业的医学数据分析工具")

if not modules_loaded:
    st.error("❌ 关键模块加载失败，应用无法正常运行")
    st.info("请检查以下文件是否存在且格式正确：")
    required_files = ['data_utils.py', 'ml_algorithms.py', 'plotting.py', 'stats_utils.py', 'interpretability.py']
    for file in required_files:
        if os.path.exists(file):
            st.success(f"✅ {file}")
        else:
            st.error(f"❌ {file} - 文件不存在")
    st.stop()

st.success("✅ 所有模块加载成功！")
st.markdown("---")

# 侧边栏配置
st.sidebar.title("📊 分析配置")
st.sidebar.markdown("---")

# 初始化session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# 主要功能选择
analysis_type = st.sidebar.selectbox(
    "选择分析类型",
    ["数据上传与预览", "数据预处理", "探索性数据分析", "统计分析", "机器学习建模", "模型解释与可视化"]
)

st.sidebar.markdown("---")

# 1. 数据上传与预览
if analysis_type == "数据上传与预览":
    st.header("📁 数据上传与预览")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传您的数据文件",
        type=['csv', 'xlsx', 'xls'],
        help="支持CSV、Excel格式，文件大小限制200MB"
    )
    
    if uploaded_file is not None:
        try:
            # 加载数据
            with st.spinner("正在加载数据..."):
                data = modules['data_utils']['load_data'](uploaded_file)
                st.session_state.data = data
            
            st.success(f"✅ 数据加载成功！共 {data.shape[0]} 行，{data.shape[1]} 列")
            
            # 数据基本信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总行数", data.shape[0])
            with col2:
                st.metric("总列数", data.shape[1])
            with col3:
                st.metric("数值列", len(data.select_dtypes(include=[np.number]).columns))
            with col4:
                st.metric("缺失值", data.isnull().sum().sum())
            
            # 数据预览
            st.subheader("📋 数据预览")
            st.dataframe(data.head(100), use_container_width=True)
            
            # 数据信息
            st.subheader("📊 数据信息")
            info_df = modules['data_utils']['get_data_info'](data)
            st.dataframe(info_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ 数据加载失败: {str(e)}")
            st.error("请检查文件格式是否正确")

# 2. 数据预处理
elif analysis_type == "数据预处理":
    st.header("🔧 数据预处理")
    
    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据")
    else:
        data = st.session_state.data.copy()
        
        st.subheader("缺失值处理")
        missing_strategy = st.selectbox(
            "选择缺失值处理策略",
            ["不处理", "删除含缺失值的行", "均值填充", "中位数填充", "众数填充"]
        )
        
        st.subheader("异常值检测")
        outlier_detection = st.checkbox("启用异常值检测")
        
        st.subheader("分类变量编码")
        categorical_encoding = st.checkbox("启用分类变量编码")
        
        if st.button("🚀 开始预处理", type="primary"):
            try:
                with st.spinner("正在预处理数据..."):
                    processed_data = modules['data_utils']['preprocess_data'](data)
                    st.session_state.processed_data = processed_data
                
                st.success("✅ 数据预处理完成！")
                
                # 显示处理结果对比
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("原始数据")
                    st.write(f"形状: {data.shape}")
                    st.write(f"缺失值: {data.isnull().sum().sum()}")
                
                with col2:
                    st.subheader("处理后数据")
                    st.write(f"形状: {processed_data.shape}")
                    st.write(f"缺失值: {processed_data.isnull().sum().sum()}")
                
                # 预览处理后的数据
                st.subheader("处理后数据预览")
                st.dataframe(processed_data.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ 数据预处理失败: {str(e)}")
                st.error(traceback.format_exc())

# 3. 探索性数据分析
elif analysis_type == "探索性数据分析":
    st.header("🔍 探索性数据分析")
    
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    
    if data is None:
        st.warning("⚠️ 请先上传数据")
    else:
        # 描述性统计
        st.subheader("📈 描述性统计")
        try:
            desc_stats = modules['stats_utils']['descriptive_statistics'](data)
            st.dataframe(desc_stats, use_container_width=True)
        except Exception as e:
            st.error(f"描述性统计计算失败: {str(e)}")
        
        # 可视化选项
        st.subheader("📊 数据可视化")
        
        plot_type = st.selectbox(
            "选择图表类型",
            ["相关性热图", "分布图", "箱线图", "散点图"]
        )
        
        try:
            if plot_type == "相关性热图":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    fig = modules['plotting']['plot_correlation_matrix'](data[numeric_cols])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("数据中数值列少于2列，无法绘制相关性热图")
            
            elif plot_type == "分布图":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("选择列", numeric_cols)
                    fig = modules['plotting']['plot_distribution'](data, selected_col)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("数据中没有数值列")
            
            elif plot_type == "箱线图":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("选择列", numeric_cols)
                    fig = modules['plotting']['plot_boxplot'](data, selected_col)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("数据中没有数值列")
            
            elif plot_type == "散点图":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X轴", numeric_cols)
                    with col2:
                        y_col = st.selectbox("Y轴", [col for col in numeric_cols if col != x_col])
                    
                    fig = modules['plotting']['plot_scatter'](data, x_col, y_col)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("数据中数值列少于2列，无法绘制散点图")
        
        except Exception as e:
            st.error(f"图表生成失败: {str(e)}")

# 4. 统计分析
elif analysis_type == "统计分析":
    st.header("📊 统计分析")
    
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    
    if data is None:
        st.warning("⚠️ 请先上传数据")
    else:
        test_type = st.selectbox(
            "选择统计检验类型",
            ["相关性分析", "t检验", "卡方检验", "方差分析(ANOVA)"]
        )
        
        try:
            if test_type == "相关性分析":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        var1 = st.selectbox("变量1", numeric_cols)
                    with col2:
                        var2 = st.selectbox("变量2", [col for col in numeric_cols if col != var1])
                    
                    method = st.selectbox("相关性方法", ["pearson", "spearman", "kendall"])
                    
                    if st.button("执行分析"):
                        result = modules['stats_utils']['correlation_analysis'](data, var1, var2, method)
                        st.write(result)
                else:
                    st.warning("需要至少2个数值变量进行相关性分析")
            
            elif test_type == "t检验":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if numeric_cols and categorical_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        numeric_var = st.selectbox("数值变量", numeric_cols)
                    with col2:
                        group_var = st.selectbox("分组变量", categorical_cols)
                    
                    if st.button("执行t检验"):
                        result = modules['stats_utils']['hypothesis_testing'](data, numeric_var, group_var, test_type="ttest")
                        st.write(result)
                else:
                    st.warning("需要数值变量和分类变量进行t检验")
        
        except Exception as e:
            st.error(f"统计分析失败: {str(e)}")

# 5. 机器学习建模
elif analysis_type == "机器学习建模":
    st.header("🤖 机器学习建模")
    
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    
    if data is None:
        st.warning("⚠️ 请先上传数据")
    else:
        # 选择目标变量
        target_col = st.selectbox("选择目标变量", data.columns.tolist())
        
        # 选择特征变量
        feature_cols = st.multiselect(
            "选择特征变量",
            [col for col in data.columns if col != target_col],
            default=[col for col in data.columns if col != target_col][:5]
        )
        
        if not feature_cols:
            st.warning("请至少选择一个特征变量")
        else:
            # 判断问题类型
            if data[target_col].dtype == 'object' or data[target_col].nunique() <= 10:
                problem_type = "classification"
                st.info("🎯 检测到分类问题")
            else:
                problem_type = "regression"
                st.info("🎯 检测到回归问题")
            
            # 选择算法
            try:
                algorithms = modules['ml_algorithms']['get_ml_algorithms'](problem_type)
                selected_algorithm = st.selectbox("选择机器学习算法", list(algorithms.keys()))
                
                # 模型参数设置
                st.subheader("模型参数")
                test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
                random_state = st.number_input("随机种子", value=42, min_value=0)
                
                if st.button("🚀 开始训练", type="primary"):
                    try:
                        with st.spinner("正在训练模型..."):
                            X = data[feature_cols]
                            y = data[target_col]
                            
                            # 训练模型
                            model_results = modules['ml_algorithms']['train_model'](
                                X, y, 
                                algorithm=selected_algorithm,
                                problem_type=problem_type,
                                test_size=test_size,
                                random_state=random_state
                            )
                            
                            st.session_state.model_results = model_results
                            st.session_state.model = model_results['model']
                        
                        st.success("✅ 模型训练完成！")
                        
                        # 显示模型性能
                        st.subheader("📈 模型性能")
                        performance_metrics = modules['ml_algorithms']['evaluate_model'](
                            model_results['model'], 
                            model_results['X_test'], 
                            model_results['y_test'],
                            problem_type
                        )
                        
                        # 创建性能指标表格
                        metrics_df = pd.DataFrame([performance_metrics])
                        st.dataframe(metrics_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ 模型训练失败: {str(e)}")
                        st.error(traceback.format_exc())
            
            except Exception as e:
                st.error(f"算法加载失败: {str(e)}")

# 6. 模型解释与可视化
elif analysis_type == "模型解释与可视化":
    st.header("🔍 模型解释与可视化")
    
    if st.session_state.model is None:
        st.warning("⚠️ 请先训练模型")
    else:
        model_results = st.session_state.model_results
        
        # 特征重要性
        st.subheader("🎯 特征重要性")
        try:
            if hasattr(model_results['model'], 'feature_importances_'):
                fig_importance = modules['plotting']['plot_feature_importance'](
                    model_results['model'], 
                    model_results.get('feature_names', [f'Feature_{i}' for i in range(len(model_results['X_test'].columns))])
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("当前模型不支持特征重要性分析")
        except Exception as e:
            st.error(f"特征重要性图绘制失败: {str(e)}")
        
        # 模型报告
        st.subheader("📋 模型报告")
        if st.button("生成完整报告"):
            try:
                report = modules['interpretability']['generate_model_report'](
                    model_results['model'],
                    model_results['X_test'],
                    model_results['y_test'],
                    model_results.get('feature_names', model_results['X_test'].columns.tolist())
                )
                
                st.markdown(report)
                
            except Exception as e:
                st.error(f"报告生成失败: {str(e)}")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏥 医学数据分析平台 v1.0.0 | Built with ❤️ using Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# 侧边栏信息
st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 支持与帮助")
st.sidebar.markdown("- 📖 使用文档")
st.sidebar.markdown("- 🐛 报告问题")
st.sidebar.markdown("- 💡 功能建议")

# 显示当前状态
if st.sidebar.checkbox("显示调试信息"):
    st.sidebar.markdown("### 🔧 调试信息")
    st.sidebar.write(f"数据状态: {'✅' if st.session_state.data is not None else '❌'}")
    st.sidebar.write(f"预处理状态: {'✅' if st.session_state.processed_data is not None else '❌'}")
    st.sidebar.write(f"模型状态: {'✅' if st.session_state.model is not None else '❌'}")
    st.sidebar.write(f"模块状态: {'✅' if modules_loaded else '❌'}")
