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

# 忽略警告
warnings.filterwarnings('ignore')

# --- 1. 标准、直接地导入你的模块 ---
# 我们不再需要复杂的 load_modules 函数
try:
    from data_utils import (handle_missing_values, scale_features,
                            encode_labels, feature_engineering, load_data)
    from ml_algorithms import MLAlgorithms
except ImportError as e:
    st.error(f"关键模块导入失败: {e}")
    st.error("请确认 data_utils.py 和 ml_algorithms.py 文件与 app.py 在 GitHub 仓库的同一目录下。")
    st.stop()


# --- 页面配置 ---
st.set_page_config(
    page_title="医学数据分析平台",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 主应用逻辑 ---
def main():
    """主应用函数"""
    st.title("📊 医学数据分析与机器学习平台")

    # 初始化 ML 管理器
    # MLAlgorithms 现在是直接导入的类，可以直接实例化
    ml_manager = MLAlgorithms()

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("操作面板")
        st.info("请按顺序操作：上传数据 -> 数据处理 -> 模型训练")
        
        uploaded_file = st.file_uploader("1. 上传数据文件 (CSV 或 Excel)", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file:
            # 使用 session_state 在不同操作间保持数据
            if 'data' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
                st.session_state.data = load_data(uploaded_file)
                st.session_state.file_name = uploaded_file.name
                st.session_state.data_processed = None # 清空旧的处理后数据
            
            st.success("数据加载成功！")
            st.write("数据预览：")
            st.dataframe(st.session_state.data.head())

    # --- 主页面内容 ---
    if 'data' in st.session_state:
        data = st.session_state.data
        
        # --- 2. 数据处理模块 ---
        st.header("数据预处理")
        
        # 缺失值处理
        with st.expander("缺失值处理", expanded=True):
            missing_strategy = st.selectbox(
                '选择缺失值填充策略',
                ['mean', 'median', 'mode', 'drop'],
                key='missing_strategy'
            )
            if st.button('处理缺失值'):
                with st.spinner('正在处理...'):
                    data_processed = handle_missing_values(data, missing_strategy)
                    st.session_state.data_processed = data_processed
                    st.success("缺失值处理完成！")
                    st.dataframe(data_processed.head())

        # 特征缩放和编码
        with st.expander("特征工程", expanded=True):
            if st.session_state.get('data_processed') is not None:
                data_to_process = st.session_state.data_processed
                
                numeric_cols = data_to_process.select_dtypes(include=np.number).columns.tolist()
                categorical_cols = data_to_process.select_dtypes(include='object').columns.tolist()

                scale_cols = st.multiselect("选择要缩放的数值特征", options=numeric_cols)
                encode_cols = st.multiselect("选择要编码的类别特征", options=categorical_cols)

                if st.button("执行缩放和编码"):
                    with st.spinner("正在处理..."):
                        processed_data = data_to_process.copy()
                        if scale_cols:
                            processed_data = scale_features(processed_data, scale_cols)
                        if encode_cols:
                            processed_data = encode_labels(processed_data, encode_cols)
                        
                        st.session_state.data_processed = processed_data
                        st.success("特征缩放和编码完成！")
                        st.dataframe(processed_data.head())
            else:
                st.warning("请先处理缺失值。")


        # --- 3. 模型训练与评估 ---
        st.header("模型训练与评估")
        if st.session_state.get('data_processed') is not None:
            final_data = st.session_state.data_processed
            
            task_type = st.selectbox("选择任务类型", list(ml_manager.algorithms.keys()))
            
            if task_type in ['分类', '回归']:
                target_col = st.selectbox("选择目标变量 (Y)", final_data.columns)
                feature_cols = [col for col in final_data.columns if col != target_col]
                
                X = final_data[feature_cols]
                y = final_data[target_col]

                algorithm_name = st.selectbox("选择算法", list(ml_manager.algorithms[task_type].keys()))
            
                st.subheader(f"配置 '{algorithm_name}' 参数")
                params = ml_manager.get_algorithm_params(task_type, algorithm_name)

                if st.button("开始训练和评估"):
                    try:
                        with st.spinner("正在训练模型..."):
                            model = ml_manager.train_model(task_type, algorithm_name, params, X, y)
                            st.session_state.model = model
                            st.success("模型训练成功！")

                        with st.spinner("正在评估模型..."):
                            report = ml_manager.evaluate_model(model, X, y, task_type)
                            st.subheader("评估报告")
                            st.json(report) if isinstance(report, dict) else st.text(report)
                    
                    except Exception as e:
                        st.error(f"发生错误: {e}")
                        st.code(traceback.format_exc())
            
            elif task_type == '聚类':
                feature_cols = st.multiselect("选择用于聚类的特征", final_data.columns)
                if feature_cols:
                    X = final_data[feature_cols]
                    algorithm_name = st.selectbox("选择算法", list(ml_manager.algorithms[task_type].keys()))
                    
                    st.subheader(f"配置 '{algorithm_name}' 参数")
                    params = ml_manager.get_algorithm_params(task_type, algorithm_name)

                    if st.button("开始聚类"):
                        try:
                            with st.spinner("正在执行聚类..."):
                                model, labels = ml_manager.train_model(task_type, algorithm_name, params, X)
                                st.session_state.model = model
                                st.success("聚类完成！")
                                
                                result_df = X.copy()
                                result_df['cluster'] = labels
                                st.dataframe(result_df.head())

                        except Exception as e:
                            st.error(f"发生错误: {e}")
                            st.code(traceback.format_exc())

        else:
            st.warning("请先完成数据预处理步骤。")


if __name__ == "__main__":
    main()
