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
    initial_sidebar_state="expanded"
)

# 导入自定义模块 - 安全导入
@st.cache_resource
def load_modules():
    """安全加载模块"""
    modules = {}
    try:
        # 逐个测试模块导入
        try:
            import data_utils
            modules['data_utils'] = data_utils
        except Exception as e:
            st.warning(f"data_utils 导入失败: {e}")
            modules['data_utils'] = None
            
        try:
            import ml_algorithms
            modules['ml_algorithms'] = ml_algorithms
        except Exception as e:
            st.warning(f"ml_algorithms 导入失败: {e}")
            modules['ml_algorithms'] = None
            
        try:
            import plotting
            modules['plotting'] = plotting
        except Exception as e:
            st.warning(f"plotting 导入失败: {e}")
            modules['plotting'] = None
            
        try:
            import stats_utils
            modules['stats_utils'] = stats_utils
        except Exception as e:
            st.warning(f"stats_utils 导入失败: {e}")
            modules['stats_utils'] = None
            
        try:
            import interpretability
            modules['interpretability'] = interpretability
        except Exception as e:
            st.warning(f"interpretability 导入失败: {e}")
            modules['interpretability'] = None
        
        return modules, True
        
    except Exception as e:
        st.error(f"❌ 模块导入失败: {str(e)}")
        return {}, False

# 加载模块
modules, modules_loaded = load_modules()

# 应用主标题
st.title("🏥 医学数据分析平台")
st.markdown("### 专业的医学数据分析工具")

# 显示模块状态
st.subheader("📦 模块状态检查")
module_status = {
    'data_utils.py': '✅' if modules.get('data_utils') else '❌',
    'ml_algorithms.py': '✅' if modules.get('ml_algorithms') else '❌',
    'plotting.py': '✅' if modules.get('plotting') else '❌',
    'stats_utils.py': '✅' if modules.get('stats_utils') else '❌',
    'interpretability.py': '✅' if modules.get('interpretability') else '❌'
}

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.write(f"data_utils {module_status['data_utils.py']}")
with col2:
    st.write(f"ml_algorithms {module_status['ml_algorithms.py']}")
with col3:
    st.write(f"plotting {module_status['plotting.py']}")
with col4:
    st.write(f"stats_utils {module_status['stats_utils.py']}")
with col5:
    st.write(f"interpretability {module_status['interpretability.py']}")

st.markdown("---")

# 初始化session state
if 'data' not in st.session_state:
    st.session_state.data = None

# 侧边栏配置
st.sidebar.title("📊 分析配置")
analysis_type = st.sidebar.selectbox(
    "选择分析类型",
    ["数据上传与预览", "数据预处理", "探索性数据分析"]
)

# 1. 数据上传与预览
if analysis_type == "数据上传与预览":
    st.header("📁 数据上传与预览")
    
    uploaded_file = st.file_uploader(
        "上传您的数据文件",
        type=['csv', 'xlsx', 'xls'],
        help="支持CSV、Excel格式"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("正在加载数据..."):
                # 使用内置方法加载数据
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
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
            st.dataframe(data.head(20), use_container_width=True)
            
            # 数据信息
            st.subheader("📊 数据信息")
            info_data = []
            for col in data.columns:
                info_data.append({
                    'Column': col,
                    'Type': str(data[col].dtype),
                    'Non-Null Count': data[col].count(),
                    'Null Count': data[col].isnull().sum(),
                    'Unique Values': data[col].nunique()
                })
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ 数据加载失败: {str(e)}")

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
            ["不处理", "删除含缺失值的行", "均值填充", "中位数填充"]
        )
        
        if st.button("🚀 开始预处理", type="primary"):
            try:
                with st.spinner("正在预处理数据..."):
                    processed_data = data.copy()
                    
                    # 处理缺失值
                    if missing_strategy == "删除含缺失值的行":
                        processed_data = processed_data.dropna()
                    elif missing_strategy == "均值填充":
                        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                        processed_data[numeric_cols] = processed_data[numeric_cols].fillna(processed_data[numeric_cols].mean())
                    elif missing_strategy == "中位数填充":
                        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                        processed_data[numeric_cols] = processed_data[numeric_cols].fillna(processed_data[numeric_cols].median())
                    
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
                
                st.subheader("处理后数据预览")
                st.dataframe(processed_data.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ 数据预处理失败: {str(e)}")

# 3. 探索性数据分析
elif analysis_type == "探索性数据分析":
    st.header("🔍 探索性数据分析")
    
    data = st.session_state.get('processed_data', st.session_state.data)
    
    if data is None:
        st.warning("⚠️ 请先上传数据")
    else:
        # 描述性统计
        st.subheader("📈 描述性统计")
        st.dataframe(data.describe(), use_container_width=True)
        
        # 可视化选项
        st.subheader("📊 数据可视化")
        
        plot_type = st.selectbox(
            "选择图表类型",
            ["相关性热图", "分布图", "散点图"]
        )
        
        try:
            if plot_type == "相关性热图":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    import plotly.express as px
                    corr_matrix = data[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                                  title="相关性热图")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("数据中数值列少于2列，无法绘制相关性热图")
            
            elif plot_type == "分布图":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("选择列", numeric_cols)
                    import plotly.express as px
                    fig = px.histogram(data, x=selected_col, marginal="box",
                                     title=f"{selected_col} 分布图")
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
                    
                    import plotly.express as px
                    fig = px.scatter(data, x=x_col, y=y_col,
                                   title=f"{x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("数据中数值列少于2列，无法绘制散点图")
        
        except Exception as e:
            st.error(f"图表生成失败: {str(e)}")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏥 医学数据分析平台 v1.0.0</p>
    </div>
    """, 
    unsafe_allow_html=True
)
