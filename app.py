# -*- coding: utf-8 -*-

# 基础导入测试
try:
    import streamlit as st
    print("✅ Streamlit 导入成功")
except ImportError as e:
    print(f"❌ Streamlit 导入失败: {e}")

try:
    import pandas as pd
    print("✅ Pandas 导入成功")
except ImportError as e:
    print(f"❌ Pandas 导入失败: {e}")

try:
    import numpy as np
    print("✅ NumPy 导入成功")
except ImportError as e:
    print(f"❌ NumPy 导入失败: {e}")

try:
    import warnings
    print("✅ Warnings 导入成功")
except ImportError as e:
    print(f"❌ Warnings 导入失败: {e}")

# 设置页面配置
st.set_page_config(
    page_title="医学数据分析平台 - 测试版",
    page_icon="🏥",
    layout="wide"
)

# 显示基本信息
st.title("🏥 医学数据分析平台 - 测试版")
st.success("✅ 应用启动成功！")

# 显示Python和环境信息
import sys
import os

st.subheader("🔧 环境信息")
st.write(f"Python版本: {sys.version}")
st.write(f"当前工作目录: {os.getcwd()}")
st.write(f"文件路径: {__file__}")

# 测试模块导入
st.subheader("📦 模块导入测试")

modules_to_test = [
    'warnings', 'datetime', 'io', 'base64', 'traceback',
    'streamlit', 'pandas', 'numpy'
]

for module_name in modules_to_test:
    try:
        __import__(module_name)
        st.success(f"✅ {module_name}")
    except ImportError as e:
        st.error(f"❌ {module_name}: {e}")

# 简单功能测试
st.subheader("🧪 功能测试")

# 文件上传测试
uploaded_file = st.file_uploader("测试文件上传", type=['csv', 'txt'])
if uploaded_file:
    st.success("文件上传功能正常")

# 基本交互测试
test_input = st.text_input("测试输入框", "Hello World")
if test_input:
    st.write(f"输入内容: {test_input}")

st.info("如果您看到这条消息，说明基本功能正常运行")
