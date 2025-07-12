#!/usr/bin/env python3
"""
医学数据分析平台启动脚本
"""

import subprocess
import sys
import os

def check_requirements():
    """检查依赖是否安装"""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import plotly
        import shap
        print("✅ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def main():
    """主函数"""
    print("🏥 医学数据分析平台")
    print("=" * 50)
    
    # 检查依赖
    if not check_requirements():
        sys.exit(1)
    
    # 启动Streamlit应用
    print("🚀 启动应用...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
