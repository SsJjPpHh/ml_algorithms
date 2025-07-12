"""
部署脚本 - 用于生产环境部署
"""

import os
import subprocess
import sys
import argparse

def install_dependencies():
    """安装依赖"""
    print("📦 安装依赖包...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ 依赖安装完成")

def setup_environment():
    """设置环境变量"""
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

def deploy_local():
    """本地部署"""
    print("🚀 本地部署...")
    setup_environment()
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def deploy_docker():
    """Docker部署"""
    print("🐳 Docker部署...")
    
    # 创建Dockerfile
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # 构建Docker镜像
    subprocess.run(["docker", "build", "-t", "medical-analysis-platform", "."])
    
    # 运行容器
    subprocess.run([
        "docker", "run", "-p", "8501:8501", 
        "--name", "medical-analysis", 
        "medical-analysis-platform"
    ])

def main():
    parser = argparse.ArgumentParser(description="医学数据分析平台部署脚本")
    parser.add_argument("--mode", choices=["local", "docker"], default="local",
                       help="部署模式")
    parser.add_argument("--install-deps", action="store_true",
                       help="是否安装依赖")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
    
    if args.mode == "local":
        deploy_local()
    elif args.mode == "docker":
        deploy_docker()

if __name__ == "__main__":
    main()
