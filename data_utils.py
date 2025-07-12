import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import io

def load_data(uploaded_file):
    """
    加载上传的数据文件
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("不支持的文件格式！请上传 CSV 或 Excel 文件。")
            return None
        
        st.success(f"数据加载成功！数据形状: {df.shape}")
        return df
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None

def get_data_info(df):
    """
    获取数据基本信息
    """
    info_dict = {
        '数据形状': df.shape,
        '列数': df.shape[1],
        '行数': df.shape[0],
        '缺失值总数': df.isnull().sum().sum(),
        '数值型列': list(df.select_dtypes(include=[np.number]).columns),
        '分类型列': list(df.select_dtypes(include=['object', 'category']).columns),
        '重复行数': df.duplicated().sum()
    }
    return info_dict

def get_missing_info(df):
    """
    获取缺失值详细信息
    """
    missing_data = pd.DataFrame({
        '缺失数量': df.isnull().sum(),
        '缺失比例(%)': (df.isnull().sum() / len(df)) * 100
    })
    missing_data = missing_data[missing_data['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
    return missing_data

def preprocess_data(df, target_col, task_type, test_size=0.2, handle_missing='mean', 
                   scale_features=True, random_state=42):
    """
    数据预处理主函数
    """
    # 复制数据避免修改原始数据
    df_processed = df.copy()
    
    # 分离特征和目标变量
    if target_col not in df.columns:
        st.error(f"目标列 '{target_col}' 不存在于数据中！")
        return None, None, None, None
    
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
    
    # 处理目标变量（如果是分类任务且目标是字符串）
    if task_type in ['分类', '二分类', '多分类'] and y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    # 处理缺失值
    if handle_missing == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif handle_missing == 'median':
        imputer = SimpleImputer(strategy='median')
    elif handle_missing == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    else:
        imputer = SimpleImputer(strategy='constant', fill_value=0)
    
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if task_type in ['分类', '二分类', '多分类'] else None
    )
    
    # 特征缩放
    if scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test

def generate_sample_data(data_type="classification"):
    """
    生成示例数据用于演示
    """
    np.random.seed(42)
    
    if data_type == "classification":
        n_samples = 1000
        # 模拟医学数据：年龄、BMI、血压、血糖等
        data = {
            '年龄': np.random.normal(50, 15, n_samples),
            'BMI': np.random.normal(25, 5, n_samples),
            '收缩压': np.random.normal(120, 20, n_samples),
            '舒张压': np.random.normal(80, 10, n_samples),
            '血糖': np.random.normal(5.5, 1.5, n_samples),
            '胆固醇': np.random.normal(200, 50, n_samples),
            '性别': np.random.choice(['男', '女'], n_samples),
            '吸烟': np.random.choice(['是', '否'], n_samples, p=[0.3, 0.7])
        }
        
        # 创建目标变量（疾病风险）
        risk_score = (data['年龄'] * 0.02 + data['BMI'] * 0.1 + 
                     data['收缩压'] * 0.01 + data['血糖'] * 0.3 + 
                     np.random.normal(0, 2, n_samples))
        data['疾病风险'] = ['高风险' if score > 8 else '低风险' for score in risk_score]
        
    elif data_type == "regression":
        n_samples = 1000
        data = {
            '年龄': np.random.normal(45, 12, n_samples),
            '教育年限': np.random.normal(12, 4, n_samples),
            '工作经验': np.random.normal(15, 8, n_samples),
            '城市规模': np.random.choice([1, 2, 3, 4], n_samples),
            '行业类型': np.random.choice(['IT', '金融', '医疗', '教育', '制造'], n_samples)
        }
        
        # 创建目标变量（收入）
        data['年收入'] = (data['年龄'] * 800 + data['教育年限'] * 2000 + 
                       data['工作经验'] * 1500 + data['城市规模'] * 5000 + 
                       np.random.normal(0, 10000, n_samples))
    
    df = pd.DataFrame(data)
    return df

def download_dataframe_as_csv(df, filename="processed_data.csv"):
    """
    将DataFrame转换为CSV下载链接
    """
    csv = df.to_csv(index=False)
    return st.download_button(
        label="下载处理后的数据",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )
