# -*- coding: utf-8 -*-
"""
ML算法模块的数据处理工具
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st

def handle_missing_values(data, strategy='mean'):
    """
    处理缺失值
    
    参数:
    - data: pandas DataFrame
    - strategy: 'mean', 'median', 'mode', 'drop'
    
    返回:
    - 处理后的DataFrame
    """
    try:
        data_copy = data.copy()
        
        if strategy == 'drop':
            return data_copy.dropna()
        
        elif strategy == 'mean':
            numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
            data_copy[numeric_cols] = data_copy[numeric_cols].fillna(data_copy[numeric_cols].mean())
            
        elif strategy == 'median':
            numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
            data_copy[numeric_cols] = data_copy[numeric_cols].fillna(data_copy[numeric_cols].median())
            
        elif strategy == 'mode':
            for col in data_copy.columns:
                if data_copy[col].isnull().any():
                    mode_value = data_copy[col].mode()
                    if not mode_value.empty:
                        data_copy[col].fillna(mode_value[0], inplace=True)
        
        return data_copy
        
    except Exception as e:
        st.error(f"缺失值处理失败: {str(e)}")
        return data

def load_data(file_path):
    """加载数据文件"""
    try:
        if hasattr(file_path, 'name'):
            if file_path.name.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("不支持的文件格式")
        else:
            # 处理字符串路径
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("不支持的文件格式")
        
        return data
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None

def get_data_info(data):
    """获取数据基本信息"""
    try:
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object']).columns)
        }
        return info
    except Exception as e:
        st.error(f"获取数据信息失败: {str(e)}")
        return None

def preprocess_data(data, numeric_strategy='mean', categorical_strategy='mode', scale_features=False):
    """数据预处理"""
    try:
        processed_data = data.copy()
        
        # 处理数值型缺失值
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if numeric_strategy == 'mean':
                processed_data[numeric_cols] = processed_data[numeric_cols].fillna(processed_data[numeric_cols].mean())
            elif numeric_strategy == 'median':
                processed_data[numeric_cols] = processed_data[numeric_cols].fillna(processed_data[numeric_cols].median())
        
        # 处理分类型缺失值
        categorical_cols = processed_data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            if categorical_strategy == 'mode':
                for col in categorical_cols:
                    if processed_data[col].isnull().any():
                        mode_value = processed_data[col].mode()
                        if not mode_value.empty:
                            processed_data[col].fillna(mode_value[0], inplace=True)
        
        # 特征缩放
        if scale_features and len(numeric_cols) > 0:
            scaler = StandardScaler()
            processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])
        
        return processed_data
        
    except Exception as e:
        st.error(f"数据预处理失败: {str(e)}")
        return data

def encode_categorical_variables(data, columns=None, method='label'):
    """编码分类变量"""
    try:
        encoded_data = data.copy()
        
        if columns is None:
            columns = encoded_data.select_dtypes(include=['object']).columns
        
        if method == 'label':
            label_encoders = {}
            for col in columns:
                if col in encoded_data.columns:
                    le = LabelEncoder()
                    encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
                    label_encoders[col] = le
            
            return encoded_data, label_encoders
        
        elif method == 'onehot':
            encoded_data = pd.get_dummies(encoded_data, columns=columns, prefix=columns)
            return encoded_data, None
        
    except Exception as e:
        st.error(f"分类变量编码失败: {str(e)}")
        return data, None

def detect_outliers(data, columns=None, method='iqr'):
    """检测异常值"""
    try:
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        
        for col in columns:
            if col in data.columns:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    outliers[col] = data[outlier_mask].index.tolist()
                
                elif method == 'zscore':
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outlier_mask = z_scores > 3
                    outliers[col] = data[outlier_mask].index.tolist()
        
        return outliers
        
    except Exception as e:
        st.error(f"异常值检测失败: {str(e)}")
        return {}

def remove_outliers(data, outliers_dict):
    """移除异常值"""
    try:
        cleaned_data = data.copy()
        all_outlier_indices = set()
        
        for col, indices in outliers_dict.items():
            all_outlier_indices.update(indices)
        
        cleaned_data = cleaned_data.drop(list(all_outlier_indices))
        cleaned_data = cleaned_data.reset_index(drop=True)
        
        return cleaned_data
        
    except Exception as e:
        st.error(f"异常值移除失败: {str(e)}")
        return data

def split_features_target(data, target_column):
    """分离特征和目标变量"""
    try:
        if target_column not in data.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在")
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        return X, y
        
    except Exception as e:
        st.error(f"特征目标分离失败: {str(e)}")
        return None, None
