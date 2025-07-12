import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class MLAlgorithms:
    """机器学习算法管理类"""
    
    def __init__(self):
        self.algorithms = {
            '分类': {
                'Logistic Regression': LogisticRegression,
                'SVM': SVC,
                'Random Forest': RandomForestClassifier,
                'XGBoost': xgb.XGBClassifier,
                'LightGBM': lgb.LGBMClassifier,
                'K-Nearest Neighbors': KNeighborsClassifier,
                'Naive Bayes': GaussianNB,
                'Decision Tree': DecisionTreeClassifier,
                'MLP Neural Network': MLPClassifier,
                'Gradient Boosting': GradientBoostingClassifier
            },
            '回归': {
                'Linear Regression': LinearRegression,
                'Elastic Net': ElasticNet,
                'SVR': SVR,
                'Random Forest': RandomForestRegressor,
                'XGBoost': xgb.XGBRegressor,
                'LightGBM': lgb.LGBMRegressor,
                'K-Nearest Neighbors': KNeighborsRegressor,
                'Decision Tree': DecisionTreeRegressor,
                'MLP Neural Network': MLPRegressor,
                'Gradient Boosting': GradientBoostingRegressor
            },
            '聚类': {
                'K-Means': KMeans,
                'DBSCAN': DBSCAN
            }
        }
    
    def get_algorithm_params(self, task_type, algorithm_name):
        """获取算法参数配置界面"""
        params = {}
        
        if algorithm_name == 'Logistic Regression':
            params['C'] = st.slider('正则化强度 (C)', 0.01, 10.0, 1.0, 0.01)
            params['max_iter'] = st.slider('最大迭代次数', 100, 2000, 1000, 100)
            params['random_state'] = 42
            
        elif algorithm_name == 'SVM' or algorithm_name == 'SVR':
            params['C'] = st.slider('正则化参数 (C)', 0.01, 10.0, 1.0, 0.01)
            params['kernel'] = st.selectbox('核函数', ['rbf', 'linear', 'poly', 'sigmoid'])
            if params['kernel'] == 'rbf':
                params['gamma'] = st.selectbox('Gamma', ['scale', 'auto'])
            if algorithm_name == 'SVM':
                params['probability'] = True
                
        elif 'Random Forest' in algorithm_name:
            params['n_estimators'] = st.slider('树的数量', 10, 500, 100, 10)
            params['max_depth'] = st.slider('最大深度', 1, 20, 10, 1)
            params['min_samples_split'] = st.slider('分割最小样本数', 2, 20, 2, 1)
            params['random_state'] = 42
            
        elif 'XGBoost' in algorithm_name:
            params['n_estimators'] = st.slider('树的数量', 10, 500, 100, 10)
            params['max_depth'] = st.slider('最大深度', 1, 10, 6, 1)
            params['learning_rate'] = st.slider('学习率', 0.01, 0.3, 0.1, 0.01)
            params['random_state'] = 42
            
        elif 'LightGBM' in algorithm_name:
            params['n_estimators'] = st.slider('树的数量', 10, 500, 100, 10)
            params['max_depth'] = st.slider('最大深度', 1, 10, 6, 1)
            params['learning_rate'] = st.slider('学习率', 0.01, 0.3, 0.1, 0.01)
            params['random_state'] = 42
            params['verbose'] = -1
            
        elif 'K-Nearest Neighbors' in algorithm_name:
            params['n_neighbors'] = st.slider('邻居数量 (K)', 1, 20, 5, 1)
            params['weights'] = st.selectbox('权重', ['uniform', 'distance'])
            
        elif algorithm_name == 'Naive Bayes':
            # Naive Bayes 通常不需要太多参数调整
            pass
            
        elif 'Decision Tree' in algorithm_name:
            params['max_depth'] = st.slider('最大深度', 1, 20, 10, 1)
            params['min_samples_split'] = st.slider('分割最小样本数', 2, 20, 2, 1)
            params['random_state'] = 42
            
        elif 'MLP' in algorithm_name:
            hidden_layers = st.selectbox('隐藏层结构', ['(100,)', '(100, 50)', '(200, 100, 50)'])
            params['hidden_layer_sizes'] = eval(hidden_layers)
            params['max_iter'] = st.slider('最大迭代次数', 200, 2000, 500, 100)
            params['random_state'] = 42
            
        elif 'Gradient Boosting' in algorithm_name:
            params['n_estimators'] = st.slider('树的数量', 10, 500, 100, 10)
            params['max_depth'] = st.slider('最大深度', 1, 10, 3, 1)
            params['learning_rate'] = st.slider('学习率', 0.01, 0.3, 0.1, 0.01)
            params['random_state'] = 42
            
        elif algorithm_name == 'Linear Regression':
            # Linear Regression 不需要参数
            pass
            
        elif algorithm_name == 'Elastic Net':
            params['alpha'] = st.slider('Alpha (正则化强度)', 0.01, 2.0, 1.0, 0.01)
            params['l1_ratio'] = st.slider('L1 比例', 0.0, 1.0, 0.5, 0.1)
            params['random_state'] = 42
            
        elif algorithm_name == 'K-Means':
            params['n_clusters'] = st.slider('聚类数量', 2, 10, 3, 1)
            params['random_state'] = 42
            
        elif algorithm_name == 'DBSCAN':
            params['eps'] = st.slider('邻域半径 (eps)', 0.1, 2.0, 0.5, 0.1)
            params['min_samples'] = st.slider('最小样本数', 2, 20, 5, 1)
            
        return params
    
    def create_model(self, task_type, algorithm_name, params):
        """创建模型实例"""
        algorithm_class = self.algorithms[task_type][algorithm_name]
        return algorithm_class(**params)
    
    def train_model(self, model, X_train, y_train):
        """训练模型"""
        try:
            model.fit(X_train, y_train)
            return model, True, "模型训练成功！"
        except Exception as e:
            return None, False, f"模型训练失败: {str(e)}"
    
    def evaluate_classification(self, model, X_test, y_test):
        """分类模型评估"""
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            # 获取预测概率（如果支持）
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # 计算AUC（仅适用于二分类）
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            return metrics, y_pred, y_pred_proba
        except Exception as e:
            st.error(f"评估过程中出现错误: {str(e)}")
            return None, None, None
    
    def evaluate_regression(self, model, X_test, y_test):
        """回归模型评估"""
        try:
            y_pred = model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred)
            }
            
            return metrics, y_pred
        except Exception as e:
            st.error(f"评估过程中出现错误: {str(e)}")
            return None, None
    
    def evaluate_clustering(self, model, X):
        """聚类模型评估"""
        try:
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(X)
            else:
                model.fit(X)
                labels = model.labels_
            
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            metrics = {}
            if len(np.unique(labels)) > 1:  # 确保有多个聚类
                metrics['silhouette_score'] = silhouette_score(X, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            
            metrics['n_clusters'] = len(np.unique(labels))
            
            return metrics, labels
        except Exception as e:
            st.error(f"聚类评估过程中出现错误: {str(e)}")
            return None, None
    
    def save_model(self, model, filename):
        """保存模型"""
        try:
            joblib.dump(model, filename)
            return True, "模型保存成功！"
        except Exception as e:
            return False, f"模型保存失败: {str(e)}"
    
    def load_model(self, filename):
        """加载模型"""
        try:
            model = joblib.load(filename)
            return model, True, "模型加载成功！"
        except Exception as e:
            return None, False, f"模型加载失败: {str(e)}"

# 创建全局实例
ml_algorithms = MLAlgorithms()
