import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataVisualizer:
    """数据可视化类"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
        
    def plot_data_overview(self, df):
        """数据概览可视化"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("数据类型分布")
            dtype_counts = df.dtypes.value_counts()
            fig_dtype = px.pie(
                values=dtype_counts.values, 
                names=dtype_counts.index.astype(str),
                title="数据类型分布"
            )
            st.plotly_chart(fig_dtype, use_container_width=True)
        
        with col2:
            st.subheader("缺失值分布")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
            
            if len(missing_data) > 0:
                fig_missing = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="各列缺失值数量",
                    labels={'x': '缺失值数量', 'y': '列名'}
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("数据无缺失值！")
    
    def plot_distribution(self, df, columns=None, plot_type="histogram"):
        """绘制数据分布图"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns[:6]  # 最多显示6列
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        if plot_type == "histogram":
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=columns,
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(columns):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(height=300*n_rows, title_text="数据分布直方图")
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "boxplot":
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=columns,
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(columns):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1
                
                fig.add_trace(
                    go.Box(y=df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(height=300*n_rows, title_text="数据分布箱线图")
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_correlation_matrix(self, df, method='pearson'):
        """绘制相关系数矩阵热图"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            st.warning("数值型变量少于2个，无法绘制相关系数矩阵")
            return
        
        corr_matrix = numeric_df.corr(method=method)
        
        # 使用Plotly绘制热图
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title=f"相关系数矩阵 ({method.capitalize()})",
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        return corr_matrix
    
    def plot_target_distribution(self, y, target_name="Target"):
        """绘制目标变量分布"""
        col1, col2 = st.columns(2)
        
        with col1:
            if pd.api.types.is_numeric_dtype(y):
                fig = px.histogram(x=y, title=f"{target_name} 分布", nbins=30)
                fig.update_layout(xaxis_title=target_name, yaxis_title="频次")
            else:
                value_counts = pd.Series(y).value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"{target_name} 分布")
                fig.update_layout(xaxis_title=target_name, yaxis_title="频次")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if pd.api.types.is_numeric_dtype(y):
                fig = px.box(y=y, title=f"{target_name} 箱线图")
                fig.update_layout(yaxis_title=target_name)
            else:
                value_counts = pd.Series(y).value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"{target_name} 比例")
            
            st.plotly_chart(fig, use_container_width=True)

class ModelVisualizer:
    """模型结果可视化类"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = [f"Class {i}" for i in range(len(cm))]
        
        # 使用Plotly绘制热图
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="混淆矩阵",
            labels=dict(x="预测标签", y="真实标签"),
            x=labels,
            y=labels,
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, pos_label=1):
        """绘制ROC曲线"""
        try:
            if len(y_pred_proba.shape) > 1:
                y_scores = y_pred_proba[:, 1]
            else:
                y_scores = y_pred_proba
            
            fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='blue', width=2)
            ))
            
            # 添加对角线
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title='ROC 曲线',
                xaxis_title='假正率 (False Positive Rate)',
                yaxis_title='真正率 (True Positive Rate)',
                width=600, height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return roc_auc
            
        except Exception as e:
            st.error(f"绘制ROC曲线时出错: {str(e)}")
            return None
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, pos_label=1):
        """绘制精确率-召回率曲线"""
        try:
            if len(y_pred_proba.shape) > 1:
                y_scores = y_pred_proba[:, 1]
            else:
                y_scores = y_pred_proba
            
            precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
            pr_auc = auc(recall, precision)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'PR Curve (AUC = {pr_auc:.3f})',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title='精确率-召回率曲线',
                xaxis_title='召回率 (Recall)',
                yaxis_title='精确率 (Precision)',
                width=600, height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return pr_auc
            
        except Exception as e:
            st.error(f"绘制PR曲线时出错: {str(e)}")
            return None
    
    def plot_feature_importance(self, model, feature_names, top_n=20):
        """绘制特征重要性"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                st.warning("该模型不支持特征重要性分析")
                return
            
            # 创建特征重要性DataFrame
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True).tail(top_n)
            
            fig = px.bar(
                feature_imp, 
                x='importance', 
                y='feature',
                orientation='h',
                title=f'Top {top_n} 特征重要性',
                labels={'importance': '重要性', 'feature': '特征'}
            )
            fig.update_layout(height=max(400, len(feature_imp) * 25))
            
            st.plotly_chart(fig, use_container_width=True)
            return feature_imp
            
        except Exception as e:
            st.error(f"绘制特征重要性时出错: {str(e)}")
            return None
    
    def plot_regression_results(self, y_true, y_pred):
        """绘制回归结果"""
        col1, col2 = st.columns(2)
        
        with col1:
            # 真实值 vs 预测值散点图
            fig = px.scatter(
                x=y_true, y=y_pred,
                title='真实值 vs 预测值',
                labels={'x': '真实值', 'y': '预测值'}
            )
            
            # 添加完美预测线
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines',
                name='完美预测',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 残差图
            residuals = y_true - y_pred
            fig = px.scatter(
                x=y_pred, y=residuals,
                title='残差图',
                labels={'x': '预测值', 'y': '残差'}
            )
            
            # 添加零线
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_clustering_results(self, X, labels, algorithm_name="Clustering"):
        """绘制聚类结果"""
        if X.shape[1] > 2:
            # 使用PCA降维到2D
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_
            title_suffix = f" (PCA: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {sum(explained_var):.2%} 方差解释)"
        else:
            X_2d = X
            title_suffix = ""
        
        fig = px.scatter(
            x=X_2d[:, 0], y=X_2d[:, 1],
            color=labels.astype(str),
            title=f'{algorithm_name} 聚类结果{title_suffix}',
            labels={'x': '第一主成分' if X.shape[1] > 2 else 'Feature 1',
                   'y': '第二主成分' if X.shape[1] > 2 else 'Feature 2',
                   'color': '聚类标签'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_learning_curve(self, train_scores, val_scores, train_sizes):
        """绘制学习曲线"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_sizes, y=train_scores,
            mode='lines+markers',
            name='训练得分',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes, y=val_scores,
            mode='lines+markers',
            name='验证得分',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='学习曲线',
            xaxis_title='训练样本数',
            yaxis_title='得分',
            width=600, height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# 创建全局实例
data_visualizer = DataVisualizer()
model_visualizer = ModelVisualizer()
