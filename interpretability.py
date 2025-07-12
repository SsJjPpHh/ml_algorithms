import shap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """模型可解释性分析类"""
    
    def __init__(self):
        self.explainer = None
        self.shap_values = None
    
    def explain_model_shap(self, model, X_train, X_test, model_type='tree'):
        """使用SHAP解释模型"""
        try:
            # 根据模型类型选择合适的explainer
            if model_type == 'tree':
                # 适用于树模型 (RandomForest, XGBoost, LightGBM等)
                self.explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                # 适用于线性模型
                self.explainer = shap.LinearExplainer(model, X_train)
            elif model_type == 'kernel':
                # 通用但较慢的explainer
                self.explainer = shap.KernelExplainer(model.predict, X_train.sample(min(100, len(X_train))))
            else:
                # 默认使用Explainer
                self.explainer = shap.Explainer(model, X_train)
            
            # 计算SHAP值
            self.shap_values = self.explainer.shap_values(X_test)
            
            return True, "SHAP分析完成"
        except Exception as e:
            return False, f"SHAP分析失败: {str(e)}"
    
    def plot_shap_summary(self, feature_names, max_display=20):
        """绘制SHAP特征重要性汇总图"""
        if self.shap_values is None:
            st.error("请先运行SHAP分析")
            return
        
        try:
            # 处理多分类情况
            if isinstance(self.shap_values, list):
                shap_values_to_plot = self.shap_values[1]  # 使用第二类的SHAP值
            else:
                shap_values_to_plot = self.shap_values
            
            # 计算特征重要性（平均绝对SHAP值）
            feature_importance = np.mean(np.abs(shap_values_to_plot), axis=0)
            
            # 创建DataFrame并排序
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True).tail(max_display)
            
            # 使用Plotly绘制
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title=f'SHAP特征重要性 (Top {max_display})',
                labels={'importance': '平均|SHAP值|', 'feature': '特征'}
            )
            fig.update_layout(height=max(400, len(importance_df) * 25))
            
            st.plotly_chart(fig, use_container_width=True)
            
            return importance_df
        except Exception as e:
            st.error(f"绘制SHAP汇总图失败: {str(e)}")
            return None
    
    def plot_shap_waterfall(self, sample_idx, feature_names, max_display=20):
        """绘制单个样本的SHAP瀑布图"""
        if self.shap_values is None:
            st.error("请先运行SHAP分析")
            return
        
        try:
            # 处理多分类情况
            if isinstance(self.shap_values, list):
                shap_values_sample = self.shap_values[1][sample_idx]
            else:
                shap_values_sample = self.shap_values[sample_idx]
            
            # 获取基线值
            if hasattr(self.explainer, 'expected_value'):
                if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                    expected_value = self.explainer.expected_value[1] if len(self.explainer.expected_value) > 1 else self.explainer.expected_value[0]
                else:
                    expected_value = self.explainer.expected_value
            else:
                expected_value = 0
            
            # 创建瀑布图数据
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_values_sample
            })
            
            # 按绝对值排序并取前N个
            shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
            shap_df = shap_df.sort_values('abs_shap', ascending=False).head(max_display)
            
            # 创建瀑布图
            fig = go.Figure()
            
            cumulative = expected_value
            x_pos = 0
            
            # 添加基线
            fig.add_trace(go.Bar(
                x=[x_pos],
                y=[expected_value],
                name='基线值',
                marker_color='lightgray'
            ))
            x_pos += 1
            
            # 添加每个特征的贡献
            for _, row in shap_df.iterrows():
                color = 'red' if row['shap_value'] > 0 else 'blue'
                fig.add_trace(go.Bar(
                    x=[x_pos],
                    y=[row['shap_value']],
                    name=row['feature'],
                    marker_color=color,
                    base=cumulative if row['shap_value'] > 0 else cumulative + row['shap_value']
                ))
                cumulative += row['shap_value']
                x_pos += 1
            
            # 添加最终预测值
            fig.add_trace(go.Bar(
                x=[x_pos],
                y=[cumulative],
                name='预测值',
                marker_color='green'
            ))
            
            fig.update_layout(
                title=f'样本 {sample_idx} 的SHAP瀑布图',
                xaxis_title='特征',
                yaxis_title='SHAP值',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"绘制SHAP瀑布图失败: {str(e)}")
    
    def plot_shap_dependence(self, feature_idx, feature_names, X_test):
        """绘制SHAP依赖图"""
        if self.shap_values is None:
            st.error("请先运行SHAP分析")
            return
        
        try:
            # 处理多分类情况
            if isinstance(self.shap_values, list):
                shap_values_to_plot = self.shap_values[1]
            else:
                shap_values_to_plot = self.shap_values
            
            feature_name = feature_names[feature_idx]
            
            fig = px.scatter(
                x=X_test.iloc[:, feature_idx],
                y=shap_values_to_plot[:, feature_idx],
                title=f'{feature_name} 的SHAP依赖图',
                labels={'x': feature_name, 'y': f'SHAP值 ({feature_name})'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"绘制SHAP依赖图失败: {str(e)}")
    
    def permutation_importance_analysis(self, model, X_test, y_test, feature_names, n_repeats=10):
        """排列重要性分析"""
        try:
            # 计算排列重要性
            perm_importance = permutation_importance(
                model, X_test, y_test, 
                n_repeats=n_repeats, 
                random_state=42
            )
            
            # 创建结果DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=True)
            
            # 绘制排列重要性
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=importance_df['importance_mean'],
                y=importance_df['feature'],
                error_x=dict(array=importance_df['importance_std']),
                mode='markers',
                marker=dict(size=8),
                name='排列重要性'
            ))
            
            fig.update_layout(
                title='排列重要性分析',
                xaxis_title='重要性得分',
                yaxis_title='特征',
                height=max(400, len(importance_df) * 25)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            return importance_df
            
        except Exception as e:
            st.error(f"排列重要性分析失败: {str(e)}")
            return None
    
    def explain_tree_model(self, model, feature_names, max_depth=3):
        """解释树模型的决策规则"""
        try:
            if hasattr(model, 'estimators_'):
                # 随机森林等集成模型，展示第一棵树
                tree_rules = export_text(
                    model.estimators_[0], 
                    feature_names=feature_names,
                    max_depth=max_depth
                )
            elif hasattr(model, 'tree_'):
                # 单个决策树
                tree_rules = export_text(
                    model, 
                    feature_names=feature_names,
                    max_depth=max_depth
                )
            else:
                return "该模型不支持树结构解释"
            
            return tree_rules
            
        except Exception as e:
            return f"树模型解释失败: {str(e)}"
    
    def local_explanation(self, model, X_test, sample_idx, feature_names, model_type='tree'):
        """单个样本的局部解释"""
        try:
            sample = X_test.iloc[sample_idx:sample_idx+1]
            
            # 获取预测结果
            prediction = model.predict(sample)[0]
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(sample)[0]
            else:
                prediction_proba = None
            
            # SHAP局部解释
            if self.explainer is None:
                self.explain_model_shap(model, X_test.sample(min(100, len(X_test))), sample, model_type)
            
            if self.shap_values is not None:
                if isinstance(self.shap_values, list):
                    local_shap = self.shap_values[1][0] if len(self.shap_values) > 1 else self.shap_values[0][0]
                else:
                    local_shap = self.shap_values[0]
                
                # 创建局部解释DataFrame
                explanation_df = pd.DataFrame({
                    'feature': feature_names,
                    'feature_value': sample.iloc[0].values,
                    'shap_value': local_shap
                }).sort_values('shap_value', key=abs, ascending=False)
                
                return {
                    'prediction': prediction,
                    'prediction_proba': prediction_proba,
                    'explanation': explanation_df,
                    'sample_data': sample
                }
            
        except Exception as e:
            st.error(f"局部解释失败: {str(e)}")
            return None
    
    def model_comparison_explanation(self, models_dict, X_test, y_test, feature_names):
        """多模型解释性比较"""
        try:
            comparison_results = {}
            
            for model_name, model in models_dict.items():
                # 计算特征重要性
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_).flatten()
                else:
                    # 使用排列重要性
                    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
                    importance = perm_imp.importances_mean
                
                comparison_results[model_name] = importance
            
            # 创建比较DataFrame
            comparison_df = pd.DataFrame(comparison_results, index=feature_names)
            
            # 绘制比较图
            fig = px.bar(
                comparison_df.T,
                title='不同模型的特征重要性比较',
                labels={'value': '重要性', 'index': '模型'}
            )
            fig.update_layout(height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            
            return comparison_df
            
        except Exception as e:
            st.error(f"模型比较失败: {str(e)}")
            return None

# 创建全局实例
model_interpreter = ModelInterpreter()
