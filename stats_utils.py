import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import pingouin as pg
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """统计分析类"""
    
    def __init__(self):
        pass
    
    def descriptive_stats(self, df, group_col=None):
        """描述性统计分析"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("没有数值型变量可进行描述性统计")
            return None
        
        if group_col and group_col in df.columns:
            # 分组描述性统计
            desc_stats = df.groupby(group_col)[numeric_cols].describe()
            desc_stats = desc_stats.round(3)
        else:
            # 整体描述性统计
            desc_stats = df[numeric_cols].describe()
            desc_stats = desc_stats.round(3)
        
        return desc_stats
    
    def normality_test(self, data, test_type='shapiro'):
        """正态性检验"""
        data_clean = pd.Series(data).dropna()
        
        if len(data_clean) < 3:
            return {"error": "样本量太小，无法进行正态性检验"}
        
        results = {}
        
        if test_type == 'shapiro':
            if len(data_clean) <= 5000:  # Shapiro-Wilk适用于小样本
                stat, p_value = stats.shapiro(data_clean)
                results = {
                    'test': 'Shapiro-Wilk',
                    'statistic': stat,
                    'p_value': p_value,
                    'interpretation': '正态分布' if p_value > 0.05 else '非正态分布'
                }
            else:
                st.warning("样本量过大，Shapiro-Wilk检验不适用，自动切换到Kolmogorov-Smirnov检验")
                test_type = 'ks'
        
        if test_type == 'ks':
            # 标准化数据
            normalized_data = (data_clean - data_clean.mean()) / data_clean.std()
            stat, p_value = stats.kstest(normalized_data, 'norm')
            results = {
                'test': 'Kolmogorov-Smirnov',
                'statistic': stat,
                'p_value': p_value,
                'interpretation': '正态分布' if p_value > 0.05 else '非正态分布'
            }
        
        return results
    
    def two_sample_test(self, group1, group2, test_type='auto', paired=False):
        """两样本检验"""
        group1_clean = pd.Series(group1).dropna()
        group2_clean = pd.Series(group2).dropna()
        
        if len(group1_clean) < 2 or len(group2_clean) < 2:
            return {"error": "样本量太小"}
        
        results = {}
        
        if test_type == 'auto':
            # 自动选择检验方法
            norm1 = self.normality_test(group1_clean)
            norm2 = self.normality_test(group2_clean)
            
            if (norm1.get('p_value', 0) > 0.05 and norm2.get('p_value', 0) > 0.05 and 
                len(group1_clean) >= 30 and len(group2_clean) >= 30):
                test_type = 'ttest'
            else:
                test_type = 'mannwhitney'
        
        if test_type == 'ttest':
            if paired:
                if len(group1_clean) != len(group2_clean):
                    return {"error": "配对t检验要求两组样本量相等"}
                stat, p_value = stats.ttest_rel(group1_clean, group2_clean)
                test_name = "配对t检验"
            else:
                stat, p_value = stats.ttest_ind(group1_clean, group2_clean, equal_var=False)
                test_name = "独立样本t检验"
            
            results = {
                'test': test_name,
                'statistic': stat,
                'p_value': p_value,
                'mean_group1': group1_clean.mean(),
                'mean_group2': group2_clean.mean(),
                'std_group1': group1_clean.std(),
                'std_group2': group2_clean.std()
            }
        
        elif test_type == 'mannwhitney':
            stat, p_value = stats.mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')
            results = {
                'test': 'Mann-Whitney U检验',
                'statistic': stat,
                'p_value': p_value,
                'median_group1': group1_clean.median(),
                'median_group2': group2_clean.median()
            }
        
        # 添加效应量计算
        if test_type == 'ttest':
            # Cohen's d
            pooled_std = np.sqrt(((len(group1_clean)-1)*group1_clean.var() + 
                                (len(group2_clean)-1)*group2_clean.var()) / 
                               (len(group1_clean)+len(group2_clean)-2))
            cohens_d = (group1_clean.mean() - group2_clean.mean()) / pooled_std
            results['effect_size'] = cohens_d
            results['effect_interpretation'] = self._interpret_cohens_d(cohens_d)
        
        return results
    
    def chi_square_test(self, data, var1, var2):
        """卡方检验"""
        try:
            contingency_table = pd.crosstab(data[var1], data[var2])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # 计算Cramer's V
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            results = {
                'test': '卡方检验',
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'cramers_v': cramers_v,
                'contingency_table': contingency_table
            }
            
            return results
        except Exception as e:
            return {"error": f"卡方检验失败: {str(e)}"}
    
    def anova_test(self, data, dependent_var, independent_var):
        """方差分析"""
        try:
            groups = [group[dependent_var].dropna() for name, group in data.groupby(independent_var)]
            
            # 检查是否至少有两个组
            if len(groups) < 2:
                return {"error": "至少需要两个组进行方差分析"}
            
            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # 使用pingouin进行更详细的分析
            aov_result = pg.anova(data=data, dv=dependent_var, between=independent_var)
            
            results = {
                'test': '单因素方差分析',
                'f_statistic': f_stat,
                'p_value': p_value,
                'detailed_results': aov_result
            }
            
            # 如果显著，进行事后检验
            if p_value < 0.05:
                posthoc = pg.pairwise_tukey(data=data, dv=dependent_var, between=independent_var)
                results['posthoc_test'] = posthoc
            
            return results
        except Exception as e:
            return {"error": f"方差分析失败: {str(e)}"}
    
    def correlation_analysis(self, data, method='pearson'):
        """相关分析"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {"error": "至少需要两个数值变量进行相关分析"}
        
        try:
            if method == 'pearson':
                corr_matrix = numeric_data.corr(method='pearson')
                # 计算p值
                corr_with_p = pg.rcorr(numeric_data, method='pearson')
            elif method == 'spearman':
                corr_matrix = numeric_data.corr(method='spearman')
                corr_with_p = pg.rcorr(numeric_data, method='spearman')
            else:  # kendall
                corr_matrix = numeric_data.corr(method='kendall')
                corr_with_p = pg.rcorr(numeric_data, method='kendall')
            
            results = {
                'correlation_matrix': corr_matrix,
                'correlation_with_pvalues': corr_with_p,
                'method': method
            }
            
            return results
        except Exception as e:
            return {"error": f"相关分析失败: {str(e)}"}
    
    def linear_regression_analysis(self, data, dependent_var, independent_vars):
        """线性回归分析"""
        try:
            # 准备数据
            y = data[dependent_var].dropna()
            X = data[independent_vars].dropna()
            
            # 确保X和y的索引匹配
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            # 添加常数项
            X_with_const = sm.add_constant(X)
            
            # 拟合模型
            model = sm.OLS(y, X_with_const).fit()
            
            results = {
                'model_summary': model.summary(),
                'r_squared': model.rsquared,
                'adjusted_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'coefficients': model.params,
                'p_values': model.pvalues,
                'confidence_intervals': model.conf_int()
            }
            
            return results
        except Exception as e:
            return {"error": f"线性回归分析失败: {str(e)}"}
    
    def logistic_regression_analysis(self, data, dependent_var, independent_vars):
        """逻辑回归分析"""
        try:
            # 准备数据
            y = data[dependent_var].dropna()
            X = data[independent_vars].dropna()
            
            # 确保X和y的索引匹配
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            # 添加常数项
            X_with_const = sm.add_constant(X)
            
            # 拟合模型
            model = sm.Logit(y, X_with_const).fit()
            
            # 计算OR值和置信区间
            odds_ratios = np.exp(model.params)
            or_ci = np.exp(model.conf_int())
            
            results = {
                'model_summary': model.summary(),
                'pseudo_r_squared': model.prsquared,
                'log_likelihood': model.llf,
                'aic': model.aic,
                'bic': model.bic,
                'coefficients': model.params,
                'p_values': model.pvalues,
                'odds_ratios': odds_ratios,
                'or_confidence_intervals': or_ci
            }
            
            return results
        except Exception as e:
            return {"error": f"逻辑回归分析失败: {str(e)}"}
    
    def survival_analysis(self, data, duration_col, event_col, group_col=None):
        """生存分析"""
        try:
            # Kaplan-Meier生存分析
            kmf = KaplanMeierFitter()
            
            if group_col:
                # 分组生存分析
                results = {'groups': {}}
                fig = go.Figure()
                
                for name, group in data.groupby(group_col):
                    kmf.fit(group[duration_col], group[event_col], label=str(name))
                    
                    # 添加到图表
                    fig.add_trace(go.Scatter(
                        x=kmf.timeline,
                        y=kmf.survival_function_[str(name)],
                        mode='lines',
                        name=f'组 {name}',
                        line=dict(width=2)
                    ))
                    
                    results['groups'][name] = {
                        'median_survival': kmf.median_survival_time_,
                        'survival_function': kmf.survival_function_
                    }
                
                # Log-rank检验
                groups = [group for name, group in data.groupby(group_col)]
                if len(groups) == 2:
                    logrank_result = logrank_test(
                        groups[0][duration_col], groups[1][duration_col],
                        groups[0][event_col], groups[1][event_col]
                    )
                    results['logrank_test'] = {
                        'test_statistic': logrank_result.test_statistic,
                        'p_value': logrank_result.p_value
                    }
                
            else:
                # 整体生存分析
                kmf.fit(data[duration_col], data[event_col])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=kmf.timeline,
                    y=kmf.survival_function_.iloc[:, 0],
                    mode='lines',
                    name='生存',
                    mode='lines',
                    name='生存曲线',
                    line=dict(width=2)
                ))
                
                results = {
                    'median_survival': kmf.median_survival_time_,
                    'survival_function': kmf.survival_function_
                }
            
            # 设置图表布局
            fig.update_layout(
                title='Kaplan-Meier 生存曲线',
                xaxis_title='时间',
                yaxis_title='生存概率',
                yaxis=dict(range=[0, 1])
            )
            
            results['survival_plot'] = fig
            return results
            
        except Exception as e:
            return {"error": f"生存分析失败: {str(e)}"}
    
    def cox_regression(self, data, duration_col, event_col, covariates):
        """Cox比例风险回归"""
        try:
            # 准备数据
            analysis_data = data[[duration_col, event_col] + covariates].dropna()
            
            # 拟合Cox模型
            cph = CoxPHFitter()
            cph.fit(analysis_data, duration_col=duration_col, event_col=event_col)
            
            results = {
                'model_summary': cph.summary,
                'concordance_index': cph.concordance_index_,
                'log_likelihood': cph.log_likelihood_,
                'aic': cph.AIC_,
                'hazard_ratios': np.exp(cph.params_),
                'confidence_intervals': np.exp(cph.confidence_intervals_),
                'p_values': cph.summary['p']
            }
            
            return results
        except Exception as e:
            return {"error": f"Cox回归分析失败: {str(e)}"}
    
    def _interpret_cohens_d(self, d):
        """解释Cohen's d效应量"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "微小效应"
        elif abs_d < 0.5:
            return "小效应"
        elif abs_d < 0.8:
            return "中等效应"
        else:
            return "大效应"

# 创建全局实例
statistical_analyzer = StatisticalAnalyzer()

