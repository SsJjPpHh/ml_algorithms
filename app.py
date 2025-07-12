import streamlit as st
import pandas as pd
import numpy as np
from src.data_utils import *
from src.ml_algorithms import ml_algorithms
from src.plotting import data_visualizer, model_visualizer
from src.stats_utils import statistical_analyzer
from src.interpretability import model_interpreter
import plotly.express as px
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="医学数据分析平台",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 主标题
    st.markdown('<h1 class="main-header">🏥 医学数据分析平台</h1>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("🔧 控制面板")
        
        # 数据加载选项
        st.subheader("📊 数据加载")
        data_source = st.radio(
            "选择数据源",
            ["上传文件", "使用示例数据"]
        )
        
        # 初始化session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None
    
    # 数据加载部分
    if data_source == "上传文件":
        uploaded_file = st.file_uploader(
            "选择CSV或Excel文件",
            type=['csv', 'xlsx', 'xls'],
            help="支持CSV和Excel格式文件"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
    
    else:  # 使用示例数据
        col1, col2 = st.columns(2)
        with col1:
            if st.button("生成分类数据"):
                df = generate_sample_data("classification")
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("分类示例数据生成成功！")
        
        with col2:
            if st.button("生成回归数据"):
                df = generate_sample_data("regression")
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("回归示例数据生成成功！")
    
    # 如果数据已加载，显示主要功能
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        # 创建标签页
        tab_data, tab_ml, tab_stats, tab_interpret = st.tabs([
            "📊 数据概览", 
            "🤖 机器学习", 
            "📈 统计分析", 
            "🔍 模型解释"
        ])
        
        # 数据概览标签页
        with tab_data:
            show_data_overview(df)
        
        # 机器学习标签页
        with tab_ml:
            show_machine_learning(df)
        
        # 统计分析标签页
        with tab_stats:
            show_statistical_analysis(df)
        
        # 模型解释标签页
        with tab_interpret:
            show_model_interpretation(df)

def show_data_overview(df):
    """显示数据概览"""
    st.markdown('<h2 class="sub-header">📊 数据概览</h2>', unsafe_allow_html=True)
    
    # 基本信息
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("数据行数", df.shape[0])
    with col2:
        st.metric("数据列数", df.shape[1])
    with col3:
        st.metric("缺失值", df.isnull().sum().sum())
    with col4:
        st.metric("重复行", df.duplicated().sum())
    
    # 数据预览
    st.subheader("🔍 数据预览")
    st.dataframe(df.head(10), use_container_width=True)
    
    # 数据信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 数据信息")
        info_dict = get_data_info(df)
        for key, value in info_dict.items():
            if isinstance(value, list):
                st.write(f"**{key}**: {', '.join(value) if value else '无'}")
            else:
                st.write(f"**{key}**: {value}")
    
    with col2:
        st.subheader("❌ 缺失值详情")
        missing_info = get_missing_info(df)
        if not missing_info.empty:
            st.dataframe(missing_info)
        else:
            st.success("数据无缺失值！")
    
    # 数据可视化
    st.subheader("📈 数据可视化")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # 数据概览图
        data_visualizer.plot_data_overview(df)
    
    with viz_col2:
        # 数值变量分布
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_cols = st.multiselect(
                "选择要可视化的数值变量",
                numeric_cols,
                default=numeric_cols[:3]
            )
            
            plot_type = st.selectbox("选择图表类型", ["histogram", "boxplot"])
            
            if selected_cols:
                data_visualizer.plot_distribution(df, selected_cols, plot_type)
    
    # 相关性分析
    if len(df.select_dtypes(include=[np.number]).columns) >= 2:
        st.subheader("🔗 相关性分析")
        corr_method = st.selectbox("选择相关性方法", ["pearson", "spearman", "kendall"])
        corr_matrix = data_visualizer.plot_correlation_matrix(df, corr_method)

def show_machine_learning(df):
    """显示机器学习功能"""
    st.markdown('<h2 class="sub-header">🤖 机器学习</h2>', unsafe_allow_html=True)
    
    # 任务配置
    st.subheader("⚙️ 任务配置")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        task_type = st.selectbox(
            "选择任务类型",
            ["分类", "回归", "聚类"]
        )
    
    with col2:
        target_col = None
        if task_type in ["分类", "回归"]:
            target_col = st.selectbox(
                "选择目标变量",
                df.columns.tolist()
            )
    
    with col3:
        if task_type in ["分类", "回归"]:
            test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
    
    # 算法选择和参数配置
    if task_type in ml_algorithms.algorithms:
        st.subheader("🔧 算法选择")
        
        algorithm_name = st.selectbox(
            "选择算法",
            list(ml_algorithms.algorithms[task_type].keys())
        )
        
        # 参数配置
        st.subheader("⚙️ 参数配置")
        params = ml_algorithms.get_algorithm_params(task_type, algorithm_name)
        
        # 数据预处理选项
        st.subheader("🔄 数据预处理")
        preprocess_col1, preprocess_col2 = st.columns(2)
        
        with preprocess_col1:
            handle_missing = st.selectbox(
                "缺失值处理",
                ["mean", "median", "most_frequent", "constant"]
            )
        
        with preprocess_col2:
            scale_features = st.checkbox("特征标准化", value=True)
        
        # 训练模型
        if st.button("🚀 开始训练", type="primary"):
            if task_type in ["分类", "回归"] and target_col:
                with st.spinner("正在训练模型..."):
                    # 数据预处理
                    X_train, X_test, y_train, y_test = preprocess_data(
                        df, target_col, task_type, test_size, 
                        handle_missing, scale_features
                    )
                    
                    if X_train is not None:
                        # 创建和训练模型
                        model = ml_algorithms.create_model(task_type, algorithm_name, params)
                        trained_model, success, message = ml_algorithms.train_model(model, X_train, y_train)
                        
                        if success:
                            st.success(message)
                            st.session_state.trained_model = trained_model
                            st.session_state.model_trained = True
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            st.session_state.task_type = task_type
                            st.session_state.algorithm_name = algorithm_name
                            
                            # 模型评估
                            st.subheader("📊 模型评估")
                            
                            if task_type == "分类":
                                metrics, y_pred, y_pred_proba = ml_algorithms.evaluate_classification(
                                    trained_model, X_test, y_test
                                )
                                
                                if metrics:
                                    # 显示评估指标
                                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                    
                                    with metric_col1:
                                        st.metric("准确率", f"{metrics['accuracy']:.4f}")
                                    with metric_col2:
                                        st.metric("精确率", f"{metrics['precision']:.4f}")
                                    with metric_col3:
                                        st.metric("召回率", f"{metrics['recall']:.4f}")
                                    with metric_col4:
                                        st.metric("F1分数", f"{metrics['f1_score']:.4f}")
                                    
                                    if 'roc_auc' in metrics:
                                        st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
                                    
                                    # 可视化结果
                                    viz_col1, viz_col2 = st.columns(2)
                                    
                                    with viz_col1:
                                        # 混淆矩阵
                                        model_visualizer.plot_confusion_matrix(y_test, y_pred)
                                    
                                    with viz_col2:
                                        # ROC曲线
                                        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                                            model_visualizer.plot_roc_curve(y_test, y_pred_proba)
                                    
                                    # 特征重要性
                                    feature_names = X_train.columns.tolist()
                                    model_visualizer.plot_feature_importance(trained_model, feature_names)
                            
                            elif task_type == "回归":
                                metrics, y_pred = ml_algorithms.evaluate_regression(trained_model, X_test, y_test)
                                
                                if metrics:
                                    # 显示评估指标
                                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                    
                                    with metric_col1:
                                        st.metric("MSE", f"{metrics['mse']:.4f}")
                                    with metric_col2:
                                        st.metric("RMSE", f"{metrics['rmse']:.4f}")
                                    with metric_col3:
                                        st.metric("MAE", f"{metrics['mae']:.4f}")
                                    with metric_col4:
                                        st.metric("R²分数", f"{metrics['r2_score']:.4f}")
                                    
                                    # 回归结果可视化
                                    model_visualizer.plot_regression_results(y_test, y_pred)
                                    
                                    # 特征重要性
                                    feature_names = X_train.columns.tolist()
                                    model_visualizer.plot_feature_importance(trained_model, feature_names)
                        else:
                            st.error(message)
            
            elif task_type == "聚类":
                with st.spinner("正在进行聚类分析..."):
                    # 聚类不需要目标变量
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) == 0:
                        st.error("聚类分析需要数值型变量")
                    else:
                        X = df[numeric_cols].fillna(df[numeric_cols].mean())
                        
                        # 特征标准化
                        if scale_features:
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                        
                        # 创建和训练聚类模型
                        model = ml_algorithms.create_model(task_type, algorithm_name, params)
                        metrics, labels = ml_algorithms.evaluate_clustering(model, X)
                        
                        if metrics:
                            st.success("聚类分析完成！")
                            
                            # 显示聚类指标
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("聚类数量", metrics['n_clusters'])
                            with metric_col2:
                                if 'silhouette_score' in metrics:
                                    st.metric("轮廓系数", f"{metrics['silhouette_score']:.4f}")
                            
                            # 聚类结果可视化
                            model_visualizer.plot_clustering_results(X, labels, algorithm_name)
                            
                            # 保存聚类结果
                            st.session_state.trained_model = model
                            st.session_state.model_trained = True
                            st.session_state.cluster_labels = labels
                            st.session_state.task_type = task_type
                            st.session_state.algorithm_name = algorithm_name

def show_statistical_analysis(df):
    """显示统计分析功能"""
    st.markdown('<h2 class="sub-header">📈 统计分析</h2>', unsafe_allow_html=True)
    
    # 描述性统计
    st.subheader("📊 描述性统计")
    
    desc_col1, desc_col2 = st.columns(2)
    
    with desc_col1:
        group_col = st.selectbox(
            "分组变量 (可选)",
            [None] + df.columns.tolist(),
            key="desc_group"
        )
    
    with desc_col2:
        if st.button("生成描述性统计"):
            desc_stats = statistical_analyzer.descriptive_stats(df, group_col)
            if desc_stats is not None:
                st.dataframe(desc_stats, use_container_width=True)
    
    st.divider()
    
    # 假设检验
    st.subheader("🔬 假设检验")
    
    test_type = st.selectbox(
        "选择检验类型",
        ["正态性检验", "两组比较", "多组比较", "相关分析", "卡方检验"]
    )
    
    if test_type == "正态性检验":
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                test_var = st.selectbox("选择检验变量", numeric_cols)
                norm_test_type = st.selectbox("检验方法", ["shapiro", "ks"])
        
        with col2:
            if st.button("运行正态性检验"):
                if numeric_cols:
                    result = statistical_analyzer.normality_test(df[test_var], norm_test_type)
                    
                    if 'error' not in result:
                        st.write("**检验结果:**")
                        st.write(f"- 检验方法: {result['test']}")
                        st.write(f"- 统计量: {result['statistic']:.6f}")
                        st.write(f"- p值: {result['p_value']:.6f}")
                        st.write(f"- 结论: {result['interpretation']}")
                        
                        # 可视化
                        fig = px.histogram(df, x=test_var, title=f"{test_var} 分布直方图")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(result['error'])
    
    elif test_type == "两组比较":
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols and categorical_cols:
                test_var = st.selectbox("数值变量", numeric_cols)
                group_var = st.selectbox("分组变量", categorical_cols)
                test_method = st.selectbox("检验方法", ["auto", "ttest", "mannwhitney"])
                paired = st.checkbox("配对检验")
        
        with col2:
            if st.button("运行两组比较"):
                if numeric_cols and categorical_cols:
                    # 检查分组变量是否只有两个组
                    groups = df[group_var].dropna().unique()
                    if len(groups) != 2:
                        st.error("分组变量必须恰好有两个组")
                    else:
                        group1_data = df[df[group_var] == groups[0]][test_var]
                        group2_data = df[df[group_var] == groups[1]][test_var]
                        
                        result = statistical_analyzer.two_sample_test(
                            group1_data, group2_data, test_method, paired
                        )
                        
                        if 'error' not in result:
                            st.write("**检验结果:**")
                            st.write(f"- 检验方法: {result['test']}")
                            st.write(f"- 统计量: {result['statistic']:.6f}")
                            st.write(f"- p值: {result['p_value']:.6f}")
                            
                            if 'mean_group1' in result:
                                st.write(f"- {groups[0]}组均值: {result['mean_group1']:.4f}")
                                st.write(f"- {groups[1]}组均值: {result['mean_group2']:.4f}")
                            
                            if 'effect_size' in result:
                                st.write(f"- 效应量 (Cohen's d): {result['effect_size']:.4f}")
                                st.write(f"- 效应量解释: {result['effect_interpretation']}")
                            
                            # 可视化
                            fig = px.box(df, x=group_var, y=test_var, title=f"{test_var} 按 {group_var} 分组")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(result['error'])
    
    elif test_type == "多组比较":
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols and categorical_cols:
                dependent_var = st.selectbox("因变量", numeric_cols)
                independent_var = st.selectbox("自变量", categorical_cols)
        
        with col2:
            if st.button("运行方差分析"):
                if numeric_cols and categorical_cols:
                    result = statistical_analyzer.anova_test(df, dependent_var, independent_var)
                    
                    if 'error' not in result:
                        st.write("**ANOVA结果:**")
                        st.write(f"- F统计量: {result['f_statistic']:.6f}")
                        st.write(f"- p值: {result['p_value']:.6f}")
                        
                        # 显示详细结果
                        if 'detailed_results' in result:
                            st.write("**详细结果:**")
                            st.dataframe(result['detailed_results'])
                        
                        # 事后检验
                        if 'posthoc_test' in result:
                            st.write("**事后检验 (Tukey HSD):**")
                            st.dataframe(result['posthoc_test'])
                        
                        # 可视化
                        fig = px.box(df, x=independent_var, y=dependent_var, 
                                   title=f"{dependent_var} 按 {independent_var} 分组")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(result['error'])
    
    elif test_type == "相关分析":
        col1, col2 = st.columns(2)
        
        with col1:
            corr_method = st.selectbox("相关方法", ["pearson", "spearman", "kendall"])
        
        with col2:
            if st.button("运行相关分析"):
                result = statistical_analyzer.correlation_analysis(df, corr_method)
                
                if 'error' not in result:
                    st.write(f"**{corr_method.capitalize()} 相关分析结果:**")
                    
                    # 显示相关系数矩阵
                    st.dataframe(result['correlation_matrix'].round(4))
                    
                    # 显示带p值的相关结果
                    if 'correlation_with_pvalues' in result:
                        st.write("**相关系数及显著性:**")
                        st.dataframe(result['correlation_with_pvalues'].round(4))
                    
                    # 热图可视化
                    corr_matrix = result['correlation_matrix']
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title=f"{corr_method.capitalize()} 相关系数热图",
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(result['error'])
    
    elif test_type == "卡方检验":
        col1, col2 = st.columns(2)
        
        with col1:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(categorical_cols) >= 2:
                var1 = st.selectbox("变量1", categorical_cols)
                var2 = st.selectbox("变量2", [col for col in categorical_cols if col != var1])
        
        with col2:
            if st.button("运行卡方检验"):
                if len(categorical_cols) >= 2:
                    result = statistical_analyzer.chi_square_test(df, var1, var2)
                    
                    if 'error' not in result:
                        st.write("**卡方检验结果:**")
                        st.write(f"- 卡方统计量: {result['chi2_statistic']:.6f}")
                        st.write(f"- p值: {result['p_value']:.6f}")
                        st.write(f"- 自由度: {result['degrees_of_freedom']}")
                        st.write(f"- Cramer's V: {result['cramers_v']:.4f}")
                        
                        # 列联表
                        st.write("**列联表:**")
                        st.dataframe(result['contingency_table'])
                        
                        # 可视化
                        fig = px.density_heatmap(
                            df, x=var1, y=var2, 
                            title=f"{var1} vs {var2} 列联表热图"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(result['error'])
    
    st.divider()
    
    # 回归分析
    st.subheader("📊 回归分析")
    
    regression_type = st.selectbox("回归类型", ["线性回归", "逻辑回归"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        all_cols = df.columns.tolist()
        dependent_var = st.selectbox("因变量", all_cols, key="reg_dependent")
        
        available_vars = [col for col in all_cols if col != dependent_var]
        independent_vars = st.multiselect("自变量", available_vars, key="reg_independent")
    
    with col2:
        if st.button("运行回归分析"):
            if independent_vars:
                if regression_type == "线性回归":
                    result = statistical_analyzer.linear_regression_analysis(
                        df, dependent_var, independent_vars
                    )
                else:  # 逻辑回归
                    result = statistical_analyzer.logistic_regression_analysis(
                        df, dependent_var, independent_vars
                    )
                
                if 'error' not in result:
                    st.write(f"**{regression_type}结果:**")
                    
                    # 显示模型摘要
                    st.text(str(result['model_summary']))
                    
                    # 显示关键指标
                    if regression_type == "线性回归":
                        st.write(f"- R²: {result['r_squared']:.4f}")
                        st.write(f"- 调整R²: {result['adjusted_r_squared']:.4f}")
                    else:
                        st.write(f"- 伪R²: {result['pseudo_r_squared']:.4f}")
                        st.write(f"- AIC: {result['aic']:.4f}")
                        
                        # 显示OR值
                        st.write("**比值比 (OR) 及置信区间:**")
                        or_results = pd.DataFrame({
                            'OR': result['odds_ratios'],
                            'CI_lower': result['or_confidence_intervals'].iloc[:, 0],
                            'CI_upper': result['or_confidence_intervals'].iloc[:, 1],
                            'p_value': result['p_values']
                        })
                        st.dataframe(or_results.round(4))
                else:
                    st.error(result['error'])
            else:
                st.warning("请选择至少一个自变量")

def show_model_interpretation(df):
    """显示模型解释功能"""
    st.markdown('<h2 class="sub-header">🔍 模型解释</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('model_trained', False):
        st.warning("请先在机器学习页面训练一个模型")
        return
    
    model = st.session_state.trained_model
    task_type = st.session_state.task_type
    algorithm_name = st.session_state.algorithm_name
    
    if task_type == "聚类":
        st.info("聚类模型的解释功能有限")
        return
    
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    feature_names = X_train.columns.tolist()
    
    st.subheader("🎯 模型解释方法")
    
    interpretation_method = st.selectbox(
        "选择解释方法",
        ["SHAP分析", "排列重要性", "局部解释", "决策树规则"]
    )
    
    if interpretation_method == "SHAP分析":
        st.subheader("🔍 SHAP (SHapley Additive exPlanations) 分析")
        
        # 确定模型类型
        model_type = "tree"
        if algorithm_name in ["Logistic Regression", "Linear Regression", "Elastic Net"]:
            model_type = "linear"
        elif algorithm_name in ["SVM", "SVR", "MLP Neural Network"]:
            model_type = "kernel"
        
        if st.button("运行SHAP分析"):
            with st.spinner("正在计算SHAP值..."):
                success, message = model_interpreter.explain_model_shap(
                    model, X_train, X_test, model_type
                )
                
                if success:
                    st.success(message)
                    
                    # SHAP特征重要性汇总
                    st.subheader("📊 特征重要性汇总")
                    importance_df = model_interpreter.plot_shap_summary(feature_names)
                    
                    # SHAP依赖图
                    st.subheader("📈 SHAP依赖图")
                    feature_idx = st.selectbox(
                        "选择特征",
                        range(len(feature_names)),
                        format_func=lambda x: feature_names[x]
                    )
                    model_interpreter.plot_shap_dependence(feature_idx, feature_names, X_test)
                    
                    # 单样本SHAP瀑布图
                    st.subheader("🌊 单样本SHAP瀑布图")
                    sample_idx = st.slider("选择样本", 0, len(X_test)-1, 0)
                    model_interpreter.plot_shap_waterfall(sample_idx, feature_names)
                    
                else:
                    st.error(message)
    
    elif interpretation_method == "排列重要性":
        st.subheader("🔄 排列重要性分析")
        
        n_repeats = st.slider("重复次数", 5, 20, 10)
        
        if st.button("运行排列重要性分析"):
            with st.spinner("正在计算排列重要性..."):
                importance_df = model_interpreter.permutation_importance_analysis(
                    model, X_test, y_test, feature_names, n_repeats
                )
                
                if importance_df is not None:
                    st.subheader("📋 排列重要性结果")
                    st.dataframe(importance_df.sort_values('importance_mean', ascending=False))
    
    elif interpretation_method == "局部解释":
        st.subheader("🎯 单样本局部解释")
        
        sample_idx = st.slider("选择样本索引", 0, len(X_test)-1, 0)
        
        # 确定模型类型
        model_type = "tree"
        if algorithm_name in ["Logistic Regression", "Linear Regression", "Elastic Net"]:
            model_type = "linear"
        elif algorithm_name in ["SVM", "SVR", "MLP Neural Network"]:
            model_type = "kernel"
        
        if st.button("生成局部解释"):
            with st.spinner("正在生成局部解释..."):
                explanation = model_interpreter.local_explanation(
                    model, X_test, sample_idx, feature_names, model_type
                )
                
                if explanation:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 样本信息")
                        st.write(f"**预测结果**: {explanation['prediction']}")
                        
                        if explanation['prediction_proba'] is not None:
                            if len(explanation['prediction_proba']) == 2:
                                st.write(f"**预测概率**: {explanation['prediction_proba'][1]:.4f}")
                            else:
                                st.write("**预测概率**:")
                                for i, prob in enumerate(explanation['prediction_proba']):
                                    st.write(f"  类别 {i}: {prob:.4f}")
                        
                        st.subheader("📋 样本特征值")
                        st.dataframe(explanation['sample_data'].T, use_container_width=True)
                    
                    with col2:
                        st.subheader("🔍 特征贡献度")
                        explanation_df = explanation['explanation']
                        
                        # 可视化特征贡献
                        fig = px.bar(
                            explanation_df.head(10),
                            x='shap_value',
                            y='feature',
                            orientation='h',
                            title='Top 10 特征贡献度',
                            color='shap_value',
                            color_continuous_scale='RdBu_r'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(explanation_df, use_container_width=True)
    
    elif interpretation_method == "决策树规则":
        st.subheader("🌳 决策树规则解释")
        
        if "Tree" in algorithm_name or "Forest" in algorithm_name:
            max_depth = st.slider("显示深度", 1, 10, 3)
            
            if st.button("提取决策规则"):
                rules = model_interpreter.explain_tree_model(model, feature_names, max_depth)
                
                st.subheader("📜 决策规则")
                st.text(rules)
        else:
            st.info("该解释方法仅适用于基于树的模型")
    
    # 模型性能总结
    st.divider()
    st.subheader("📈 模型性能总结")
    
    if task_type == "分类":
        from sklearn.metrics import classification_report
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("分类报告:")
            report = classification_report(y_test, y_pred)
            st.text(report)
        
        with col2:
            # 混淆矩阵
            model_visualizer.plot_confusion_matrix(y_test, y_pred)
    
    elif task_type == "回归":
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("均方误差 (MSE)", f"{mse:.4f}")
        with col2:
            st.metric("R² 分数", f"{r2:.4f}")
        
        # 回归结果可视化
        model_visualizer.plot_regression_results(y_test, y_pred)

if __name__ == "__main__":
    main()

