import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 多语言文本字典
lang_dict = {
    'zh': {
        'title': 'IV 理论模拟器：纯理论 IV 模型',
        'exclusion_condition': '排他性条件: 假设工具变量 Z 仅通过内生变量 X 影响被解释变量 Y，不存在直接影响。',
        'original_model': '原始模型',
        'first_stage': '第一阶段',
        'second_stage': '第二阶段',
        'mu1_unbiased': '$\\mu_1$ 是无偏估计',
        'param_control': '模型参数控制',
        'gamma_label': 'γ (IV 强度)',
        'gamma_help': '控制工具变量 Z 对 X 的影响强度',
        'delta_label': 'δ (误差传导)',
        'delta_help': '控制误差项 U 对 X 的影响',
        'phi_label': 'φ (排他性违反)',
        'phi_help': '控制 Z 对 Y 的直接影响',
        'exclusion_violation': '排他性违反：φ = {:.2f}，Z 直接影响 Y，IV 一致性崩塌！',
        'weak_iv': '弱工具变量风险：γ = {:.2f}，IV 强度不足，估计量方差将很大！',
        'endogeneity_bias': '内生性偏差较大：δ = {:.2f}，误差项对 X 影响显著，OLS 将严重有偏！',
        'model_preview': '📋 理论模型预览',
        'param_detail': '📚 参数详解',
        'variable_def': '变量定义',
        'param_meaning': '参数合义',
        'error_term': '误差项',
        'instrument': '工具变量',
        'endogenous': '内生变量',
        'explained': '被解释变量',
        'iv_strength': '工具变量强度',
        'error_transmission': '误差传导系数',
        'exclusion_violation_degree': '排他性违反程度',
        'true_effect': '真实因果效应',
        'regression_comparison': '📊 实时回归结果对比',
        'ols_regression': 'OLS 回归',
        'tsls_regression': '2SLS 回归',
        'true_value': '真实值',
        'model': '模型',
        'iv_diagnosis': '🔍 工具变量诊断',
        'first_stage_f': '第一阶段 F 统计量',
        'iv_weak': 'IV 较弱 (F < 10)',
        'iv_strong': 'IV 强度足够',
        'correlation': 'Corr(X, Z)',
        'covariance': 'Cov(X, Z)',
        'visualization': '📈 数据可视化',
        'scatter_plot': 'X vs Y 散点图与回归线',
        'data_point': '数据点',
        'insight': '💡 关键洞察',
        'ols_bias': 'OLS 偏差',
        'tsls_bias': '2SLS 偏差',
        'improvement': '改善程度',
        'explanation': '解释',
        # 新增：异质性处理效应相关文本
        'hte_section': '🎯 异质性处理效应与四类个体',
        'scenario_choice': '选择实验场景',
            'scenario_basic': '基础模型',
            'scenario_one_option': '场景一：无违抗者 (Defiers = 0%)',
            'scenario_two_option': '场景二：引入违抗者 (Defiers = 20%)',
        'scenario_hte': '异质性处理效应模型',
        'compliers_label': '依从者 (Compliers) 比例',
        'always_takers_label': '始终接受者 (Always-takers) 比例',
        'never_takers_label': '从不接受者 (Never-takers) 比例',
        'defiers_label': '违抗者 (Defiers) 比例',
        'compliers': '依从者 (Compliers)',
        'always_takers': '始终接受者 (Always-takers)',
        'never_takers': '从不接受者 (Never-takers)',
        'defiers': '违抗者 (Defiers)',
        'treatment_effect_compliers': '依从者真实处理效应 (β_C)',
        'treatment_effect_always': '始终接受者真实处理效应 (β_A)',
        'treatment_effect_never': '从不接受者真实处理效应 (β_N)',
        'treatment_effect_defiers': '违抗者真实处理效应 (β_D)',
        'scenario_one': '场景一：无违抗者 (Defiers = 0%)',
        'scenario_one_desc': '验证 LATE (Local Average Treatment Effect) 定理 - IV 估计应完美恢复 Compliers 的处理效应',
        'scenario_two': '场景二：引入违抗者 (Defiers = 20%)',
        'scenario_two_desc': '展示单调性假设违反的后果 - 违抗者的存在如何扭曲 IV 估计量',
        'individual_type': '个体类型',
        'proportion': '比例',
        'true_effect': '真实处理效应',
        'late_theorem': '🔬 LATE 定理验证',
        'late_explanation': 'LATE (Local Average Treatment Effect) 承诺在以下假设下，2SLS 估计的是 Compliers 的平均处理效应：',
        'late_assumption_1': '1. 排他性：Z 只通过 D 影响 Y',
        'late_assumption_2': '2. 相关性：Z 与 D 相关',
        'late_assumption_3': '3. 单调性：不存在违抗者 (Defiers)',
        'late_result_scenario1': '场景一结果：Z→D→Y 的单向因果链，无 Defiers，满足所有 LATE 假设',
        'late_result_scenario2': '场景二结果：Defiers 的存在违反単调性假设，导致 IV 估计不再等于任何组的单一处理效应',
        'monotonicity_violation': '⚠️ 单调性假设违反：当 Z=1 时部分个体不接受处理，当 Z=0 时又接受处理',
        'hte_results': '异质性处理效应结果对比',
        'scenario_label': '实验场景',
    },
    'en': {
        'title': 'IV Theory Simulator: Pure Theoretical IV Model',
        'exclusion_condition': 'Exclusion restriction: Instrument Z affects dependent variable Y only through endogenous variable X, with no direct effect.',
        'original_model': 'Original model',
        'first_stage': 'First stage',
        'second_stage': 'Second stage',
        'mu1_unbiased': '$\\mu_1$ is unbiased estimator',
        'param_control': 'Model Parameter Control',
        'gamma_label': 'γ (IV Strength)',
        'gamma_help': 'Control the effect of instrument Z on X',
        'delta_label': 'δ (Error Transmission)',
        'delta_help': 'Control the effect of error term U on X',
        'phi_label': 'φ (Exclusion Restriction Violation)',
        'phi_help': 'Control direct effect of Z on Y',
        'exclusion_violation': 'Exclusion Restriction Violated: φ = {:.2f}, Z directly affects Y, IV consistency collapsed!',
        'weak_iv': 'Weak Instrument Risk: γ = {:.2f}, insufficient IV strength, estimator variance will be large!',
        'endogeneity_bias': 'Large Endogeneity Bias: δ = {:.2f}, error term has significant effect on X, OLS will be severely biased!',
        'model_preview': '📋 Theoretical Model Preview',
        'param_detail': '📚 Parameter Details',
        'variable_def': 'Variable Definitions',
        'param_meaning': 'Parameter Meanings',
        'error_term': 'Error term',
        'instrument': 'Instrument variable',
        'endogenous': 'Endogenous variable',
        'explained': 'Dependent variable',
        'iv_strength': 'Instrument strength',
        'error_transmission': 'Error transmission coefficient',
        'exclusion_violation_degree': 'Exclusion violation degree',
        'true_effect': 'True causal effect',
        'regression_comparison': '📊 Real-time Regression Comparison',
        'ols_regression': 'OLS Regression',
        'tsls_regression': '2SLS Regression',
        'true_value': 'True value',
        'model': 'Model',
        'iv_diagnosis': '🔍 Instrument Variable Diagnosis',
        'first_stage_f': 'First-Stage F-Statistic',
        'iv_weak': 'Weak IV (F < 10)',
        'iv_strong': 'IV Strength Sufficient',
        'correlation': 'Corr(X, Z)',
        'covariance': 'Cov(X, Z)',
        'visualization': '📈 Data Visualization',
        'scatter_plot': 'Scatter Plot: X vs Y with Regression Lines',
        'data_point': 'Data Points',
        'insight': '💡 Key Insights',
        'ols_bias': 'OLS Bias',
        'tsls_bias': '2SLS Bias',
        'improvement': 'Improvement',
        'explanation': 'Explanation',
        # New additions: Heterogeneous Treatment Effects related text
        'hte_section': '🎯 Heterogeneous Treatment Effects and Four Individual Types',
        'scenario_choice': 'Choose Experiment Scenario',
            'scenario_basic': 'Basic Model',
            'scenario_one_option': 'Scenario One: No Defiers (Defiers = 0%)',
            'scenario_two_option': 'Scenario Two: With Defiers (Defiers = 20%)',
        'scenario_hte': 'Heterogeneous Treatment Effects Model',
        'compliers_label': 'Compliers Proportion',
        'always_takers_label': 'Always-takers Proportion',
        'never_takers_label': 'Never-takers Proportion',
        'defiers_label': 'Defiers Proportion',
        'compliers': 'Compliers',
        'always_takers': 'Always-takers',
        'never_takers': 'Never-takers',
        'defiers': 'Defiers',
        'treatment_effect_compliers': 'Compliers True Treatment Effect (β_C)',
        'treatment_effect_always': 'Always-takers True Treatment Effect (β_A)',
        'treatment_effect_never': 'Never-takers True Treatment Effect (β_N)',
        'treatment_effect_defiers': 'Defiers True Treatment Effect (β_D)',
        'scenario_one': 'Scenario One: No Defiers (Defiers = 0%)',
        'scenario_one_desc': 'Verify LATE (Local Average Treatment Effect) Theorem - IV estimate should perfectly recover Compliers effect',
        'scenario_two': 'Scenario Two: With Defiers (Defiers = 20%)',
        'scenario_two_desc': 'Demonstrate consequences of monotonicity violation - how Defiers distort IV estimates',
        'individual_type': 'Individual Type',
        'proportion': 'Proportion',
        'true_effect': 'True Treatment Effect',
        'late_theorem': '🔬 LATE Theorem Verification',
        'late_explanation': 'LATE (Local Average Treatment Effect) guarantees that under the following assumptions, 2SLS estimates the average treatment effect for Compliers:',
        'late_assumption_1': '1. Exclusion: Z affects Y only through D',
        'late_assumption_2': '2. Relevance: Z is correlated with D',
        'late_assumption_3': '3. Monotonicity: No Defiers exist',
        'late_result_scenario1': 'Scenario One Result: Unidirectional causal chain Z→D→Y, no Defiers, all LATE assumptions satisfied',
        'late_result_scenario2': 'Scenario Two Result: Defiers violate monotonicity, IV estimate no longer equals any single group\'s treatment effect',
        'monotonicity_violation': '⚠️ Monotonicity Assumption Violated: When Z=1 some individuals reject treatment, when Z=0 some still accept',
        'hte_results': 'Heterogeneous Treatment Effects Results Comparison',
        'scenario_label': 'Experiment Scenario',
    }
}

# 侧边栏语言选择
language = st.sidebar.selectbox('Language / 语言', ['中文', 'English'], key='language_select')
lang = 'zh' if language == '中文' else 'en'
text = lang_dict[lang]

# 设置页面标题
st.title(text['title'])

# 在侧边栏添加滑块控制参数
st.sidebar.header(text['param_control'])
gamma = st.sidebar.slider(text['gamma_label'], min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                          help=text['gamma_help'])
delta = st.sidebar.slider(text['delta_label'], min_value=0.0, max_value=2.0, value=0.5, step=0.1,
                         help=text['delta_help'])

phi = st.sidebar.slider(
    text['phi_label'],
    min_value=0.0,
    max_value=2.0,
    value=0.0,
    step=0.1,
    help=text['phi_help']
)

# Add option to choose experiment scenario in sidebar
st.sidebar.markdown("---")
st.sidebar.header(text['hte_section'])
scenario_options = [text['scenario_basic'], text['scenario_one_option'], text['scenario_two_option']]
scenario_choice = st.sidebar.radio(text['scenario_choice'], scenario_options)

# 根据选择的场景显示不同的参数
# Show different parameters based on selected scenario
if '基础模型' in scenario_choice or 'Basic Model' in scenario_choice:
    use_hte = False
else:
    use_hte = True
    
    # 异质性处理效应参数设置
    # HTEs parameter settings  
    st.sidebar.markdown("**四类个体比例设置 (Individual Type Proportions)**")
    st.sidebar.markdown("*注：比例总和将自动调整为100%*")
    
    # 使用数值输入的方式，确保总和为100%
    # Use number input to ensure proportions sum to 100%
    col_prop = st.sidebar.columns([1, 1])
    
    with col_prop[0]:
        prop_compliers_temp = st.number_input(
            '依从者 (Compliers) %', 
            min_value=0.0, max_value=100.0, value=40.0, step=1.0, 
            key='prop_compliers'
        )
        prop_always_temp = st.number_input(
            '始终接受者 (Always-takers) %', 
            min_value=0.0, max_value=100.0, value=30.0, step=1.0,
            key='prop_always'
        )
    
    with col_prop[1]:
        prop_never_temp = st.number_input(
            '从不接受者 (Never-takers) %', 
            min_value=0.0, max_value=100.0, value=30.0, step=1.0,
            key='prop_never'
        )
        prop_defiers_temp = st.number_input(
            '违抗者 (Defiers) %', 
            min_value=0.0, max_value=100.0, value=0.0, step=1.0,
            key='prop_defiers'
        )
    
    # 计算总和并自动调整
    # Calculate sum and auto-adjust
    total = prop_compliers_temp + prop_always_temp + prop_never_temp + prop_defiers_temp
    
    if total > 0:
        # 按比例缩放所有值，使总和为100
        # Scale all values proportionally to sum to 100
        prop_compliers = prop_compliers_temp / total
        prop_always = prop_always_temp / total
        prop_never = prop_never_temp / total
        prop_defiers = prop_defiers_temp / total
    else:
        # 如果全是0，使用默认值
        prop_compliers = 0.4
        prop_always = 0.3
        prop_never = 0.3
        prop_defiers = 0.0
    
    # 显示调整后的比例
    # Display adjusted proportions
    st.sidebar.info(f"""
**调整后的比例 (Adjusted Proportions)**:
- 依从者 (Compliers): {prop_compliers:.1%}
- 始终接受者 (Always-takers): {prop_always:.1%}
- 从不接受者 (Never-takers): {prop_never:.1%}
- 违抗者 (Defiers): {prop_defiers:.1%}
- **总计**: {prop_compliers + prop_always + prop_never + prop_defiers:.1%}
    """)
    
    # 如果不是场景一（无Defiers验证），则锁定Defiers为0或显示警告
    # If not Scenario I, lock Defiers or show warning
    if '无Defiers' in scenario_choice and prop_defiers > 0.01:
        st.sidebar.warning("⚠️ 场景一应使用 0% Defiers 来验证 LATE 定理")
    elif '含Defiers' in scenario_choice and prop_defiers < 0.01:
        st.sidebar.info("ℹ️ 场景二建议设置 Defiers > 0 来观察其影响")
    
    # 异质性处理效应大小设置
    # HTE magnitude settings - 固定预设值，用户无需修改
    st.sidebar.markdown("**处理效应预设值 (Treatment Effect Preset Values)**")
    st.sidebar.info("""
根据潜在结果框架 (Potential Outcomes Framework):
- **Compliers (β_comp) = 5.0** 
- **Always-takers (β_always) = 2.0**
- **Never-takers (β_never) = 2.0**
- **Defiers (β_defiers) = 2.0**
    """)
    
    # 固定处理效应值
    # Fixed treatment effect values
    beta_compliers = 5.0
    beta_always = 2.0
    beta_never = 2.0
    beta_defiers = 2.0

# 模型预览区
st.markdown(f"### {text['model_preview']}")
st.markdown("---")

# 根据场景显示不同的模型
# Show different models based on scenario
if use_hte:
    st.markdown("#### 潜在结果框架中的 LATE 模型与异质性处理效应")
    st.markdown("(LATE Model with Heterogeneous Treatment Effects in Potential Outcomes Framework)")
    
    st.markdown("""
**二元工具变量模型 (Binary Instrumental Variable Model)**:

**结构式 (Structural Form):**
$$Y_i = \\beta_0 + \\beta_1 X_{1i} + \\boldsymbol{\\beta} \\mathbf{X} + \\epsilon_i$$

其中：
- $Y_i$ 是被解释变量 (dependent variable)
- $X_{1i}$ 是内生处理变量 (endogenous treatment variable)
- $\\mathbf{X}$ 是其他外生变量向量 (other exogenous variables)
- $\\beta_1$ 是处理效应 $X_{1i}$ 的系数
- $\\boldsymbol{\\beta}$ 是其他变量系数的**向量** (parameter vector)

**第一阶段 (First Stage):**
$$X_{1i} = \\gamma_0 + \\gamma_1 Z + \\boldsymbol{\\gamma} \\mathbf{X} + v_i$$

其中：
- $Z$ 是二元工具变量 (binary instrument)
- $\\gamma_1$ 是工具变量 $Z$ 对 $X_{1i}$ 的影响（**需检验其显著性**)
- $\\boldsymbol{\\gamma}$ 是其他外生变量系数的**向量** (parameter vector)

**第二阶段 (Second Stage) / 2SLS估计:**
$$Y_i = \\mu_0 + \\mu_1 \\hat{X}_{1i} + \\boldsymbol{\\mu} \\mathbf{X} + e_i$$

其中：
- $\\hat{X}_{1i}$ 是第一阶段对 $X_{1i}$ 的**预测值** (fitted value from first stage)
- $\\mu_1$ 是处理效应的**无偏估计** (unbiased estimate of treatment effect)
- $\\boldsymbol{\\mu}$ 是其他变量系数的**向量** (parameter vector)
    """)
    
    st.markdown("---")
    st.markdown("**潜在结果框架中的四类个体与异质性处理效应** (Four Types with Heterogeneous Effects in Potential Outcomes Framework):")
    
    # 创建表格
    table_md = """
| 个体类型 | Z→D 关系 | 数学表达 | 真实处理效应 | 说明 |
|---------|---------|--------|-----------|------|
| **Compliers** (依从者) | 完全遵照 | $D_i = Z$ | $\\beta_{1,comp} = 5.0$ | 受工具变量影响，Z=1时接受处理 |
| **Always-takers** | 始终接受 | $D_i = 1$ | $\\beta_{1,always} = 2.0$ | 无论Z如何都接受处理 |
| **Never-takers** | 始终不接受 | $D_i = 0$ | $\\beta_{1,never} = 2.0$ | 无论Z如何都不接受处理 |
| **Defiers** (违抗者) | 违抗指导 | $D_i = 1 - Z$ | $\\beta_{1,defiers} = 2.0$ | 违背工具变量指导的个体 |
    """
    st.markdown(table_md)
    
    st.markdown("---")
    st.markdown("""
**LATE 定理在2SLS框架中的应用**:

当满足 LATE 假设时，2SLS第二阶段估计量收敛到 Compliers 的平均处理效应：

$$\\hat{\\mu}_1^{2SLS} \\xrightarrow{p} E[\\beta_{1,i} \\mid \\text{Complier}] = \\beta_{1,comp} = 5.0$$

**关键假设**：
1. **排他性 (Exclusion Restriction)**: $Z$ 只通过 $X_{1i}$ 影响 $Y_i$
2. **相关性 (Relevance)**: $\\gamma_1 \\neq 0$，即 $Z$ 与 $X_{1i}$ 相关
3. **单调性 (Monotonicity)**: 不存在 Defiers，即 $P(\\text{Defier}) = 0$

当违反单调性假设时（存在 Defiers），第二阶段的 $\\hat{\\mu}_1$ 不再等于任何单一群体的处理效应。
    """)
    
    # 显示参数设置
    st.markdown("---")
    st.markdown(f"#### 模型参数 (Model Parameters)")
    
    # 创建更清晰的展示
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**四类个体比例**")
        rows_data = []
        types = ['Compliers (依从者)', 'Always-takers (始终接受)', 'Never-takers (从不接受)', 'Defiers (违抗者)']
        proportions = [prop_compliers, prop_always, prop_never, prop_defiers]
        for t, p in zip(types, proportions):
            rows_data.append({'个体类型': t, '比例': f'{p:.1%}'})
        
        df_types = pd.DataFrame(rows_data)
        st.dataframe(df_types, use_container_width=True)
        
        if prop_defiers > 0.01 and '无Defiers' in scenario_choice:
            st.warning("⚠️ 检测到 Defiers。这会违反单调性假设！")
    
    with col2:
        st.markdown("**异质性处理效应**")
        effect_data = []
        effects = [
            ('Compliers', beta_compliers),
            ('Always-takers', beta_always),
            ('Never-takers', beta_never),
            ('Defiers', beta_defiers)
        ]
        for t, e in effects:
            effect_data.append({'个体类型': t, '$\\\\beta_i$': f'{e:.1f}'})
        
        df_effects = pd.DataFrame(effect_data)
        st.dataframe(df_effects, use_container_width=True)

else:
    # 原始模型显示
    # Original model display
    st.markdown(f"**{text.get('original_model','原始模型 / Original model')}:**")
    st.latex(r"Y_i = \beta_0 + \beta_1 X_{1i} + \mathbf{\beta} \mathbf{X} + \varepsilon_i")
    st.markdown(f"**{text.get('first_stage','第一阶段 / First stage')}:**")
    st.latex(r"X_{1i} = \pi_1 Z_i + \mathbf{\pi} \mathbf{X} + v_i")
    st.markdown(f"**{text.get('second_stage','第二阶段 / Second stage')}:**")
    st.latex(r"Y_i = \mu_0 + \mu_1 \widehat{X_{1i}} + \mathbf{\mu} \mathbf{X} + e_i")
    st.markdown(text.get('mu1_unbiased', r"$\\mu_1$ 是无偏估计 / $\\mu_1$ is unbiased estimator"))
    st.markdown("---")
    st.markdown(f"### {text['param_detail']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {text['variable_def']}")
        st.markdown(f"""
    - **U**: {text['error_term']}，$U \\sim N(0, 1)$
    - **Z**: {text['instrument']}，$Z \\sim N(0, 1)$
    - **X**: {text['endogenous']}
    - **Y**: {text['explained']}
        """)
    
    with col2:
        st.markdown(f"#### {text['param_meaning']}")
        st.markdown(f"""
    - **γ (gamma)** = {gamma:.2f}: {text['iv_strength']}
    - **δ (delta)** = {delta:.2f}: {text['error_transmission']}
    - **β (beta)** = 1.0: {text['true_effect']}
    
    {text['exclusion_condition']}
        """)

# ======================== 数据生成与回归分析部分 (Data Generation & Regression Analysis) ========================
# 设置随机种子以确保结果可重复
# Set random seed for reproducibility
np.random.seed(42)

# 模拟数据
# Simulate data
n = 1000

if use_hte:
    # ======================== 异质性处理效应数据生成 (HTE Data Generation) ========================
    # HTE数据生成过程说明 (HTE Data Generation Process)
    # 1. 生成工具变量 Z (binary)
    # 2. 根据预设比例随机分配个体类型
    # 3. 根据个体类型和 Z 值确定处理变量 D
    # 4. 根据个体类型对应的 beta 生成 Y
    
    # 1. 生成工具变量 Z (0 或 1)
    # 1. Generate instrument Z (binary: 0 or 1)
    Z = np.random.binomial(1, 0.5, n)
    
    # 2. 随机分配个体类型
    # 2. Randomly assign individual types
    type_probs = [prop_compliers, prop_always, prop_never, prop_defiers]
    individual_types = np.random.choice(['Compliers', 'Always-takers', 'Never-takers', 'Defiers'], 
                                        size=n, p=type_probs)
    
    # 3. 根据类型和 Z 确定处理变量 D
    # 3. Determine treatment variable D based on type and Z
    D = np.zeros(n)
    for i in range(n):
        if individual_types[i] == 'Compliers':
            D[i] = Z[i]  # Compliers: D = Z
        elif individual_types[i] == 'Always-takers':
            D[i] = 1  # Always-takers: D = 1
        elif individual_types[i] == 'Never-takers':
            D[i] = 0  # Never-takers: D = 0
        elif individual_types[i] == 'Defiers':
            D[i] = 1 - Z[i]  # Defiers: D = 1 - Z
    
    # 4. 生成误差项和结果变量 Y
    # 4. Generate error term and outcome variable Y
    U = np.random.normal(0, 1, n)
    
    # 根据个体类型获取对应的处理效应
    # Get treatment effect corresponding to individual type
    betas = np.zeros(n)
    for i in range(n):
        if individual_types[i] == 'Compliers':
            betas[i] = beta_compliers
        elif individual_types[i] == 'Always-takers':
            betas[i] = beta_always
        elif individual_types[i] == 'Never-takers':
            betas[i] = beta_never
        elif individual_types[i] == 'Defiers':
            betas[i] = beta_defiers
    
    # 生成结果变量：Y = beta_i * D + U
    # Generate outcome variable: Y = beta_i * D + U
    Y = betas * D + U
    
    # 创建数据框
    # Create dataframe
    data = pd.DataFrame({
        'Y': Y,
        'D': D,
        'Z': Z,
        'U': U,
        'type': individual_types,
        'beta': betas
    })
    
    # 用于回归的 X 在 HTE 模型中实际上就是 D
    # For regression in HTE model, X is actually D
    X = D
else:
    # ======================== 原始 IV 模型数据生成 (Original IV Model Data Generation) ========================
    # 1. 生成 U (误差项) 和 Z (工具变量)
    # 1. Generate U (error term) and Z (instrument)
    U = np.random.normal(0, 1, n)
    Z = np.random.normal(0, 1, n)
    
    # 2. X = γ·Z + δ·U + e₁
    e1 = np.random.normal(0, 1, n)
    X = gamma * Z + delta * U + e1
    
    # 3. Y = β₀ + β₁X₁ᵢ + β·X + ε (β = 1.0 为真实值)
    # Y = β₀ + β₁X₁ᵢ + β·X + ε (β = 1.0 is true value)
    # 注：排他性条件假设Z不直接影响Y，仅通过X影响Y
    # Note: Exclusion restriction assumes Z affects Y only through X
    alpha = 1.0  # α 系数 / coefficient
    beta_true = 1.0  # β 真实值 / true value = 1.0
    e2 = np.random.normal(0, 1, n)
    Y = beta_true * X + alpha * U + e2
    
    # 创建数据框
    # Create dataframe
    data = pd.DataFrame({
        'Y': Y,
        'X': X,
        'Z': Z,
        'U': U
    })

# ======================== 回归分析部分 (Regression Analysis) ========================

if use_hte:
    # ======================== HTE 模型回归分析 (HTE Model Regression Analysis) ========================
    # 在 HTE 模型中：
    # - X 实际上就是 D (处理变量)
    # - Y 是根据亚群特定的处理效应生成的
    # In HTE model:
    # - X is actually D (treatment variable)
    # - Y is generated with subgroup-specific treatment effects
    
    # OLS 回归: Y = b0 + b1*D
    # OLS Regression: Y = b0 + b1*D
    D_ols = np.column_stack([np.ones(n), D])
    beta_ols = np.linalg.lstsq(D_ols, Y, rcond=None)[0]
    beta_ols_coef = beta_ols[1]
    Y_pred_ols = D_ols @ beta_ols
    
    # 2SLS 回归 (使用 Z 作为工具变量)
    # 2SLS Regression (using Z as instrument)
    # 第一阶段: D = f(Z)
    # First stage: D = f(Z)
    Z_first = np.column_stack([np.ones(n), Z])
    pi_hat = np.linalg.lstsq(Z_first, D, rcond=None)[0]
    D_pred = Z_first @ pi_hat
    
    # 第二阶段: Y = b0 + b1*D_pred
    # Second stage: Y = b0 + b1*D_pred
    D_second = np.column_stack([np.ones(n), D_pred])
    beta_2sls = np.linalg.lstsq(D_second, Y, rcond=None)[0]
    beta_2sls_coef = beta_2sls[1]
    Y_pred_2sls = D_second @ beta_2sls
    
    # 计算 R² 
    ssr_ols = np.sum((Y - Y_pred_ols)**2)
    tss = np.sum((Y - np.mean(Y))**2)
    r2_ols = 1 - (ssr_ols / tss) if tss > 0 else 0
    
    ssr_2sls = np.sum((Y - Y_pred_2sls)**2)
    r2_2sls = 1 - (ssr_2sls / tss) if tss > 0 else 0
    
    # 计算第一阶段 F 统计量
    try:
        u_first = np.linalg.lstsq(Z_first, D, rcond=None)[0]
        D_pred_first = Z_first @ u_first
        ssr_first = np.sum((D - D_pred_first)**2)
        msr_z = np.sum((D_pred_first - np.mean(D))**2)
        if ssr_first / (n - 2) > 1e-10:
            f_stat = (msr_z / 1) / (ssr_first / (n - 2))
        else:
            f_stat = np.inf
    except:
        f_stat = np.nan
    
    # 计算个体类型的加权平均处理效应
    # Calculate weighted average treatment effects for each type
    compliers_ate = beta_compliers if prop_compliers > 0 else 0
    always_ate = beta_always if prop_always > 0 else 0
    never_ate = beta_never if prop_never > 0 else 0
    defiers_ate = beta_defiers if prop_defiers > 0 else 0
    
    # LATE (Local Average Treatment Effect) 理论值
    # LATE Theoretical value
    # 根据 LATE 定理，2SLS 应该估计的是 Compliers 的平均处理效应
    # 如果有 Defiers，LATE 的解释会改变
    # According to LATE theorem, 2SLS should estimate the average treatment effect for Compliers
    # If Defiers exist, the interpretation of LATE changes
    
    if prop_defiers == 0:
        # 无 Defiers：LATE 就是 Compliers 的 ATE
        # No Defiers: LATE is exactly Compliers' ATE
        late_theoretical = compliers_ate
    else:
        # 有 Defiers：LATE 的定义变得复杂
        # With Defiers: LATE definition becomes complex
        # LATE = [E[Y|Z=1] - E[Y|Z=0]] / [E[D|Z=1] - E[D|Z=0]]
        # 计算工具变量的效应
        # Calculate the effect of instrument
        y_given_z1 = np.mean(Y[Z == 1])
        y_given_z0 = np.mean(Y[Z == 0])
        d_given_z1 = np.mean(D[Z == 1])
        d_given_z0 = np.mean(D[Z == 0])
        
        if abs(d_given_z1 - d_given_z0) > 1e-6:
            late_theoretical = (y_given_z1 - y_given_z0) / (d_given_z1 - d_given_z0)
        else:
            late_theoretical = 0
    
    # 显示结果对比
    st.markdown("---")
    st.subheader(text['hte_results'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {text['ols_regression']}")
        st.metric("β̂_OLS", f"{beta_ols_coef:.4f}", delta=f"{beta_ols_coef - compliers_ate:.4f}")
        st.metric("R²", f"{r2_ols:.4f}")
        st.markdown(f"**{text['model']}**: Y = {beta_ols[0]:.4f} + {beta_ols_coef:.4f}·D")
    
    with col2:
        st.markdown(f"### {text['tsls_regression']}")
        st.metric("β̂_2SLS (LATE)", f"{beta_2sls_coef:.4f}", delta=f"{beta_2sls_coef - late_theoretical:.4f}")
        st.metric("R²", f"{r2_2sls:.4f}")
        st.markdown(f"**{text['model']}**: Y = {beta_2sls[0]:.4f} + {beta_2sls_coef:.4f}·D_pred")
    
    # 显示工具变量强度
    st.markdown("---")
    st.subheader(text['iv_diagnosis'])
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(text['first_stage_f'], f"{f_stat:.2f}")
        if f_stat < 10:
            st.warning(f"⚠️ {text['iv_weak']}")
        else:
            st.success(f"✓ {text['iv_strong']}")
    
    with col2:
        correlation_dz = np.corrcoef(D, Z)[0, 1]
        st.metric(text['correlation'], f"{correlation_dz:.4f}")
    
    with col3:
        covariance_dz = np.cov(D, Z)[0, 1]
        st.metric(text['covariance'], f"{covariance_dz:.4f}")
    
    # ======================== LATE 定理验证部分 (LATE Theorem Verification) ========================
    st.markdown("---")
    st.subheader(text['late_theorem'])
    
    if lang == 'zh':
        st.markdown(f"""
{text['late_explanation']}

{text['late_assumption_1']}
{text['late_assumption_2']}
{text['late_assumption_3']}

**实验场景分析**：

{text['late_result_scenario1'] if prop_defiers == 0 else text['late_result_scenario2']}
        """)
    else:
        st.markdown(f"""
{text['late_explanation']}

{text['late_assumption_1']}
{text['late_assumption_2']}
{text['late_assumption_3']}

**Scenario Analysis**:

{text['late_result_scenario1'] if prop_defiers == 0 else text['late_result_scenario2']}
        """)
    
    if prop_defiers > 0:
        st.warning(f"{text['monotonicity_violation']}")
    
    # 显示各类型的ATE和加权效应
    st.markdown("---")
    st.subheader("异质性处理效应对比" if lang == 'zh' else "Heterogeneous Treatment Effects Comparison")
    
    hte_comparison_data = []
    hte_comparison_data.append({
        '个体类型' if lang == 'zh' else 'Individual Type': text['compliers'],
        '比例' if lang == 'zh' else 'Proportion': f'{prop_compliers:.0%}',
        '真实处理效应' if lang == 'zh' else 'True Effect': f'{beta_compliers:.4f}',
        '加权贡献' if lang == 'zh' else 'Weighted Contribution': f'{beta_compliers * prop_compliers:.4f}'
    })
    hte_comparison_data.append({
        '个体类型' if lang == 'zh' else 'Individual Type': text['always_takers'],
        '比例' if lang == 'zh' else 'Proportion': f'{prop_always:.0%}',
        '真实处理效应' if lang == 'zh' else 'True Effect': f'{beta_always:.4f}',
        '加权贡献' if lang == 'zh' else 'Weighted Contribution': f'{beta_always * prop_always:.4f}'
    })
    hte_comparison_data.append({
        '个体类型' if lang == 'zh' else 'Individual Type': text['never_takers'],
        '比例' if lang == 'zh' else 'Proportion': f'{prop_never:.0%}',
        '真实处理效应' if lang == 'zh' else 'True Effect': f'{beta_never:.4f}',
        '加权贡献' if lang == 'zh' else 'Weighted Contribution': f'{beta_never * prop_never:.4f}'
    })
    if prop_defiers > 0:
        hte_comparison_data.append({
            '个体类型' if lang == 'zh' else 'Individual Type': text['defiers'],
            '比例' if lang == 'zh' else 'Proportion': f'{prop_defiers:.0%}',
            '真实处理效应' if lang == 'zh' else 'True Effect': f'{beta_defiers:.4f}',
            '加权贡献' if lang == 'zh' else 'Weighted Contribution': f'{beta_defiers * prop_defiers:.4f}'
        })
    
    df_hte = pd.DataFrame(hte_comparison_data)
    st.dataframe(df_hte, use_container_width=True)
    
    # 计算理论的人口平均处理效应 (Population ATE)
    pop_ate = (beta_compliers * prop_compliers + beta_always * prop_always + 
               beta_never * prop_never + beta_defiers * prop_defiers)
    
    if lang == 'zh':
        st.markdown(f"""
**关键结果**:
- **人口平均处理效应 (Population ATE)**: {pop_ate:.4f}
- **OLS 估计**: {beta_ols_coef:.4f}
- **2SLS 估计 (LATE)**: {beta_2sls_coef:.4f}
- **理论 LATE 值**: {late_theoretical:.4f}
- **2SLS 偏差**: {abs(beta_2sls_coef - late_theoretical):.4f}

**解释**:
        """)
        
        if prop_defiers == 0:
            st.success(f"""
✓ **场景一验证成功**：无违抗者存在
- 2SLS 完美恢复了依从者的真实处理效应 ({beta_compliers:.4f})
- IV 估计值 ({beta_2sls_coef:.4f}) ≈ 理论 LATE 值 ({late_theoretical:.4f})
- 所有 LATE 假设得到满足，LATE 定理完全适用
            """)
        else:
            st.error(f"""
⚠️ **场景二结果展示**：违抗者的破坏性影响
- Defiers (占 {prop_defiers:.0%}) 的存在违反了单调性假设
- 2SLS 估计值 ({beta_2sls_coef:.4f}) 不再对应任何单一群体的处理效应
- 违抗者对工具变量效应的中断导致 IV 估计被扭曲
- 这证明单调性假设对于 LATE 定理的有效性是必要的
            """)
    else:
        st.markdown(f"""
**Key Results**:
- **Population Average Treatment Effect (ATE)**: {pop_ate:.4f}
- **OLS Estimate**: {beta_ols_coef:.4f}
- **2SLS Estimate (LATE)**: {beta_2sls_coef:.4f}
- **Theoretical LATE Value**: {late_theoretical:.4f}
- **2SLS Deviation**: {abs(beta_2sls_coef - late_theoretical):.4f}

**Explanation**:
        """)
        
        if prop_defiers == 0:
            st.success(f"""
✓ **Scenario I Verification Success**: No Defiers present
- 2SLS perfectly recovers Compliers' true treatment effect ({beta_compliers:.4f})
- IV estimate ({beta_2sls_coef:.4f}) ≈ Theoretical LATE value ({late_theoretical:.4f})
- All LATE assumptions are satisfied, LATE theorem fully applicable
            """)
        else:
            st.error(f"""
⚠️ **Scenario II Results**: Destructive Impact of Defiers
- Defiers (comprising {prop_defiers:.0%}) violate the monotonicity assumption
- 2SLS estimate ({beta_2sls_coef:.4f}) no longer corresponds to any single group's effect
- Defiers' disruption of the instrument effect causes IV estimates to be distorted
- This proves monotonicity assumption is necessary for LATE theorem validity
            """)
    
    # 可视化：不同处理值下的Y分布
    st.markdown("---")
    st.subheader("数据可视化" if lang == 'zh' else "Data Visualization")
    
    # 按处理状态和类型分组的Y值分布
    fig_box = go.Figure()
    
    colors = {'Compliers': 'blue', 'Always-takers': 'green', 'Never-takers': 'orange', 'Defiers': 'red'}
    
    for dtype in ['Compliers', 'Always-takers', 'Never-takers', 'Defiers']:
        if dtype == 'Defiers' and prop_defiers == 0:
            continue
        mask = individual_types == dtype
        if np.any(mask):
            # D = 0 的情况
            mask_d0 = mask & (D == 0)
            if np.any(mask_d0):
                fig_box.add_trace(go.Box(
                    y=Y[mask_d0],
                    name=f'{dtype} (D=0)',
                    marker_color=colors[dtype],
                    opacity=0.7
                ))
            
            # D = 1 的情况
            mask_d1 = mask & (D == 1)
            if np.any(mask_d1):
                fig_box.add_trace(go.Box(
                    y=Y[mask_d1],
                    name=f'{dtype} (D=1)',
                    marker_color=colors[dtype],
                    opacity=1.0
                ))
    
    fig_box.update_layout(
        title="按个体类型和处理状态分组的结果变量(Y)分布" if lang == 'zh' else "Outcome Distribution by Type and Treatment Status",
        yaxis_title="Y",
        xaxis_title="类型和处理状态" if lang == 'zh' else "Type and Treatment Status",
        boxmode='group',
        height=500
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    
else:
    # ======================== 原始 IV 模型回归分析 (Original IV Model Regression) ========================
    # OLS 回归: Y = b0 + b1*X
    X_ols = np.column_stack([np.ones(n), X])
    beta_ols = np.linalg.lstsq(X_ols, Y, rcond=None)[0]
    beta_ols_coef = beta_ols[1]
    Y_pred_ols = X_ols @ beta_ols

    # 2SLS 回归
    # 第一阶段: X = f(Z)
    X_first = np.column_stack([np.ones(n), Z])
    gamma_hat = np.linalg.lstsq(X_first, X, rcond=None)[0]
    X_pred = X_first @ gamma_hat

    # 第二阶段: Y = b0 + b1*X_pred
    X_second = np.column_stack([np.ones(n), X_pred])
    beta_2sls = np.linalg.lstsq(X_second, Y, rcond=None)[0]
    beta_2sls_coef = beta_2sls[1]
    Y_pred_2sls = X_second @ beta_2sls

    # 计算 R² 和其他统计量
    ssr_ols = np.sum((Y - Y_pred_ols)**2)
    tss = np.sum((Y - np.mean(Y))**2)
    r2_ols = 1 - (ssr_ols / tss)

    ssr_2sls = np.sum((Y - Y_pred_2sls)**2)
    r2_2sls = 1 - (ssr_2sls / tss)

    # 计算第一阶段 F 统计量
    try:
        Z_with_const = np.column_stack([np.ones(n), Z])
        u_first = np.linalg.lstsq(Z_with_const, X, rcond=None)[0]
        X_pred_first = Z_with_const @ u_first
        ssr_first = np.sum((X - X_pred_first)**2)
        msr_z = np.sum((X_pred_first - np.mean(X))**2)
        # 防止分母为 0
        if ssr_first / (n - 2) > 1e-10:
            f_stat = (msr_z / 1) / (ssr_first / (n - 2))
        else:
            f_stat = np.inf
    except:
        f_stat = np.nan

    # 设定原始模型的真实值
    beta_true = 1.0

    # 显示结果对比
    st.markdown("---")
    st.subheader(text['regression_comparison'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {text['ols_regression']}")
        st.metric("β̂_OLS", f"{beta_ols_coef:.4f}", delta=f"{beta_ols_coef - beta_true:.4f} ({text['true_value']}: 1.0)")
        st.metric("R²", f"{r2_ols:.4f}")
        st.markdown(f"**{text['model']}**: Y = {beta_ols[0]:.4f} + {beta_ols_coef:.4f}·X")

    with col2:
        st.markdown(f"### {text['tsls_regression']}")
        st.metric("β̂_2SLS", f"{beta_2sls_coef:.4f}", delta=f"{beta_2sls_coef - beta_true:.4f} ({text['true_value']}: 1.0)")
        st.metric("R²", f"{r2_2sls:.4f}")
        st.markdown(f"**{text['model']}**: Y = {beta_2sls[0]:.4f} + {beta_2sls_coef:.4f}·X_pred")

    # 显示工具变量强度
    st.markdown("---")
    st.subheader(text['iv_diagnosis'])
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(text['first_stage_f'], f"{f_stat:.2f}")
        if f_stat < 10:
            st.warning(f"⚠️ {text['iv_weak']}")
        else:
            st.success(f"✓ {text['iv_strong']}")

    with col2:
        correlation_xz = np.corrcoef(X, Z)[0, 1]
        st.metric(text['correlation'], f"{correlation_xz:.4f}")

    with col3:
        covariance_xz = np.cov(X, Z)[0, 1]
        st.metric(text['covariance'], f"{covariance_xz:.4f}")

    # 可视化
    st.markdown("---")
    st.subheader(text['visualization'])

    # 创建 X vs Y 散点图，附带拟合线
    fig = go.Figure()

    # 散点
    fig.add_trace(go.Scatter(
        x=X, y=Y,
        mode='markers',
        name=text['data_point'],
        marker=dict(color='rgba(0, 100, 200, 0.5)', size=4)
    ))

    # OLS 拟合线
    X_sort_idx = np.argsort(X)
    X_sort = X[X_sort_idx]
    Y_pred_ols_sort = Y_pred_ols[X_sort_idx]
    fig.add_trace(go.Scatter(
        x=X_sort, y=Y_pred_ols_sort,
        mode='lines',
        name=f'OLS (β̂={beta_ols_coef:.4f})',
        line=dict(color='red', width=2)
    ))

    # 2SLS 拟合线
    Y_pred_2sls_sort = Y_pred_2sls[X_sort_idx]
    fig.add_trace(go.Scatter(
        x=X_sort, y=Y_pred_2sls_sort,
        mode='lines',
        name=f'2SLS (β̂={beta_2sls_coef:.4f})',
        line=dict(color='green', width=2)
    ))

    fig.update_layout(
        title=text['scatter_plot'],
        xaxis_title='X',
        yaxis_title='Y',
        hovermode='closest',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # 显示关键洞察
    st.markdown("---")
    st.subheader(text['insight'])

    bias_ols = beta_ols_coef - beta_true
    bias_2sls = beta_2sls_coef - beta_true

    if lang == 'zh':
        st.markdown(f"""
- **{text['ols_bias']}**: {bias_ols:.4f} ({(bias_ols/beta_true)*100:.2f}%)
- **{text['tsls_bias']}**: {bias_2sls:.4f} ({(bias_2sls/beta_true)*100:.2f}%)
- **{text['improvement']}**: {abs(bias_ols - bias_2sls):.4f}

**{text['explanation']}**:
- 当 φ > 0 时，Z 直接影响 Y，违反排他性假设，导致 OLS 有偏差
- 2SLS 通过工具变量法消除这种偏差
- IV 强度 (γ) 越大，2SLS 估计越精确
- 误差传导 (δ) 影响 X 和 U 的相关性，影响 OLS 的有偏程度
        """)
    else:
        st.markdown(f"""
- **{text['ols_bias']}**: {bias_ols:.4f} ({(bias_ols/beta_true)*100:.2f}%)
- **{text['tsls_bias']}**: {bias_2sls:.4f} ({(bias_2sls/beta_true)*100:.2f}%)
- **{text['improvement']}**: {abs(bias_ols - bias_2sls):.4f}

**{text['explanation']}**:
- When φ > 0, Z directly affects Y, violating exclusion restriction, causing OLS bias
- 2SLS eliminates this bias through instrumental variable method
- Larger IV strength (γ) leads to more precise 2SLS estimates
- Error transmission (δ) affects correlation between X and U, influencing OLS bias magnitude
        """)
