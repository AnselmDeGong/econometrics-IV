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
        'param_meaning': '参数含义',
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
        'hte_section': '🎯 异质性处理效应与四类个体',
        'scenario_choice': '选择实验场景',
        'scenario_basic': '基础模型',
        'scenario_one_option': '场景一：无违抗者 (Defiers = 0%)',
        'scenario_two_option': '场景二：引入违抗者 (Defiers > 0%)',
        'scenario_hte': '异质性处理效应模型',
        'compliers_label': '依从者 (Compliers) %',
        'always_takers_label': '始终接受者 (Always-takers) %',
        'never_takers_label': '从不接受者 (Never-takers) %',
        'defiers_label': '违抗者 (Defiers) %',
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
        'scenario_two': '场景二：引入违抗者 (Defiers > 0%)',
        'scenario_two_desc': '展示单调性假设违反的后果 - 违抗者的存在如何扭曲 IV 估计量',
        'individual_type': '个体类型',
        'proportion': '比例',
        'true_effect_col': '真实处理效应',
        'late_theorem': '🔬 LATE 定理验证',
        'late_explanation': 'LATE (Local Average Treatment Effect) 承诺在以下假设下，2SLS 估计的是 Compliers 的平均处理效应：',
        'late_assumption_1': '1. 排他性：Z 只通过 D 影响 Y',
        'late_assumption_2': '2. 相关性：Z 与 D 相关',
        'late_assumption_3': '3. 单调性：不存在违抗者 (Defiers)',
        'late_result_scenario1': '场景一结果：Z→D→Y 的单向因果链，无 Defiers，满足所有 LATE 假设',
        'late_result_scenario2': '场景二结果：Defiers 的存在违反单调性假设，导致 IV 估计不再等于任何组的单一处理效应',
        'monotonicity_violation': '⚠️ 单调性假设违反：当 Z=1 时部分个体不接受处理，当 Z=0 时又接受处理',
        'hte_results': '异质性处理效应结果对比',
        'scenario_label': '实验场景',
        'prop_setting_title': '**四类个体比例设置**',
        'prop_setting_note': '*注：比例总和将自动调整为100%*',
        'adjusted_prop': '**调整后的比例**',
        'total': '**总计**',
        'defier_warn_scen1': '⚠️ 场景一应使用 0% Defiers 来验证 LATE 定理',
        'defier_info_scen2': 'ℹ️ 场景二建议设置 Defiers > 0 来观察其影响',
        'effect_preset_title': '**处理效应预设值**',
        'effect_preset_info': '根据潜在结果框架：\n- 依从者 β_comp = 5.0\n- 始终接受者 β_always = 2.0\n- 从不接受者 β_never = 2.0\n- 违抗者 β_defiers = 2.0',
        'hte_model_title': '#### 潜在结果框架中的 LATE 模型与异质性处理效应',
        'four_types_title': '**潜在结果框架中的四类个体与异质性处理效应**',
        'model_params': '#### 模型参数',
        'defier_detect_warn': '⚠️ 检测到 Defiers。这会违反单调性假设！',
        'weighted_contrib': '加权贡献',
        'pop_ate': '人口平均处理效应 (Population ATE)',
        'ols_est': 'OLS 估计',
        'tsls_est': '2SLS 估计 (LATE)',
        'theoretical_late': '理论 LATE 值',
        'tsls_dev': '2SLS 偏差',
        'key_results': '**关键结果**',
        'explain_title': '**解释**',
        'scen1_success': '✓ **场景一验证成功**：无违抗者存在\n- 2SLS 完美恢复了依从者的真实处理效应 ({:.4f})\n- IV 估计值 ({:.4f}) ≈ 理论 LATE 值 ({:.4f})\n- 所有 LATE 假设得到满足，LATE 定理完全适用',
        'scen2_error': '⚠️ **场景二结果展示**：违抗者的破坏性影响\n- Defiers (占 {:.0%}) 的存在违反了单调性假设\n- 2SLS 估计值 ({:.4f}) 不再对应任何单一群体的处理效应\n- 违抗者对工具变量效应的中断导致 IV 估计被扭曲\n- 这证明单调性假设对于 LATE 定理的有效性是必要的',
        'dist_title': '按个体类型和处理状态分组的结果变量(Y)分布',
        'dist_xaxis': '类型和处理状态'
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
        'hte_section': '🎯 Heterogeneous Treatment Effects (HTE)',
        'scenario_choice': 'Choose Experiment Scenario',
        'scenario_basic': 'Basic Model',
        'scenario_one_option': 'Scenario I: No Defiers (Defiers = 0%)',
        'scenario_two_option': 'Scenario II: With Defiers (Defiers > 0%)',
        'scenario_hte': 'Heterogeneous Treatment Effects Model',
        'compliers_label': 'Compliers %',
        'always_takers_label': 'Always-takers %',
        'never_takers_label': 'Never-takers %',
        'defiers_label': 'Defiers %',
        'compliers': 'Compliers',
        'always_takers': 'Always-takers',
        'never_takers': 'Never-takers',
        'defiers': 'Defiers',
        'treatment_effect_compliers': 'Compliers True Treatment Effect (β_C)',
        'treatment_effect_always': 'Always-takers True Treatment Effect (β_A)',
        'treatment_effect_never': 'Never-takers True Treatment Effect (β_N)',
        'treatment_effect_defiers': 'Defiers True Treatment Effect (β_D)',
        'scenario_one': 'Scenario I: No Defiers (Defiers = 0%)',
        'scenario_one_desc': 'Verify LATE Theorem - IV estimate should perfectly recover Compliers effect',
        'scenario_two': 'Scenario II: With Defiers (Defiers > 0%)',
        'scenario_two_desc': 'Demonstrate consequences of monotonicity violation',
        'individual_type': 'Individual Type',
        'proportion': 'Proportion',
        'true_effect_col': 'True Effect',
        'late_theorem': '🔬 LATE Theorem Verification',
        'late_explanation': 'LATE guarantees that under the following assumptions, 2SLS estimates the average treatment effect for Compliers:',
        'late_assumption_1': '1. Exclusion: Z affects Y only through D',
        'late_assumption_2': '2. Relevance: Z is correlated with D',
        'late_assumption_3': '3. Monotonicity: No Defiers exist',
        'late_result_scenario1': 'Scenario I Result: Unidirectional causal chain Z→D→Y, no Defiers, all LATE assumptions satisfied',
        'late_result_scenario2': "Scenario II Result: Defiers violate monotonicity, IV estimate no longer equals any single group's treatment effect", 
        'monotonicity_violation': '⚠️ Monotonicity Assumption Violated: When Z=1 some individuals reject treatment, when Z=0 some still accept',
        'hte_results': 'Heterogeneous Treatment Effects Results',
        'scenario_label': 'Experiment Scenario',
        'prop_setting_title': '**Individual Type Proportions**',
        'prop_setting_note': '*Note: The sum will be automatically adjusted to 100%*',
        'adjusted_prop': '**Adjusted Proportions**',
        'total': '**Total**',
        'defier_warn_scen1': '⚠️ Scenario I should use 0% Defiers to verify LATE',
        'defier_info_scen2': 'ℹ️ Scenario II recommends Defiers > 0 to observe impact',
        'effect_preset_title': '**Treatment Effect Preset Values**',
        'effect_preset_info': 'Based on Potential Outcomes Framework:\n- Compliers (β_comp) = 5.0\n- Always-takers (β_always) = 2.0\n- Never-takers (β_never) = 2.0\n- Defiers (β_defiers) = 2.0',
        'hte_model_title': '#### LATE Model with Heterogeneous Treatment Effects',
        'four_types_title': '**Four Types with Heterogeneous Effects in Potential Outcomes Framework**',
        'model_params': '#### Model Parameters',
        'defier_detect_warn': '⚠️ Defiers detected. This violates the monotonicity assumption!',
        'weighted_contrib': 'Weighted Contribution',
        'pop_ate': 'Population Average Treatment Effect (ATE)',
        'ols_est': 'OLS Estimate',
        'tsls_est': '2SLS Estimate (LATE)',
        'theoretical_late': 'Theoretical LATE Value',
        'tsls_dev': '2SLS Deviation',
        'key_results': '**Key Results**',
        'explain_title': '**Explanation**',
        'scen1_success': '✓ **Scenario I Verification Success**: No Defiers present\n- 2SLS perfectly recovers Compliers\' true treatment effect ({:.4f})\n- IV estimate ({:.4f}) ≈ Theoretical LATE value ({:.4f})\n- All LATE assumptions are satisfied, LATE theorem fully applicable',
        'scen2_error': '⚠️ **Scenario II Results**: Destructive Impact of Defiers\n- Defiers (comprising {:.0%}) violate the monotonicity assumption\n- 2SLS estimate ({:.4f}) no longer corresponds to any single group\'s effect\n- Defiers\' disruption of the instrument effect causes IV estimates to be distorted\n- This proves monotonicity assumption is necessary for LATE theorem validity',
        'dist_title': 'Outcome Distribution by Type and Treatment Status',
        'dist_xaxis': 'Type and Treatment Status'
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
gamma = st.sidebar.slider(text['gamma_label'], min_value=0.1, max_value=2.0, value=1.0, step=0.1, help=text['gamma_help'])
delta = st.sidebar.slider(text['delta_label'], min_value=0.0, max_value=2.0, value=0.5, step=0.1, help=text['delta_help'])
phi = st.sidebar.slider(text['phi_label'], min_value=0.0, max_value=2.0, value=0.0, step=0.1, help=text['phi_help'])

# ======================== 异质性处理效应部分 (HTE Section) ========================
st.sidebar.markdown("---")
st.sidebar.header(text['hte_section'])

# 动态生成单选框选项，确保纯净的对应语言
scenario_options = [text['scenario_basic'], text['scenario_one_option'], text['scenario_two_option']]
scenario_choice_str = st.sidebar.radio(text['scenario_choice'], scenario_options)

# 判断是否使用基础模型
use_hte = scenario_choice_str != text['scenario_basic']

if use_hte:
    st.sidebar.markdown(text['prop_setting_title'])
    st.sidebar.markdown(text['prop_setting_note'])
    col_prop = st.sidebar.columns([1, 1])
    
    with col_prop[0]:
        prop_compliers_temp = st.number_input(text['compliers_label'], min_value=0.0, max_value=100.0, value=40.0, step=1.0, key='prop_compliers')
        prop_always_temp = st.number_input(text['always_takers_label'], min_value=0.0, max_value=100.0, value=30.0, step=1.0, key='prop_always')
    with col_prop[1]:
        prop_never_temp = st.number_input(text['never_takers_label'], min_value=0.0, max_value=100.0, value=30.0, step=1.0, key='prop_never')
        prop_defiers_temp = st.number_input(text['defiers_label'], min_value=0.0, max_value=100.0, value=0.0, step=1.0, key='prop_defiers')
        
    total = prop_compliers_temp + prop_always_temp + prop_never_temp + prop_defiers_temp
    if total > 0:
        prop_compliers = prop_compliers_temp / total
        prop_always = prop_always_temp / total
        prop_never = prop_never_temp / total
        prop_defiers = prop_defiers_temp / total
    else:
        prop_compliers, prop_always, prop_never, prop_defiers = 0.4, 0.3, 0.3, 0.0
        
    st.sidebar.info(f"""
{text['adjusted_prop']}:
- {text['compliers']}: {prop_compliers:.1%}
- {text['always_takers']}: {prop_always:.1%}
- {text['never_takers']}: {prop_never:.1%}
- {text['defiers']}: {prop_defiers:.1%}
- {text['total']}: {prop_compliers + prop_always + prop_never + prop_defiers:.1%}
    """)
    
    # 警告提示
    if scenario_choice_str == text['scenario_one_option'] and prop_defiers > 0.01:
        st.sidebar.warning(text['defier_warn_scen1'])
    elif scenario_choice_str == text['scenario_two_option'] and prop_defiers < 0.01:
        st.sidebar.info(text['defier_info_scen2'])
    
    # 异质性处理效应大小设置
    st.sidebar.markdown(text['effect_preset_title'])
    st.sidebar.info(text['effect_preset_info'])
    
    # 固定处理效应值
    beta_compliers, beta_always, beta_never, beta_defiers = 5.0, 2.0, 2.0, 2.0

# 模型预览区
st.markdown(f"### {text['model_preview']}")
st.markdown("---")

if use_hte:
    st.markdown(text['hte_model_title'])
    st.markdown("""
**Structural Form:**
$$Y_i = \\beta_0 + \\beta_1 X_{1i} + \\boldsymbol{\\beta} \\mathbf{X} + \\epsilon_i$$

**First Stage:**
$$X_{1i} = \\gamma_0 + \\gamma_1 Z + \\boldsymbol{\\gamma} \\mathbf{X} + v_i$$

**Second Stage (2SLS):**
$$Y_i = \\mu_0 + \\mu_1 \\hat{X}_{1i} + \\boldsymbol{\\mu} \\mathbf{X} + e_i$$
    """)
    
    st.markdown("---")
    st.markdown(text['four_types_title'])
    
    table_md = """
| 个体类型 | Z→D 关系 | 数学表达 | 真实处理效应 | 说明 |
|---------|---------|--------|-----------|------|
| **Compliers** | 完全遵照 | $D_i = Z$ | $\\beta_{1,comp} = 5.0$ | 受工具变量影响，Z=1时接受处理 |
| **Always-takers** | 始终接受 | $D_i = 1$ | $\\beta_{1,always} = 2.0$ | 无论Z如何都接受处理 |
| **Never-takers** | 始终不接受 | $D_i = 0$ | $\\beta_{1,never} = 2.0$ | 无论Z如何都不接受处理 |
| **Defiers** | 违抗指导 | $D_i = 1 - Z$ | $\\beta_{1,defiers} = 2.0$ | 违背工具变量指导的个体 |
    """ if lang == 'zh' else """
| Type | Z→D Relation | Math | True Effect | Description |
|---------|---------|--------|-----------|------|
| **Compliers** | Follows | $D_i = Z$ | $\\beta_{1,comp} = 5.0$ | Affected by IV, accepts when Z=1 |
| **Always-takers** | Always accepts | $D_i = 1$ | $\\beta_{1,always} = 2.0$ | Accepts regardless of Z |
| **Never-takers** | Never accepts | $D_i = 0$ | $\\beta_{1,never} = 2.0$ | Rejects regardless of Z |
| **Defiers** | Defies | $D_i = 1 - Z$ | $\\beta_{1,defiers} = 2.0$ | Does opposite of IV assignment |
    """
    st.markdown(table_md)
    
    st.markdown("---")
    st.markdown(f"**{text['late_theorem']}**:\n\n$$\\hat{{\\mu}}_1^{{2SLS}} \\xrightarrow{{p}} E[\\beta_{{1,i}} \\mid \\text{{Complier}}] = \\beta_{{1,comp}} = 5.0$$")
    
    st.markdown("---")
    st.markdown(text['model_params'])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(text['prop_setting_title'])
        rows_data = [
            {text['individual_type']: 'Compliers', text['proportion']: f'{prop_compliers:.1%}'},
            {text['individual_type']: 'Always-takers', text['proportion']: f'{prop_always:.1%}'},
            {text['individual_type']: 'Never-takers', text['proportion']: f'{prop_never:.1%}'},
            {text['individual_type']: 'Defiers', text['proportion']: f'{prop_defiers:.1%}'}
        ]
        st.dataframe(pd.DataFrame(rows_data), use_container_width=True)
        if prop_defiers > 0.01 and scenario_choice_str == text['scenario_one_option']:
            st.warning(text['defier_detect_warn'])
            
    with col2:
        st.markdown(text['effect_preset_title'])
        effect_data = [
            {text['individual_type']: 'Compliers', '$\beta_i$': f'{beta_compliers:.1f}'},
            {text['individual_type']: 'Always-takers', '$\beta_i$': f'{beta_always:.1f}'},
            {text['individual_type']: 'Never-takers', '$\beta_i$': f'{beta_never:.1f}'},
            {text['individual_type']: 'Defiers', '$\beta_i$': f'{beta_defiers:.1f}'}
        ]
        st.dataframe(pd.DataFrame(effect_data), use_container_width=True)

else:
    st.markdown(f"**{text['original_model']}:**")
    st.latex(r"Y_i = \beta_0 + \beta_1 X_{1i} + \mathbf{\beta} \mathbf{X} + \varepsilon_i")
    st.markdown(f"**{text['first_stage']}:**")
    st.latex(r"X_{1i} = \pi_1 Z_i + \mathbf{\pi} \mathbf{X} + v_i")
    st.markdown(f"**{text['second_stage']}:**")
    st.latex(r"Y_i = \mu_0 + \mu_1 \widehat{X_{1i}} + \mathbf{\mu} \mathbf{X} + e_i")
    st.markdown(text['mu1_unbiased'])
    st.markdown("---")
    st.markdown(f"### {text['param_detail']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {text['variable_def']}")
        st.markdown(f"- **U**: {text['error_term']}，$U \\sim N(0, 1)$\n- **Z**: {text['instrument']}，$Z \\sim N(0, 1)$\n- **X**: {text['endogenous']}\n- **Y**: {text['explained']}")
    with col2:
        st.markdown(f"#### {text['param_meaning']}")
        st.markdown(f"- **γ (gamma)** = {gamma:.2f}: {text['iv_strength']}\n- **δ (delta)** = {delta:.2f}: {text['error_transmission']}\n- **β (beta)** = 1.0: {text['true_effect']}\n\n{text['exclusion_condition']}")

# ======================== 数据生成与回归分析部分 ========================
np.random.seed(42)
n = 1000

if use_hte:
    Z = np.random.binomial(1, 0.5, n)
    type_probs = [prop_compliers, prop_always, prop_never, prop_defiers]
    individual_types = np.random.choice(['Compliers', 'Always-takers', 'Never-takers', 'Defiers'], size=n, p=type_probs)
    
    D = np.zeros(n)
    for i in range(n):
        if individual_types[i] == 'Compliers': D[i] = Z[i]
        elif individual_types[i] == 'Always-takers': D[i] = 1
        elif individual_types[i] == 'Never-takers': D[i] = 0
        elif individual_types[i] == 'Defiers': D[i] = 1 - Z[i]
    
    U = np.random.normal(0, 1, n)
    betas = np.zeros(n)
    for i in range(n):
        if individual_types[i] == 'Compliers': betas[i] = beta_compliers
        elif individual_types[i] == 'Always-takers': betas[i] = beta_always
        elif individual_types[i] == 'Never-takers': betas[i] = beta_never
        elif individual_types[i] == 'Defiers': betas[i] = beta_defiers
    
    Y = betas * D + U
    X = D
else:
    U = np.random.normal(0, 1, n)
    Z = np.random.normal(0, 1, n)
    e1 = np.random.normal(0, 1, n)
    X = gamma * Z + delta * U + e1
    
    alpha = 1.0
    beta_true = 1.0
    e2 = np.random.normal(0, 1, n)
    # 修复了原代码中未将 phi 纳入 Y 生成的逻辑问题
    Y = beta_true * X + alpha * U + phi * Z + e2

# ======================== 回归分析部分 ========================
if use_hte:
    D_ols = np.column_stack([np.ones(n), D])
    beta_ols = np.linalg.lstsq(D_ols, Y, rcond=None)[0]
    beta_ols_coef = beta_ols[1]
    Y_pred_ols = D_ols @ beta_ols
    
    Z_first = np.column_stack([np.ones(n), Z])
    pi_hat = np.linalg.lstsq(Z_first, D, rcond=None)[0]
    D_pred = Z_first @ pi_hat
    
    D_second = np.column_stack([np.ones(n), D_pred])
    beta_2sls = np.linalg.lstsq(D_second, Y, rcond=None)[0]
    beta_2sls_coef = beta_2sls[1]
    Y_pred_2sls = D_second @ beta_2sls
    
    ssr_ols = np.sum((Y - Y_pred_ols)**2)
    tss = np.sum((Y - np.mean(Y))**2)
    r2_ols = 1 - (ssr_ols / tss) if tss > 0 else 0
    
    ssr_2sls = np.sum((Y - Y_pred_2sls)**2)
    r2_2sls = 1 - (ssr_2sls / tss) if tss > 0 else 0
    
    try:
        u_first = np.linalg.lstsq(Z_first, D, rcond=None)[0]
        D_pred_first = Z_first @ u_first
        ssr_first = np.sum((D - D_pred_first)**2)
        msr_z = np.sum((D_pred_first - np.mean(D))**2)
        f_stat = (msr_z / 1) / (ssr_first / (n - 2)) if ssr_first / (n - 2) > 1e-10 else np.inf
    except:
        f_stat = np.nan
    
    compliers_ate = beta_compliers if prop_compliers > 0 else 0
    
    if prop_defiers == 0:
        late_theoretical = compliers_ate
    else:
        y_given_z1, y_given_z0 = np.mean(Y[Z == 1]), np.mean(Y[Z == 0])
        d_given_z1, d_given_z0 = np.mean(D[Z == 1]), np.mean(D[Z == 0])
        late_theoretical = (y_given_z1 - y_given_z0) / (d_given_z1 - d_given_z0) if abs(d_given_z1 - d_given_z0) > 1e-6 else 0
    
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
    
    st.markdown("---")
    st.subheader(text['iv_diagnosis'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(text['first_stage_f'], f"{f_stat:.2f}")
        if f_stat < 10: st.warning(f"⚠️ {text['iv_weak']}")
        else: st.success(f"✓ {text['iv_strong']}")
    with col2:
        st.metric(text['correlation'], f"{np.corrcoef(D, Z)[0, 1]:.4f}")
    with col3:
        st.metric(text['covariance'], f"{np.cov(D, Z)[0, 1]:.4f}")
    
    st.markdown("---")
    st.subheader(text['late_theorem'])
    st.markdown(f"{text['late_explanation']}\n\n{text['late_assumption_1']}\n{text['late_assumption_2']}\n{text['late_assumption_3']}\n\n**Analysis**:\n\n{text['late_result_scenario1'] if prop_defiers == 0 else text['late_result_scenario2']}")
    
    if prop_defiers > 0:
        st.warning(text['monotonicity_violation'])
    
    st.markdown("---")
    st.subheader(text['hte_results'])
    
    hte_comparison_data = [
        {text['individual_type']: 'Compliers', text['proportion']: f'{prop_compliers:.0%}', text['true_effect_col']: f'{beta_compliers:.4f}', text['weighted_contrib']: f'{beta_compliers * prop_compliers:.4f}'},
        {text['individual_type']: 'Always-takers', text['proportion']: f'{prop_always:.0%}', text['true_effect_col']: f'{beta_always:.4f}', text['weighted_contrib']: f'{beta_always * prop_always:.4f}'},
        {text['individual_type']: 'Never-takers', text['proportion']: f'{prop_never:.0%}', text['true_effect_col']: f'{beta_never:.4f}', text['weighted_contrib']: f'{beta_never * prop_never:.4f}'}
    ]
    if prop_defiers > 0:
        hte_comparison_data.append({text['individual_type']: 'Defiers', text['proportion']: f'{prop_defiers:.0%}', text['true_effect_col']: f'{beta_defiers:.4f}', text['weighted_contrib']: f'{beta_defiers * prop_defiers:.4f}'})
    
    st.dataframe(pd.DataFrame(hte_comparison_data), use_container_width=True)
    
    pop_ate = (beta_compliers * prop_compliers + beta_always * prop_always + beta_never * prop_never + beta_defiers * prop_defiers)
    
    st.markdown(f"""
{text['key_results']}:
- **{text['pop_ate']}**: {pop_ate:.4f}
- **{text['ols_est']}**: {beta_ols_coef:.4f}
- **{text['tsls_est']}**: {beta_2sls_coef:.4f}
- **{text['theoretical_late']}**: {late_theoretical:.4f}
- **{text['tsls_dev']}**: {abs(beta_2sls_coef - late_theoretical):.4f}

{text['explain_title']}:
    """)
    if prop_defiers == 0:
        st.success(text['scen1_success'].format(beta_compliers, beta_2sls_coef, late_theoretical))
    else:
        st.error(text['scen2_error'].format(prop_defiers, beta_2sls_coef))
    
    st.markdown("---")
    st.subheader(text['visualization'])
    
    fig_box = go.Figure()
    colors = {'Compliers': 'blue', 'Always-takers': 'green', 'Never-takers': 'orange', 'Defiers': 'red'}
    
    for dtype in ['Compliers', 'Always-takers', 'Never-takers', 'Defiers']:
        if dtype == 'Defiers' and prop_defiers == 0: continue
        mask = individual_types == dtype
        if np.any(mask):
            mask_d0 = mask & (D == 0)
            if np.any(mask_d0):
                fig_box.add_trace(go.Box(y=Y[mask_d0], name=f'{dtype} (D=0)', marker_color=colors[dtype], opacity=0.7))
            mask_d1 = mask & (D == 1)
            if np.any(mask_d1):
                fig_box.add_trace(go.Box(y=Y[mask_d1], name=f'{dtype} (D=1)', marker_color=colors[dtype], opacity=1.0))
    
    fig_box.update_layout(title=text['dist_title'], yaxis_title="Y", xaxis_title=text['dist_xaxis'], boxmode='group', height=500)
    st.plotly_chart(fig_box, use_container_width=True)
    
else:
    X_ols = np.column_stack([np.ones(n), X])
    beta_ols = np.linalg.lstsq(X_ols, Y, rcond=None)[0]
    beta_ols_coef = beta_ols[1]
    Y_pred_ols = X_ols @ beta_ols

    X_first = np.column_stack([np.ones(n), Z])
    gamma_hat = np.linalg.lstsq(X_first, X, rcond=None)[0]
    X_pred = X_first @ gamma_hat

    X_second = np.column_stack([np.ones(n), X_pred])
    beta_2sls = np.linalg.lstsq(X_second, Y, rcond=None)[0]
    beta_2sls_coef = beta_2sls[1]
    Y_pred_2sls = X_second @ beta_2sls

    ssr_ols = np.sum((Y - Y_pred_ols)**2)
    tss = np.sum((Y - np.mean(Y))**2)
    r2_ols = 1 - (ssr_ols / tss)

    ssr_2sls = np.sum((Y - Y_pred_2sls)**2)
    r2_2sls = 1 - (ssr_2sls / tss)

    try:
        Z_with_const = np.column_stack([np.ones(n), Z])
        u_first = np.linalg.lstsq(Z_with_const, X, rcond=None)[0]
        X_pred_first = Z_with_const @ u_first
        ssr_first = np.sum((X - X_pred_first)**2)
        msr_z = np.sum((X_pred_first - np.mean(X))**2)
        f_stat = (msr_z / 1) / (ssr_first / (n - 2)) if ssr_first / (n - 2) > 1e-10 else np.inf
    except:
        f_stat = np.nan

    beta_true = 1.0

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

    st.markdown("---")
    st.subheader(text['iv_diagnosis'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(text['first_stage_f'], f"{f_stat:.2f}")
        if f_stat < 10: st.warning(f"⚠️ {text['iv_weak']}")
        else: st.success(f"✓ {text['iv_strong']}")
    with col2:
        st.metric(text['correlation'], f"{np.corrcoef(X, Z)[0, 1]:.4f}")
    with col3:
        st.metric(text['covariance'], f"{np.cov(X, Z)[0, 1]:.4f}")

    st.markdown("---")
    st.subheader(text['visualization'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=Y, mode='markers', name=text['data_point'], marker=dict(color='rgba(0, 100, 200, 0.5)', size=4)))

    X_sort_idx = np.argsort(X)
    X_sort = X[X_sort_idx]
    
    fig.add_trace(go.Scatter(x=X_sort, y=Y_pred_ols[X_sort_idx], mode='lines', name=f'OLS (β̂={beta_ols_coef:.4f})', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=X_sort, y=Y_pred_2sls[X_sort_idx], mode='lines', name=f'2SLS (β̂={beta_2sls_coef:.4f})', line=dict(color='green', width=2)))

    fig.update_layout(title=text['scatter_plot'], xaxis_title='X', yaxis_title='Y', hovermode='closest', height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader(text['insight'])

    bias_ols = beta_ols_coef - beta_true
    bias_2sls = beta_2sls_coef - beta_true

    st.markdown(f"""

**{text['explanation']}**:
    """)

    if lang == 'en':
            st.markdown("""
            - When φ > 0, Z directly affects Y, violating the exclusion restriction and causing OLS bias.
            - 2SLS eliminates this bias using the instrumental variable method.
            - The stronger the IV (γ), the more precise the 2SLS estimate.
            - Error transmission (δ) affects the correlation between X and U, impacting the degree of OLS bias.
            """
            )
    elif lang == 'zh':
        st.markdown("""
        - 当 φ > 0 时，Z 会直接影响 Y，违反排他性条件，导致 OLS 回归产生偏差。
        - 2SLS 利用工具变量方法消除该偏差。
        - 工具变量越强（γ 越大），2SLS 估计越精确。
        - 误差传导（δ）影响 X 与 U 的相关性，进而影响 OLS 偏差程度。
        """
        )