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
    },
    'en': {
        'title': 'IV Theory Simulator: Pure Theoretical IV Model',
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
phi = st.sidebar.slider(text['phi_label'], min_value=0.0, max_value=2.0, value=0.0, step=0.1,
                       help=text['phi_help'])

# 动态诊断提示
if phi > 0:
    st.error(f'⚠️ {text["exclusion_violation"].format(phi)}')
elif gamma < 0.5:
    st.warning(f'⚠️ {text["weak_iv"].format(gamma)}')
elif delta > 1.0:
    st.info(f'ℹ️ {text["endogeneity_bias"].format(delta)}')

# 模型预览区
st.markdown(f"### {text['model_preview']}")
st.markdown("---")

# 显示 X 的生成方程
st.latex(r"X = \gamma \cdot Z + \delta \cdot U + e_1, \quad e_1 \sim N(0, 1)")

# 显示 Y 的生成方程
st.latex(r"Y = \beta \cdot X + \alpha \cdot U + \phi \cdot Z + e_2, \quad e_2 \sim N(0, 1)")

# 参数说明
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
    - **φ (phi)** = {phi:.2f}: {text['exclusion_violation_degree']}
    - **β (beta)** = 1.0: {text['true_effect']}
    """)

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 模拟数据
n = 1000

# 1. 生成 U (误差项) 和 Z (工具变量)
U = np.random.normal(0, 1, n)
Z = np.random.normal(0, 1, n)

# 2. X = γ·Z + δ·U + e₁
e1 = np.random.normal(0, 1, n)
X = gamma * Z + delta * U + e1

# 3. Y = β·X + α·U + φ·Z + e₂ (β = 1.0 为真实值)
alpha = 1.0  # α 系数
beta_true = 1.0  # β 真实值 = 1.0
e2 = np.random.normal(0, 1, n)
Y = beta_true * X + alpha * U + phi * Z + e2

# 创建数据框
data = pd.DataFrame({
    'Y': Y,
    'X': X,
    'Z': Z,
    'U': U
})

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