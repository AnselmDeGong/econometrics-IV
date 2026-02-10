import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置页面标题
st.title("IV 理论模拟器：纯理论 IV 模型")

# 在侧边栏添加滑块控制参数
st.sidebar.header("模型参数控制")
gamma = st.sidebar.slider('γ (IV 强度)', min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                          help="控制工具变量 Z 对 X 的影响强度")
delta = st.sidebar.slider('δ (误差传导)', min_value=0.0, max_value=2.0, value=0.5, step=0.1,
                         help="控制误差项 U 对 X 的影响")
phi = st.sidebar.slider('φ (排他性违反)', min_value=0.0, max_value=2.0, value=0.0, step=0.1,
                       help="控制 Z 对 Y 的直接影响 (排他性违反程度)")

# 动态诊断提示
if phi > 0:
    st.error(f'⚠️ 排他性违反：φ = {phi:.2f}，Z 直接影响 Y，IV 一致性崩塌！')
elif gamma < 0.5:
    st.warning(f'⚠️ 弱工具变量风险：γ = {gamma:.2f}，IV 强度不足，估计量方差将很大！')
elif delta > 1.0:
    st.info(f'ℹ️ 内生性偏差较大：δ = {delta:.2f}，误差项对 X 影响显著，OLS 将严重有偏！')

# 模型预览区
st.markdown("### 📋 理论模型预览")
st.markdown("---")

# 显示 X 的生成方程
st.latex(r"X = \gamma \cdot Z + \delta \cdot U + e_1, \quad e_1 \sim N(0, 1)")

# 显示 Y 的生成方程
st.latex(r"Y = \beta \cdot X + \alpha \cdot U + \phi \cdot Z + e_2, \quad e_2 \sim N(0, 1)")

# 参数说明
st.markdown("---")
st.markdown("### 📚 参数详解")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 变量定义")
    st.markdown(f"""
    - **U**: 误差项，$U \\sim N(0, 1)$
    - **Z**: 工具变量，$Z \\sim N(0, 1)$
    - **X**: 内生变量
    - **Y**: 被解释变量
    """)

with col2:
    st.markdown("#### 参数合义")
    st.markdown(f"""
    - **γ (gamma)** = {gamma:.2f}: 工具变量强度
    - **δ (delta)** = {delta:.2f}: 误差传导系数
    - **φ (phi)** = {phi:.2f}: 排他性违反程度
    - **β (beta)** = 1.0: 真实因果效应
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
st.subheader("📊 实时回归结果对比")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### OLS 回归")
    st.metric("β̂_OLS", f"{beta_ols_coef:.4f}", delta=f"{beta_ols_coef - beta_true:.4f} (真实值: 1.0)")
    st.metric("R²", f"{r2_ols:.4f}")
    st.markdown(f"**模型**: Y = {beta_ols[0]:.4f} + {beta_ols_coef:.4f}·X")

with col2:
    st.markdown("### 2SLS 回归")
    st.metric("β̂_2SLS", f"{beta_2sls_coef:.4f}", delta=f"{beta_2sls_coef - beta_true:.4f} (真实值: 1.0)")
    st.metric("R²", f"{r2_2sls:.4f}")
    st.markdown(f"**模型**: Y = {beta_2sls[0]:.4f} + {beta_2sls_coef:.4f}·X_pred")

# 显示工具变量强度
st.markdown("---")
st.subheader("🔍 工具变量诊断")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("第一阶段 F 统计量", f"{f_stat:.2f}")
    if f_stat < 10:
        st.warning("⚠️ IV 较弱 (F < 10)")
    else:
        st.success("✓ IV 强度足够")

with col2:
    correlation_xz = np.corrcoef(X, Z)[0, 1]
    st.metric("Corr(X, Z)", f"{correlation_xz:.4f}")

with col3:
    covariance_xz = np.cov(X, Z)[0, 1]
    st.metric("Cov(X, Z)", f"{covariance_xz:.4f}")

# 可视化
st.markdown("---")
st.subheader("📈 数据可视化")

# 创建 X vs Y 散点图，附带拟合线
fig = go.Figure()

# 散点
fig.add_trace(go.Scatter(
    x=X, y=Y,
    mode='markers',
    name='数据点',
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
    title='X vs Y 散点图与回归线',
    xaxis_title='X',
    yaxis_title='Y',
    hovermode='closest',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# 显示关键洞察
st.markdown("---")
st.subheader("💡 关键洞察")

bias_ols = beta_ols_coef - beta_true
bias_2sls = beta_2sls_coef - beta_true

st.markdown(f"""
- **OLS 偏差**: {bias_ols:.4f} ({(bias_ols/beta_true)*100:.2f}%)
- **2SLS 偏差**: {bias_2sls:.4f} ({(bias_2sls/beta_true)*100:.2f}%)
- **改善程度**: {abs(bias_ols - bias_2sls):.4f}

**解释**:
- 当 φ > 0 时，Z 直接影响 Y，违反排他性假设，导致 OLS 有偏差
- 2SLS 通过工具变量法消除这种偏差
- IV 强度 (γ) 越大，2SLS 估计越精确
- 误差传导 (δ) 影响 X 和 U 的相关性，影响 OLS 的有偏程度
""")