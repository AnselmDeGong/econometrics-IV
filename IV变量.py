# 现在请帮我把上述 Jupyter 逻辑转化为一个 Streamlit 应用。要求如下：
# 1. 交互滑块：允许用户调整『合规率 (Compliance Rate)』——即  对  的影响有多大。
# 2. 对比展示：在一个页面上并排对比 OLS 回归和 2SLS 回归的结果。
# 3. 可视化：使用 Plotly 绘制散点图，并根据  的值（0或1）对点进行着色。
# 4. 公式：在页面顶部使用 LaTeX 展示 IV 的估计公式
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
# 设置页面标题
st.title("IV 教学 App: 服役抽签 (Draft Lottery) 模拟")

# 显示 IV 估计公式
st.latex(r"""
\hat{\beta}_{IV} = \frac{Cov(Y, Z)}{Cov(X, Z)}
""")   

# 交互滑块：调整合规率（在侧边栏中）
compliance_rate = st.sidebar.slider('Compliance Rate', min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# 检查工具变量强度
if compliance_rate < 0.3:
    st.warning('工具变量过弱，结果不可信！')

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 重新模拟数据根据合规率
n = 1000
Z = np.random.binomial(1, 0.5, n)
U = np.random.normal(0, 1, n)
X = compliance_rate * Z + (1 - compliance_rate) * np.random.binomial(1, 0.5, n) + U * 0.1
X = (X > 0.5).astype(int)
Y = 2 * X + U + np.random.normal(0, 1, n)
data = pd.DataFrame({'Y': Y, 'X': X, 'Z': Z})
data = sm.add_constant(data)
# OLS 回归
ols_model = sm.OLS(data['Y'], data[['const', 'X']])
ols_results = ols_model.fit()
# 2SLS 回归
iv_model = IV2SLS(dependent=data['Y'], exog=data[['const']], endog=data['X'], instruments=data['Z'])
iv_results = iv_model.fit()
# 显示回归结果对比
st.subheader("回归结果对比")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### OLS 回归结果")
    st.text(ols_results.summary().as_text())
with col2:
    st.markdown("### 2SLS 回归结果")
    # linearmodels 的 summary 没有 as_text() 方法，直接转换为字符串
    st.text(str(iv_results.summary))
# 显示第一阶段 F 统计量
st.subheader("工具变量强度检验")
first_stage_results = iv_results.first_stage
f_stat_value = float(first_stage_results.individual['X'].f_statistic.stat)
st.metric("第一阶段 F 统计量", f"{f_stat_value:.2f}")
# 可视化散点图
st.subheader("散点图可视化")
fig = px.scatter(data, x='X', y='Y', color='Z', title='散点图: X vs Y (按 Z 着色)', labels={'X': '实际参军 (X)', 'Y': '收入 (Y)', 'Z': '抽签结果 (Z)'})
st.plotly_chart(fig)    
