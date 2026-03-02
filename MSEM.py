import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial']  # 解决中文显示问题

# --------------------------
# 1. 加载数据并验证核心变量
# --------------------------
# 加载数据（使用用户上传的文件路径）
df = pd.read_csv("D:/校务/Projects/problematic-papers/results/paper_rates_newest.csv")

# 定义核心变量（确保与CSV列名一致）
X_var = "citation_ai_rate"  # 自变量：引用AI率（0/1/2）
M_vars = [                  # 中介变量：4个分项质量评分（1-10，越高越差）
    "empirical_clarity_score",
    "explanation_vs_speculation_score",
    "language_misuse_score",
    "math_quality_score"
]
Y_var = "all_score"         # 因变量：论文总质量（0-1，越高越好）
group_var = "domain"        # 分组变量（层2：学科类型）
control_var = "year"        # 控制变量：年份

# 验证变量是否存在
missing_vars = [var for var in [X_var, Y_var, control_var] + M_vars if var not in df.columns]
if missing_vars:
    raise ValueError(f"CSV文件中缺失以下核心变量：{', '.join(missing_vars)}")

# 查看数据基本信息
print("=== 数据基本信息 ===")
print(df[[X_var, Y_var, control_var] + M_vars + [group_var]].info())
print("\n=== 数据前5行 ===")
print(df[[X_var, Y_var, control_var] + M_vars + [group_var]].head())
print(f"\n=== 学科类型（{group_var}）分布 ===")
print(df[group_var].value_counts().sort_index())
print(f"\n=== 自变量（{X_var}）分布 ===")
print(df[X_var].value_counts().sort_index())

# 2.1 年份编码（控制变量处理）
# --------------------------
df["year_encoded"] = df[control_var] - 2017  
print(f"\n=== 年份编码后范围：{df['year_encoded'].min()}~{df['year_encoded'].max()} ===")

# --------------------------
# 2.2 中心化处理（MSEM核心：区分层1/层2）
# --------------------------
# 层1变量：所有变量按学科（group_var）分组做组内中心化（WC）
# 目的：分离层1（组内）和层2（组间）变异
layer1_vars = [X_var, Y_var, "year_encoded"] + M_vars
for var in layer1_vars:
    df[f"{var}_wc"] = df.groupby(group_var)[var].transform(lambda x: x - x.mean())
    # 验证：每组中心化后均值≈0
    print(f"\n=== {var}组内中心化后各组均值 ===")
    print(df.groupby(group_var)[f"{var}_wc"].mean().round(4))

# 层2变量：学科类型虚拟编码（用于控制层2效应）
# 关键1：使用更独特的前缀，避免列名冲突
# 关键2：drop_first=True 避免虚拟变量陷阱（解决后续共线性）
df_dummies = pd.get_dummies(df[group_var], prefix="domain_dummy", drop_first=True)
df = pd.concat([df, df_dummies], axis=1)
layer2_vars = df_dummies.columns.tolist()  # 层2变量：学科虚拟变量

# 关键检查：在生成虚拟变量后，检查列名是否重复
print(f"\n=== 列名是否重复：{df.columns.duplicated().any()} ===")
# 如果输出True，执行以下代码去重（保留最后一列）
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()]
    print("=== 已自动删除重复列名 ===")

# 层2变量组间中心化（BC）
for var in layer2_vars:
    # 关键修复1：强制通过列名精确索引，确保取到单列（避免模糊匹配多列）
    # 先检查var是否在df.columns中（避免列名不存在）
    if var not in df.columns:
        print(f"警告：列名{var}不存在，跳过该变量")
        continue
    
    # 关键修复2：用iloc[:, df.columns.get_loc(var)] 强制取单列Series
    # df.columns.get_loc(var) 返回var列的索引位置，iloc按位置取列（避免模糊匹配）
    var_series = df.iloc[:, df.columns.get_loc(var)]
    
    # 验证：确保var_series是Series（一维）
    if not isinstance(var_series, pd.Series):
        raise ValueError(f"变量{var}不是单列Series，请检查列名是否重复/多列匹配")
    
    # 组间中心化（减总均值）
    df[f"{var}_bc"] = var_series - var_series.mean()
    
    print(f"\n=== {var}组间中心化后总均值 ===")
    print(df[f"{var}_bc"].mean().round(4))

# --------------------------
# 2.3 缺失值处理（按变量类型适配）
# --------------------------
print("\n=== 缺失值统计 ===")
# 需处理的变量：所有中心化后的层1变量 + 层2变量
target_vars = [f"{var}_wc" for var in layer1_vars] + [f"{var}_bc" for var in layer2_vars]
missing_rate = df[target_vars].isnull().mean()
print(missing_rate.round(4))

# 填充策略：离散变量（X_var）用众数，连续变量用均值
for var in target_vars:
    if df[var].isnull().sum() > 0:
        original_var = var.replace("_wc", "").replace("_bc", "")
        if original_var == X_var:  # 引用AI率（离散）：众数填充
            fill_val = df[original_var].mode()[0]
        else:  # 其他变量：均值填充
            fill_val = df[var].mean()
        df[var].fillna(fill_val, inplace=True)
        print(f"\n=== {var}缺失值用{fill_val:.2f}填充 ===")

# --------------------------
# 3.1 ICC检验（层间变异验证）
# --------------------------
def calculate_icc(df, dependent_var_wc, group_var):
    """计算ICC(1)：层2变异占比（>0.05需用多层模型）"""
    # 空模型：仅含截距，随机效应为学科
    model_null = MixedLM.from_formula(
        formula=f"{dependent_var_wc} ~ 1",
        data=df,
        groups=df[group_var]
    )
    result_null = model_null.fit(disp=False)  # 关闭收敛信息
    
    # 提取方差组件
    tau00 = result_null.cov_re.iloc[0, 0]  # 层2（学科）方差
    sigma2 = result_null.resid.var()       # 层1（个体）方差
    icc = tau00 / (tau00 + sigma2)         # ICC(1)
    
    return icc, tau00, sigma2

# 对因变量和所有中介变量做ICC检验
print("\n=== ICC(1)检验结果（层间变异占比）===")
icc_results = {}
for var in [Y_var] + M_vars:
    icc, tau, sigma = calculate_icc(df, f"{var}_wc", group_var)
    icc_results[var] = icc
    print(f"{var}：ICC={icc:.4f}（{'需多层模型' if icc>0.05 else '可简化为单层模型'}）")

# --------------------------
# 3.2 正态性检验（连续变量）
# --------------------------
def normality_test(df, var_wc):
    """检验正态性：偏度绝对值<3且峰度绝对值<3为近似正态"""
    skew = stats.skew(df[var_wc])
    kurt = stats.kurtosis(df[var_wc], fisher=True)  # Fisher峰度（减3）
    is_normal = (abs(skew) < 3) & (abs(kurt) < 3)
    return pd.Series([skew, kurt, is_normal], index=["偏度", "峰度", "是否正态"])

print("\n=== 正态性检验结果（组内中心化后）===")
norm_results = pd.DataFrame()
for var in [X_var, Y_var] + M_vars:
    norm_results[var] = normality_test(df, f"{var}_wc")
print(norm_results.round(4))

# --------------------------
# 3.3 多重共线性检验（VIF）
# --------------------------
def calculate_vif(df, var_list):
    """计算VIF：<5无严重共线性，<10可接受"""
    from statsmodels.formula.api import ols
    vif_data = pd.DataFrame(index=var_list, columns=["VIF"])
    for var in var_list:
        # 其他变量作为自变量
        other_vars = [v for v in var_list if v != var]
        formula = f"{var} ~ {' + '.join(other_vars)}"
        model = ols(formula, data=df).fit()
        vif_data.loc[var, "VIF"] = 1 / (1 - model.rsquared)
    return vif_data.astype(float).round(2)

# 需检验的变量：自变量+中介变量+控制变量（均为中心化后）
vif_vars = [f"{X_var}_wc", "year_encoded_wc"] + [f"{m}_wc" for m in M_vars]
vif_results = calculate_vif(df, vif_vars)
print("\n=== 多重共线性检验（VIF）===")
print(vif_results)
print(f"结论：{'无严重共线性' if vif_results['VIF'].max() < 5 else '存在严重共线性（需删除高VIF变量）'}")

# --------------------------
# 4.1 模型1：中介变量回归（X→M路径）
# --------------------------
print("\n" + "="*50)
print("=== 模型1：中介变量回归（X→M路径）===")
print("="*50)


def mixed_lm_r2(result):
    """
    计算混合线性模型（MixedLM）的Marginal R²和Conditional R²
    参数：
        result: MixedLM.fit()返回的结果对象
    返回：
        r2_marginal: 固定效应解释的方差比例
        r2_conditional: 固定+随机效应解释的方差比例
    """
    # 提取方差组件
    # 随机效应方差（层2方差）
    var_random = result.cov_re.iloc[0, 0] if result.cov_re is not None else 0
    # 残差方差（层1方差）
    var_residual = result.scale
    
    # 固定效应的预测值方差
    y_hat = result.fittedvalues
    var_fixed = np.var(y_hat)
    
    # 计算R²
    r2_marginal = var_fixed / (var_fixed + var_random + var_residual)
    r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_residual)
    
    return r2_marginal, r2_conditional


# 存储X→M路径系数（a）和p值
a_coefs = {}  # 格式：{中介变量: (a系数, p值)}
for m_var in M_vars:
    # 模型公式：中介变量 ~ 自变量 + 控制变量 + 学科虚拟变量（层2控制）
    formula = f"{m_var}_wc ~ {X_var}_wc + year_encoded_wc + {' + '.join([f'{v}_bc' for v in layer2_vars])}"
    model = MixedLM.from_formula(
        formula=formula,
        data=df,
        groups=df[group_var]  # 随机效应：学科截距
    )
    result = model.fit(disp=False)
    
    # 提取X→M的系数（a）和p值
    a_coef = result.params[f"{X_var}_wc"]
    a_p = result.pvalues[f"{X_var}_wc"]
    a_coefs[m_var] = (a_coef, a_p)
    
    # 计算混合模型专用R²（替换rsquared_adj）
    r2_marginal, r2_conditional = mixed_lm_r2(result)

    # 输出关键结果
    print(f"\n【{m_var}】")
    print(f"X→M路径系数（a）：{a_coef:.4f}（p={a_p:.4f}）")
    print(f"Marginal R²（固定效应）：{r2_marginal:.4f}")
    print(f"Conditional R²（固定+随机效应）：{r2_conditional:.4f}")

# --------------------------
# 4.2 模型2：因变量回归（X→Y直接效应 + M→Y路径）
# --------------------------
print("\n" + "="*50)
print("=== 模型2：因变量回归（X→Y直接效应 + M→Y路径）===")
print("="*50)

# 模型公式：总质量 ~ 自变量 + 所有中介变量 + 控制变量 + 学科虚拟变量
formula = f"{Y_var}_wc ~ {X_var}_wc + {' + '.join([f'{m}_wc' for m in M_vars])} + year_encoded_wc + {' + '.join([f'{v}_bc' for v in layer2_vars])}"
model_y = MixedLM.from_formula(
    formula=formula,
    data=df,
    groups=df[group_var]
)
result_y = model_y.fit(disp=False)

# 提取关键系数
c_prime = result_y.params[f"{X_var}_wc"]  # X→Y直接效应（c'）
c_prime_p = result_y.pvalues[f"{X_var}_wc"]
b_coefs = {}  # M→Y路径系数（b1-b4）
for m_var in M_vars:
    b_coef = result_y.params[f"{m_var}_wc"]
    b_p = result_y.pvalues[f"{m_var}_wc"]
    b_coefs[m_var] = (b_coef, b_p)

r2_marginal_2, r2_conditional_2 = mixed_lm_r2(result_y)
# 输出关键结果
print(f"X→Y直接效应（c'）：{c_prime:.4f}（p={c_prime_p:.4f}）")
for m_var in M_vars:
    b_coef, b_p = b_coefs[m_var]
    print(f"M→Y路径系数（{m_var}，b）：{b_coef:.4f}（p={b_p:.4f}）")
    
print(f"Marginal R²（固定效应）：{r2_marginal_2:.4f}")
print(f"Conditional R²（固定+随机效应）：{r2_conditional_2:.4f}")

# --------------------------
# 5.1 中介效应分解
# --------------------------
print("\n" + "="*50)
print("=== 联合中介效应分解结果 ===")
print("="*50)

# 计算各中介的单独效应、总中介效应、总效应
mediation_results = pd.DataFrame(
    index=M_vars,
    columns=["X→M系数（a）", "a的p值", "M→Y系数（b）", "b的p值", "单独中介效应（a*b）", "贡献比例（%）"]
)

total_mediation = 0  # 总中介效应
for m_var in M_vars:
    a_coef, a_p = a_coefs[m_var]
    b_coef, b_p = b_coefs[m_var]
    individual_effect = a_coef * b_coef  # 单独中介效应
    total_mediation += individual_effect
    
    # 填充结果
    mediation_results.loc[m_var, "X→M系数（a）"] = round(a_coef, 4)
    mediation_results.loc[m_var, "a的p值"] = round(a_p, 4)
    mediation_results.loc[m_var, "M→Y系数（b）"] = round(b_coef, 4)
    mediation_results.loc[m_var, "b的p值"] = round(b_p, 4)
    mediation_results.loc[m_var, "单独中介效应（a*b）"] = round(individual_effect, 4)

# 计算贡献比例（避免总中介效应为0的情况）
if abs(total_mediation) > 1e-8:
    mediation_results["贡献比例（%）"] = round((mediation_results["单独中介效应（a*b）"] / total_mediation) * 100, 2)
else:
    mediation_results["贡献比例（%）"] = 0.0

# 输出汇总结果
print(mediation_results)
print(f"\n【汇总】")
print(f"总中介效应（各单独效应之和）：{total_mediation:.4f}")
print(f"直接效应（c'）：{c_prime:.4f}")
print(f"总效应（总中介效应 + 直接效应）：{total_mediation + c_prime:.4f}")
print(f"中介效应占总效应比例：{round((total_mediation / (total_mediation + c_prime)) * 100, 2)}%")
print(f"\n注：1. 中介变量（{', '.join(M_vars)}）越高表示该维度质量越差；2. 因变量（{Y_var}）越高表示总质量越好；3. 负中介效应表示自变量通过该中介提升总质量。")

# --------------------------
# 5.2 中介效应贡献比例可视化
# --------------------------
plt.figure(figsize=(12, 6))
# 准备数据（排除贡献为0的中介）
plot_data = mediation_results[mediation_results["贡献比例（%）"] != 0]
if not plot_data.empty:
    # 简化中介变量名（便于显示）
    var_names = {
        "empirical_clarity_score": "实证清晰度",
        "explanation_vs_speculation_score": "解释vs推测",
        "language_misuse_score": "语言误用",
        "math_quality_score": "数学质量"
    }
    plot_labels = [var_names.get(idx, idx) for idx in plot_data.index]
    plot_values = plot_data["贡献比例（%）"].values
    
    # 绘制条形图
    bars = plt.bar(plot_labels, plot_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    # 添加数值标签
    for bar, value in zip(bars, plot_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f"{value:.1f}%", ha='center', va='bottom', fontsize=10)
    
    plt.title("各中介变量对总中介效应的贡献比例", fontsize=14, pad=20)
    plt.ylabel("贡献比例（%）", fontsize=12)
    plt.ylim(0, max(plot_values) * 1.2)  # 调整y轴范围
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("D:/校务/Projects/problematic-papers/results/mediation_contribution.png", dpi=300, bbox_inches='tight')
    print(f"\n=== 可视化结果已保存为：mediation_contribution.png ===")
else:
    print(f"\n=== 无有效中介效应，未生成可视化 ===")