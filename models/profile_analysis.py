# --------------------------
# 1. 导入通用库
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, chi2
from scipy.linalg import det
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（避免画图乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 2. 数据加载与预处理
# --------------------------
# 替换为你的CSV文件路径
df = pd.read_csv("D:/校务/Projects/problematic-papers/results/paper_rates_critic_score.csv")
print("=== 筛选domain=1的论文 ===")
# 检查domain列是否存在
if 'domain' not in df.columns:
    raise ValueError("数据中未找到'domain'列，请确认列名是否正确！")
# 筛选domain=1的行
#df = df[df['domain'] == 1]

# 核心变量定义（根据实际列名修改）
group_var = 'author_type'  # 分组变量：1=学术界，2=工业界
quality_vars = ['empirical_clarity_score','explanation_vs_speculation_score','language_misuse_score','math_quality_score']
control_var = 'year'  # 控制变量：发表年份

# 样本量分布检查
print("=== 样本量分布 ===")
sample_dist = df[group_var].value_counts().sort_index()
print(f"学术界（author_type=1）样本量：{sample_dist.get(1, 0)}")
print(f"工业界（author_type=2）样本量：{sample_dist.get(2, 0)}")

# 工业界样本量风险提示
if sample_dist.get(2, 0) < 30:
    print("\n⚠️ 警告：工业界样本量<30，参数检验结果可能不稳定，建议重点参考非参数/置换检验结果！")
else:
    print("\n✅ 工业界样本量≥30，可进行参数+非参数联合分析。")

# 数据清洗：删除缺失值 + 分组标签转换
df_clean = df.dropna(subset=[group_var] + quality_vars + [control_var])
df_clean[group_var] = df_clean[group_var].map({1: '学术界', 2: '工业界'})
print(f"\n清洗后总样本量：{len(df_clean)}")
print(f"清洗后分组分布：\n{df_clean[group_var].value_counts()}")

# --------------------------
# 3. MANCOVA前的模型假设检验
# --------------------------
print("\n" + "="*60)
print("=== MANCOVA 模型假设检验 ===")
print("="*60)

# 3.1 单变量正态性检验（Shapiro-Wilk，工业界）
print("\n1. 单变量正态性检验（Shapiro-Wilk，工业界）")
industry_data = df_clean[df_clean[group_var] == '工业界'][quality_vars]
norm_results = []
for col in quality_vars:
    stat, p = shapiro(industry_data[col])
    norm_results.append({
        '质量指标': col,
        'Shapiro统计量': round(stat, 4),
        'p值': round(p, 4),
        '是否满足正态性': p > 0.05
    })
norm_df = pd.DataFrame(norm_results)
print(norm_df)

# 3.2 方差-协方差矩阵齐性检验（Box's M）
print("\n2. 方差-协方差矩阵齐性检验（Box's M）")
def box_m_test(group1_data, group2_data):
    n1, p = group1_data.shape
    n2, _ = group2_data.shape
    cov1 = group1_data.cov()
    cov2 = group2_data.cov()
    cov_pooled = ((n1-1)*cov1 + (n2-1)*cov2) / (n1 + n2 - 2)
    m_stat = (n1-1)*np.log(det(cov_pooled)) + (n2-1)*np.log(det(cov_pooled)) - \
             (n1-1)*np.log(det(cov1)) - (n2-1)*np.log(det(cov2))
    df_box = p*(p+1) // 2
    p_box = 1 - chi2.cdf(m_stat, df_box)
    return m_stat, p_box, df_box

group1 = df_clean[df_clean[group_var] == '学术界'][quality_vars]
group2 = df_clean[df_clean[group_var] == '工业界'][quality_vars]
box_stat, box_p, box_df = box_m_test(group1, group2)
print(f"Box's M统计量：{box_stat:.4f}")
print(f"p值：{box_p:.4f}，自由度：{box_df}")
print(f"是否满足齐性假设：{'是' if box_p > 0.05 else '否'}")

# 3.3 单变量方差齐性检验（Levene）
print("\n3. 单变量方差齐性检验（Levene）")
levene_results = []
for col in quality_vars:
    stat, p = levene(
        df_clean[df_clean[group_var] == '学术界'][col],
        df_clean[df_clean[group_var] == '工业界'][col]
    )
    levene_results.append({
        '质量指标': col,
        'Levene统计量': round(stat, 4),
        'p值': round(p, 4),
        '是否满足方差齐性': p > 0.05
    })
levene_df = pd.DataFrame(levene_results)
print(levene_df)

# --------------------------
# 4. 基础差异检验：MANCOVA + 置换检验
# --------------------------
print("\n" + "="*60)
print("=== 基础差异检验：MANCOVA + 多变量置换检验 ===")
print("="*60)

# 4.1 普通MANCOVA（控制变量：年份）
print("\n【1. 普通MANCOVA结果（控制变量：年份）】")
formula = f"{' + '.join(quality_vars)} ~ C({group_var}) + {control_var}"
manova = MANOVA.from_formula(formula, data=df_clean)
manova_result = manova.mv_test()
print("分组变量（学术界vs工业界）主效应：")
print(manova_result.results['C(author_type)'])

# 4.2 多变量置换检验（稳健版）
print("\n【2. 多变量置换检验（无分布假设，适配样本量不均衡）】")
def multivariate_permutation_test(df, group_col, y_cols, covar_col, n_permutations=1000):
    keep_cols = y_cols + [group_col]
    if covar_col is not None:
        keep_cols.append(covar_col)
    
    df_filtered = df[keep_cols].dropna()
    Y = df_filtered[y_cols].values.astype(np.float64)
    
    # 构建自变量矩阵X
    group_dummies = pd.get_dummies(df_filtered[group_col], drop_first=True)
    covar_data = []
    if covar_col is not None:
        covar_series = df_filtered[covar_col]
        if covar_series.dtype == 'object' or len(covar_series.unique()) < 10:
            covar_dummies = pd.get_dummies(covar_series, drop_first=True)
            covar_data.append(covar_dummies)
        else:
            covar_data.append(covar_series.values.reshape(-1, 1))
    
    X_parts = [group_dummies.values] + covar_data
    X = np.hstack(X_parts).astype(np.float64)
    
    # 计算原始F统计量
    n, p = Y.shape
    k = X.shape[1]
    try:
        XTX = X.T @ X
        XTX += 1e-8 * np.eye(XTX.shape[0])
        XTX_inv = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        raise ValueError("矩阵不可逆，请检查自变量是否存在多重共线性")
    
    H = X @ XTX_inv @ X.T
    T = Y.T @ H @ Y
    SSE = Y.T @ (np.eye(n) - H) @ Y
    f_stat_original = (np.trace(T) / k) / (np.trace(SSE) / (n - k - 1))
    
    # 置换检验
    perm_f_stats = []
    for _ in range(n_permutations):
        permuted_groups = np.random.permutation(df_filtered[group_col].values)
        perm_group_dummies = pd.get_dummies(permuted_groups, drop_first=True)
        perm_X_parts = [perm_group_dummies.values] + covar_data
        perm_X = np.hstack(perm_X_parts).astype(np.float64)
        
        perm_XTX_inv = np.linalg.inv(perm_X.T @ perm_X + 1e-8 * np.eye(k))
        perm_H = perm_X @ perm_XTX_inv @ perm_X.T
        perm_T = Y.T @ perm_H @ Y
        perm_SSE = Y.T @ (np.eye(n) - perm_H) @ Y
        perm_f = (np.trace(perm_T) / k) / (np.trace(SSE) / (n - k - 1))
        perm_f_stats.append(perm_f)
    
    perm_f_stats = np.array(perm_f_stats)
    perm_p = np.sum(perm_f_stats >= f_stat_original) / n_permutations
    return f_stat_original, perm_p, perm_f_stats

# 运行置换检验
f_original, perm_p, perm_f = multivariate_permutation_test(
    df=df_clean,
    group_col=group_var,
    y_cols=quality_vars,
    covar_col=control_var,
    n_permutations=1000
)

print(f"原始F统计量：{f_original:.4f}")
print(f"置换检验p值（n=1000）：{perm_p:.4f}")
if perm_p < 0.05:
    # 【修改】调整表述：差异是“质量更差”的维度差异
    print("✅ 置换检验结论：控制年份后，两组在4个质量指标（越高质量越差）整体上存在显著差异！")
else:
    print("❌ 置换检验结论：控制年份后，两组在4个质量指标（越高质量越差）整体上无显著差异。")

# 4.3 单维度事后检验（Mann-Whitney U + Bonferroni校正）
print("\n【3. 单维度差异检验（Mann-Whitney U + Bonferroni校正）】")
u_results = []
alpha = 0.05
corrected_alpha = alpha / len(quality_vars)

for col in quality_vars:
    acad_data = df_clean[df_clean[group_var] == '学术界'][col]
    ind_data = df_clean[df_clean[group_var] == '工业界'][col]
    u_stat, p_raw = stats.mannwhitneyu(acad_data, ind_data, alternative='two-sided')
    
    # 计算效应量r
    n_total = len(acad_data) + len(ind_data)
    z = stats.norm.ppf(p_raw / 2) if p_raw < 0.05 else stats.norm.ppf(0.025)
    effect_size_r = abs(z) / np.sqrt(n_total)
    p_corrected = p_raw * len(quality_vars)
    
    u_results.append({
        '质量指标': col,
        'U统计量': round(u_stat, 4),
        '原始p值': round(p_raw, 4),
        '校正后p值': round(p_corrected, 4),
        '效应量r': round(effect_size_r, 4),
        '是否显著（α=0.05）': p_corrected < 0.05
    })
u_df = pd.DataFrame(u_results)
print(u_df)

# --------------------------
# 5. 核心占优维度分析：轮廓分析
# --------------------------
print("\n" + "="*60)
print("=== 核心占优维度分析：轮廓分析 ===")
print("="*60)

# 5.1 质量指标标准化
scaler = StandardScaler()
df_clean[quality_vars] = scaler.fit_transform(df_clean[quality_vars])

# 5.2 转换为长格式
df_long = pd.melt(
    df_clean,
    id_vars=[group_var, control_var],
    value_vars=quality_vars,
    var_name='质量指标',
    value_name='标准化得分'
)

# 5.3 轮廓分析三步检验
print("\n【1. 平行性检验（核心：占优维度是否不同）】")
model_formula = "标准化得分 ~ C(author_type) + C(质量指标) + C(author_type):C(质量指标) + C(year)"
profile_model = ols(model_formula, data=df_long).fit()
anova_results = anova_lm(profile_model, typ=2)

# 平行性检验（交互项）
interact_term = 'C(author_type):C(质量指标)'
if interact_term in anova_results.index:
    interact_f = anova_results.loc[interact_term, 'F']
    interact_p = anova_results.loc[interact_term, 'PR(>F)']
else:
    raise ValueError(f"交互项 '{interact_term}' 未在 ANOVA 结果中找到")

print(f"分组×指标交互项F值：{float(interact_f):.4f}")
print(f"p值：{float(interact_p):.4f}")
if float(interact_p) < 0.05:
    print("✅ 平行性检验拒绝H0：两组质量轮廓不平行 → 质量劣势维度存在显著差异（得分越高质量越差）！")
else:
    print("❌ 平行性检验接受H0：两组质量轮廓平行 → 质量劣势维度无显著差异（得分越高质量越差）。")

# 水平性检验（分组主效应：整体质量水平）
print("\n【2. 水平性检验（整体质量水平是否不同）】")
group_term = 'C(author_type)'
if group_term in anova_results.index:
    group_f = anova_results.loc[group_term, 'F']
    group_p = anova_results.loc[group_term, 'PR(>F)']
else:
    raise ValueError(f"分组项 '{group_term}' 未在 ANOVA 结果中找到")

print(f"分组主效应F值：{float(group_f):.4f}，p值：{float(group_p):.4f}")
if float(group_p) < 0.05:
    print("✅ 结论：两组整体质量水平存在显著差异（得分越高质量越差）")
else:
    print("❌ 结论：两组整体质量水平无显著差异（得分越高质量越差）")

# 扁平性检验（指标主效应）
print("\n【3. 扁平性检验（单组内维度是否有差异）】")
index_term = 'C(质量指标)'
if index_term in anova_results.index:
    index_f = anova_results.loc[index_term, 'F']
    index_p = anova_results.loc[index_term, 'PR(>F)']
else:
    raise ValueError(f"指标项 '{index_term}' 未在 ANOVA 结果中找到")

print(f"质量指标主效应F值：{float(index_f):.4f}，p值：{float(index_p):.4f}")
if float(index_p) < 0.05:
    print("✅ 结论：单组内各质量维度（越高质量越差）存在显著差异")
else:
    print("❌ 结论：单组内各质量维度（越高质量越差）无显著差异")

# ====================== 新增代码开始 ======================
# 5.3.1 打印两组各指标的标准化后具体值
print("\n【3.1 两组各质量指标的标准化后均值（得分越高质量越差）】")
# 计算各分组各指标的标准化均值
std_mean_df = df_clean.groupby(group_var)[quality_vars].mean().round(4)
# 重命名列名，让输出更易读
std_mean_df.columns = [
    '实证清晰度得分', 
    '解释-推测区分度得分', 
    '语言规范性得分', 
    '数学合理性得分'
]
# 打印标准化均值表格
print(std_mean_df)

# 5.3.1 打印两组各指标的标准化后具体值
print("\n【3.1 两组各质量指标的标准化后均值（得分越高质量越差）】")
# 计算各分组各指标的标准化均值
std_mean_df = df_clean.groupby(group_var)[quality_vars].mean().round(4)
# 重命名列名，让输出更易读
std_mean_df.columns = [
    '实证清晰度得分', 
    '解释-推测区分度得分', 
    '语言规范性得分', 
    '数学合理性得分'
]
# 核心修复：定义 profile_mean 变量（转置后适配雷达图/排名逻辑）
# 转置原因：原std_mean_df是“行=分组，列=指标”，转置后变为“行=指标，列=分组”，方便按指标遍历
profile_mean = std_mean_df.T  # T表示转置，关键！
print("标准化均值表（行=指标，列=分组）：")
print(profile_mean.round(4))

# 可选：打印更详细的描述性统计（均值、标准差、最小值、最大值）
print("\n【3.2 两组各质量指标的标准化后详细统计】")
std_detail_df = df_clean.groupby(group_var)[quality_vars].describe().round(4)
print(std_detail_df)
# ====================== 新增代码结束 ======================

print("\n【4.1 绘制质量维度雷达图】")
# 准备雷达图数据
labels = [
    '实证清晰度', 
    '解释-推测\n区分度', 
    '语言规范性', 
    '数学\n合理性'
]  # 简化指标名称，更适合图表展示
acad_scores = profile_mean['学术界'].values  # 现在profile_mean已定义，不会报错
ind_scores = profile_mean['工业界'].values   # 同上

# 计算雷达图角度（等分360度，闭合图形）
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # 闭合角度
acad_scores = np.append(acad_scores, acad_scores[0])  # 闭合学术界得分
ind_scores = np.append(ind_scores, ind_scores[0])  # 闭合工业界得分
labels += [labels[0]]  # 闭合标签

# 创建雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制学术界轮廓
ax.plot(angles, acad_scores, color='#2E86AB', linewidth=2.5, label='学术界', marker='o', markersize=8)
ax.fill(angles, acad_scores, color='#2E86AB', alpha=0.2)

# 绘制工业界轮廓
ax.plot(angles, ind_scores, color='#A23B72', linewidth=2.5, label='工业界', marker='s', markersize=8)
ax.fill(angles, ind_scores, color='#A23B72', alpha=0.2)

# 设置雷达图样式
ax.set_theta_offset(np.pi / 2)  # 旋转角度，让第一个维度在顶部
ax.set_theta_direction(-1)  # 顺时针显示维度
ax.set_xticks(angles[:-1])  # 设置维度刻度
ax.set_xticklabels(labels[:-1], fontsize=16)  # 设置维度标签
ax.set_ylabel('', fontsize=10, labelpad=20)
ax.grid(True, alpha=0.3)

# 添加标题和图例
plt.title('学术界 vs 工业界论文质量雷达图（得分越高质量越差）\n', 
          fontsize=14, fontweight='bold', pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)

# 保存雷达图
plt.tight_layout()
plt.savefig('quality_radar_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 雷达图已保存为 quality_radar_plot.png")

# 5.5 占优维度总结
print("\n【5. 劣势维度总结（标准化均值，得分越高质量越差）】")
# 核心：调整排名逻辑——ascending=True表示“得分越低（质量越好）排名越靠前”
profile_mean['学术界质量最优维度排名'] = profile_mean['学术界'].rank(ascending=True).astype(int)
profile_mean['工业界质量最优维度排名'] = profile_mean['工业界'].rank(ascending=True).astype(int)
# 补充：质量最差维度排名（原逻辑）
profile_mean['学术界质量最差维度排名'] = profile_mean['学术界'].rank(ascending=False).astype(int)
profile_mean['工业界质量最差维度排名'] = profile_mean['工业界'].rank(ascending=False).astype(int)
print(profile_mean.round(4))