import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer  # 需安装：pip install factor-analyzer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1. 读取数据，提取4个质量指标
df = pd.read_csv("D:/校务/Projects/problematic-papers/results/paper_rates.csv")
quality_cols = ['empirical_clarity_score', 'explanation_vs_speculation_score', 'language_misuse_score', 'math_quality_score']  # 替换为你的4个质量指标列名
df_quality = df[quality_cols].dropna()

# 初始化相关系数矩阵和p值矩阵
n = len(quality_cols)
pearson_corr = np.zeros((n, n))  # 存储Pearson相关系数（替换原Spearman）
p_values = np.zeros((n, n))       # 存储对应的p值

# 遍历计算每对指标的Pearson相关系数和p值
for i, col1 in enumerate(quality_cols):
    for j, col2 in enumerate(quality_cols):
        # scipy.stats.pearsonr返回相关系数和p值（适配连续数据）
        corr, p = stats.pearsonr(df_quality[col1], df_quality[col2])
        pearson_corr[i, j] = corr
        p_values[i, j] = p

# 转换为DataFrame方便查看和绘图
pearson_corr_df = pd.DataFrame(
    pearson_corr,  
    index=quality_cols, 
    columns=quality_cols
).round(3)

p_values_df = pd.DataFrame(
    p_values, 
    index=quality_cols, 
    columns=quality_cols
).round(3)

# 打印结果
print("4个质量指标的pearson相关系数矩阵：")
print(pearson_corr_df)
print("\n各相关系数对应的p值矩阵（p<0.05为显著）：")
print(p_values_df)

# -------------------------- 3. 绘制Spearman相关系数热力图（含显著性标注） --------------------------
# 设置绘图风格和画布大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.figure(figsize=(10, 8))

# 绘制热力图（仅展示相关系数）
ax = sns.heatmap(
    pearson_corr,
    annot=True,        # 显示相关系数数值
    fmt='.3f',         # 数值保留3位小数
    cmap='RdBu_r',     # 配色方案（红-蓝，正负相关区分）
    vmin=-1, vmax=1,   # 颜色范围（Spearman取值-1到1）
    square=True,       # 单元格为正方形
    linewidths=0.5,    # 单元格边框宽度
    cbar_kws={"shrink": 0.8}  # 颜色条缩放比例
)

# 标注显著性（p<0.05标*，p<0.01标**，p<0.001标***）
for i in range(n):
    for j in range(n):
        p = p_values[i, j]
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''
        # 在热力图数值下方标注显著性（调整位置避免重叠）
        ax.text(
            j+0.5, i+0.2, sig, 
            ha='center', va='center', 
            color='black', fontsize=12, fontweight='bold'
        )

# 设置标题和坐标轴标签
ax.set_title('4个论文质量指标的Spearman相关系数热力图', fontsize=14, pad=20)
ax.set_xlabel('质量指标', fontsize=12)
ax.set_ylabel('质量指标', fontsize=12)

# 调整坐标轴标签显示（避免重叠）
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 保存图片（可选，建议保存为高清格式）
plt.tight_layout()  # 自动调整布局
plt.savefig("pearson_corr_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 4. 关联性判断（适配Spearman） --------------------------
print("\n关联性判断：")
# 统计显著相关（p<0.05）且|相关系数|>0.3的指标对数量
significant_corr_pairs = 0
total_pairs = n*(n-1)//2  # 总两两配对数（排除对角线）
for i in range(n):
    for j in range(i+1, n):
        corr = pearson_corr[i, j]
        p = p_values[i, j]
        if abs(corr) > 0.3 and p < 0.05:
            significant_corr_pairs += 1
            print(f"{quality_cols[i]} 与 {quality_cols[j]}: Spearman相关系数={corr:.3f}, p值={p:.3f}（显著相关）")
        else:
            print(f"{quality_cols[i]} 与 {quality_cols[j]}: Spearman相关系数={corr:.3f}, p值={p:.3f}（不显著/关联度低）")

print(f"\n显著且关联度>0.3的指标对数量：{significant_corr_pairs}/{total_pairs}")
if significant_corr_pairs >= total_pairs * 0.5:  # 多数（≥50%）
    print("结论：多数指标对关联性高且显著，指标间存在合理的线性趋势关联")
else:
    print("结论：多数指标对关联性较低/不显著，指标间独立性较强但仍属同一构念")

# 3. 方法2：信度检验（Cronbach's α，衡量内部一致性）
def cronbach_alpha(df):
    """计算Cronbach's α系数（>0.6说明一致性可接受，>0.7说明良好）"""
    n_items = df.shape[1]
    item_variances = df.var(axis=0, ddof=1)
    total_variance = df.sum(axis=1).var(ddof=1)
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha

alpha = cronbach_alpha(df_quality)
print(f"\nCronbach's α系数：{alpha:.3f}")

# 4. 方法3：探索性因子分析（EFA，看是否能提取1个主因子）
# 标准化指标（消除量纲影响）
scaler = StandardScaler()
df_quality_scaled = scaler.fit_transform(df_quality)
# 执行EFA
fa = FactorAnalyzer(n_factors=1, rotation=None)
fa.fit(df_quality_scaled)
# 查看因子载荷（>0.5说明指标能被主因子解释）
factor_loadings = pd.DataFrame(fa.loadings_, index=quality_cols, columns=["主因子载荷"])
print("\n探索性因子分析：主因子载荷（>0.5为良好）")
print(factor_loadings.round(3))