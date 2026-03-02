import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr
from scipy.stats.mstats import winsorize
warnings.filterwarnings('ignore')

# ===================== 1. 读取数据 & 定义参数 =====================
df = pd.read_csv('D:/校务/Projects/problematic-papers/results/paper_rates_new_author.csv')
indicator_cols = [
    'empirical_clarity_score',
    'explanation_vs_speculation_score',
    'language_misuse_score',
    'math_quality_score'
]

# ===================== 2. 预处理：负向转正向 + 缩尾 + 分位数标准化 =====================
# 2.1 负向指标转正向
def reverse_negative_indicator(df, cols):
    df_processed = df.copy()
    positive_cols = []
    for col in cols:
        max_val = df[col].max()
        min_val = df[col].min()
        if max_val == min_val:
            df_processed[col + '_positive'] = 0
        else:
            df_processed[col + '_positive'] = (max_val - df[col]) / (max_val - min_val)
        positive_cols.append(col + '_positive')
    return df_processed, positive_cols

df, positive_cols = reverse_negative_indicator(df, indicator_cols)

# 2.2 缩尾处理（降低极端值对CV的干扰，保留极端样本）
for col in positive_cols:
    df[col + '_winsor'] = winsorize(df[col], limits=(0.01, 0.01))  # 1%/99%分位数缩尾
winsor_cols = [col + '_winsor' for col in positive_cols]

# 2.3 分位数标准化（彻底规避偏态，确保CV可靠）
def quantile_standardize(df, cols):
    df_std = df.copy()
    std_cols = []
    for col in cols:
        df_std[col + '_std'] = df_std[col].rank(pct=True)  # 排名→百分位（0~1）
        std_cols.append(col + '_std')
    return df_std, std_cols

df, std_cols = quantile_standardize(df, winsor_cols)

# ===================== 3. CRITIC法核心：计算权重 =====================
def critic_weight(df, cols):
    X = df[cols].values
    n, m = X.shape  # n=样本数，m=指标数（4）
    
    # 步骤1：计算每个指标的变异系数（CVj）
    cv = []
    for j in range(m):
        mean_j = np.mean(X[:, j])
        std_j = np.std(X[:, j])
        cv_j = std_j / mean_j if mean_j != 0 else 0  # 避免除以0
        cv.append(cv_j)
    cv = np.array(cv)
    
    # 步骤2：计算指标间的相关矩阵（rjk）
    corr_matrix = np.zeros((m, m))
    for j in range(m):
        for k in range(m):
            r, _ = pearsonr(X[:, j], X[:, k])
            corr_matrix[j, k] = r  # rjk为指标j和k的相关系数
    
    # 步骤3：计算每个指标的冲突性（Cj = sum(1 - rjk)）
    conflict = np.sum(1 - corr_matrix, axis=1)  # 按行求和（与所有其他指标的冲突性）
    
    # 步骤4：计算信息量（Ij = CVj * Cj）和权重（wj）
    information = cv * conflict
    weight = information / np.sum(information)  # 权重归一化（总和=1）
    
    # 输出结果（便于验证）
    print("=== CRITIC法权重结果 ===")
    for col, w, cv_j, c_j, info_j in zip(cols, weight, cv, conflict, information):
        indicator_name = col.replace('_positive_winsor_std', '')
        print(f"{indicator_name}：变异系数={cv_j:.4f}，冲突性={c_j:.4f}，信息量={info_j:.4f}，权重={w:.4f}")
    print(f"权重总和：{np.sum(weight):.4f}")
    return weight

# 计算CRITIC权重
weights = critic_weight(df, std_cols)

# ===================== 4. 计算综合得分 all_score =====================
df['all_score'] = np.dot(df[std_cols].values, weights)

# ===================== 5. 保存结果 & 验证 =====================
output_path = 'D:/校务/Projects/problematic-papers/results/paper_rates_newest.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

# 结果统计（验证极端值保留+区分度）
print("\n=== 综合得分统计（极端偏态适配） ===")
print(f"样本总数：{len(df)}（无丢失，保留所有极端样本）")
print(f"all_score最大值：{df['all_score'].max():.4f}（质量最高）")
print(f"all_score最小值：{df['all_score'].min():.4f}（质量最低，极端高扣分样本）")
print(f"all_score标准差：{df['all_score'].std():.4f}（标准差越大，区分度越好）")

# 查看极端高扣分样本的得分（原始指标值大→正向值小→all_score小）
print("\n极端高扣分样本（前5行，原始指标值最大）：")
high_defect_samples = df.sort_values(by=indicator_cols[0], ascending=False).head(5)
print(high_defect_samples[indicator_cols + ['all_score']])

print(f"\n新文件已保存至：{output_path}")