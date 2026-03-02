import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# 1. 数据预处理（增加年份中心化，以2017为基准）
df = pd.read_csv("D:/校务/Projects/problematic-papers/results/paper_rates.csv")
# 将domain转为哑变量（math=0，cs.ai=1）
df['domain_ai'] = df['domain'].map({'math': 0, 'cs.ai': 1})
# 确保year是数值型
df['year'] = pd.to_numeric(df['year'])
# 年份中心化：以2017年为基准（2017=0，2018=1，…，2026=9）
df['year_centered'] = df['year'] - 2017  # 核心修正：中心化

# 2. 构建带交互的混合效应模型（改用中心化后的年份）
model = smf.mixedlm(
    formula='citation_ai_rate ~ domain_ai + year_centered + domain_ai:year_centered',  # 改用year_centered
    data=df,
    groups=df['year']  # 随机效应仍用原始year分组，不影响
)
result = model.fit()

# 3. 输出核心结果
print("混合效应模型核心结果（年份中心化，2017=0）：")
print(result.summary())

# 4. 提取关键系数和显著性
print("\n=== 核心解读 ===")
domain_coef = result.params['domain_ai']
domain_p = result.pvalues['domain_ai']
year_coef = result.params['year_centered']  # 对应中心化年份
year_p = result.pvalues['year_centered']
interact_coef = result.params['domain_ai:year_centered']
interact_p = result.pvalues['domain_ai:year_centered']

print(f"1. 领域主效应：系数={domain_coef:.4f}，p值={domain_p:.4f} → {'显著' if domain_p<0.05 else '不显著'}")
print(f"   解读：2017年（基准年），cs.ai的citation_ai_rate比math高{domain_coef:.4f}")
print(f"2. 年份主效应：系数={year_coef:.4f}，p值={year_p:.4f} → {'显著' if year_p<0.05 else '不显著'}")
print(f"   解读：math领域的citation_ai_rate每年变化{year_coef:.4f}（无显著变化）")
print(f"3. 领域×年份交互效应：系数={interact_coef:.4f}，p值={interact_p:.4f} → {'显著' if interact_p<0.05 else '不显著'}")
print(f"   解读：cs.ai相对math的优势每年扩大{interact_coef:.4f}")

# 5. 可视化（不变）
plt.rcParams['font.size'] = 12
yearly_mean = df.groupby(['year', 'domain'])['citation_ai_rate'].mean().reset_index()
ai_data = yearly_mean[yearly_mean['domain'] == 'cs.ai']
math_data = yearly_mean[yearly_mean['domain'] == 'math']

plt.figure(figsize=(10, 6))
plt.plot(ai_data['year'], ai_data['citation_ai_rate'], 'o-', label='cs.ai', color='blue', linewidth=2)
plt.plot(math_data['year'], math_data['citation_ai_rate'], 's-', label='math', color='orange', linewidth=2)
plt.xlabel('年份')
plt.ylabel('citation_ai_rate 均值')
plt.title('cs.ai vs math 引用率随年份的变化趋势')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("D:/校务/images/yearly_trend.png", dpi=300, bbox_inches='tight')
plt.show()