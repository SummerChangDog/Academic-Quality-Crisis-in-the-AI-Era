import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. 基础设置
# --------------------------
# 设置中文字体
plt.rcParams.update({
    'font.sans-serif': ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.antialiased': True,
    'lines.antialiased': True,
    'patch.antialiased': True
})

# --------------------------
# 2. 数据读取与预处理
# --------------------------
# 读取CSV文件（请替换为您的文件路径）
df = pd.read_csv("D:/校务/Projects/problematic-papers/results/paper_rates_newest.csv")

# 定义质量指标列和中文标签映射
quality_cols = ['empirical_clarity_score', 'explanation_vs_speculation_score', 
                'language_misuse_score', 'math_quality_score']

cn_labels = {
    'empirical_clarity_score': '实证清晰度',
    'explanation_vs_speculation_score': '解释-推测区分度',
    'language_misuse_score': '语言规范性',
    'math_quality_score': '数学合理性'
}

# 按年份和领域分组计算均值
yearly_mean = df.groupby(['year', 'domain'])[quality_cols].mean().reset_index()

# --------------------------
# 3. 绘图参数设置
# --------------------------
# 莫兰蒂色系
colors = {
    'math': '#8FAADC',    # 莫兰蒂蓝
    'cs.ai': '#E8DCCA'    # 莫兰蒂黄
}

# 创建2x2分面折线图
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300, sharex=True)
axes = axes.flatten()  # 转换为1D数组便于循环

# --------------------------
# 4. 绘制各子图
# --------------------------
for i, indicator in enumerate(quality_cols):
    ax = axes[i]
    
    # 绘制math领域折线
    math_data = yearly_mean[yearly_mean['domain'] == 'math'].sort_values('year')
    ax.plot(math_data['year'], math_data[indicator], 
            color=colors['math'], linewidth=4, marker='o', markersize=8, 
            label='Math', markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors['math'])
    
    # 绘制cs.ai领域折线
    ai_data = yearly_mean[yearly_mean['domain'] == 'cs.ai'].sort_values('year')
    ax.plot(ai_data['year'], ai_data[indicator], 
            color=colors['cs.ai'], linewidth=4, marker='s', markersize=8, 
            label='CS.AI', markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors['cs.ai'])
    
    # 设置子图标题和标签
    ax.set_title(cn_labels[indicator], fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('得分', fontsize=14, fontweight='bold')
    
    # 设置网格和边框
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#333333')
    
    # 设置图例
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9, 
              facecolor='white', edgecolor='gray', frameon=True)
    
    # 自适应Y轴范围
    all_data = pd.concat([math_data[indicator], ai_data[indicator]])
    y_min = max(0, all_data.min() - 0.5)
    y_max = all_data.max() + 0.5
    ax.set_ylim(y_min, y_max)
    
    # 设置刻度字体大小
    ax.tick_params(axis='y', labelsize=12)

# --------------------------
# 5. 统一设置X轴
# --------------------------
for ax in axes:
    ax.set_xlim(2016.5, 2026.5)
    ax.set_xticks(range(2017, 2027))
    ax.set_xticklabels(range(2017, 2027), rotation=45, ha='right', fontsize=12)

# 只为最下方子图添加X轴标签
axes[2].set_xlabel('年份', fontsize=16, fontweight='bold', labelpad=15)
axes[3].set_xlabel('年份', fontsize=16, fontweight='bold', labelpad=15)

# --------------------------
# 6. 调整布局并保存
# --------------------------
plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
plt.savefig('quality_indicators_faceted_plot.png', 
            dpi=300, bbox_inches='tight', facecolor='white', 
            edgecolor='none', pil_kwargs={'compression': 0})
plt.close()

print("✅ 2x2分面折线图已成功生成并保存！")