import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# -------------------------- 基础设置（莫兰蒂色+高清+中文） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示
plt.rcParams['figure.figsize'] = (12, 7)      # 画布尺寸
plt.rcParams['savefig.dpi'] = 600             # 高清保存
plt.rcParams['figure.dpi'] = 150              # 显示高清

# 莫兰蒂色系（双变量箱线图用）
morandi_blue = '#6D9EAF'       # math箱体颜色
morandi_yellow = '#E4D192'     # cs.ai箱体颜色（新增莫兰蒂黄）
morandi_deepblue = '#003366'   # 边框/中位数线颜色
morandi_orange = '#C09F80'     # 异常值颜色

# -------------------------- 1. 读取csv数据 --------------------------
csv_abs_path = r"D:/校务/Projects/problematic-papers/results/paper_rates.csv"
# 读取CSV文件
try:
    df = pd.read_csv(csv_abs_path, encoding='utf-8')
    print(f"✅ 成功读取CSV文件：{csv_abs_path}")
    print(f"📊 数据总行数：{len(df)}，列名：{list(df.columns)}")
except FileNotFoundError:
    print(f"❌ 错误：未找到CSV文件，请检查路径 → {csv_abs_path}")
    exit(1)  # 路径错误则退出程序
except Exception as e:
    print(f"❌ 读取CSV文件出错：{str(e)}")
    exit(1)

# -------------------------- 2. 绘制双变量按年箱线图 --------------------------
# 定义两个要对比的domain
target_domains = ['math', 'cs.ai']  
domain_names = {'math': '数学领域', 'cs.ai': 'AI领域'}  # 用于图表标注

# 获取两个领域共有的年份（保证横轴年份一致）
df_math = df[df['domain'] == 'math']
df_cs_ai = df[df['domain'] == 'cs.ai']
common_years = sorted(list(set(df_math['year'].unique()) & set(df_cs_ai['year'].unique())))
print(f"📅 两个领域共有的年份：{common_years}")

# 准备双变量箱线图数据
# 格式：[[math_2020, cs.ai_2020], [math_2021, cs.ai_2021], ...]
box_data = []
for year in common_years:
    math_data = df_math[df_math['year'] == year]['citation_ai_rate'].values
    cs_ai_data = df_cs_ai[df_cs_ai['year'] == year]['citation_ai_rate'].values
    box_data.extend([math_data, cs_ai_data])  # 每个年份添加两个领域的数据

# 创建画布
fig, ax = plt.subplots(figsize=(12, 7))

# 设置箱体位置：每个年份下两个箱体并列，间距0.3，组内间距0.1
box_width = 0.3
positions = []
for i, year in enumerate(common_years):
    # 每个年份的两个箱体位置：i*2+1, i*2+2（调整间距）
    positions.append(i * 2 + 1 - box_width/2)  # math位置
    positions.append(i * 2 + 1 + box_width/2)  # cs.ai位置

# 绘制双变量箱线图
box_plot = ax.boxplot(
    box_data,
    positions=positions,      # 自定义箱体位置
    labels=['']*len(positions),  # 先清空标签，后续手动设置
    patch_artist=True,       # 允许填充颜色
    widths=box_width,        # 箱体宽度
    showfliers=True,         # 显示异常值
    flierprops={             # 异常值样式（两个领域保持一致）
        'marker': 'o',
        'markerfacecolor': morandi_orange,
        'markeredgecolor': morandi_deepblue,
        'markersize': 6
    },
    medianprops={            # 中位数线样式
        'color': morandi_deepblue,
        'linewidth': 1.5
    },
    whiskerprops={           # 须线样式
        'color': morandi_deepblue,
        'linewidth': 1.2
    },
    capprops={               # 端线样式
        'color': morandi_deepblue,
        'linewidth': 1.2
    }
)

# 为不同领域的箱体设置不同颜色
for i, box in enumerate(box_plot['boxes']):
    if i % 2 == 0:  # 偶数索引：math领域 → 莫兰蒂蓝
        box.set_facecolor(morandi_blue)
        box.set_edgecolor(morandi_deepblue)
        box.set_linewidth(1.2)
    else:  # 奇数索引：cs.ai领域 → 莫兰蒂黄
        box.set_facecolor(morandi_yellow)
        box.set_edgecolor(morandi_deepblue)
        box.set_linewidth(1.2)

# 设置横轴标签（只在每个年份组的中间显示年份）
ax.set_xticks([i * 2 + 1 for i in range(len(common_years))])
ax.set_xticklabels(common_years, fontsize=11)

# 设置图表标注
ax.set_title(f'math与cs.ai领域不同年份引用AI率分布对比箱线图', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('年份', fontsize=12, labelpad=10)
ax.set_ylabel('AI率', fontsize=12, labelpad=10)
ax.set_ylim(-0.1, 2.1)  # 纵轴范围适配0-2的AI率
ax.grid(axis='y', alpha=0.3, linestyle='--', color='#CCCCCC')  # 横向网格线
ax.tick_params(axis='both', labelsize=11)

# 添加图例（区分两个领域）
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=morandi_blue, edgecolor=morandi_deepblue, label='math（数学领域）'),
    Patch(facecolor=morandi_yellow, edgecolor=morandi_deepblue, label='cs.ai（AI领域）')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

# 保存图片
plt.savefig(f'math_cs_ai_rate_boxplot_by_year.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()