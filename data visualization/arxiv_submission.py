import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告

# -------------------------- 1. 数据读取与清洗 --------------------------
# 读取CSV文件（请确保文件路径正确）
df = pd.read_csv("D:\校务\images\get_monthly_submissions.csv", encoding='gbk')


# 日期格式转换（适配“1992/5/1”这类“年份-月份”格式）
def parse_date(str):

    year_str, month_str, date_str = str.split('/')
    return f"{year_str}-{month_str}-{date_str}"  
# 应用日期转换并验证
df['Date'] = df['month'].apply(parse_date)
df['Date'] = pd.to_datetime(df['Date'])  # 转为datetime类型便于排序
df = df.sort_values('Date').reset_index(drop=True)  # 按日期排序

# 检查数据完整性（确保覆盖Jul-91至Jan26）
print("数据时间范围：")
print(f"起始日期：{df['Date'].min().strftime('%Y-%m')}")
print(f"结束日期：{df['Date'].max().strftime('%Y-%m')}")
print(f"总数据量：{len(df)} 个月")

# -------------------------- 2. 莫兰蒂蓝色柱状图绘制 --------------------------
# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 定义莫兰蒂蓝色（低饱和柔和蓝，符合莫兰蒂风格）
morandi_blue = '#7B9E87'

# 创建高清画布（16x8英寸，300dpi确保清晰度）
fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

# 绘制柱状图
bars = ax.bar(
    df['Date'],  # x轴：日期
    df['submissions'],  # y轴：投稿量
    color=morandi_blue,  # 莫兰蒂蓝色
    alpha=0.8,  # 轻微透明（增强视觉层次）
    width=20,  # 柱子宽度（适配月份数据，避免过密）
    edgecolor='#5A7A67',  # 柱子边框色（深色版莫兰蒂蓝，增强轮廓）
    linewidth=0.5
)

# -------------------------- 3. 图表美化（提升可读性） --------------------------
ax.set_ylabel(
    'Number of Submissions',
    fontsize=14,
    fontweight='medium',
    labelpad=12,
    color='#4A5A50'
)

# 优化x轴（避免日期重叠）
ax.xaxis.set_major_locator(mdates.YearLocator(2))  # 每2年显示一个刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 仅显示年份
plt.xticks(rotation=45, ha='right', fontsize=10)  # 这里的fontsize是plt.xticks的合法参数

# 优化y轴（修复核心：修正tick_params参数名）
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # 只显示整数刻度
ax.tick_params(
    axis='y',
    labelsize=10,        # 替换fontsize → labelsize（刻度标签字体大小）
    labelcolor='#4A5A50',# 替换colors → labelcolor（刻度标签颜色）
    color='#8FA89B'      # 可选：设置刻度线颜色（与淡蓝色适配）
)

# 添加网格线（辅助读数，不干扰主体）
ax.grid(
    axis='y',
    alpha=0.3,
    linestyle='--',
    color='#CBD8D0',  # 网格线也适配淡蓝色调
    linewidth=0.8
)
ax.set_axisbelow(True)  # 网格线置于柱子下方

# 其他美化代码（如标题、x轴标签等，按需补充）
ax.set_title('Monthly Submissions Trend (Jul 1991 - Jan 2026)', fontsize=18, fontweight='bold', pad=20, color='#3A4A40')
ax.set_xlabel('Month', fontsize=14, fontweight='medium', labelpad=12, color='#4A5A50')

plt.tight_layout()

# 保存高清图片（可直接下载使用）
plt.savefig(
    'D:/校务/images/monthly_submissions_trend.png',
    dpi=300,
    bbox_inches='tight',  # 完整保存所有元素
    facecolor='white'  # 白色背景，适配打印
)
plt.close()

print("\n图表已保存为：monthly_submissions_trend.png")
print("代码运行完成！")