import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取Excel文件
# load the demographics
current_directory = os.getcwd()
new_directory = "/mnt/data/T7_Shield/CSVD"
os.chdir(new_directory)
updated_directory = os.getcwd()
file_path = "demo_CSVD.xls"
df = pd.read_excel(file_path)

# 打印出DataFrame的前几行以检查数据结构
print(df.head())

# 假设 'Group' 是包含分组信息的列名，'Age' 是包含年龄数据的列名
group_column = 'group'
age_column = 'age'

# 检查是否存在指定的列
if group_column not in df.columns or age_column not in df.columns:
    print(f"警告: 列 '{group_column}' 或 '{age_column}' 不存在于数据集中。")
else:
    # 分离数据到不同组
    groups = df[group_column].unique()

    # 创建一个新的图形
    plt.figure(figsize=(14, 8))

    # 使用循环来绘制每个组的直方图
    for group in groups:
        # 筛选出属于当前组的数据
        group_data = df[df[group_column] == group][age_column]

        # 绘制直方图
        sns.histplot(group_data, bins=30, kde=True, label=group, alpha=0.6)

    # 设置图表标题和轴标签
    plt.title('age Distribution by group', fontsize=16)
    plt.xlabel('age', fontsize=14)
    plt.ylabel('frequency', fontsize=14)

    # 添加图例
    plt.legend(title='groups')

    # 显示网格
    plt.grid(True, alpha=0.3, linestyle='--')

    # 保存图形到文件（可选）
    plt.savefig('age_distribution_by_group_histogram.png', dpi=300)

    # 显示图形
    plt.show()

    # 如果需要进行统计学分析，可以计算并打印每个组的描述性统计量
    for group in groups:
        group_data = df[df[group_column] == group][age_column]
        print(f"Statistics for {group}:")
        print(group_data.describe())

        import pandas as pd
import matplotlib.pyplot as plt

PLT_SHOW = plt.show()
import seaborn as sns

# 读取Excel文件
file_path = 'demo1126.xlsx'
df = pd.read_excel(file_path)

# 打印出DataFrame的前几行以检查数据结构
print(df.head())

# 假设 'Group' 是包含分组信息的列名，'Age' 是包含年龄数据的列名
group_column = 'group'
age_column = 'age'

# 检查是否存在指定的列
if group_column not in df.columns or age_column not in df.columns:
    print(f"警告: 列 '{group_column}' 或 '{age_column}' 不存在于数据集中。")
else:
    # 定义年龄区间
    age_bins = range(40, 91, 5)  # 从40岁到90岁，每隔5岁一个区间
    bin_labels = [f'{i}-{i+4}' for i in age_bins[:-1]]

    # 将年龄数据分段
    df['Age Bin'] = pd.cut(df[age_column], bins=age_bins, labels=bin_labels, right=False)

    # 使用FacetGrid创建分面网格
    g = sns.FacetGrid(df, col=group_column, height=6, aspect=0.75)
    g.map(sns.countplot, 'Age Bin', order=bin_labels)

    # 设置图表标题和轴标签
    g.fig.suptitle('Age Distribution by Group with Facet Grid', fontsize=16, y=1.05)
    g.set_axis_labels('Age Bin', 'Frequency (Count)')

    # 显示网格
    for ax in g.axes.flat:
        ax.grid(True, alpha=0.3, linestyle='--')

    # 保存图形到文件（可选）
    plt.savefig('age_distribution_by_group_facet_grid.png', dpi=300)

    # 显示图形

    # 如果需要进行统计学分析，可以计算并打印每个组的描述性统计量
    for group in df[group_column].unique():
        group_data = df[df[group_column] == group][age_column]
        print(f"Statistics for {group}:")
        print(group_data.describe())

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置与图片完全相同的样式参数
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3
})

# 加载数据（请替换为您的实际数据路径）
df = pd.read_excel('demo_CSVD.xls')

# 创建与图片相同的年龄分段
age_bins = np.arange(40, 90, 5)
bin_labels = [f'{i}-{i+4}' for i in age_bins[:-1]]
df['Age Bin'] = pd.cut(df['age'], bins=age_bins, labels=bin_labels, right=False)

# 创建三个子图（与图片布局完全一致）
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# 定义与图片相同的组顺序
groups = ['HC', 'CSVD']

# 绘制每个组的直方图（精确匹配原图样式）
for ax, group in zip(axes, groups):
    # 筛选组数据
    group_data = df[df['group'] == group]

    # 绘制直方图（使用相同的蓝色）
    sns.histplot(
        data=group_data,
        x='Age Bin',
        bins=len(bin_labels),
        color='#1f77b4',
        ax=ax,
        edgecolor='white',
        linewidth=0.5
    )

    # 设置与图片相同的标题格式
    ax.set_title(f'group={group}', fontsize=14, pad=10)

    # 设置与图片相同的X轴刻度
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')

    # 设置与图片相同的Y轴范围
    ax.set_ylim(0, 40)

    # 设置网格样式（与图片完全一致）
    ax.grid(True, linestyle='--', alpha=0.3)

# 设置共享的Y轴标签（位置与图片相同）
axes[0].set_ylabel('Frequency (Count)', fontsize=14, labelpad=10)

# 设置共享的X轴标签（位置与图片相同）
for ax in axes:
    ax.set_xlabel('Age Bin', fontsize=14, labelpad=10)

# 调整子图间距（匹配原图布局）
plt.tight_layout(pad=3.0)

# 保存图像（600DPI，与出版质量要求一致）
plt.savefig('age_distribution_histograms.png', dpi=600, bbox_inches='tight')
plt.show()