import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

# 设置全局字体和样式
matplotlib.rcParams['axes.grid'] = True  # 启用网格
matplotlib.rcParams['grid.alpha'] = 0.3  # 网格透明度设置为 0.3

# 假设你有一个目录包含所有CSV文件
directory = 'test_experiment/ablation'

# 读取所有CSV文件
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# 准备绘图数据
data = {}

# 遍历所有CSV文件并读取数据
for file in csv_files:
    # 提取 method 和 dataset，去掉.csv后缀
    method, dataset_with_extension = file.split('_')[:2]  # 假设文件名格式是 method_dataset.csv
    dataset = dataset_with_extension.replace('.csv', '')  # 去掉 .csv 后缀

    # 读取CSV文件
    df = pd.read_csv(os.path.join(directory, file))

    # 如果dataset不存在于data字典中，初始化
    if dataset not in data:
        data[dataset] = {}

    # 如果method不存在于dataset的字典中，初始化
    if method not in data[dataset]:
        if dataset == 'medmnistA' or dataset == 'medmnistC':
            # Calculate the moving average (MA), window size of 3
            ma_before = df['accuracy_test_before'].rolling(window=25).mean()
            ma_after = df['accuracy_test_before'].rolling(window=25).mean()
        else:
            ma_before = df['accuracy_test_before'].rolling(window=5).mean()
            ma_after = df['accuracy_test_before'].rolling(window=5).mean()

        data[dataset][method] = {
            'epoch': df['epoch'],
            'test_before': ma_before,
            'test_after': ma_after
        }


# 按照字母顺序排序数据集，并转换为大写字母
datasets = sorted(data.keys(), key=lambda x: x.upper())  # 字母排序并转换为大写

# 自定义配色方案：为每个方法分配固定的颜色
method_color_map = {
    'None': '#f032e6',  # 紫色
    'PCA': '#3cb44b',  # 绿色
    'PFA': '#4363d8', # 水蓝色
    'ALL': '#e6194B',  # 红色
    'Non-Linear': '#f58231',  # 橙色
}

# 创建子图 2 行 4 列，5:3的比例
# fig, axes = plt.subplots(1, 1, figsize=(8, 5))
linewidth = 1.5

# 迭代每个数据集（假设有8个不同的数据集）
for i, dataset in enumerate(datasets):
    # ax = axes[i // 4, i % 4]  # 计算子图的位置

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # 绘制每个方法的test_before，设置线宽为1
    for method, values in data[dataset].items():
        # 获取该方法的固定颜色
        color = method_color_map.get(method, '#000000')  # 如果方法没有定义颜色，默认使用黑色
        if method == 'None':
            ax.plot(values['epoch'], values['test_before'], label='None', linewidth=linewidth, color=color)  # 线条粗细设置为 1
        if method == 'PCA':
            ax.plot(values['epoch'], values['test_before'], label='PCA', linewidth=linewidth, color=color)  # 线条粗细设置为 1
        if method == 'PFA':
            ax.plot(values['epoch'], values['test_before'], label='PFA', linewidth=linewidth, color=color)  # 线条粗细设置为 1
        if method == 'ALL':
            ax.plot(values['epoch'], values['test_before'], label='ALL', linewidth=linewidth + 0.8, color=color)  # 线条粗细设置为 1
        if method == 'Non-Linear':
            ax.plot(values['epoch'], values['test_before'], label='PFA(Non)', linewidth=linewidth, color=color)  # 线条粗细设置为 1

    # 设置y轴从40开始（仅针对指定的数据集）
    if dataset in ['emnist', 'fmnist', 'svhn']:
        ax.set_ylim(bottom=70)

    if dataset in ['mnist', ]:
        ax.set_ylim(bottom=90)

    # 获取并排序图例（按字母顺序）
    handles, labels = ax.get_legend_handles_labels()
    # 按照标签（method）字母顺序排序
    sorted_handles_labels = sorted(zip(labels, handles))
    sorted_labels, sorted_handles = zip(*sorted_handles_labels)
    ax.legend(sorted_handles, sorted_labels, fontsize=18)

    # 设置图表标题和标签
    ax.set_title(f'{dataset.upper()}', fontsize=20)  # 设置为大写字母
    ax.set_xlabel('Epochs', fontsize=20)
    ax.set_ylabel('Accuracy (%)', fontsize=20)
    ax.tick_params(labelsize=16)

# 确保图像保存的文件夹存在
output_dir = 'chart'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.tight_layout()
# 调整子图之间的距离
plt.subplots_adjust(wspace=0.2, hspace=0.3)  # 调整宽度和高度的间距

# 保存图像到文件夹
output_path = os.path.join(output_dir, 'ablation.pdf')
plt.savefig(output_path)

# 显示图形
plt.show()
