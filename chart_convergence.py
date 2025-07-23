import pandas as pd
import matplotlib.pyplot as plt
import os, matplotlib

# 设置全局字体和样式
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

matplotlib.rcParams['axes.linewidth'] = 1  # 线型粗细为 1
matplotlib.rcParams['axes.grid'] = True  # 启用网格

# 假设最外层的目录
root_directory = 'test_experiment/baseline/alpha=0.1/convergence'  # 请替换为实际的根目录路径

# 准备绘图数据
data = {}

# 遍历所有方法文件夹（第二层文件夹）
for method in os.listdir(root_directory):
    method_path = os.path.join(root_directory, method)

    if os.path.isdir(method_path):
        # 遍历每个数据集文件夹（第三层文件夹）
        for dataset in os.listdir(method_path):
            dataset_path = os.path.join(method_path, dataset)

            if os.path.isdir(dataset_path):
                # 找到存放csv文件的子文件夹
                csv_folder = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

                for subfolder in csv_folder:
                    # 构造CSV文件路径
                    csv_file_path = os.path.join(dataset_path, subfolder)
                    # 在子文件夹中找到csv文件
                    csv_files = [f for f in os.listdir(csv_file_path) if f.endswith('.csv')]

                    for csv_file in csv_files:
                        # 读取CSV文件
                        df = pd.read_csv(os.path.join(csv_file_path, csv_file))

                        # 确保dataset在data字典中存在
                        dataset_upper = dataset.upper()  # 将数据集名称转换为大写
                        if dataset_upper not in data:
                            data[dataset_upper] = {}

                        if dataset_upper == 'MEDMNISTA' or dataset_upper == 'MEDMNISTC':
                            # Calculate the moving average (MA), window size of 3
                            ma_before = df['accuracy_test_before'].rolling(window=70).mean()
                            ma_after = df['accuracy_test_after'].rolling(window=70).mean()
                        else:
                            ma_before = df['accuracy_test_before'].rolling(window=15).mean()
                            ma_after = df['accuracy_test_after'].rolling(window=15).mean()

                        # 将method转为小写，并存储数据
                        method_lower = method.lower()
                        if method_lower not in data[dataset_upper]:
                            data[dataset_upper][method_lower] = {
                                'epoch': df['epoch'],
                                'test_before': ma_before,
                                'test_after': ma_after,
                            }

# 按照字母顺序排序数据集
datasets = sorted(data.keys())

method_color_map = {
    'fedavg': 'saddlebrown',
    'local': 'g',
    'fedah': 'gray',
    'fedala': 'c',
    'fedas': 'lightsteelblue',
    'feddpa': 'y',
    'fedfed': 'brown',
    'fedpac': 'orange',
    'fedproto': 'purple',
    'fedrod': 'b',
    'ours': 'r',
    # 'psfl+obp': 'r',
    # 'sfl': 'rosybrown',
    # 'fedproto': 'teal',
}

linewidth = 1

# 创建子图，假设最多有8个数据集
fig, axes = plt.subplots(1, 4, figsize=(28, 6))

# 迭代每个数据集
for i, dataset in enumerate(datasets):
    # ax = axes[i // 4, i % 4]  # 计算子图的位置,适合多行
    ax = axes[i]

    # 绘制每个方法的test_before
    for method, values in data[dataset].items():
        color = method_color_map.get(method, '#000000')  # 如果方法没有定义颜色，默认使用黑色
        if method == 'fedavg':
            ax.plot(values['epoch'], values['test_before'], label='FedAvg', linewidth=linewidth, color=color)
        if method == 'local':
            ax.plot(values['epoch'], values['test_before'], label='Local-Only', linewidth=linewidth, color=color)
        if method == 'fedah':
            ax.plot(values['epoch'], values['test_before'], label='FedAH', linewidth=linewidth, color=color)
        if method == 'fedala':
            ax.plot(values['epoch'], values['test_before'], label='FedALA', linewidth=linewidth, color=color)
        if method == 'fedas':
            ax.plot(values['epoch'], values['test_before'], label='FedAS', linewidth=linewidth, color=color)
        if method == 'fedrod':
            ax.plot(values['epoch'], values['test_before'], label='FedRoD', linewidth=linewidth, color=color)
        if method == 'feddpa':
            ax.plot(values['epoch'], values['test_before'], label='FedDPA', linewidth=linewidth, color=color)
        if method == 'fedfed':
            ax.plot(values['epoch'], values['test_before'], label='FedFed', linewidth=linewidth, color=color)
        if method == 'fedpac':
            ax.plot(values['epoch'], values['test_before'], label='FedPAC', linewidth=linewidth, color=color)
        if method == 'fedproto':
            ax.plot(values['epoch'], values['test_before'], label='FedProto', linewidth=linewidth, color=color)
        if method == 'ours':
            ax.plot(values['epoch'], values['test_before'], label='Ours', linewidth=linewidth + 0.6, color=color)
        # if method == 'psfl+fisher':
        #     ax.plot(values['epoch'], values['test_before'], label='S.FedEDT+Fisher', linewidth=linewidth + 0.6, color=color)
        # if method == 'psfl+obp':
        #     ax.plot(values['epoch'], values['test_before'], label='S.FedEDT+OBD', linewidth=linewidth + 0.6, color=color)
        # if method == 'sfl':
        #     ax.plot(values['epoch'], values['test_before'], label='SFL', linewidth=linewidth, color=color)

    # 设置y轴从40开始（仅针对指定的数据集）
    if dataset in ['EMNIST', 'FMNIST', ]:
        ax.set_ylim(bottom=70)
    if dataset in ['MNIST']:
        ax.set_ylim(bottom=90)
    if dataset in ['SVHN']:
        ax.set_ylim(bottom=65)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.5)

    # 设置图表标题和标签
    ax.set_title(f'{dataset}', fontsize=14)  # 设置为大写字母
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)

    # 排序并显示图例（按字母顺序排列方法）
    handles, labels = ax.get_legend_handles_labels()
    # 按照标签（method）字母顺序排序
    sorted_handles_labels = sorted(zip(labels, handles))
    sorted_labels, sorted_handles = zip(*sorted_handles_labels)
    ax.legend(sorted_handles, sorted_labels, ncol=2, fontsize=9, )

# 调整布局
plt.tight_layout()

# 确保图像保存的文件夹存在
output_dir = 'chart'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存图像到文件夹
output_path = os.path.join(output_dir, 'convergence_0.1.pdf')
plt.savefig(output_path)

# 显示图形
plt.show()
