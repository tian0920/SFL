import os
import re, matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

# 假设所有 log 文件存放在一个目录下
log_directory = 'experiment_logs/alpha=0.1_1/join_ratio'

# 存放每个数据集的准确率和join_ratio
dataset_results = {}

# 遍历 log 文件并提取数据
for log_file in os.listdir(log_directory):
    if log_file.endswith('.log'):
        # 提取数据集名称和 join_ratio
        match = re.match(r'(\w+)_(\w+)_([\d.]+)_([\d.]+)\.log', log_file)
        if match:
            dataset_name = match.group(2).upper()  # 数据集名称大写
            join_ratio = round(float(match.group(4)), 1)  # 保留一位小数

            # 读取文件并提取准确率
            with open(os.path.join(log_directory, log_file), 'r', encoding='utf-8') as file:
                accuracy = None
                for line in file:
                    if '(test) before fine-tuning:' in line:
                        accuracy_match = re.search(r'(\d+\.\d+)%', line)
                        if accuracy_match:
                            accuracy = float(accuracy_match.group(1))
                            break

                if dataset_name not in dataset_results:
                    dataset_results[dataset_name] = {'join_ratio': [], 'accuracy': []}

                dataset_results[dataset_name]['join_ratio'].append(join_ratio)
                dataset_results[dataset_name]['accuracy'].append(accuracy)

# 创建一个图
plt.figure(figsize=(10, 6))

# 定义不同的颜色和线型
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

# 对数据集进行按字母排序
sorted_datasets = sorted(dataset_results.items(), key=lambda x: x[0])

# 绘制每个数据集的准确率 vs. join_ratio
for idx, (dataset_name, results) in enumerate(sorted_datasets):
    # 按照 join_ratio 排序
    sorted_join_ratios, sorted_accuracies = zip(*sorted(zip(results['join_ratio'], results['accuracy'])))

    # 使用不同的颜色和线型绘制折线图，并添加标记
    plt.plot(sorted_join_ratios, sorted_accuracies,
             marker='o' if idx % 2 == 0 else '^',  # 每个数据集使用不同的标记
             color=colors[idx],
             linestyle=linestyles[idx % len(linestyles)],  # 循环使用线型
             label=dataset_name)

# 添加标题和标签
# plt.title('Accuracy vs. Join Ratio for Different Datasets')
plt.xlabel('Join Ratio', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)

# 设置横轴刻度，仅显示实际出现的 join_ratio
join_ratios = sorted(set(join_ratio for results in dataset_results.values() for join_ratio in results['join_ratio']))
plt.xticks(join_ratios)

# 绘制网格
plt.grid(True)

# 添加图例
plt.legend()

# 确保图像保存的文件夹存在
output_dir = 'chart'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存图像到文件夹
output_path = os.path.join(output_dir, 'scalability_new.pdf')
plt.savefig(output_path)

# 显示图形
plt.tight_layout()
plt.show()
