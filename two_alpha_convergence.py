import pandas as pd
import matplotlib.pyplot as plt
import os, matplotlib

# 设置全局字体和样式（保留你的设置）
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['axes.grid'] = True

# 定义所有alpha路径
root_directories = {
    '0.1': 'test_experiment/baseline/alpha=0.1/convergence',
    '0.5': 'test_experiment/baseline/alpha=0.5/convergence'
}

data = {}

for alpha, root_directory in root_directories.items():
    for method in os.listdir(root_directory):
        method_path = os.path.join(root_directory, method)
        if os.path.isdir(method_path):
            for dataset in os.listdir(method_path):
                if dataset.upper() != 'CIFAR100':
                    continue  # 只处理 CIFAR100

                dataset_path = os.path.join(method_path, dataset)
                if os.path.isdir(dataset_path):
                    csv_folder = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

                    for subfolder in csv_folder:
                        csv_file_path = os.path.join(dataset_path, subfolder)
                        csv_files = [f for f in os.listdir(csv_file_path) if f.endswith('.csv')]

                        for csv_file in csv_files:
                            df = pd.read_csv(os.path.join(csv_file_path, csv_file))
                            ma_before = df['accuracy_test_before'].rolling(window=5).mean()

                            key = (dataset.upper(), alpha)
                            if key not in data:
                                data[key] = {}
                            method_lower = method.lower()
                            if method_lower not in data[key]:
                                data[key][method_lower] = {
                                    'epoch': df['epoch'],
                                    'test_before': ma_before
                                }

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 高度是宽度的60%

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
}

linewidth = 1.5
alphas = ['0.1', '0.5']
all_handles = []
all_labels = []

for i, alpha in enumerate(alphas):
    key = ('CIFAR100', alpha)
    ax = axes[i]
    ax.set_title(f'CIFAR100 ($\\alpha$ = {alpha})', fontsize=22)

    for method, values in data[key].items():
        color = method_color_map.get(method, '#000000')
        label_map = {
            'fedavg': 'FedAvg',
            'local': 'Local-Only',
            'fedah': 'FedAH',
            'fedala': 'FedALA',
            'fedas': 'FedAS',
            'fedrod': 'FedRoD',
            'feddpa': 'FedDPA',
            'fedfed': 'FedFed',
            'fedpac': 'FedPAC',
            'fedproto': 'FedProto',
            'ours': 'Ours'
        }
        label = label_map.get(method, method)

        lw = linewidth + 0.8 if method == 'ours' else linewidth
        line, = ax.plot(values['epoch'], values['test_before'], label=label, linewidth=lw, color=color)

        # 仅第一次添加 handle 和 label（避免重复）
        if label not in all_labels:
            all_handles.append(line)
            all_labels.append(label)

    ax.set_xlabel('Epochs', fontsize=18)
    if i == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=18)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.5)

# 添加统一图例
fig.legend(all_handles, all_labels, loc='upper center', ncol=5, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 留出顶部空间给图例
output_dir = 'chart'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'cifar100_convergence.pdf'))
plt.show()
