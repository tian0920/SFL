import os
import re
from collections import defaultdict

# 设置日志目录
log_dir = "infocom_experiment/motivation/down and upload"  # ← 修改为你的文件夹路径
# log_dir = "experiment_logs/alpha=1.0/psfl+obp"  # ← 修改为你的文件夹路径

# 存储结构：dataset → [(value, parameter)]
dataset_values = defaultdict(list)

# 遍历所有log文件
for filename in os.listdir(log_dir):
    if filename.endswith(".log"):
        # 提取数据集名和参数值
        match_name = re.match(r"pspfl\+diff_(\w+)_il_([\d.]+)_ig_([\d.]+)\.log", filename)
        if match_name:
            dataset = match_name.group(1)
            param = match_name.group(2)

            # 读取文件内容
            with open(os.path.join(log_dir, filename), "r", encoding='utf-8') as f:
                content = f.read()

            # 提取 before fine-tuning 的数值
            match_value = re.search(r"before fine-tuning:\s*([\d.]+)%", content)
            if match_value:
                value = float(match_value.group(1))
                dataset_values[dataset].append((value, param))

# 查找每个数据集最大值对应的参数
for dataset, values in dataset_values.items():
    max_value, best_param = max(values, key=lambda x: x[0])
    print(f"Dataset: {dataset} | Max Value: {max_value}% | Best Param: {best_param}")
