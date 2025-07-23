import os
import re
import numpy as np
from collections import defaultdict

# 指定log文件夹路径
log_dir = "test_experiment/epoch_cifar100"  # /baseline/alpha=0.5

# 初始化存储结构：方法 → 数据集 → 数值
data = defaultdict(dict)
datasets = set()

# 遍历log文件夹
for filename in os.listdir(log_dir):
    if filename.endswith(".log"):
        # 提取方法名和数据集名
        name_parts = filename.replace(".log", "").split("_")
        # print(name_parts)
        if len(name_parts) < 2:
            continue  # 不符合命名规则跳过
        method, dataset = name_parts[0], name_parts[-1]
        datasets.add(dataset)

        # 读取文件内容
        with open(os.path.join(log_dir, filename), "r", encoding='utf-8') as f:
            content = f.read()

        # 正则提取百分比数值
        match = re.search(r"before fine-tuning:\s*([\d.]+)%", content)
        if match:
            value = match.group(1)
            data[method][dataset] = value
        else:
            data[method][dataset] = "N/A"  # 没有找到则标N/A

# 构建Markdown表格
datasets = sorted(datasets)
header = "| Method | " + " | ".join(datasets) + " | Mean | Std |"
separator = "|--------|" + "|".join(["--------"] * len(datasets)) + "|--------|--------|"

rows = [header, separator]
for method in sorted(data.keys()):
    row = [method]
    values = []
    for dataset in datasets:
        val_str = data[method].get(dataset, "N/A")
        row.append(val_str)
        try:
            val = float(val_str)
            values.append(val)
        except:
            continue  # Skip "N/A"
    # 计算均值和标准差
    if values:
        mean = np.mean(values)
        std = np.std(values)
        row.append(f"{mean:.2f}")
        row.append(f"{std:.2f}")
    else:
        row.append("N/A")
        row.append("N/A")
    rows.append("| " + " | ".join(row) + " |")

# 输出Markdown表格
markdown_table = "\n".join(rows)
print(markdown_table)