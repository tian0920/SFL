import os
import re
import numpy as np
from collections import defaultdict

# 指定log文件夹路径
log_dir = "test_experiment/FedOBP/res18_baseline"  # /baseline/alpha=0.5
# log_dir = "test_experiment/FedPDAv2/hyperparam"

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
        # if name_parts[2] == "0":
        #     # 读取文件内容
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
    val_map = {}  # 存放每个 dataset 的 float 值，方便后续加粗

    # 先提取所有有效 float 值
    for dataset in datasets:
        val_str = data[method].get(dataset, "N/A")
        try:
            val = float(val_str)
            values.append(val)
            val_map[dataset] = val
        except:
            val_map[dataset] = "N/A"

    # 找出最大值（本行中）
    max_val = max(values) if values else None

    # 填写表格内容，并加粗最大值
    for dataset in datasets:
        val = val_map[dataset]
        if isinstance(val, float) and val == max_val:
            row.append(f"**{val:.3f}**")
        elif isinstance(val, float):
            row.append(f"{val:.3f}")
        else:
            row.append("N/A")

    # # 计算均值和标准差
    # if values:
    #     mean = np.mean(values)
    #     std = np.std(values)
    #     row.append(f"{mean:.2f}")
    #     row.append(f"{std:.2f}")
    # else:
    #     row.append("N/A")
    #     row.append("N/A")

    rows.append("| " + " | ".join(row) + " |")

# 输出Markdown表格
markdown_table = "\n".join(rows)
print(markdown_table)
