import os
import re
from collections import defaultdict

# 新的log根目录
log_dir = "test_experiment/sflas"

# 数据结构：文件夹名 → log文件名 → 数值
data = defaultdict(dict)
log_names = set()

# 遍历子文件夹
for subfolder in os.listdir(log_dir):
    subfolder_path = os.path.join(log_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # 遍历子文件夹下的log文件
    for filename in os.listdir(subfolder_path):
        if filename.endswith(".log"):
            log_name = filename.replace(".log", "")
            log_name = log_name.split("_")[1]  # 提取最后一部分
            log_names.add(log_name)

            file_path = os.path.join(subfolder_path, filename)
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read()

            # 提取指标数值
            match = re.search(r"before fine-tuning:\s*([\d.]+)%", content)
            if match:
                value = match.group(1)
            else:
                value = "N/A"

            data[subfolder][log_name] = value

# 构建Markdown表格
log_names = sorted(log_names)
header = "| Folder | " + " | ".join(log_names) + " |"
separator = "|--------|" + "|".join(["--------"] * len(log_names)) + "|"

rows = [header, separator]
for folder in sorted(data.keys()):
    row = [folder]
    for log_name in log_names:
        row.append(data[folder].get(log_name, "N/A"))
    rows.append("| " + " | ".join(row) + " |")

# 输出Markdown表格
markdown_table = "\n".join(rows)
print(markdown_table)
