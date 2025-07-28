###### 用于寻找convergence的目标文件，若有许多空main.log文件，则删除对应文件夹

import os
import re
import shutil

# 设置根目录
root_dir = 'D://SFL//out//fedpdav2//cifar100'

# 遍历子文件夹
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        log_file = None

        # 寻找log文件
        for file in os.listdir(folder_path):
            if file.endswith('.log'):
                log_file = os.path.join(folder_path, file)
                break

        if not log_file:
            continue  # 没有log文件跳过

        with open(log_file, 'r') as f:
            content = f.read()

        # 匹配alpha和before fine-tuning值
        alpha_match = re.search(r"'alpha'\s*:\s*([0-9.]+)", content)
        before_match = re.search(r"\(test\) before fine-tuning:\s*([0-9.]+)%\s+at epoch", content)

        if before_match:
            if alpha_match:
                alpha = alpha_match.group(1)
            else:
                alpha = 'NA'

            score = before_match.group(1)
            new_name = f"{alpha}_{score}"
            new_path = os.path.join(root_dir, new_name)

            # 重命名文件夹（防止同名冲突）
            if not os.path.exists(new_path):
                os.rename(folder_path, new_path)
                print(f"Renamed: {folder} -> {new_name}")
            else:
                print(f"Skipped (name exists): {new_name}")
        else:
            # 没找到before fine-tuning，删除文件夹
            shutil.rmtree(folder_path)
            print(f"Deleted folder (no fine-tune info): {folder}")

