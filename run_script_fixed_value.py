# Comparison of Importance Scores for FedFew (改为手动配置 dataset-param 映射)

import subprocess
import sys
from pathlib import Path


def run_command(command, log_file):
    """
    运行命令并将输出写入日志文件。
    """
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            try:
                decoded_line = line.decode('utf-8')
            except UnicodeDecodeError:
                decoded_line = line.decode('utf-8', errors='ignore')
            print(decoded_line, end='')  # 实时输出到控制台
            f.write(decoded_line)
        process.wait()
        if process.returncode != 0:
            print(f"命令失败，查看日志文件: {log_file}")
        else:
            print(f"命令成功完成，日志文件: {log_file}")


def main():
    # 手动配置：每个 dataset 对应一个 ig_ratio（或其他参数）
    #### alpha=0.1
    obp_dataset_map = {'medmnistA': 0.99, 'medmnistC': 0.9}
    diff_dataset_map = {'medmnistA': 0.9998, 'medmnistC': 0.9993,} # 'medmnistC': 0.9993,
    fisher_dataset_map = {'medmnistC': 0.1, 'medmnistA': 0} #
    # fisher_dataset_map = {'cifar10': 0.999, 'cifar100': 0.9991, 'emnist': 0.9, 'fmnist': 0.99, 'mnist': 0.99, 'svhn': 0.99} #


    #### alpha=0.5
    # obp_dataset_map = {'medmnistA': 0.99997, 'medmnistC': 0.99997}  #
    # diff_dataset_map = {'medmnistC': 0.99997}
    # fisher_dataset_map = {'medmnistA': 0}

    #### alpha=1.0
    # obp_dataset_map = {'cifar10': 0.9999, 'cifar100': 0.9991, 'emnist': 0.99, 'fmnist': 0.999, 'mnist': 0.99, 'svhn': 0.9995}
    # diff_dataset_map = {'cifar10': 0.9993, 'cifar100': 0.9991, 'emnist': 0.993, 'fmnist': 0.9993, 'mnist': 0.99993, 'svhn': 0.999999}
    # obp_dataset_map = {'medmnistA': 0.99, 'medmnistC': 0.9993}
    # diff_dataset_map = {'medmnistA': 0.99, 'medmnistC': 0.9993}


    method_list = ['psfl',] # 'feddpa',
    score_list = ['diff', 'obp'] # 'obp'
    alpha = 0.0

    # 创建日志目录
    log_dir = Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)

    # 遍历 dataset-param 对
    for method in method_list:
        # 构建命令
        if method == 'feddpa':
            for dataset, ig_ratio in fisher_dataset_map.items():
                command = [
                    sys.executable,
                    'main.py',
                    f'method={method}',
                    f'dataset.name={dataset}',
                    f'{method}.fisher_threshold={ig_ratio}',
                    # 'model.name=lenet5',
                ]
                # 构建日志路径
                log_filename = f"psfl+fisher_{dataset}.log"
                log_path = log_dir / log_filename
                print(f"运行命令: {' '.join(command)}")
                run_command(command, log_path)

        if method == 'psfl':
            for score in score_list:
                if score == 'diff':
                    for dataset, ig_ratio in diff_dataset_map.items():
                        command = [
                            sys.executable,
                            'main.py',
                            f'method={method}',
                            f'dataset.name={dataset}',
                            f'{method}.ig_ratio={ig_ratio}',
                            f'{method}.alpha={alpha}',
                            f'{method}.score={score}',
                        ]

                        if dataset in ['medmnistA', 'medmnistC']:
                            command.append('model.name=lenet5')
                            command.append('common.join_ratio=0.5')

                        # 构建日志路径
                        log_filename = f"{method}+{score}_{dataset}.log"
                        log_path = log_dir / log_filename
                        print(f"运行命令: {' '.join(command)}")
                        run_command(command, log_path)

                if score == 'obp':
                    for dataset, ig_ratio in obp_dataset_map.items():
                        command = [
                            sys.executable,
                            'main.py',
                            f'method={method}',
                            f'dataset.name={dataset}',
                            f'{method}.ig_ratio={ig_ratio}',
                            f'{method}.alpha={alpha}',
                            f'{method}.score={score}',
                        ]

                        if dataset in ['medmnistA', 'medmnistC']:
                            command.append('model.name=lenet5')
                            command.append('common.join_ratio=0.5')

                        # 构建日志路径
                        log_filename = f"{method}+{score}_{dataset}.log"
                        log_path = log_dir / log_filename
                        print(f"运行命令: {' '.join(command)}")
                        run_command(command, log_path)

    print("\n所有实验已完成。")


if __name__ == '__main__':
    main()
