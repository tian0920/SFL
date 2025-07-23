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
                # 尝试用 UTF-8 解码
                decoded_line = line.decode('utf-8')
            except UnicodeDecodeError:
                # 如果解码失败，可以选择忽略或使用其他编码
                decoded_line = line.decode('utf-8', errors='ignore')
            print(decoded_line, end='')  # 实时输出到控制台
            f.write(decoded_line)
        process.wait()
        if process.returncode != 0:
            print(f"命令失败，查看日志文件: {log_file}")
        else:
            print(f"命令成功完成，日志文件: {log_file}")


def build_command(method, dataset, ig_ratio, alpha_tmp, score=None):
    """
    根据参数构建命令
    """
    if method == 'psfl':
        return [
            sys.executable,
            'main.py',
            f'method={method}',
            f'dataset.name={dataset}',
            f'{method}.ig_ratio={ig_ratio}',
            f'{method}.alpha={alpha_tmp}',
            f'{method}.score={score}',
            'model.name=lenet5',
            'common.join_ratio=0.5'
        ]
    else:
        # return [
        #     sys.executable,
        #     'main.py',
        #     f'method={method}',
        #     f'dataset.name={dataset}',
        #     f'{method}.fisher_threshold={ig_ratio}',
        #     'model.name=lenet5',
        # ]
        return [
            sys.executable,
            'main.py',
            f'method={method}',
            f'dataset.name={dataset}',
            f'{method}.headfinetune_epoch={ig_ratio}',
        ]


def build_log_filename(method, dataset, ig_ratio, score=None):
    """
    根据实验参数构建日志文件名
    """
    if method == 'psfl':
        return f"{method}+{score}_{dataset}_{ig_ratio}.log"
    else:
        # return f"psfl+fisher_{dataset}_{ig_ratio}.log"
        return f"fedas_{dataset}_epoch_{ig_ratio}.log"


def main():
    # 定义参数
    datasets_name = ['cifar100', ] # 'cifar10', 'cifar100', 'svhn', 'fmnist', 'medmnistC', 'mnist', 'emnist'
    ig_values = [1, 2, 3, 4, 5,] #
    # tem: 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
    # pro: 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 3, 4, 5,
    # epoch: 1, 2, 3, 4, 5,
    methods = ['fedas',] #  'feddpa', 'psfl',
    alpha = [0.0]
    score_list = ['obp', 'diff']

    # 创建一个目录来保存所有日志
    log_dir = Path("test_experiment")
    log_dir.mkdir(exist_ok=True)

    # 遍历所有组合并运行实验
    for dataset in datasets_name:
        for method in methods:
            for alpha_tmp in alpha:
                for ig_ratio in ig_values:
                    if method == 'psfl':
                        for score in score_list:
                            command = build_command(method, dataset, ig_ratio, alpha_tmp, score)
                            log_filename = build_log_filename(method, dataset, ig_ratio, score)
                            log_path = log_dir / log_filename
                            print(f"运行命令: {' '.join(command)}")
                            run_command(command, log_path)
                    else:
                        command = build_command(method, dataset, ig_ratio, alpha_tmp)
                        log_filename = build_log_filename(method, dataset, ig_ratio)
                        log_path = log_dir / log_filename
                        print(f"运行命令: {' '.join(command)}")
                        run_command(command, log_path)

    print("\n所有实验已完成。")


if __name__ == '__main__':
    main()
