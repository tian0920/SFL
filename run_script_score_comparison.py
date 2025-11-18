import subprocess
import sys
from pathlib import Path
import numpy as np


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


def build_command(method, dataset, align, proto, tem, score=None):
    """
    根据参数构建命令
    """
    # if dataset == 'cifar10':
    #     return [
    #         sys.executable,
    #         'main.py',
    #         f'method={method}',
    #         f'dataset.name={dataset}',
    #         f'{method}.lambda_align={align}',
    #         # f'{method}.temperature=0.2',
    #         f'{method}.lambda_proto={proto}',
    #     ]
    # elif dataset == 'cifar100':
    #     return [
    #         sys.executable,
    #         'main.py',
    #         f'method={method}',
    #         f'dataset.name={dataset}',
    #         f'{method}.lambda_align={align}',
    #         # f'{method}.temperature=0.2',
    #         f'{method}.lambda_proto={proto}',
    #     ]
    # elif dataset == 'fmnist':
    #     return [
    #         sys.executable,
    #         'main.py',
    #         f'method={method}',
    #         f'dataset.name={dataset}',
    #         f'{method}.lambda_align={align}',
    #         # f'{method}.temperature=0.2',
    #         f'{method}.lambda_proto={proto}',
    #     ]
    if method in ['fedobp']:
        return [
            sys.executable,
            'main.py',
            f'method={method}',
            f'dataset.name={dataset}',
            f'model.name=res18',
            f'common.global_epoch=400',
            f'{method}.ig_ratio={align}',]
    else:
        return [
            sys.executable,
            'main.py',
            f'method={method}',
            f'dataset.name={dataset}',
            # f'{method}.lambda_align={align}',
            f'{method}.gen_mult={align}',
            f'{method}.temperature={tem}',
            f'{method}.lambda_proto={proto}',
        ]

def build_log_filename(method, dataset, align, proto, tem, score=None):
    """
    根据实验参数构建日志文件名
    """
    if method == 'psfl':
        return f"{method}+{score}_{dataset}_{align}.log"
    else:
        # return f"psfl+fisher_{dataset}_{ig_ratio}.log"
        return f"nonlinear_{method}_{dataset}_{align}_{proto}_{tem}.log"
        # return f"{method}_{dataset}_{align}_{proto}.log"

def should_skip(log_path):
    """
    判断是否应该跳过当前实验：若日志文件存在并包含特定内容，则跳过。
    """
    if log_path.exists():
        content = log_path.read_text(encoding='utf-8', errors='ignore')
        if "(test) before fine-tuning:" in content:
            return True
    return False


def main():
    # 定义参数
    datasets_name = ['fmnist',] # 'cifar10', 'cifar100', 'svhn', 'fmnist', 'medmnistC', 'mnist', 'emnist'
    # aligns = np.round(np.arange(0.1, 1, 0.2), 2).tolist()
    # aligns = [] # 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
    protos = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]  # 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 3, 4, 5
    tems = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 3, 5, 7, 9]
    aligns= [1, 2, 3]  # gen_mult: nonlinear( 2, 3)
    # tem: 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 3, 5
    # pro: 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 3, 4, 5,
    # epoch: 1, 2, 3, 4, 5,
    # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999
    methods = ['fedpdav2'] #  'feddpa', 'psfl',
    alpha = [0.0]
    score_list = ['obp',]

    # 创建一个目录来保存所有日志
    log_dir = Path("CSIS/nonlinear")  # /FedPDAv2/hyperparam
    log_dir.mkdir(exist_ok=True)

    # 遍历所有组合并运行实验
    for dataset in datasets_name:
        for method in methods:
            for align in aligns:
                for proto in protos:
                    for tem in tems:
                        # if method == 'psfl':
                        #     for score in score_list:
                        #         command = build_command(method, dataset, ig_ratio, alpha_tmp, score)
                        #         log_filename = build_log_filename(method, dataset, ig_ratio, score)
                        #         log_path = log_dir / log_filename
                        #         print(f"运行命令: {' '.join(command)}")
                        #         run_command(command, log_path)
                        # else:
                            log_filename = build_log_filename(method, dataset, align, proto, tem)
                            log_path = log_dir / log_filename

                            # 如果日志文件已存在并包含指定信息，跳过
                            if should_skip(log_path):
                                print(f"跳过已完成的实验日志: {log_filename}")
                                continue

                            command = build_command(method, dataset, align, proto, tem)
                            print(f"运行命令: {' '.join(command)}")
                            run_command(command, log_path)

    print("\n所有实验已完成。")


if __name__ == '__main__':
    main()
