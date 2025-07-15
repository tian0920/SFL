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

def main():
    # 定义参数
    # datasets_name = ['tiny_imagenet', ]
    # datasets_name = ['cifar10', 'cifar100', 'cinic10', ]  # 'svhn', 'fmnist', 'mnist',
    datasets_name = ['mnist', 'fmnist', 'svhn', ]

    # methods = ['fedpac', 'fedfed', 'fedproto', ]  #  'fedavg', 'local', 'fedproto', 'fedpac', 'fedfed',
    # methods = [ 'feddpa', 'fedala', 'fedah', 'metafed', 'fedpac', 'fedfed', 'fedproto', ]
    methods = ['sflas']

    # SeqFedEDT: 'floco', 'fedavg', 'local', 'fedper', 'apfl', 'lgfedavg', 'fedrep', 'pfedfda', 'fedrod', 'fedproto',
    # SeqFedRPC: 'fedavg', 'local', 'sfl', 'cfl', 'feddyn', 'fedfomo', 'fedper', 'fedrep', 'pfedsim', 'lgfedavg', 'flute',

    # for logging
    log_dir = Path("test_experiment")
    log_dir.mkdir(exist_ok=True)

    # 遍历每个方法和数据集的组合
    for method in methods:
        for dataset in datasets_name:
            if dataset in ['tiny_imagenet']:
                command = [
                    sys.executable,
                    'main.py',
                    f'method={method}',
                    f'dataset.name={dataset}',
                    'model.name=vgg11',
                ]
            else:
                command = [
                    sys.executable,
                    'main.py',
                    f'method={method}',
                    f'dataset.name={dataset}',
                ]
            # logging
            log_filename = f"{method}_{dataset}.log"
            log_path = log_dir / log_filename

            print(f"运行命令: {' '.join(command)}")
            run_command(command, log_path)

    print("\n所有实验已完成。")


if __name__ == '__main__':
    main()

