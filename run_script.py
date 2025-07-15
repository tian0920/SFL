import subprocess
import sys
from pathlib import Path

# 客户端类别数（cc）字典
client_classes = {
    'cifar10': [4,5,6],
    'cifar100': [10, 20, 30, 40, 50, 60]
}

def run_command(command, log_file):
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
        return process.returncode == 0  # 返回是否成功

def main():
    methods = ['fedavg', 'sfl']
    log_dir = Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)

    for dataset, cc_values in client_classes.items():
        for cc in cc_values:
            if dataset == 'cifar10':
                oc_values = [1]  # list(range(0, cc))
            elif dataset == 'cifar100':
                oc_values = [5]  # [x for x in range(5, cc, 5)]
            else:
                continue

            for oc in oc_values:
                data_command = [
                    sys.executable,
                    "generate_data.py",
                    "-d", dataset,
                    "-cc", str(cc),
                    "-oc", str(oc),
                    "-cn", "100"
                ]

                data_log_path = log_dir / f"generate_{dataset}_cc{cc}_oc{oc}.log"
                success = run_command(data_command, data_log_path)

                if not success:
                    print(f"❌ 数据生成失败: {dataset} (cc={cc}, oc={oc})，跳过对应实验")
                    continue  # 跳过后续实验，直接进入下一个oc

                print(f"\n📌 数据划分已完成: {dataset} (cc={cc}, oc={oc})")

                for method in methods:
                    command = [
                        sys.executable,
                        'main.py',
                        f'method={method}',
                        f'dataset.name={dataset}'
                    ]

                    log_filename = f"{method}_{dataset}_cc{cc}_oc{oc}.log"
                    log_path = log_dir / log_filename

                    print(f"\n🚀 运行实验: {' '.join(command)}")
                    run_command(command, log_path)

    print("\n✅ 所有实验已完成。")

if __name__ == '__main__':
    main()
