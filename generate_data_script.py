import subprocess
import sys

def run_command(command):
    """
    运行命令并将输出写入到日志文件（不写日志时传入 None）。
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in process.stdout:
        try:
            decoded_line = line.decode('utf-8')
        except UnicodeDecodeError:
            decoded_line = line.decode('utf-8', errors='ignore')
        print(decoded_line, end='')  # 实时输出到控制台
    process.wait()

    return process.returncode == 0  # 返回是否成功

def generate_data(cn, a, d_list):
    """
    根据输入的 cn, a, d 参数运行 generate_data.py 文件。
    """
    # 运行命令，传入参数 cn, a, d
    for d in d_list:
        data_command = [
            sys.executable,
            "generate_data.py",
            "-cn", str(cn),
            "-a", str(a),
            "-d", str(d)
        ]

        success = run_command(data_command)

        if not success:
            print(f"❌ 数据生成失败: (cn={cn}, a={a}, d={d})，跳过对应实验")

        print(f"\n📌 数据划分已完成: (cn={cn}, a={a}, d={d})")

def main():
    # 在这里输入你的 cn, a, d 参数
    cn = 100
    a = 0.1
    d = ['cifar10', 'cifar100', 'svhn', 'emnist',] # 'cifar10', 'cifar100', 'svhn', 'fmnist', 'medmnistA', 'medmnistC', 'mnist', 'cifar100', 'cinic10', 'fmnist',
    # d = ['fmnist', ]

    success = generate_data(cn, a, d)

if __name__ == '__main__':
    main()
