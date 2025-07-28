import pandas as pd
import numpy as np
from pathlib import Path

def read_all_data(files, methods, target_datasets):
    """
    只读取指定数据集的数据，返回 {method: {dataset: [values]}} 格式
    """
    data = {method: {ds: [] for ds in target_datasets} for method in methods}

    for file in files:
        df = pd.read_excel(file)
        df.columns = [col.lower() for col in df.columns]

        for method in methods:
            method_data = df[df['method'] == method]
            if not method_data.empty:
                for dataset in target_datasets:
                    dataset = dataset.lower()
                    if dataset in method_data.columns:
                        numeric_values = pd.to_numeric(method_data[dataset], errors='coerce').dropna()
                        data[method][dataset].extend(numeric_values.tolist())

    return data

def calculate_statistics(data, methods, target_datasets, decimal_places=2):
    statistics = {method: {} for method in methods}

    for method in methods:
        for dataset in target_datasets:
            values = data[method].get(dataset, [])
            if values:
                mean_value = round(np.mean(values), decimal_places)
                std_value = round(np.std(values), decimal_places)
                statistics[method][dataset] = f"{mean_value}({std_value})"
            else:
                statistics[method][dataset] = "N/A"

    return statistics

def generate_markdown_table(statistics, methods, target_datasets):
    header = '| Method | ' + ' | '.join([ds.upper() for ds in target_datasets]) + ' |\n'
    separator = '|--------' + '|--------' * len(target_datasets) + '|\n'
    body = ''
    for method in methods:
        row = f"| {method} | " + ' | '.join([statistics[method].get(ds, 'N/A') for ds in target_datasets]) + ' |\n'
        body += row
    return header + separator + body

def main():
    folder_path = Path("D://SFL//test_experiment//baseline//alpha=0.1")
    files = list(folder_path.glob("*.xlsx"))

    methods = ['fedavg', 'local', 'fedah', 'fedala', 'fedas', 'feddpa', 'fedfed', 'fedpac', 'fedproto', 'fedrod']

    # target_datasets = ['mnist', 'fmnist', 'medmnista', 'medmnistc',]  # 👈 可以修改为任意多个数据集
    target_datasets = ['cifar10', 'cifar100', 'cinic10', 'fmnist', ]  # 👈 可以修改为任意多个数据集


    data = read_all_data(files, methods, target_datasets)
    statistics = calculate_statistics(data, methods, target_datasets)
    markdown_result = generate_markdown_table(statistics, methods, target_datasets)

    print(markdown_result)

if __name__ == '__main__':
    main()
