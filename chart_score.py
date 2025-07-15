import os
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib, glob
import matplotlib.ticker as ticker  # Import the ticker module

# Set dpi and font style
dpi = 300
matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["axes.linewidth"] = 1
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.labelsize'] = 'x-large'
matplotlib.rcParams['ytick.labelsize'] = 'x-large'

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

# Set base font size
base_font_size = 20  # Base font size, other font sizes will be adjusted based on this

# Set scale factor (e.g., 1.5 means font size is increased by 1.5 times)
scale_factor = 1.2

# Adjust font sizes using the scale factor
plt_dict = {
    'label.size': base_font_size,  # Label font size
    'title.size': base_font_size * scale_factor,  # Title font size
    'legend.size': base_font_size * 0.8,  # Legend font size (reduced)
    'xtick.size': base_font_size * 0.5,  # x-axis tick font size
    'ytick.size': base_font_size * 0.7,  # y-axis tick font size
    'legend.loc': 'lower left'  # Legend position
}

# Set method folder path
base_folder = "experiment_logs/alpha=0.1/score"  # Change to the actual folder path
# dataset_names = ["cifar10", "cifar100"]  # Add multiple dataset names
# dataset_names = ["mnist", "svhn"]  # Add multiple dataset names
dataset_names = ["medmnistA", "medmnistC"]  # Add multiple dataset names
# dataset_names = ["emnist", "fmnist"]  # Add multiple dataset names

# Custom method names (LaTeX format)
custom_method_names = [
    r"$I_F(\cdot)$",
    r"$I_G(\cdot)$",
    r"$I_O(\cdot)$"
    # Add more custom LaTeX formatted method names
]

# Store x and y values for each method
methods_data = {dataset_name: {} for dataset_name in dataset_names}

# Iterate through each dataset
for dataset_name in dataset_names:
    base_folder_path = os.path.join(base_folder, dataset_name)

    # Iterate through method folders
    for method_name in os.listdir(base_folder_path):
        method_folder = os.path.join(base_folder_path, method_name)

        if os.path.isdir(method_folder):
            # Initialize x and y lists for the current method
            x_vals = []
            y_vals = []

            # Iterate through each result folder
            for result_folder in os.listdir(method_folder):
                result_folder_path = os.path.join(method_folder, result_folder)

                if os.path.isdir(result_folder_path):
                    # Use regex to extract the last number from the folder name (x coordinate)
                    match = re.search(r'_(\d+\.\d+)$', result_folder)
                    if match:
                        x_val = float(match.group(1))
                        x_vals.append(x_val)

                        # Look for and read the main.log file
                        log_files = glob.glob(os.path.join(result_folder_path, '*.log'))
                        for file in log_files:
                            with open(file, 'r', encoding='utf-8') as log_file:  # 打开每个 log 文件
                                # 读取文件中的每一行
                                for line in log_file:
                                    if "before fine-tuning" in line:  # 查找包含 "before fine-tuning" 的行
                                        match = re.search(r'before fine-tuning: (\d+\.\d+)%', line)  # 正则提取准确率
                                        if match:
                                            y_val = float(match.group(1))  # 提取出的准确率转为浮动数值
                                            y_vals.append(y_val)  # 将准确率添加到 y_vals 列表
                                            break  # 找到目标数据后跳出循环

            # Store the current method's data
            methods_data[dataset_name][method_name] = (x_vals, y_vals)

# Create main figure and subplots
fig, axs = plt.subplots(1, 4, figsize=(28, 6), dpi=dpi)

index = 0
# Define colors and markers
colors = ['blue', 'orange', 'red']  # Use blue, orange, and red
markers = ['o', 's', '^']  # Circle, square, triangle markers

# Plot for each dataset
for dataset_idx, dataset_name in enumerate(dataset_names):
    # Get the method data for the current dataset
    dataset_methods = methods_data[dataset_name]

    # Plot the curve from 0 to 1.0
    ax1 = axs[dataset_idx + index]
    for idx, (method_name, (x_vals, y_vals)) in enumerate(dataset_methods.items()):
        # Sort x_vals and y_vals in ascending order
        sorted_data = sorted(zip(x_vals, y_vals))  # Sort based on x_vals
        x_vals_sorted, y_vals_sorted = zip(*sorted_data)  # Unzip the sorted values

        color = colors[idx % len(colors)]  # Get the color for the current curve
        ax1.plot(x_vals_sorted, y_vals_sorted, marker=markers[idx % len(markers)], color=color,
                 label=f'{custom_method_names[idx]}')

        # Find the highest point and highlight it
        max_index = np.argmax(y_vals_sorted)
        max_x = x_vals_sorted[max_index]
        max_y = y_vals_sorted[max_index]
        ax1.plot(max_x, max_y, 'o', markersize=10, color=color)  # Mark the highest point with the current curve color
        ax1.plot(max_x, max_y, 'o', markersize=15, color=color,
                 alpha=0.5)  # Highlight the highest point with larger size

    # Set x-axis to have denser ticks
    x_ticks = np.linspace(0, 1, 11)  # Create x ticks from 0 to 1
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{tick:.1f}' for tick in x_ticks], fontsize=plt_dict['xtick.size'])

    # Set y-axis tick label font size and format
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))  # Format y-axis to 2 decimal places
    ax1.set_yticklabels([f'{tick:.0f}' for tick in ax1.get_yticks()], fontsize=plt_dict['ytick.size'])

    # Set title and labels
    ax1.set_title(f'{dataset_name.upper()} (0.0 to 1.0)', fontsize=plt_dict['title.size'], fontweight='bold')
    ax1.set_xlabel('Quantile', fontsize=plt_dict['label.size'], fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=plt_dict['label.size'], fontweight='bold')
    ax1.legend(fontsize=plt_dict['legend.size'], loc=plt_dict['legend.loc'], title='Scores',
               title_fontsize=plt_dict['legend.size'], frameon=True, markerscale=1.5)
    ax1.grid(True)

    # Plot the curve from 0.99 to 1.0
    ax2 = axs[dataset_idx + index + 1]
    for idx, (method_name, (x_vals, y_vals)) in enumerate(dataset_methods.items()):
        # Filter x values between 0.99 and 1.0
        filtered_x_vals = [x for x in x_vals if 0.999 <= x <= 1.0]
        filtered_y_vals = [y_vals[i] for i in range(len(x_vals)) if 0.999 <= x_vals[i] <= 1.0]

        # Sort filtered x_vals and y_vals in ascending order
        sorted_data = sorted(zip(filtered_x_vals, filtered_y_vals))  # Sort based on filtered_x_vals
        filtered_x_vals_sorted, filtered_y_vals_sorted = zip(*sorted_data)  # Unzip the sorted values

        color = colors[idx % len(colors)]  # Get the color for the current curve
        ax2.plot(filtered_x_vals_sorted, filtered_y_vals_sorted, marker=markers[idx % len(markers)], color=color,
                 label=f'{custom_method_names[idx]}', linestyle='--')

        # Find the highest point and highlight it
        if filtered_y_vals_sorted:  # Ensure there is data
            max_index = np.argmax(filtered_y_vals_sorted)
            max_x = filtered_x_vals_sorted[max_index]
            max_y = filtered_y_vals_sorted[max_index]
            ax2.plot(max_x, max_y, 'o', markersize=10,
                     color=color)  # Mark the highest point with the current curve color
            ax2.plot(max_x, max_y, 'o', markersize=15, color=color,
                     alpha=0.5)  # Highlight the highest point with larger size

    # Set x-axis to have denser ticks
    x_ticks = np.linspace(0.999, 1.0, 11)  # Create x ticks from 0.99 to 1.0
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{tick:.4f}' for tick in x_ticks], fontsize=plt_dict['xtick.size'])

    # For the second subplot (0.9 to 1.0)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))  # Format y-axis to 2 decimal places
    ax2.set_yticklabels([f'{tick:.0f}' for tick in ax2.get_yticks()], fontsize=plt_dict['ytick.size'])

    # Set title and labels
    ax2.set_title(f'{dataset_name.upper()} (0.999 to 1.0)', fontsize=plt_dict['title.size'], fontweight='bold')
    ax2.set_xlabel('Quantile', fontsize=plt_dict['label.size'], fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=plt_dict['label.size'], fontweight='bold')
    ax2.legend(fontsize=plt_dict['legend.size'], loc=plt_dict['legend.loc'], title='Scores',
               title_fontsize=plt_dict['legend.size'], frameon=True, markerscale=1.5)
    ax2.grid(True)

    index += 1

# Adjust spacing between subplots
plt.tight_layout()

# Save the image
output_dir = 'chart'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存图像到文件夹
output_path = os.path.join(output_dir, 'score_medmnistA.pdf')
plt.savefig(output_path)

# Show the image
plt.show()
