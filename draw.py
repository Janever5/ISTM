import os
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

def plot_data_from_files(folder,inside_name):
    """从指定文件夹中的文件绘制数据"""
    for file_name in os.listdir(folder):
        if inside_name != '' and inside_name not in file_name:
            continue
        if file_name.startswith('filtered_') and file_name.endswith('.txt'):
            file_path = os.path.join(folder, file_name)
            x_values = []
            y_values = []
            
            with open(file_path, 'r') as file:
                for line in file:
                    x, y = map(float, line.strip().split('\t'))
                    x_values.append(x)
                    y_values.append(y)
            plt.plot(x_values, y_values, label=file_name)

    plt.xlabel('X轴标签')
    plt.ylabel('Y轴标签')
    plt.title('数据曲线')
    plt.legend()
    plt.show()


def filter_and_save_data_per_file(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if not file_name.endswith('.txt') or file_name == "finger cramp1.txt":  # 确保只处理文本文件
            continue
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, f"filtered_{file_name}")
        
        with open(input_file_path, 'r') as input_file, \
             open(output_file_path, 'w') as output_file:
            for line in input_file:
                if line.strip():  # 跳过空行
                    columns = line.split('\t')
                    x = float(columns[0])
                    y = float(columns[1])
                    if 10 <= x <= 22:
                        # 将符合条件的数据点写入对应的新文件
                        output_file.write(f"{x}\t{y}\n")
            
            print(f"文件 {file_name} 的筛选数据已保存至: {output_file_path}")

input_folder = "./data"
output_folder = "./data_filtered"
filter_and_save_data_per_file(input_folder, output_folder)
plot_data_from_files(output_folder,"")