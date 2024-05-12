import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_data_from_files(folder):
    """从指定文件夹中的文件绘制数据"""
    for file_name in os.listdir(folder):
        if file_name.startswith('filtered_') and file_name.endswith('.txt'):
            if 'elbow' not in file_name:
                continue 
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

if __name__ == "__main__":
    folder_to_plot = './Dataset_Folders/Test_Set'  # 替换为你的TXT文件夹路径
    plot_data_from_files(folder_to_plot)