import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_data_from_files(folder, time_range=(16, 22)):
    """从指定文件夹中的文件绘制指定时间范围内的数据"""
    filtered_data = []  # 存储筛选出的数据，以便打印
    
    for file_name in os.listdir(folder):
        if file_name.endswith('.txt') and 'knee' in file_name:
            file_path = os.path.join(folder, file_name)
            x_values = []
            y_values = []
            in_time_range = False
            
            with open(file_path, 'r') as file:
                for line in file:
                    x, y = map(float, line.strip().split('\t'))
                    if time_range[0] <= x <= time_range[1]:
                        in_time_range = True
                        x_values.append(x)
                        y_values.append(y)
                    elif in_time_range:  # 一旦超出范围，停止记录
                        break
            
            if x_values:  # 确保有数据才进行操作
                plt.plot(x_values, y_values, label=f'{file_name} ({time_range[0]}-{time_range[1]}s)')
                
                # 将数据添加到filtered_data以供打印
                filtered_data.append((file_name, x_values, y_values))
    
    # # 打印数据点到控制台
    # print('选定时间范围内的数据点:')
    # for name, xs, ys in filtered_data:
    #     print(f'\n{name}:')
    #     for x, y in zip(xs, ys):
    #         print(f'({x:.2f}, {y:.2f})', end=' ')
    
    # plt.xlabel('X轴标签 (秒)')
    # plt.ylabel('Y轴标签')
    # plt.title('prediction:finger_bending')
    # 在图像下方添加文本
    plt.figtext(0.5, 0.02, 'prediction: knee_bending', ha='center', fontsize=12) 
    plt.legend()
    plt.xlim(time_range[0], time_range[1])  # 设置x轴显示范围
    plt.show()

if __name__ == "__main__":
    folder_to_plot = './data'  # 替换为你的TXT文件夹路径
    plot_data_from_files(folder_to_plot)
