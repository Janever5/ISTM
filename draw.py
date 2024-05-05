import os
import matplotlib.pyplot as plt

def plot_data_from_folder(input_folder):
    for file_name in os.listdir(input_folder):
        if file_name != "knee.txt":
            continue
        file_path = os.path.join(input_folder, file_name)  
        x_values = []
        y_values = []

        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # 跳过空行
                    columns = line.split('\t')
                    x_values.append(float(columns[0]))
                    y_values.append(float(columns[1]))

        plt.plot(x_values, y_values, label=file_name)

    plt.xlabel('X轴标签')
    plt.ylabel('Y轴标签')
    plt.title('数据曲线')
    plt.legend()
    plt.show()

input_folder = "./data"
plot_data_from_folder(input_folder)
