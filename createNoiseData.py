import os
import numpy as np
import pandas as pd

def generate_and_split_noisy_files(input_folder, num_variations=10, noise_std=0.01):
    """
    为指定文件夹中的每个txt文件创建训练集和测试集文件夹，
    并在这些文件夹中生成带有噪声的数据文件。只对文件中的第二列数据添加高斯噪声。
    
    参数:
    input_folder (str): 包含txt文件的输入文件夹路径。
    num_variations (int): 每个原始文件生成的带有噪声的文件数量。
    noise_std (float): 添加到第二列数据的高斯噪声标准差。
    """
    output_root = 'Dataset_Folders'
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    train_folder = os.path.join(output_root, 'Training_Set')
    test_folder = os.path.join(output_root, 'Test_Set')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            
            data = pd.read_csv(filepath, sep='\t', header=None)
            if data.shape[1] != 2:
                print(f"警告：{filename} 不符合预期的两列格式，跳过处理。")
                continue
            
            for i in range(1, num_variations + 1):
                noise = np.random.normal(0, noise_std, len(data))
                noisy_data = data.copy()
                noisy_data.iloc[:, 1] += noise
                
                # 确定文件的目标文件夹（训练集或测试集）
                target_folder = train_folder if i <= 8 else test_folder
                
                # 构建新文件名并保存
                new_filename = f"{base_name}_noisy_{i}.txt"
                new_filepath = os.path.join(target_folder, new_filename)
                
                # 保存到相应文件夹
                noisy_data.to_csv(new_filepath, sep='\t', index=False, header=False)

if __name__ == "__main__":
    input_folder_path = './data_filtered'  # 请替换为您的输入文件夹路径
    generate_and_split_noisy_files(input_folder_path)