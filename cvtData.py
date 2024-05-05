import os

def remove_header(input_folder, output_folder):
    # 获取输入文件夹中的所有文件
    input_files = os.listdir(input_folder)
    
    for input_file in input_files:
        # 构建输入文件的完整路径
        input_file_path = os.path.join(input_folder, input_file)
        # 构建输出文件的完整路径
        output_file_path = os.path.join(output_folder, input_file)
        
        with open(input_file_path, 'r', encoding='utf-8') as f:  # 显式指定使用GBK编码打开文件
            lines = f.readlines()
        
        # 去除前三行
        lines = lines[3:]

        with open(output_file_path, 'w') as f:
            f.writelines(lines)

input_folder = "./数据/传感文件"
output_folder = "./data"

remove_header(input_folder, output_folder)
