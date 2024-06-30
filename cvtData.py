import os

def adjust_column_within_range(input_file_path, output_file_path, start_row=1000, end_row=2000):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 分离指定行范围内的第二列数据
    second_column_values = [float(line.split()[1]) for i, line in enumerate(lines) if start_row <= i < end_row]
    min_value = min(second_column_values) if second_column_values else 0

    # 处理指定范围内的数据，修改第二列
    adjusted_lines = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) > 1 and start_row <= i < end_row:  # 确保行至少有两列且在指定范围内
            original_value = float(parts[1])
            adjusted_value = ((original_value - min_value) / min_value) * 100
            parts[1] = str(adjusted_value)
        else:
            continue
        adjusted_lines.append('\t'.join(parts) + '\n')

    # 写入处理后的数据
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(adjusted_lines)

def remove_header_and_adjust_data(input_folder, output_folder, start_row=1000, end_row=2000):
    input_files = os.listdir(input_folder)
    
    for input_file in input_files:
        input_file_path = os.path.join(input_folder, input_file)
        output_file_path = os.path.join(output_folder, input_file)
        
        # 去除文件头
        with open(input_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = lines[3:]  # 假定前三行为头部
        
        # 写入处理前的文件（去除头部）
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        # 调整指定范围内的第二列数据
        adjust_column_within_range(output_file_path, output_file_path, start_row=start_row, end_row=end_row)

input_folder = "./数据/传感文件"
output_folder = "./data"

remove_header_and_adjust_data(input_folder, output_folder)