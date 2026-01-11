import os
import zipfile
from pathlib import Path

def package_dataset(input_dir, output_path):
    """
    将数据集目录打包成ZIP文件
    
    Args:
        input_dir (str): 输入数据集目录路径
        output_path (str): 输出ZIP文件路径
    """
    input_path = Path(input_dir)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in input_path.rglob('*'):
            if file_path.is_file():
                # 计算相对路径
                arcname = file_path.relative_to(input_path)
                zipf.write(file_path, arcname)
    
    print(f"数据集已打包到: {output_path}")

if __name__ == "__main__":
    # 获取当前目录下的"元数据"文件夹
    metadata_dir = Path("元数据")
    if metadata_dir.exists():
        package_dataset(str(metadata_dir), "dataset.zip")
    else:
        print("未找到'元数据'目录，请确认目录存在")