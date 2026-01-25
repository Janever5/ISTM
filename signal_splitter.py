import pandas as pd
import numpy as np
import os
from typing import Dict, List
import json
from pathlib import Path


class SignalSplitter:
    """
    传感信号数据分割工具
    """
    
    def __init__(self):
        pass
    
gi    def _read_csv_with_encoding(self, file_path: str):
        """
        尝试多种编码格式读取CSV文件
        """
        encodings = ['utf-8', 'gbk', 'latin1', 'cp1252', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # 如果不是编码错误，继续尝试其他编码
                if "UnicodeDecodeError" not in str(type(e)):
                    continue
                continue
        
        # 如果所有编码都失败，抛出异常
        raise ValueError(f"无法使用常见编码格式读取文件: {file_path}")
    
    def process_file(self, file_path: str, params: List[Dict], output_dir: str) -> Dict:
        """
        处理文件分割
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 使用改进的编码检测方法读取CSV文件
            df = self._read_csv_with_encoding(file_path)
            
            # 验证数据列
            if len(df.columns) < 3:
                return {'success': False, 'error': '数据列数不足，需要至少3列（序号、时间、电阻）'}
            
            # 检查参数有效性
            for param in params:
                start = param.get('start', 0)
                end = param.get('end', 0)
                name = param.get('name', '')
                
                if start >= end:
                    return {'success': False, 'error': f'起始点必须小于结束点'}
                
                if not name:
                    return {'success': False, 'error': f'段名称不能为空'}
                
                # 检查文件名是否包含非法字符
                invalid_chars = '<>:"/\\|?*'
                for char in invalid_chars:
                    if char in name:
                        return {'success': False, 'error': f'段名称不能包含字符: {invalid_chars}'}
            
            # 根据时间范围分割数据
            time_col = df.columns[1]  # 假设第二列为时间
            signal_col = df.columns[2]  # 假设第三列为信号
            
            saved_files = []
            for i, param in enumerate(params):
                start = param['start']
                end = param['end']
                name = param['name']
                
                # 筛选时间段内的数据
                mask = (df[time_col] >= start) & (df[time_col] <= end)
                segment_df = df[mask].copy()
                
                if segment_df.empty:
                    continue
                
                # 重置索引，但保留原始时间戳
                # 如果DataFrame中已经有名为'index'的列，则先重命名它
                if 'index' in segment_df.columns:
                    # 找一个不会冲突的列名来临时存储原来的index列
                    temp_col_name = 'original_index'
                    counter = 1
                    while temp_col_name in segment_df.columns:
                        temp_col_name = f'original_index_{counter}'
                        counter += 1
                    segment_df.rename(columns={'index': temp_col_name}, inplace=True)
                
                # 重置索引，这会创建一个新的从0开始的整数索引
                segment_df = segment_df.reset_index(drop=True)
                
                # 插入新的从0开始的index列
                segment_df.insert(0, 'index', range(len(segment_df)))
                
                # 保存为Excel文件
                output_path = os.path.join(output_dir, f"{name}.xlsx")
                segment_df.to_excel(output_path, index=False)
                saved_files.append(output_path)
            
            return {
                'success': True,
                'message': f'成功分割数据为{len(saved_files)}个文件',
                'file_count': len(saved_files),
                'saved_files': saved_files
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def preview_split(self, file_path: str, params: List[Dict]) -> Dict:
        """
        预览分割效果
        """
        try:
            # 使用改进的编码检测方法读取CSV文件
            df = self._read_csv_with_encoding(file_path)
            
            # 验证数据列
            if len(df.columns) < 3:
                return {'success': False, 'error': '数据列数不足，需要至少3列（序号、时间、电阻）'}
            
            time_col = df.columns[1]  # 假设第二列为时间
            signal_col = df.columns[2]  # 假设第三列为信号
            index_col = df.columns[0]  # 假设第一列为索引
            
            min_time = df[time_col].min()
            max_time = df[time_col].max()
            
            # 检查参数有效性
            for param in params:
                start = param.get('start', 0)
                end = param.get('end', 0)
                name = param.get('name', '')
                
                if start >= end:
                    return {'success': False, 'error': f'起始点必须小于结束点'}
                
                if start < min_time or end > max_time:
                    return {'success': False, 'error': f'分割点超出数据范围（有效范围：{min_time}-{max_time}）'}
                
                if not name:
                    return {'success': False, 'error': f'段名称不能为空'}
            
            # 分析各段数据
            preview_info = []
            for param in params:
                start = param['start']
                end = param['end']
                name = param['name']
                
                # 筛选时间段内的数据
                mask = (df[time_col] >= start) & (df[time_col] <= end)
                segment_df = df[mask].copy()
                
                if not segment_df.empty:
                    # 获取前10行数据用于预览
                    first_10_rows = segment_df.head(10).to_dict('records')
                    
                    preview_info.append({
                        'name': name,
                        'time_range': [start, end],
                        'time_span': end - start,
                        'data_points': len(segment_df),
                        'first_10_rows': first_10_rows
                    })
            
            return {
                'success': True,
                'message': '预览成功，数据格式正确',
                'data_range': {'min': min_time, 'max': max_time},
                'segments_count': len(params),
                'time_range': [min_time, max_time],
                'total_data_points': len(df),
                'columns': {
                    'index': index_col,
                    'time': time_col,
                    'resistance': signal_col
                },
                'preview_info': preview_info
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def load_csv(self, file_path: str):
        """
        加载CSV文件并返回时间、信号数据
        """
        try:
            # 使用改进的编码检测方法读取CSV文件
            df = self._read_csv_with_encoding(file_path)
            if len(df.columns) < 3:
                raise ValueError('数据列数不足，需要至少3列（序号、时间、电阻）')
            
            index_col = df.columns[0]  # 第一列为序号
            time_col = df.columns[1]   # 第二列为时间
            resistance_col = df.columns[2]  # 第三列为电阻
            
            index = df[index_col].values
            time = df[time_col].values
            resistance = df[resistance_col].values
            
            return index, time, resistance
        except Exception as e:
            raise ValueError(f"读取文件失败：{str(e)}")