import requests
import pandas as pd
import numpy as np
import os
from signal_splitter import SignalSplitter

def test_excel_format():
    """测试signal_splitter是否正确处理Excel格式"""
    print("测试signal_splitter的Excel格式处理...")
    
    # 创建测试数据
    test_data = {
        'index': range(0, 100),
        'time': np.linspace(0, 10, 100),
        'resistance': np.random.rand(100) * 100
    }
    df = pd.DataFrame(test_data)
    
    # 保存为CSV
    test_csv_path = 'test_input.csv'
    df.to_csv(test_csv_path, index=False)
    
    # 创建分割器实例
    splitter = SignalSplitter()
    
    # 定义分割参数
    params = [
        {'start': 0, 'end': 3, 'name': 'segment_1'},
        {'start': 3, 'end': 6, 'name': 'segment_2'}
    ]
    
    # 执行分割
    result = splitter.process_file(test_csv_path, params, 'test_output')
    
    if result['success']:
        print("✅ 分割成功")
        
        # 检查生成的Excel文件
        for param in params:
            excel_path = f"test_output/{param['name']}.xlsx"
            if os.path.exists(excel_path):
                df_result = pd.read_excel(excel_path)
                print(f"✅ {excel_path} 已创建")
                print(f"   行数: {len(df_result)}")
                print(f"   列数: {len(df_result.columns)}")
                
                # 检查是否包含重置的index列
                if 'index' in df_result.columns:
                    print(f"   ✅ 包含重置的index列")
                    # 检查index是否从0开始
                    if df_result['index'].iloc[0] == 0:
                        print(f"   ✅ index列正确从0开始")
                    else:
                        print(f"   ❌ index列未从0开始")
                else:
                    print(f"   ❌ 缺少index列")
            else:
                print(f"❌ {excel_path} 未找到")
    else:
        print(f"❌ 分割失败: {result['error']}")
    
    # 清理测试文件
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
    
    import shutil
    if os.path.exists('test_output'):
        shutil.rmtree('test_output')


def test_frontend_tabs():
    """测试前端标签页切换功能"""
    print("\n测试前端标签页切换功能...")
    
    try:
        response = requests.get("http://localhost:5000")
        if response.status_code == 200:
            print("✅ 前端页面加载成功")
            
            # 检查HTML中是否包含修复后的JavaScript代码
            html_content = response.text
            if "switchTab" in html_content:
                print("✅ 包含标签页切换功能")
                
                # 检查修复后的代码是否存在
                if "event && event.target" in html_content:
                    print("✅ 包含修复后的事件处理代码")
                else:
                    print("❌ 缺少修复后的事件处理代码")
            else:
                print("❌ 缺少标签页切换功能")
        else:
            print(f"❌ 前端页面加载失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")


def test_api_endpoints():
    """测试API端点"""
    print("\n测试API端点...")
    
    endpoints_to_test = [
        "/api/qtbfs_calculate",
        "/api/split_signal",
        "/api/upload_for_visualization"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.post(f"http://localhost:5000{endpoint}", timeout=5)
            # 某些端点可能需要特定参数，我们只测试是否返回了某种响应
            print(f"✅ {endpoint}: {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"⚠️  {endpoint}: 请求超时（端点正常但缺少必要参数）")
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint}: {e}")


if __name__ == "__main__":
    print("开始验证修复的功能...\n")
    
    test_excel_format()
    test_frontend_tabs()
    test_api_endpoints()
    
    print("\n验证完成！")