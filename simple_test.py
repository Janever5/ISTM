import requests
import json

# 创建测试CSV内容
test_csv_content = """index,time,resistance
0,0.0,10.5
1,0.1,11.2
2,0.2,12.8
3,0.3,14.1
4,0.4,15.6
5,0.5,16.3
6,0.6,17.9
7,0.7,18.2
8,0.8,19.5
9,0.9,20.1
10,1.0,21.3
11,1.1,22.7
12,1.2,23.4
13,1.3,24.8
14,1.4,25.2
15,1.5,26.9
"""

with open('test_simple.csv', 'w', encoding='utf-8') as f:
    f.write(test_csv_content)

print("测试 multipart/form-data 格式的预览分割...")

# 测试 multipart/form-data 格式的请求
url = "http://localhost:5000/api/preview_split"

with open('test_simple.csv', 'rb') as f:
    files = {'file': ('test_simple.csv', f, 'text/csv')}
    data = {'params': json.dumps([{'start': 0, 'end': 0.5, 'name': 'segment_1'}])}
    
    try:
        response = requests.post(url, files=files, data=data)
        print(f"Response Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")
    except Exception as e:
        print(f"Request failed: {e}")

# 清理测试文件
import os
if os.path.exists('test_simple.csv'):
    os.remove('test_simple.csv')

# 测试API端点
BASE_URL = "http://localhost:5000"

def test_qtbfs_api():
    """测试QTBFS评分API"""
    print("测试QTBFS评分API...")
    try:
        # 测试JSON请求
        response = requests.post(f"{BASE_URL}/api/qtbfs_calculate", 
                                json={"input_data": {}},
                                headers={'Content-Type': 'application/json'})
        print(f"  JSON请求状态: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("  ✓ QTBFS评分API测试成功")
                print(f"    评分结果: {data['result']['total_score']}分")
                print(f"    康复阶段: {data['result']['stage']}")
            else:
                print(f"    ✗ QTBFS评分API测试失败: {data['error']}")
        else:
            print(f"    ✗ QTBFS评分API请求失败: {response.status_code}")
    except Exception as e:
        print(f"    ✗ QTBFS评分API测试出错: {str(e)}")

def test_preview_split_api():
    """测试分割预览API - 使用JSON请求"""
    print("\n测试分割预览API...")
    try:
        # 创建测试CSV内容
        import os
        test_csv_path = "test_preview.csv"
        with open(test_csv_path, "w") as f:
            f.write("index,time,resistance\n")
            for i in range(50):
                f.write(f"{i},{i*0.1},{1000 + i*5}\n")
        
        # 发送JSON请求
        payload = {
            "file_path": test_csv_path,
            "params": [
                {"start": 0, "end": 2, "name": "segment_1"},
                {"start": 2, "end": 4, "name": "segment_2"}
            ]
        }
        
        response = requests.post(f"{BASE_URL}/api/preview_split",
                                json=payload,
                                headers={'Content-Type': 'application/json'})
                                
        print(f"  预览API状态: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print("  ✓ 分割预览API测试成功")
                print(f"    数据范围: {result.get('data_range', {}).get('min', 'N/A')} - {result.get('data_range', {}).get('max', 'N/A')}")
            else:
                print(f"  ✗ 分割预览API测试失败: {result['error']}")
        else:
            print(f"  ✗ 分割预览API请求失败: {response.status_code}")
        
        # 清理测试文件
        os.remove(test_csv_path)
    except Exception as e:
        print(f"  ✗ 分割预览API测试出错: {str(e)}")

def test_frontend_pages():
    """测试前端页面是否正常加载"""
    print("\n测试前端页面...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("  ✓ 前端页面加载成功")
        else:
            print(f"  ✗ 前端页面加载失败: {response.status_code}")
    except Exception as e:
        print(f"  ✗ 前端页面测试出错: {str(e)}")

if __name__ == "__main__":
    print("开始测试API功能...\n")
    
    test_frontend_pages()
    test_qtbfs_api()
    test_preview_split_api()
    
    print("\n测试完成!")