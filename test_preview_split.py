import requests
import json
import os

def test_preview_split_api():
    """测试预览分割API"""
    print("测试预览分割API...")
    
    # 创建测试CSV文件
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
    
    with open('test_preview.csv', 'w', encoding='utf-8') as f:
        f.write(test_csv_content)
    
    # 测试multipart/form-data格式的请求
    print("测试 multipart/form-data 格式的预览分割...")
    url = "http://localhost:5000/api/preview_split"
    
    with open('test_preview.csv', 'rb') as f:
        files = {'file': f}
        data = {'params': json.dumps([{'start': 0, 'end': 0.5, 'name': 'segment_1'}])}
        
        try:
            response = requests.post(url, files=files, data=data)
            print(f"Response Status Code: {response.status_code}")
            print(f"Response JSON: {response.json()}")
        except Exception as e:
            print(f"Request failed: {e}")
    
    # 测试application/json格式的请求
    print("\n测试 application/json 格式的预览分割...")
    headers = {'Content-Type': 'application/json'}
    json_data = {
        'file_path': 'test_preview.csv',
        'params': [{'start': 0, 'end': 0.5, 'name': 'segment_1'}]
    }
    
    try:
        response = requests.post(url, json=json_data, headers=headers)
        print(f"Response Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    # 清理测试文件
    if os.path.exists('test_preview.csv'):
        os.remove('test_preview.csv')

def test_split_signal_api():
    """测试分割信号API"""
    print("\n测试分割信号API...")
    
    # 创建测试CSV文件
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
    
    with open('test_split.csv', 'w', encoding='utf-8') as f:
        f.write(test_csv_content)
    
    # 测试直接上传文件进行分割
    print("测试直接上传文件进行分割...")
    url = "http://localhost:5000/api/split_signal"
    
    with open('test_split.csv', 'rb') as f:
        files = {'file': f}
        data = {'params': json.dumps([{'start': 0, 'end': 0.5, 'name': 'segment_1'}, {'start': 0.5, 'end': 1.0, 'name': 'segment_2'}])}
        
        try:
            response = requests.post(url, files=files, data=data)
            print(f"Response Status Code: {response.status_code}")
            print(f"Response JSON: {response.json()}")
        except Exception as e:
            print(f"Request failed: {e}")
    
    # 清理测试文件
    if os.path.exists('test_split.csv'):
        os.remove('test_split.csv')

if __name__ == "__main__":
    test_preview_split_api()
    test_split_signal_api()