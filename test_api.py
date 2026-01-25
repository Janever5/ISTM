import requests
import json
import os

# 测试API端点
BASE_URL = "http://localhost:5000"

def test_server_connection():
    """测试服务器连接"""
    print("测试服务器连接...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print(f"✓ 服务器连接成功: {response.status_code}")
            return True
        else:
            print(f"✗ 服务器连接失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 服务器连接失败: {str(e)}")
        return False

def test_qtbfs_api():
    """测试QTBFS评分API"""
    print("\n测试QTBFS评分API...")
    try:
        response = requests.post(f"{BASE_URL}/api/qtbfs_calculate", 
                               json={"input_data": {}})
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("✓ QTBFS评分API测试成功")
                print(f"  评分结果: {data['result']['total_score']}分")
                print(f"  康复阶段: {data['result']['stage']}")
            else:
                print(f"✗ QTBFS评分API测试失败: {data['error']}")
        else:
            print(f"✗ QTBFS评分API请求失败: {response.status_code}")
    except Exception as e:
        print(f"✗ QTBFS评分API测试出错: {str(e)}")

def test_upload_api():
    """测试上传API"""
    print("\n测试上传API...")
    try:
        # 创建一个简单的CSV测试文件
        test_csv = "test_upload.csv"
        with open(test_csv, "w") as f:
            f.write("index,time,resistance\n")
            for i in range(50):
                f.write(f"{i},{i*0.1},{1000 + i*5}\n")
        
        # 测试上传功能
        with open(test_csv, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/api/upload_for_visualization", files=files)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print("✓ 上传API测试成功")
                print(f"  文件信息: {result['file_info']}")
            else:
                print(f"✗ 上传API测试失败: {result['error']}")
        else:
            print(f"✗ 上传API请求失败: {response.status_code}")
        
        # 清理测试文件
        os.remove(test_csv)
    except Exception as e:
        print(f"✗ 上传API测试出错: {str(e)}")

def test_split_preview_api():
    """测试分割预览API"""
    print("\n测试分割预览API...")
    try:
        # 创建一个简单的CSV测试文件
        test_csv = "test_split_preview.csv"
        with open(test_csv, "w") as f:
            f.write("index,time,resistance\n")
            for i in range(100):
                f.write(f"{i},{i*0.1},{1000 + i*10}\n")
        
        # 先上传文件
        with open(test_csv, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/api/upload_for_visualization", files=files)
            file_path = response.json().get('file_info', {}).get('path', 'dummy.csv')
        
        # 测试分割预览功能
        data = {
            'file_path': file_path,
            'segments': json.dumps([
                {"start": 0, "end": 5, "name": "segment_1"},
                {"start": 5, "end": 10, "name": "segment_2"}
            ])
        }
        response = requests.post(f"{BASE_URL}/api/visualize_split_data", 
                               json=data)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print("✓ 分割预览API测试成功")
                print(f"  预览数据点数: {len(result['preview_data'])}")
            else:
                print(f"✗ 分割预览API测试失败: {result['error']}")
        else:
            print(f"✗ 分割预览API请求失败: {response.status_code}")
        
        # 清理测试文件
        if os.path.exists(test_csv):
            os.remove(test_csv)
    except Exception as e:
        print(f"✗ 分割预览API测试出错: {str(e)}")

def test_frontend_pages():
    """测试前端页面是否正常加载"""
    print("\n测试前端页面...")
    endpoints = [
        ("/", "主页"),
        ("/qtbfs", "QTBFS评分页面"), 
        ("/split", "数据分割页面")
    ]
    
    success_count = 0
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            if response.status_code == 200:
                print(f"✓ {name}加载成功")
                success_count += 1
            else:
                print(f"✗ {name}加载失败: {response.status_code}")
        except Exception as e:
            print(f"✗ {name}测试出错: {str(e)}")
    
    if success_count == len(endpoints):
        print("✓ 所有前端页面测试通过")
    else:
        print(f"✗ 前端页面测试部分失败: {success_count}/{len(endpoints)}")

if __name__ == "__main__":
    print("开始测试API功能...\n")
    
    # 按顺序执行测试
    if test_server_connection():
        test_qtbfs_api()
        test_upload_api() 
        test_split_preview_api()
        test_frontend_pages()
    
    print("\n测试完成!")