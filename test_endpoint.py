from backend_server import api_qtbfs_calculate
import json

# 模拟Flask请求对象
class MockRequest:
    def __init__(self, json_data):
        self._json = json_data
        self.is_json = True

    def get_json(self):
        return self._json

print("Testing QTBFS endpoint directly...")
try:
    # 模拟请求对象
    import backend_server
    backend_server.request = MockRequest({"input_data": {}})
    
    result = api_qtbfs_calculate()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()