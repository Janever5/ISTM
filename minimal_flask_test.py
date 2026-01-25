from flask import Flask, request, jsonify
import json
from qtbfs_scorer import QTBFSScorer

app = Flask(__name__)

@app.route('/api/qtbfs_calculate', methods=['POST'])
def api_qtbfs_calculate():
    """计算QTBFS康复评分"""
    try:
        print("收到QTBFS请求")
        # 检查是否是JSON请求
        if request.is_json:
            data = request.get_json()
            print(f"收到JSON数据: {data}")
        else:
            print("请求不是JSON格式")
            # 如果不是JSON，尝试从form获取
            try:
                data_str = request.form.get('data', '{}')
                data = json.loads(data_str) if data_str else {'input_data': {}}
            except:
                data = {'input_data': {}}
        
        input_data = data.get('input_data', {})
        print(f"输入数据: {input_data}")
        
        # 创建评分器并计算
        scorer = QTBFSScorer()
        result = scorer.calculate_qtbfs_score(input_data)
        print(f"计算结果: {result}")
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        print(f"QTBFS评分计算出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'details': str(type(e).__name__)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # 使用不同的端口避免冲突