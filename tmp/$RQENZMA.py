import json
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import sys
import zipfile
import threading
import time

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型定义
from waveform_classifier import MGTransformer, WaveformDataset, train_model, predict_waveform

# 创建两个Flask应用实例：中文版和英文版
chinese_app = Flask(__name__)
english_app = Flask(__name__)

# 为两个应用启用CORS
CORS(chinese_app)
CORS(english_app)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
chinese_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
english_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 全局变量存储模型和相关组件
model = None
scaler = None
label_names = None
model_config = None

# 训练状态跟踪
training_status = {
    'status': 'idle',  # idle, training, completed
    'progress': 0,
    'message': '等待开始训练',
    'logs': [],
    'chart_data': {
        'epochs': [],
        'losses': [],
        'accuracies': []
    }
}

# 中文版路由
@chinese_app.route('/')
def chinese_index():
    return send_from_directory('.', 'waveform_miniprogram.html')

@chinese_app.route('/<path:path>')
def chinese_static_files(path):
    return send_from_directory('.', path)

# 英文版路由
@english_app.route('/')
def english_index():
    return send_from_directory('.', 'waveform_miniprogram_en.html')

@english_app.route('/<path:path>')
def english_static_files(path):
    return send_from_directory('.', path)

# API路由（两个应用共享）
def register_api_routes(app_instance):
    @app_instance.route('/api/upload_dataset', methods=['POST'])
    def upload_dataset():
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': '未找到文件'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': '未选择文件'}), 400
            
            # 保存上传的文件
            filename = file.filename
            file_path = os.path.join(app_instance.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 如果是zip文件，解压
            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(app_instance.config['UPLOAD_FOLDER'], 'dataset'))
                os.remove(file_path)  # 删除zip文件
                dataset_path = os.path.join(app_instance.config['UPLOAD_FOLDER'], 'dataset')
            else:
                dataset_path = file_path
            
            return jsonify({
                'success': True, 
                'message': '数据集上传成功',
                'dataset_path': dataset_path
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app_instance.route('/api/upload_model', methods=['POST'])
    def upload_model():
        global model, scaler, label_names, model_config
        
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': '未找到模型文件'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': '未选择模型文件'}), 400
            
            # 保存上传的模型文件
            model_filename = file.filename
            model_path = os.path.join(app_instance.config['UPLOAD_FOLDER'], model_filename)
            file.save(model_path)
            
            # 加载模型
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            label_names = checkpoint["label_names"]
            model_config = checkpoint["config"]
            
            model = MGTransformer(
                d_model=model_config["d_model"],
                nhead=model_config["nhead"],
                dim_feedforward=model_config["dim_feedforward"],
                num_layers=model_config["num_layers"],
                num_classes=len(label_names),
                cnn_channels=model_config["cnn_channels"],
                dropout=model_config["dropout"]
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            # 加载标准化器
            scaler_path = model_path.replace('.pth', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            else:
                # 尝试默认的标准化器路径
                default_scaler_path = 'waveform_scaler.pkl'
                if os.path.exists(default_scaler_path):
                    with open(default_scaler_path, "rb") as f:
                        scaler = pickle.load(f)
            
            return jsonify({
                'success': True, 
                'message': '模型加载成功',
                'label_names': label_names
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app_instance.route('/api/upload_file', methods=['POST'])
    def upload_file():
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': '未找到文件'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': '未选择文件'}), 400
            
            # 保存上传的文件
            filename = file.filename
            file_path = os.path.join(app_instance.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            return jsonify({
                'success': True, 
                'message': '文件上传成功',
                'file_path': file_path
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app_instance.route('/api/train', methods=['POST'])
    def train():
        global training_status
        
        try:
            data = request.json
            data_dir = data.get('data_dir')
            model_path = data.get('model_path', 'waveform_model.pth')
            epochs = data.get('epochs', 50)
            batch_size = data.get('batch_size', 64)
            
            if not data_dir or not os.path.exists(data_dir):
                return jsonify({'success': False, 'error': f'数据目录不存在: {data_dir}'}), 400
            
            # 更新训练状态
            training_status['status'] = 'training'
            training_status['progress'] = 0
            training_status['message'] = '开始训练...'
            training_status['logs'] = []
            training_status['chart_data'] = {
                'epochs': [],
                'losses': [],
                'accuracies': []
            }
            
            # 在后台线程中运行训练
            def run_training():
                global model, scaler, label_names, model_config, training_status
                try:
                    # 训练过程中收集日志
                    def progress_callback(epoch, total_epochs, loss, accuracy):
                        progress = (epoch + 1) / total_epochs * 100
                        training_status['progress'] = progress
                        training_status['message'] = f'训练中... Epoch: {epoch+1}/{total_epochs}'
                        log_msg = f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
                        log_msg = f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
                        training_status['logs'].append(log_msg)
                        
                        # 更新图表数据
                        training_status['chart_data']['epochs'].append(epoch + 1)
                        training_status['chart_data']['losses'].append(loss)
                        training_status['chart_data']['accuracies'].append(accuracy)
                    
                    # 执行训练
                    train_model(
                        data_dir=data_dir,
                        model_save_path=model_path,
                        epochs=epochs,
                        batch_size=batch_size,
                        progress_callback=progress_callback
                    )
                    
                    # 训练完成
                    training_status['status'] = 'completed'
                    training_status['progress'] = 100
                    training_status['message'] = '训练完成!'
                    training_status['logs'].append('✅ 模型训练完成!')
                    
                    # 加载训练好的模型
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                    label_names = checkpoint["label_names"]
                    model_config = checkpoint["config"]
                    
                    model = MGTransformer(
                        d_model=model_config["d_model"],
                        nhead=model_config["nhead"],
                        dim_feedforward=model_config["dim_feedforward"],
                        num_layers=model_config["num_layers"],
                        num_classes=len(label_names),
                        cnn_channels=model_config["cnn_channels"],
                        dropout=model_config["dropout"]
                    )
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.eval()
                    
                    # 加载标准化器
                    scaler_path = model_path.replace('.pth', '_scaler.pkl')
                    if os.path.exists(scaler_path):
                        with open(scaler_path, "rb") as f:
                            scaler = pickle.load(f)
                    else:
                        # 尝试默认的标准化器路径
                        default_scaler_path = 'waveform_scaler.pkl'
                        if os.path.exists(default_scaler_path):
                            with open(default_scaler_path, "rb") as f:
                                scaler = pickle.load(f)
                                
                except Exception as e:
                    training_status['status'] = 'error'
                    training_status['message'] = f'训练出错: {str(e)}'
                    training_status['logs'].append(f'❌ 训练出错: {str(e)}')
            
            # 启动训练线程
            training_thread = threading.Thread(target=run_training)
            training_thread.start()
            
            return jsonify({
                'success': True,
                'message': '训练已启动'
            })
        except Exception as e:
            training_status['status'] = 'error'
            training_status['message'] = f'启动训练失败: {str(e)}'
            return jsonify({'success': False, 'error': str(e)}), 500

    @app_instance.route('/api/train_status', methods=['GET'])
    def get_train_status():
        return jsonify({
            'success': True,
            'status': training_status['status'],
            'progress': training_status['progress'],
            'message': training_status['message'],
            'logs': training_status['logs'],
            'chart_data': training_status['chart_data']
        })

    @app_instance.route('/api/predict', methods=['POST'])
    def predict():
        global model, scaler, label_names
        
        if model is None or scaler is None or label_names is None:
            return jsonify({'success': False, 'error': '请先加载模型'}), 400
        
        try:
            # 获取上传的文件
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': '未找到文件'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': '未选择文件'}), 400
            
            # 保存临时文件
            temp_path = os.path.join(app_instance.config['UPLOAD_FOLDER'], f"temp_{file.filename}")
            file.save(temp_path)
            
            # 进行预测
            predicted_class, confidence, all_probabilities = predict_waveform(
                model=model,
                scaler=scaler,
                label_names=label_names,
                file_path=temp_path
            )
            
            # 删除临时文件
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'probabilities': all_probabilities
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app_instance.route('/api/visualize', methods=['POST'])
    def visualize():
        try:
            # 获取上传的文件
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': '未找到文件'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': '未选择文件'}), 400
            
            # 保存临时文件
            temp_path = os.path.join(app_instance.config['UPLOAD_FOLDER'], f"temp_visualize_{file.filename}")
            file.save(temp_path)
            
            # 读取文件并提取波形数据
            try:
                if temp_path.endswith('.csv'):
                    df = pd.read_csv(temp_path)
                elif temp_path.endswith('.xlsx') or temp_path.endswith('.xls'):
                    df = pd.read_excel(temp_path)
                elif temp_path.endswith('.txt'):
                    df = pd.read_csv(temp_path, delimiter=r'\s+', engine='python')
                else:
                    return jsonify({'success': False, 'error': '不支持的文件格式'}), 400
                
                # 尝试找到电流或电阻列
                current_col = None
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['电流', 'current', '电阻', 'resistance', 'r', 'a']):
                        current_col = col
                        break
                
                if current_col is None:
                    return jsonify({'success': False, 'error': '未找到电流或电阻列'}), 400
                
                # 提取波形数据
                waveform_data = df[current_col].dropna().tolist()
                
                # 删除临时文件
                os.remove(temp_path)
                
                return jsonify({
                    'success': True,
                    'waveform_data': waveform_data[:500]  # 限制数据点数量
                })
            except Exception as e:
                # 确保临时文件被删除
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

# 为两个应用注册API路由
register_api_routes(chinese_app)
register_api_routes(english_app)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--english':
        # 启动英文版应用（8000端口）
        english_app.run(host='127.0.0.1', port=8000, debug=False)
    else:
        # 启动中文版应用（5000端口）
        chinese_app.run(host='127.0.0.1', port=5000, debug=False)