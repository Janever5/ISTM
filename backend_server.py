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

# 统一的波形序列长度（保留更多时间细节）
TARGET_LENGTH = 100

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型定义
from waveform_classifier import MGTransformer, WaveformDataset, train_model, predict_waveform, load_standard_waveforms

# 在文件开头添加标准波形库加载
def initialize_system():
    """初始化系统，加载标准波形库"""
    try:
        load_standard_waveforms()
        print("✅ 标准波形库加载成功")
    except Exception as e:
        print(f"⚠️  标准波形库加载失败: {e}")

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

@app.route('/')
def index():
    # 检查是否请求英文版
    lang = request.args.get('lang', 'zh')
    if lang == 'en':
        return send_from_directory('.', 'waveform_miniprogram_en.html')
    else:
        return send_from_directory('.', 'waveform_miniprogram.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        if 'files' not in request.files and 'file' not in request.files:
            return jsonify({'success': False, 'error': '未找到文件'}), 400
        
        # 处理单个文件上传的情况（原有的逻辑）
        uploaded_files = []
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                uploaded_files.append(file)
        
        # 处理多个文件上传的情况
        if 'files' in request.files:
            files = request.files.getlist('files')
            uploaded_files.extend(files)
        
        if not uploaded_files:
            return jsonify({'success': False, 'error': '未选择文件'}), 400
        
        # 创建一个临时数据集目录
        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset')
        os.makedirs(dataset_path, exist_ok=True)
        
        for file in uploaded_files:
            filename = file.filename
            file_path = os.path.join(dataset_path, filename)
            file.save(file_path)
        
        return jsonify({
            'success': True, 
            'message': f'成功上传 {len(uploaded_files)} 个文件',
            'dataset_path': dataset_path
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload_model', methods=['POST'])
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
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
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

@app.route('/api/upload_file', methods=['POST'])
def upload_file():
    try:
        if 'files' not in request.files and 'file' not in request.files:
            return jsonify({'success': False, 'error': '未找到文件'}), 400
        
        # 处理单个文件上传的情况
        uploaded_files = []
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                uploaded_files.append(file)
        
        # 处理多个文件上传的情况
        if 'files' in request.files:
            files = request.files.getlist('files')
            uploaded_files.extend(files)
        
        if not uploaded_files:
            return jsonify({'success': False, 'error': '未选择文件'}), 400
        
        file_paths = []
        for file in uploaded_files:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
        
        return jsonify({
            'success': True, 
            'message': f'成功上传 {len(uploaded_files)} 个文件',
            'file_paths': file_paths
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
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

@app.route('/api/train_status', methods=['GET'])
def get_train_status():
    return jsonify({
        'success': True,
        'status': training_status['status'],
        'progress': training_status['progress'],
        'message': training_status['message'],
        'logs': training_status['logs'],
        'chart_data': training_status['chart_data']
    })

@app.route('/api/predict', methods=['POST'])
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
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{file.filename}")
        file.save(temp_path)
        
        try:
            # 使用预测函数
            result = predict_waveform(
                file_path=temp_path,
                loaded_model=model,
                loaded_scaler=scaler,
                label_names=label_names
            )
            
            # 删除临时文件
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'all_probabilities': result['all_probabilities']
            })
        except Exception as e:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'success': False, 'error': str(e)}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def predict_waveform(file_path, model_path=None, scaler_path=None, loaded_model=None, loaded_scaler=None, label_names=None):
    """对单个波形文件进行分类预测"""
    
    # 加载并预处理数据
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        df = pd.read_excel(file_path, engine='openpyxl')
    elif file_extension == '.xls':
        df = pd.read_excel(file_path, engine='xlrd')
    elif file_extension == '.txt':
        df = pd.read_csv(file_path, sep='\s+')  # 空格分隔
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")
    
    # 获取列名（不区分大小写）
    columns = [col.lower() for col in df.columns]
    
    # 查找电流或电阻列（支持多种命名方式）
    current_col = None
    for col in columns:
        if 'current' in col or '电流' in col or 'r' in col or '电阻' in col or 'resistance' in col:
            current_col = df.columns[columns.index(col)]
            break
    
    if current_col is None:
        raise ValueError(f"未找到电流或电阻列，可用列: {df.columns.tolist()}")
    
    current = df[current_col].values
    
    # 统一长度为TARGET_LENGTH（100）
    if len(current) >= TARGET_LENGTH:
        current = current[:TARGET_LENGTH]
    else:
        current = np.pad(current, (0, TARGET_LENGTH - len(current)), mode="constant")
    
    # 添加增强特征
    def _add_enhanced_features(seq):
        """增强版特征提取，与waveform_classifier.py中的一致"""
        # 原有特征
        diff1 = np.diff(seq, prepend=seq[0])  # 一阶差分（速度）
        diff2 = np.diff(diff1, prepend=diff1[0])  # 二阶差分（加速度）
        win_mean = np.convolve(seq, np.ones(5)/5, mode='same')  # 平滑
        
        # 新增特征 - 对区分速度很重要
        # 1. 过零率（反映运动频率）
        zero_crossings = np.where(np.diff(np.signbit(seq - np.mean(seq))))[0]
        zero_cross_rate = len(zero_crossings) / len(seq)
        zcr_seq = np.full_like(seq, zero_cross_rate)
        
        # 2. 能量包络
        energy = np.abs(seq) ** 2
        energy_envelope = np.convolve(energy, np.ones(10)/10, mode='same')
        
        # 3. 峰值特征
        peak_val = np.max(seq)
        peak_pos = np.argmax(seq) / len(seq)
        valley_val = np.min(seq)
        
        # 4. 范围特征（区分30/60/90度）
        range_val = peak_val - valley_val
        range_seq = np.full_like(seq, range_val)
        
        return np.stack([
            seq,              # 原始信号
            diff1,            # 速度
            diff2,            # 加速度
            win_mean,         # 平滑信号
            energy_envelope,  # 能量包络
            zcr_seq,          # 过零率
            range_seq,        # 幅度范围
            np.full_like(seq, peak_pos),  # 峰值位置
        ], axis=1)
    
    current = _add_enhanced_features(current)
    current = loaded_scaler.transform(current.reshape(-1, current.shape[-1])).reshape(current.shape)
    x = torch.FloatTensor(current)  # shape: (seq_len, features)
    
    # 预测
    with torch.no_grad():
        outputs = loaded_model(x.unsqueeze(0))  # 添加batch维度
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)
        
    # 返回结果
    results = {}
    for i, (label, prob) in enumerate(zip(label_names, probs)):
        results[label] = float(prob)
    
    predicted_label = label_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    return {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "all_probabilities": results
    }

if __name__ == '__main__':
    # 初始化系统
    initialize_system()
    
    print("膝关节康复角度波形分类系统后端服务器启动中...")
    print("请访问 http://localhost:5000 查看前端界面")
    print("访问 http://localhost:5000?lang=en 查看英文界面")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
