import json
import os
import zipfile
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import logging
import re
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, send_file, render_template
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
import shutil
import uuid
from werkzeug.utils import secure_filename

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 统一的波形序列长度（保留更多时间细节）
TARGET_LENGTH = 100

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型定义
from waveform_classifier import MGTransformer, WaveformDataset, train_model, predict_waveform, load_standard_waveforms

# 添加新的导入
from qtbfs_scorer import QTBFSScorer
from signal_splitter import SignalSplitter

# 在文件开头添加标准波形库加载
def initialize_system():
    """初始化系统，加载标准波形库"""
    try:
        load_standard_waveforms()
        logger.info("✅ 标准波形库加载成功")
    except Exception as e:
        logger.warning(f"⚠️  标准波形库加载失败: {e}")

# 初始化系统
try:
    initialize_system()
except Exception as e:
    logger.error(f"系统初始化失败: {e}")
    raise

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # 允许跨域请求

# 目录配置
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
TEMP_FOLDER = 'tmp'

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER

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
        return render_template('waveform_miniprogram_en.html')
    else:
        return render_template('waveform_miniprogram.html')

@app.route('/api/upload_model', methods=['POST'])
def api_upload_model():
    """上传模型文件API"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件被上传'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        # 保存模型文件
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(model_path)
        
        return jsonify({
            'success': True,
            'path': model_path,
            'message': f'模型已保存至 {model_path}'
        })
    except Exception as e:
        logger.error(f"上传模型时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload_dataset', methods=['POST'])
def api_upload_dataset():
    """上传数据集ZIP文件API"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件被上传'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        if not file.filename.lower().endswith('.zip'):
            return jsonify({'success': False, 'error': '只支持ZIP格式的数据集文件'})
        
        # 保存数据集ZIP文件
        zip_filename = secure_filename(file.filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
        file.save(zip_path)
        
        # 创建唯一的解压目录名
        extract_dir_name = f"dataset_{uuid.uuid4().hex}"
        extract_dir = os.path.join(app.config['UPLOAD_FOLDER'], extract_dir_name)
        os.makedirs(extract_dir, exist_ok=True)
        
        # 解压ZIP文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        return jsonify({
            'success': True,
            'dataset_path': extract_dir,
            'message': f'数据集已解压至 {extract_dir}'
        })
    except Exception as e:
        logger.error(f"上传数据集时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload_csv', methods=['POST'])
def api_upload_csv():
    """上传CSV文件API"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件被上传'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        if not file.filename.lower().endswith(('.csv', '.txt')):
            return jsonify({'success': False, 'error': '只支持CSV或TXT文件'})
        
        # 保存CSV文件
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(csv_path)
        
        return jsonify({
            'success': True,
            'path': csv_path,
            'message': f'CSV文件已保存至 {csv_path}'
        })
    except Exception as e:
        logger.error(f"上传CSV文件时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

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

@app.route('/api/qtbfs_calculate', methods=['POST'])
def api_qtbfs_calculate():
    """计算QTBFS康复评分，支持多文件上传"""
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(app.config['TEMP_FOLDER'], session_id)
    state0_dir = os.path.join(session_dir, 'state0')
    current_dir = os.path.join(session_dir, 'current')
    
    os.makedirs(state0_dir)
    os.makedirs(current_dir)
    
    try:
        # 1. 保存上传的文件
        state0_files = request.files.getlist('state0_files')
        current_files = request.files.getlist('current_files')
        
        if not state0_files or not current_files:
            return jsonify({'success': False, 'error': '缺少必要的文件'})

        # 构建输入数据字典 (供 scorer 使用)
        input_data = {'state0': {}, 'current_state': {}}
        
        def save_and_map(files, target_dir, map_dict):
            for file in files:
                if not file.filename: continue
                # 确保使用 secure_filename 安全保存
                safe_name = secure_filename(file.filename)
                save_path = os.path.join(target_dir, safe_name)
                file.save(save_path)
                
                # --- 关键修改：更强的正则匹配，不区分大小写，忽略文件扩展名 ---
                
                # 1. 尝试匹配 angle_XX (如 angle_30.xlsx, ANGLE_30.csv)
                # 使用 re.IGNORECASE 忽略大小写
                match_angle = re.search(r'(angle_\d+)', safe_name, re.IGNORECASE)
                if match_angle:
                    # 统一转为小写 key，例如 'angle_30'
                    key = match_angle.group(1).lower()
                    map_dict[key] = save_path
                    continue
                
                # 2. 尝试匹配 speed_XX_Ys (如 speed_30deg_1s.xlsx)
                match_speed = re.search(r'speed_(\d+)(?:deg)?_(\d+)s', safe_name, re.IGNORECASE)
                if match_speed:
                    deg = match_speed.group(1)
                    sec = match_speed.group(2)
                    key = f"speed_{deg}_{sec}s"
                    map_dict[key] = save_path
                    continue
                
                # 3. 默认回退：使用文件名（去掉后缀）
                map_dict[os.path.splitext(safe_name)[0].lower()] = save_path

        save_and_map(state0_files, state0_dir, input_data['state0'])
        save_and_map(current_files, current_dir, input_data['current_state'])
        
        # 2. 调用评分器
        scorer = QTBFSScorer()
        result = scorer.calculate_qtbfs_score(input_data)
        
        # 3. 清理临时文件
        shutil.rmtree(session_dir, ignore_errors=True)
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        # 出错也要清理
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.error(f"QTBFS评分计算出错: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'details': str(type(e).__name__)
        }), 500


@app.route('/api/upload_for_visualization', methods=['POST'])
def api_upload_for_visualization():
    """上传文件用于可视化API，支持CSV和Excel"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件被上传'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        # 检查文件扩展名
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            return jsonify({'success': False, 'error': '只支持CSV和Excel文件'})
        
        # 保存文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'path': file_path,
            'message': f'文件已保存至 {file_path}'
        })
    except Exception as e:
        logger.error(f"上传文件时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/visualize_split_data', methods=['POST'])
def api_visualize_split_data():
    """可视化分割数据波形图API"""
    try:
        data = request.json
        file_path = data.get('file_path')
        x_axis_column = data.get('x_axis_column', 1)  # 默认为第2列（0索引）
        y_axis_column = data.get('y_axis_column', 2)  # 默认为第3列（0索引）
        
        if not file_path:
            return jsonify({'success': False, 'error': '没有提供文件路径'})
        
        # 根据文件扩展名读取文件
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.csv':
            # 尝试多种编码格式读取CSV文件
            encodings = ['utf-8', 'gbk', 'latin1', 'cp1252', 'utf-8-sig']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break  # 成功读取就跳出循环
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue
            
            if df is None:
                return jsonify({'success': False, 'error': '无法使用常见编码格式读取CSV文件'})
                
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            return jsonify({'success': False, 'error': '不支持的文件格式'})
        
        # 检查列数是否足够
        if x_axis_column >= len(df.columns) or y_axis_column >= len(df.columns):
            return jsonify({'success': False, 'error': f'列索引超出范围，文件只有{len(df.columns)}列'})
        
        # 提取指定列的数据
        x_values = df.iloc[:, x_axis_column].values
        y_values = df.iloc[:, y_axis_column].values
        
        # 为了性能考虑，如果数据点过多，进行降采样
        max_points = 1000
        if len(x_values) > max_points:
            step = len(x_values) // max_points
            x_values = x_values[::step]
            y_values = y_values[::step]
        
        # 准备返回给前端的数据
        result_data = {
            'labels': [f'{x:.2f}' for x in x_values],
            'data': y_values.tolist(),
            'x_axis_label': f'第{x_axis_column+1}列',
            'y_axis_label': f'第{y_axis_column+1}列',
            'success': True
        }
        
        return jsonify(result_data)
    except Exception as e:
        logger.error(f"可视化分割数据时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/split_signal', methods=['POST'])
def split_signal():
    """信号数据分割API，支持ZIP下载，现在支持直接上传文件进行分割"""
    try:
        # 检查是否有文件上传（multipart/form-data请求）
        if 'file' in request.files:
            # 直接上传文件进行分割
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': '没有选择文件'})
            
            # 保存上传的文件到临时位置
            filename = secure_filename(file.filename)
            temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_split_{uuid.uuid4()}_{filename}")
            file.save(temp_file_path)
            
            # 获取分割参数
            params_str = request.form.get('params')
            if params_str:
                try:
                    params = json.loads(params_str)
                except json.JSONDecodeError:
                    return jsonify({'success': False, 'error': '分割参数格式错误'})
            else:
                return jsonify({'success': False, 'error': '没有提供分割参数'})
        else:
            # 检查Content-Type是否为application/json
            content_type = request.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                # 从JSON请求获取数据
                data = request.get_json()
                if data is None:
                    return jsonify({'success': False, 'error': '无效的JSON数据'})
            else:
                # 如果不是application/json，尝试从form中获取
                data_str = request.form.get('data', '{}')
                if data_str:
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        data = {}
                else:
                    data = {}
            
            temp_file_path = data.get('file_path')
            params = data.get('params')  # List of {start, end, name}
        
        if not temp_file_path or not params:
            return jsonify({'success': False, 'error': '缺少文件路径或分割参数'})
        
        # 1. 创建临时输出目录
        session_id = str(uuid.uuid4())
        session_out_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        os.makedirs(session_out_dir)
        
        # 2. 执行分割
        splitter = SignalSplitter()
        result = splitter.process_file(temp_file_path, params, session_out_dir)
        
        if not result.get('success'):
            return jsonify(result)
            
        # 3. 打包成 ZIP
        zip_filename = f"split_{session_id}.zip"
        zip_path = os.path.join(app.config['OUTPUT_FOLDER'], zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(session_out_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), file)
        
        # 4. 清理临时文件
        if os.path.exists(temp_file_path) and 'temp_' in temp_file_path:
            os.remove(temp_file_path)
        shutil.rmtree(session_out_dir)
        
        return jsonify({
            'success': True, 
            'download_url': f'/api/download/{zip_filename}',
            'message': result.get('message', ''),
            'file_count': result.get('file_count', 0)
        })
        
    except Exception as e:
        logger.error(f"信号数据分割时出错: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.errorhandler(415)
def handle_unsupported_media_type(error):
    """处理不支持的媒体类型错误"""
    return jsonify({
        'success': False,
        'error': '不支持的媒体类型，请检查请求格式'
    }), 415

@app.route('/api/preview_split', methods=['POST'])
def api_preview_split():
    """分割预览API"""
    try:
        # 创建信号分割器实例
        splitter = SignalSplitter()
        
        # 检查是否有文件上传（multipart/form-data请求）
        if 'file' in request.files:
            # 处理multipart/form-data格式
            file = request.files['file']
            if file.filename != '':
                # 保存上传的文件
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_preview_{uuid.uuid4()}_{file.filename}")
                file.save(file_path)
                
                # 获取分割参数
                params_str = request.form.get('params')
                if params_str:
                    try:
                        params = json.loads(params_str)
                    except json.JSONDecodeError:
                        params = []
                else:
                    params = []
            else:
                return jsonify({'success': False, 'error': '没有选择文件'})
        else:
            # 检查Content-Type是否为application/json
            content_type = request.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                # 处理application/json格式
                data = request.get_json()
                if data is None:
                    return jsonify({'success': False, 'error': '无效的JSON数据'})
            else:
                # 如果不是application/json，尝试从form中获取
                data_str = request.form.get('data', '{}')
                if data_str:
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        data = {}
                else:
                    data = {}
            
            file_path = data.get('file_path')
            params = data.get('params', [])
            
            if not file_path:
                return jsonify({'success': False, 'error': '没有提供文件路径'})
        
        if not params:
            return jsonify({'success': False, 'error': '没有提供分割参数'})
        
        # 执行预览
        result = splitter.preview_split(file_path, params)
        
        # 清理临时文件
        if 'file' in request.files and os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"分割预览时出错: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

def predict_waveform(file_path, model_path=None, scaler_path=None, loaded_model=None, loaded_scaler=None, label_names=None):
    """对单个波形文件进行分类预测"""
    
    # 加载并预处理数据
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.csv':
        # 尝试多种编码格式读取CSV文件
        encodings = ['utf-8', 'gbk', 'latin1', 'cp1252', 'utf-8-sig']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break  # 成功读取就跳出循环
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        if df is None:
            raise ValueError(f"无法使用常见编码格式读取CSV文件: {file_path}")
    elif file_extension == '.xlsx':
        df = pd.read_excel(file_path, engine='openpyxl')
    elif file_extension == '.xls':
        df = pd.read_excel(file_path, engine='xlrd')
    elif file_extension == '.txt':
        df = pd.read_csv(file_path, sep=r'\s+')  # 空格分隔
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

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """下载文件API"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    logger.info("膝关节康复角度波形分类系统后端服务器启动中...")
    logger.info("请访问 http://localhost:5000 查看前端界面")
    logger.info("访问 http://localhost:5000?lang=en 查看英文界面")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
