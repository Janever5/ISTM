import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import argparse
import json
import re
from scipy.signal import find_peaks

# 导入所需的库
from scipy import signal
from scipy.signal import butter, filtfilt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 标准波形库（需要预先录制9种标准动作）
STANDARD_WAVEFORMS = {}  # 在初始化时加载
TARGET_LENGTH = 100  # 增加到100点以保留更多细节

# 定义模型（MG-Transformer）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


def load_standard_waveforms(standard_dir='standard_waveforms'):
    """加载9类标准波形用于评分比对"""
    global STANDARD_WAVEFORMS
    for label in ['30度快', '30度中', '30度慢', 
                  '60度快', '60度中', '60度慢',
                  '90度快', '90度中', '90度慢']:
        file_path = os.path.join(standard_dir, f'{label}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            STANDARD_WAVEFORMS[label] = df.iloc[:, 0].values

def calculate_rehab_score(patient_waveform, predicted_class):
    """
    计算康复评分 (0-100分)
    使用DTW算法比较患者波形与标准波形的相似度
    """
    if predicted_class not in STANDARD_WAVEFORMS:
        return -1, "无标准波形可供比对"
    
    standard = STANDARD_WAVEFORMS[predicted_class]
    
    # 统一长度
    patient_resampled = signal.resample(patient_waveform, len(standard))
    
    # 计算DTW距离
    distance, _ = fastdtw(patient_resampled, standard, dist=euclidean)
    
    # 距离转分数（需要根据实际数据范围调整）
    max_distance = 1000  # 根据实际数据范围调整
    score = max(0, 100 * (1 - distance / max_distance))
    
    # 生成评语
    if score >= 90:
        comment = "优秀！动作非常标准"
    elif score >= 75:
        comment = "良好，动作基本标准"
    elif score >= 60:
        comment = "一般，需要继续练习"
    else:
        comment = "需改进，建议在医生指导下练习"
    
    return round(score, 1), comment

def preprocess_signal(data, fs=100):
    """
    信号预处理
    fs: 采样频率 (Hz)
    """
    # 1. 去除NaN
    data = np.nan_to_num(data, nan=np.nanmean(data))
    
    # 如果数据太短，跳过滤波直接返回
    if len(data) < 15:
        # 对于短信号，只做简单的基线校正
        if len(data) > 0:
            data = data - np.mean(data)
        return data
    
    # 2. 低通滤波去除高频噪声（人体运动很少超过20Hz）
    nyquist = fs / 2
    cutoff = 20  # Hz
    b, a = butter(4, cutoff / nyquist, btype='low')
    data_filtered = filtfilt(b, a, data)
    
    # 3. 基线漂移校正
    data_filtered = data_filtered - np.mean(data_filtered[:min(10, len(data_filtered))])  # 减去起始基线
    
    return data_filtered

def normalize_length(data_seq, target_length=TARGET_LENGTH):
    """
    使用重采样将波形统一到指定长度
    - 保留完整波形形状
    - 快速动作会被"拉伸"
    - 慢速动作会被"压缩"
    """
    if len(data_seq) == target_length:
        return data_seq
    
    # scipy.signal.resample 使用傅里叶方法进行重采样
    resampled = signal.resample(data_seq, target_length)
    return resampled

def find_data_column(df):
    """智能识别数据列"""
    # 膝盖康复运动的典型列名
    VALID_COLUMNS = [
        'angle', '角度', 'deg', 'degree',           # 角度
        'acc', 'accel', 'acceleration', '加速度',    # 加速度
        'gyro', 'angular_velocity', '角速度',        # 角速度
        'emg', '肌电',                               # 肌电信号
        'force', '力', 'torque', '扭矩',             # 力/扭矩
        'value', 'data', '数据'                      # 通用名
    ]

    target_col = None
    columns_lower = [c.lower() for c in df.columns]

    for valid_name in VALID_COLUMNS:
        for i, col in enumerate(columns_lower):
            if valid_name in col:
                target_col = df.columns[i]
                break
        if target_col:
            break

    # 如果都没找到，取第一个数值列
    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
        else:
            # 如果没有数值列，尝试使用第一列（可能是字符串但可转换为数值）
            if len(df.columns) > 0:
                first_col = df.columns[0]
                # 尝试转换第一列为数值
                try:
                    pd.to_numeric(df[first_col], errors='raise')
                    target_col = first_col
                except (ValueError, TypeError):
                    raise ValueError(f"未找到有效数据列，可用列: {df.columns.tolist()}")
            else:
                raise ValueError(f"DataFrame为空或没有列，可用列: {df.columns.tolist()}")
    
    return target_col

def _add_enhanced_features(seq):
    """增强版特征提取，返回8个特征维度"""
    # 基础微分特征
    diff1 = np.diff(seq, prepend=seq[0])  # 一阶差分（速度）
    diff2 = np.diff(diff1, prepend=diff1[0])  # 二阶差分（加速度）
    
    # 平滑处理
    win_mean = np.convolve(seq, np.ones(5)/5, mode='same')  # 滑动平均平滑
    
    # 运动频率特征
    zero_crossings = np.where(np.diff(np.signbit(seq - np.mean(seq))))[0]
    zero_cross_rate = len(zero_crossings) / len(seq)
    zcr_seq = np.full_like(seq, zero_cross_rate)
    
    # 能量特征
    energy = np.abs(seq) ** 2
    energy_envelope = np.convolve(energy, np.ones(10)/10, mode='same')
    
    # 极值特征
    peak_val = np.max(seq)
    peak_pos = np.argmax(seq) / len(seq)  # 归一化峰值位置
    valley_val = np.min(seq)
    
    # 范围特征（区分30/60/90度）
    range_val = peak_val - valley_val
    range_seq = np.full_like(seq, range_val)
    
    # 组合所有特征，确保返回8个特征维度
    features = np.stack([
        seq,                # 0: 原始信号
        diff1,             # 1: 速度（一阶差分）
        diff2,             # 2: 加速度（二阶差分）
        win_mean,          # 3: 平滑信号
        energy_envelope,   # 4: 能量包络
        zcr_seq,           # 5: 过零率
        range_seq,         # 6: 幅度范围
        np.full_like(seq, peak_pos),  # 7: 峰值位置（归一化）
    ], axis=1)
    
    return features


class WaveformDataset(Dataset):
    def __init__(self, data_dir, scaler=None):
        self.data_dir = Path(data_dir)
        self.scaler = scaler
        self.samples = []
        self.labels = []
        self.label_names = []
        
        # 定义有效的数据文件扩展名
        data_extensions = {'.csv', '.xlsx', '.xls', '.txt'}
        
        # 获取所有子目录作为类别
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if not class_dirs:  # 如果没有子目录，则遍历所有有效数据文件
            # 只获取有效的数据文件
            valid_files = []
            for file_path in self.data_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in data_extensions:
                    valid_files.append(file_path)
            
            if valid_files:
                labels = list(set([f.stem for f in valid_files]))  # 使用文件名作为标签
                self.label_names = sorted(labels)
            else:
                self.label_names = []
        else:
            self.label_names = sorted([d.name for d in class_dirs])
        
        # 创建标签到索引的映射
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}
        
        # 加载数据
        self._load_data()
        
        # 检查是否加载了任何样本
        if len(self.samples) == 0:
            raise ValueError(f"未找到有效的数据文件。支持的格式: {data_extensions}")
        
        # 如果没有传入scaler，则创建一个新的
        if self.scaler is None:
            # 收集所有样本以拟合标准化器
            all_samples_for_fitting = []
            for sample in self.samples:
                # 确保样本形状是 (sequence_length, n_features)，然后重塑为 (n_samples, n_features)
                sample_reshaped = sample.reshape(-1, sample.shape[-1])  # (seq_len * n_channels, n_features)
                all_samples_for_fitting.append(sample_reshaped)
            
            # 合并所有样本
            all_data_for_fitting = np.vstack(all_samples_for_fitting)
            
            # 创建并拟合标准化器
            self.scaler = StandardScaler()
            self.scaler.fit(all_data_for_fitting)  # 现在标准化器知道正确的特征数量
        
        # 标准化所有样本
        for i in range(len(self.samples)):
            orig_shape = self.samples[i].shape
            reshaped_sample = self.samples[i].reshape(-1, orig_shape[-1])
            self.samples[i] = self.scaler.transform(reshaped_sample).reshape(orig_shape)

    def _load_data(self):
        """加载数据"""
        # 只处理有效的数据文件格式
        data_extensions = {'.csv', '.xlsx', '.xls', '.txt'}
        
        # 检查是否有子目录结构 - 只检查真正的目录（排除隐藏目录等）
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        has_subdirs = len(subdirs) > 0
        print(f"Debug: has_subdirs = {has_subdirs}")
        print(f"Debug: data_dir = {self.data_dir}")
        print(f"Debug: label_names = {self.label_names}")
        print(f"Debug: subdirs found = {[d.name for d in subdirs]}")
        
        if has_subdirs:
            # 有子目录结构，按子目录分类
            print("Debug: Loading with subdirectory structure")
            for file_path in self.data_dir.glob('**/*.csv'):
                print(f"Debug: Loading CSV file: {file_path}")
                self._load_file(file_path, file_path.parent.name)
            
            for file_path in self.data_dir.glob('**/*.xlsx'):
                print(f"Debug: Loading XLSX file: {file_path}")
                self._load_file(file_path, file_path.parent.name)
                
            for file_path in self.data_dir.glob('**/*.xls'):
                print(f"Debug: Loading XLS file: {file_path}")
                self._load_file(file_path, file_path.parent.name)
                
            for file_path in self.data_dir.glob('**/*.txt'):
                print(f"Debug: Loading TXT file: {file_path}")
                self._load_file(file_path, file_path.parent.name)
        else:
            # 没有子目录结构，直接使用文件名作为标签
            print("Debug: Loading without subdirectory structure")
            for file_path in self.data_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in data_extensions:
                    label = file_path.stem  # 使用文件名作为标签
                    print(f"Debug: Loading file: {file_path} with label: {label}")
                    self._load_file(file_path, label)

    def _load_file(self, file_path, label):
        """加载单个文件"""
        try:
            # 根据文件扩展名选择读取方法
            file_extension = file_path.suffix.lower()
            print(f"Debug _load_file: Processing file {file_path} with extension {file_extension}")
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
                print(f"Debug _load_file: CSV loaded, shape: {df.shape}, columns: {df.columns.tolist()}")
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file_extension == '.xls':
                df = pd.read_excel(file_path, engine='xlrd')
            elif file_extension == '.txt':
                df = pd.read_csv(file_path, sep='\\s+')  # 空格分隔
            else:
                print(f"跳过不支持的文件格式: {file_path}")
                return  # 跳过不支持的文件格式
            
            # 检查DataFrame是否为空
            if df.empty:
                print(f"跳过空文件: {file_path}")
                return
            
            print(f"Debug _load_file: DataFrame info - rows: {len(df)}, cols: {len(df.columns)}")
            
            # 智能识别数据列
            target_col = find_data_column(df)
            print(f"Debug _load_file: Found target column: {target_col}")
            current = df[target_col].values.astype(float)
            print(f"Debug _load_file: Data values shape: {current.shape}")
            
            # 信号预处理
            processed_data = preprocess_signal(current)
            print(f"Debug _load_file: Processed data shape: {processed_data.shape}")
            
            # 长度归一化
            normalized_data = signal.resample(processed_data, TARGET_LENGTH)
            print(f"Debug _load_file: Normalized data shape: {normalized_data.shape}")
            
            # 特征工程
            features = _add_enhanced_features(normalized_data)
            print(f"Debug _load_file: Features shape: {features.shape}")
            
            # 添加到样本列表
            self.samples.append(features)
            self.labels.append(label)
            print(f"Debug _load_file: Successfully added sample for label {label}")
            
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}，跳过该文件")
            import traceback
            traceback.print_exc()
            return

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.samples[idx])  # shape: (seq_len, features)
        label_str = self.labels[idx]
        label_idx = self.label_to_idx[label_str]
        y = torch.LongTensor([label_idx])   # 保持为二维张量 [1]
        return x, y.squeeze()                      # squeeze后变为标量


# 训练函数
def train_model(data_dir, model_save_path="waveform_model.pth", 
                scaler_save_path="waveform_scaler.pkl", epochs=50, batch_size=64, 
                progress_callback=None, existing_scaler=None):
    """训练波形分类模型
    
    Args:
        data_dir: 数据目录路径
        model_save_path: 模型保存路径
        scaler_save_path: 标准化器保存路径
        epochs: 训练轮数
        batch_size: 批次大小
        progress_callback: 进度回调函数
        existing_scaler: 已存在的标准化器，如果为None则新建
    """
    print("开始训练波形分类模型...")
    
    # 创建数据集
    dataset = WaveformDataset(data_dir, scaler=existing_scaler)
    
    # 保存标准化器
    with open(scaler_save_path, "wb") as f:
        pickle.dump(dataset.scaler, f)
    
    # 保存类别名称
    with open("label_names.json", "w") as f:
        json.dump(dataset.label_names, f, ensure_ascii=False)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = MGTransformer(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        num_layers=3,
        num_classes=len(dataset.label_names),
        cnn_channels=128,
        dropout=0.2
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_preds, all_labels = [], []
        
        for x, y in dataloader:
            # 确保输入数据维度正确
            if x.dim() == 2:
                x = x.unsqueeze(0)  # 添加batch维度
            if y.dim() == 0:
                y = y.unsqueeze(0)  # 添加batch维度
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(dataset)
        
        # 调用进度回调函数
        if progress_callback is not None:
            progress_callback(epoch, epochs, avg_loss, acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    
    # 保存模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_names": dataset.label_names,
        "config": {
            "d_model": 256,
            "nhead": 8,
            "dim_feedforward": 1024,
            "num_layers": 3,
            "cnn_channels": 128,
            "dropout": 0.2
        },
        "input_shape": (TARGET_LENGTH, 8)  # 记录输入形状信息，8个特征维度
    }, model_save_path)
    
    print(f"模型已保存到: {model_save_path}")
    print(f"标准化器已保存到: {scaler_save_path}")
    print(f"类别名称已保存到: label_names.json")


def predict_waveform(file_path, model_path="waveform_model.pth", 
                     scaler_path="waveform_scaler.pkl",
                     loaded_model=None, loaded_scaler=None, label_names=None, return_score=True):
    """
    膝关节康复波形预测与评分
    """
    # 加载模型和标准化器（如果未提供）
    if loaded_model is None or loaded_scaler is None or label_names is None:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_config = checkpoint["config"]
        
        # 重新创建模型
        model = MGTransformer(
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            dim_feedforward=model_config["dim_feedforward"],
            num_layers=model_config["num_layers"],
            num_classes=len(checkpoint["label_names"]),
            cnn_channels=model_config["cnn_channels"],
            dropout=model_config["dropout"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # 加载标准化器
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        label_names = checkpoint["label_names"]
    else:
        model = loaded_model
        scaler = loaded_scaler
        model.eval()

    # 1. 读取数据
    file_extension = Path(file_path).suffix.lower()
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif file_extension == '.txt':
        df = pd.read_csv(file_path, sep=r'\\s+')
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")
    
    # 2. 智能列名识别
    target_col = find_data_column(df)  # 使用上面定义的函数
    raw_data = df[target_col].values.astype(float)
    
    # 3. 信号预处理
    processed_data = preprocess_signal(raw_data)
    
    # 4. 长度归一化（重采样，不是截断！）
    normalized_data = signal.resample(processed_data, TARGET_LENGTH)
    
    # 5. 特征工程
    features = _add_enhanced_features(normalized_data)
    
    # 6. 标准化 - 需要适配8维特征
    features_flat = features.reshape(-1, features.shape[-1])  # reshape为(序列长度*T, 8)
    features_scaled = scaler.transform(features_flat).reshape(features.shape)
    
    # 7. 模型预测
    x = torch.FloatTensor(features_scaled).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)
    
    predicted_label = label_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    # 8. 康复评分
    rehab_score, score_comment = -1, ""
    if return_score and STANDARD_WAVEFORMS:
        rehab_score, score_comment = calculate_rehab_score(
            normalized_data, predicted_label
        )
    
    return {
        "predicted_class": predicted_label,
        "confidence": round(confidence, 4),
        "rehab_score": rehab_score,
        "score_comment": score_comment,
        "all_probabilities": {l: round(float(p), 4) for l, p in zip(label_names, probs)}
    }


class MGTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, num_classes, 
                 cnn_channels=128, dropout=0.1):
        super().__init__()
        # CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=cnn_channels//2, kernel_size=5, padding=2),  # 修改为8个输入通道
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels//2),
            nn.Conv1d(cnn_channels//2, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Conv1d(cnn_channels, d_model, kernel_size=3, padding=1),
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 多尺度池化
        self.pool = lambda x: torch.cat([
            x.mean(dim=1),    # 平均池化
            x.max(dim=1)[0],  # 最大池化
            x.min(dim=1)[0]   # 最小池化
        ], dim=1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(3*d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        initrange = 0.1
        self.classifier[0].weight.data.uniform_(-initrange, initrange)
        self.classifier[0].bias.data.zero_()
        self.classifier[3].weight.data.uniform_(-initrange, initrange)
        self.classifier[3].bias.data.zero_()
        self.classifier[5].weight.data.uniform_(-initrange, initrange)
        self.classifier[5].bias.data.zero_()
        
    def forward(self, src):
        """
        src: (batch_size, seq_len, feature_dim)
        """
        # CNN特征提取
        src = src.permute(0, 2, 1)  # (batch_size, feature_dim, seq_len)
        src = self.cnn(src)
        src = src.permute(0, 2, 1)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        src = self.pos_encoder(src)
        src = self.layer_norm(src)
        
        # Transformer编码器
        output = self.transformer_encoder(src)
        
        # 多尺度池化
        output = self.pool(output)
        
        # 分类器
        output = self.classifier(output)
        return output
