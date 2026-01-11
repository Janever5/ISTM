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

class MGTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, num_classes, 
                 cnn_channels=128, dropout=0.1):
        super().__init__()
        # CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=7, out_channels=cnn_channels//2, kernel_size=5, padding=2),
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
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # 转换维度以适应CNN (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # 确保CNN输出形状正确
        if x.dim() == 3:
            # 转换回Transformer所需的维度 (batch, features, seq_len) -> (batch, seq_len, features)
            x = x.transpose(1, 2)
        # 位置编码和Transformer编码
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        # 池化和分类
        x = self.pool(x)
        logits = self.classifier(x)
        return logits

# 波形数据集处理类
class WaveformDataset(Dataset):
    def __init__(self, data_dir, seq_len=50, scaler=None):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.samples = []
        self.labels = []
        self.label_names = []
        
        # 收集所有文件
        self._collect_files()
        
        # 标准化
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            
        if self.samples:
            # 将所有样本合并进行标准化
            all_samples = np.array(self.samples)
            all_samples_flat = all_samples.reshape(-1, all_samples.shape[-1])
            if scaler is None:  # 只有在没有提供现有标准化器时才进行拟合
                self.scaler.fit(all_samples_flat)
            
            # 对每个样本进行标准化
            normalized_samples = []
            for sample in self.samples:
                sample_normalized = self.scaler.transform(sample)
                normalized_samples.append(sample_normalized)
            self.samples = normalized_samples
    
    def _collect_files(self):
        """收集所有文件"""
        # 获取所有支持的文件
        supported_extensions = {'.csv', '.xlsx', '.xls', '.txt'}
        files = []
        
        for file_path in self.data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
        
        # 提取标签（文件名不带扩展名）
        labels = []
        for file_path in files:
            label = file_path.stem  # 文件名不带扩展名
            if label not in self.label_names:
                self.label_names.append(label)
            labels.append(label)
        
        # 加载所有文件
        for file_path, label_name in zip(files, labels):
            label = self.label_names.index(label_name)
            self._load_file(file_path, label)
    
    def _load_file(self, file_path, label):
        """加载单个文件"""
        try:
            # 根据文件扩展名选择读取方法
            file_extension = file_path.suffix.lower()
            
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
            
            # 时间归一化处理
            if len(df.columns) > 1:
                time_col = df.columns[0]  # 假设第一列是时间
                time_values = df[time_col].values
                # 归一化时间到0-1区间
                if len(time_values) > 1:
                    time_values = (time_values - time_values.min()) / (time_values.max() - time_values.min())
            
            # 统一序列长度
            if len(current) >= self.seq_len:
                current = current[:self.seq_len]
            else:
                current = np.pad(current, (0, self.seq_len - len(current)), mode="constant")
                
            # 添加增强特征
            def _add_enhanced_features(seq):
                current = seq
                diff1 = np.diff(current, prepend=current[0])
                diff2 = np.diff(diff1, prepend=diff1[0])
                win_mean = np.convolve(current, np.ones(3)/3, mode='same')
                peak_val = np.max(current)
                valley_val = np.min(current)
                peak_pos = np.argmax(current) / len(current)
                
                peak_val_seq = np.full_like(current, peak_val)
                peak_pos_seq = np.full_like(current, peak_pos)
                valley_val_seq = np.full_like(current, valley_val)
                
                return np.stack([
                    current, diff1, diff2, win_mean,
                    peak_val_seq, peak_pos_seq, valley_val_seq
                ], axis=1)
            
            enhanced_features = _add_enhanced_features(current)
            self.samples.append(enhanced_features)
            self.labels.append(label)
        except Exception as e:
            print(f"读取 {file_path} 失败：{e}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.samples[idx])  # shape: (seq_len, features)
        y = torch.LongTensor([self.labels[idx]])   # 保持为二维张量 [1]
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
        "input_shape": (50, 7)  # 记录输入形状信息
    }, model_save_path)
    
    print(f"模型已保存到: {model_save_path}")
    print(f"标准化器已保存到: {scaler_save_path}")
    print(f"类别名称已保存到: label_names.json")

# 预测函数
def predict_waveform(file_path, model_path="waveform_model.pth", 
                     scaler_path="waveform_scaler.pkl",
                     loaded_model=None, loaded_scaler=None, label_names=None):
    """对单个波形文件进行分类预测
    
    Args:
        file_path: 待预测文件路径
        model_path: 模型文件路径
        scaler_path: 标准化器文件路径
        loaded_model: 已加载的模型对象，如果为None则从文件加载
        loaded_scaler: 已加载的标准化器对象，如果为None则从文件加载
        label_names: 类别名称列表，如果为None则从文件加载
    
    Returns:
        预测结果字典
    """
    # 加载模型
    if loaded_model is not None and label_names is not None:
        model = loaded_model
        model.eval()
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if label_names is None:
            with open("label_names.json", "r") as f:
                label_names = json.load(f)
        
        model = MGTransformer(
            d_model=checkpoint["config"]["d_model"],
            nhead=checkpoint["config"]["nhead"],
            dim_feedforward=checkpoint["config"]["dim_feedforward"],
            num_layers=checkpoint["config"]["num_layers"],
            num_classes=len(label_names),
            cnn_channels=checkpoint["config"]["cnn_channels"],
            dropout=checkpoint["config"]["dropout"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    
    # 加载标准化器
    if loaded_scaler is not None:
        scaler = loaded_scaler
    else:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    
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
    
    # 统一长度为50
    if len(current) >= 50:
        current = current[:50]
    else:
        current = np.pad(current, (0, 50 - len(current)), mode="constant")
    
    # 添加增强特征
    def _add_enhanced_features(seq):
        current = seq
        diff1 = np.diff(current, prepend=current[0])
        diff2 = np.diff(diff1, prepend=diff1[0])
        win_mean = np.convolve(current, np.ones(3)/3, mode='same')
        peak_val = np.max(current)
        valley_val = np.min(current)
        peak_pos = np.argmax(current) / len(current)
        
        peak_val_seq = np.full_like(current, peak_val)
        peak_pos_seq = np.full_like(current, peak_pos)
        valley_val_seq = np.full_like(current, valley_val)
        
        return np.stack([
            current, diff1, diff2, win_mean,
            peak_val_seq, peak_pos_seq, valley_val_seq
        ], axis=1)
    
    current = _add_enhanced_features(current)
    current = scaler.transform(current.reshape(-1, current.shape[-1])).reshape(current.shape)
    x = torch.FloatTensor(current)  # shape: (seq_len, features)
    
    # 预测
    with torch.no_grad():
        # 确保输入维度正确
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (seq_len, features) -> (1, seq_len, features)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(2)  # 处理一维情况
        
        outputs = model(x)  # x已经是正确的三维形状
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

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="波形分类系统")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True,
                        help="运行模式: train(训练) 或 predict(预测)")
    parser.add_argument("--data_dir", type=str, default="./waveform_data",
                        help="训练数据目录路径")
    parser.add_argument("--model_path", type=str, default="waveform_model.pth",
                        help="模型保存/加载路径")
    parser.add_argument("--scaler_path", type=str, default="waveform_scaler.pkl",
                        help="标准化器保存/加载路径")
    parser.add_argument("--file_path", type=str, default="",
                        help="待预测的CSV文件路径（预测模式下必需）")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(
            data_dir=args.data_dir,
            model_save_path=args.model_path,
            scaler_save_path=args.scaler_path,
            epochs=args.epochs
        )
    elif args.mode == "predict":
        if not args.file_path:
            raise ValueError("预测模式下必须提供 --file_path 参数")
        result = predict_waveform(
            file_path=args.file_path,
            model_path=args.model_path,
            scaler_path=args.scaler_path
        )
        print("预测结果:")
        print(f"预测类别: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("\n各类别概率:")
        for label, prob in result['all_probabilities'].items():
            print(f"  {label}: {prob:.4f}")