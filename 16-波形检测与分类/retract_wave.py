import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from scipy.signal import find_peaks, welch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------------- 配置与日志 --------------------------
# 设置matplotlib字体，仅保留Windows常用字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置日志（指定UTF-8编码，避免中文问题）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("waveform_processing.log", encoding='utf-8'),  # 关键：添加编码设置
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 替换特殊符号为ASCII兼容符号
SUCCESS = "[SUCCESS]"
ERROR = "[ERROR]"
WARNING = "[WARNING]"
INFO = "[INFO]"

class WaveformClusterClassifier:
    def __init__(self, csv_path, output_dir="output_cluster_ml", 
                 fixed_len=200, use_pca=False, pca_components=5):
        """初始化参数，增加更多可配置选项"""
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.fixed_len = fixed_len  # 允许外部指定周期长度
        self.use_pca = use_pca      # 是否使用PCA降维
        self.pca_components = pca_components  # PCA主成分数量
        
        # 创建分层目录保存不同阶段的中间结果
        self.subdirs = {
            "raw": os.path.join(output_dir, "0_raw_data"),
            "cycles": os.path.join(output_dir, "1_split_cycles"),
            "cluster": os.path.join(output_dir, "2_cluster_results"),
            "ml": os.path.join(output_dir, "3_ml_results"),
            "models": os.path.join(output_dir, "4_trained_models")
        }
        for dir_path in self.subdirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 原始数据
        self.time = None
        self.current = None
        # 周期数据（原始+归一化）
        self.cycles = []  # 每个元素：(原始时间, 原始电流, 归一化电流)
        # 聚类相关
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.cluster_labels = None
        self.normal_cluster_id = None
        self.pca_model = None  # PCA模型
        # 分类结果
        self.cluster_normal = []
        self.cluster_abnormal = []
        self.ml_normal = []
        self.ml_abnormal = []
        # 模型训练记录
        self.train_losses = []
        self.test_accuracies = []
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"{INFO} 使用设备: {self.device}")

    # -------------------------- 1. 数据加载与预处理 --------------------------
    def load_csv_data(self,start_time, end_time):
        """加载数据并保存原始数据可视化结果（仅保留30-50秒的数据）"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                data_header_idx = next(i for i, line in enumerate(lines) 
                                      if line.strip().startswith("Time(S),Current(A)"))

            # 读取完整数据
            df = pd.read_csv(self.csv_path, skiprows=data_header_idx, header=0, dtype=np.float64)

            # 过滤30-50秒的数据
            time_filter = (df["Time(S)"] >= start_time) & (df["Time(S)"] <= end_time)
            df_filtered = df[time_filter].copy()

            # 检查过滤后的数据是否为空
            if df_filtered.empty:
                logger.warning(f"{WARNING} 30-50秒范围内未找到数据")
                return False

            # 保存过滤后的数据
            self.time = df_filtered["Time(S)"].values
            self.current = df_filtered["Current(A)"].values

            # 保存过滤后的原始数据CSV
            df_filtered.to_csv(os.path.join(self.subdirs["raw"], "raw_data_filtered.csv"), index=False)
            # 同时保存完整原始数据供参考
            df.to_csv(os.path.join(self.subdirs["raw"], "raw_data_full.csv"), index=False)

            # 绘制过滤后的原始数据波形图
            plt.figure(figsize=(12, 5))
            plt.plot(self.time, self.current, color="#3498db")
            plt.axvline(x=30, color='r', linestyle='--', label='30s')
            plt.axvline(x=50, color='g', linestyle='--', label='50s')
            plt.title("30-50秒原始电流波形数据", fontsize=14)
            plt.xlabel("时间(S)"), plt.ylabel("电流(A)")
            plt.legend(), plt.grid(alpha=0.3), plt.tight_layout()
            plt.savefig(os.path.join(self.subdirs["raw"], "raw_waveform_filtered.png"))
            plt.close()

            logger.info(f"{SUCCESS} 加载并过滤数据：{len(self.time)}个采样点，时间范围 {self.time[0]:.2f}~{self.time[-1]:.2f}S")
            return True
        except Exception as e:
            logger.error(f"{ERROR} 数据加载失败：{str(e)}")
            return False

    def detect_peaks_valleys(self, peak_thr=0.2, valley_thr=0.2, min_dist=20):
        """检测波峰波谷并可视化"""
        try:
            current_max, current_min = np.max(self.current), np.min(self.current)
            current_range = current_max - current_min
            peak_threshold = current_min + current_range * peak_thr
            valleys_threshold = current_max - current_range * valley_thr
            
            self.peaks, _ = find_peaks(self.current, height=peak_threshold, distance=min_dist)
            self.valleys, _ = find_peaks(-self.current, height=-valleys_threshold, distance=min_dist)
            
            # 可视化波峰波谷检测结果
            plt.figure(figsize=(12, 5))
            plt.plot(self.time, self.current, color="#3498db", label="原始波形")
            plt.scatter(self.time[self.peaks], self.current[self.peaks], 
                       color="#e74c3c", s=50, marker='^', label="波峰")
            plt.scatter(self.time[self.valleys], self.current[self.valleys], 
                       color="#2ecc71", s=50, marker='v', label="波谷")
            plt.title("波峰波谷检测结果", fontsize=14)
            plt.xlabel("时间(S)"), plt.ylabel("电流(A)")
            plt.legend(), plt.grid(alpha=0.3), plt.tight_layout()
            plt.savefig(os.path.join(self.subdirs["raw"], "peaks_valleys_detection.png"))
            plt.close()
            
            if len(self.peaks) < 3 or len(self.valleys) < 3:
                logger.warning(f"{WARNING} 波峰({len(self.peaks)}个)或波谷({len(self.valleys)}个)数量不足")
            else:
                logger.info(f"{SUCCESS} 检测到波峰{len(self.peaks)}个，波谷{len(self.valleys)}个")
            return True
        except Exception as e:
            logger.error(f"{ERROR} 波峰波谷检测失败：{str(e)}")
            return False

    def split_cycles_by_extremes(self):
        """分割周期并保存所有周期的原始数据与图形（修复数组长度问题）"""
        try:
            all_extremes = np.sort(np.concatenate([self.peaks, self.valleys]))
            cycle_start = 0
            peak_count = 0
            valley_count = 0
            cycle_idx = 1  # 周期编号
            
            for idx in all_extremes:
                if idx in self.peaks:
                    peak_count += 1
                else:
                    valley_count += 1
                
                if peak_count >= 2 and valley_count >= 2:
                    # 提取周期数据
                    cycle_time = self.time[cycle_start:idx+1]
                    cycle_current = self.current[cycle_start:idx+1]
                    
                    # 增加长度检查，避免空周期或过短周期
                    if len(cycle_time) < 10 or len(cycle_current) < 10:  # 过滤过短周期
                        logger.warning(f"{WARNING} 跳过过短周期（长度：{len(cycle_time)}）")
                        cycle_start = idx + 1
                        peak_count = 0
                        valley_count = 0
                        continue
                    
                    # 确保时间和电流数组长度一致
                    if len(cycle_time) != len(cycle_current):
                        min_len = min(len(cycle_time), len(cycle_current))
                        cycle_time = cycle_time[:min_len]
                        cycle_current = cycle_current[:min_len]
                        logger.warning(f"{WARNING} 周期{cycle_idx}时间和电流长度不匹配，已截断到最短长度")
                    
                    # 归一化周期
                    cycle_current_norm = self._normalize_cycle(cycle_current)
                    self.cycles.append((cycle_time, cycle_current, cycle_current_norm))
                    
                    # 修复：分离保存原始数据和归一化数据（避免长度不匹配）
                    # 1. 保存原始周期数据（time和current长度一致）
                    raw_df = pd.DataFrame({
                        "Time(S)": cycle_time,
                        "Current(A)": cycle_current
                    })
                    raw_df.to_csv(os.path.join(self.subdirs["cycles"], f"cycle_{cycle_idx}_raw.csv"), index=False)
                    
                    # 2. 单独保存归一化数据（长度为fixed_len）
                    norm_df = pd.DataFrame({
                        "Normalized_Time": np.linspace(0, 1, self.fixed_len),
                        "Normalized_Current": cycle_current_norm
                    })
                    norm_df.to_csv(os.path.join(self.subdirs["cycles"], f"cycle_{cycle_idx}_normalized.csv"), index=False)
                    
                    # 绘制单个周期波形图
                    plt.figure(figsize=(10, 4))
                    plt.plot(cycle_time, cycle_current, color="#3498db")
                    plt.title(f"分割的周期 #{cycle_idx}", fontsize=12)
                    plt.xlabel("时间(S)"), plt.ylabel("电流(A)")
                    plt.grid(alpha=0.3), plt.tight_layout()
                    plt.savefig(os.path.join(self.subdirs["cycles"], f"cycle_{cycle_idx}.png"))
                    plt.close()
                    
                    # 更新计数器
                    cycle_start = idx + 1
                    peak_count = 0
                    valley_count = 0
                    cycle_idx += 1
            
            # 如果没有检测到周期
            if not self.cycles:
                logger.error(f"{ERROR} 未检测到有效周期")
                return False
            
            # 绘制所有周期的归一化对比图
            plt.figure(figsize=(12, 6))
            for i, (_, _, norm_curr) in enumerate(self.cycles):
                plt.plot(np.linspace(0, 1, self.fixed_len), norm_curr, 
                         alpha=0.6, linewidth=1, label=f"周期 {i+1}" if i < 10 else "")
            plt.title("所有周期的归一化波形", fontsize=14)
            plt.xlabel("归一化时间"), plt.ylabel("归一化电流")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'), plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.subdirs["cycles"], "all_cycles_normalized.png"))
            plt.close()
            
            logger.info(f"{SUCCESS} 分割完成：共{len(self.cycles)}个完整周期")
            return True
        except Exception as e:
            logger.error(f"{ERROR} 周期分割失败：{str(e)}")
            return False

    def _normalize_cycle(self, cycle):
        """周期归一化：插值到固定长度+标准化"""
        norm_x = np.linspace(0, 1, self.fixed_len)
        raw_x = np.linspace(0, 1, len(cycle))
        cycle_interp = np.interp(norm_x, raw_x, cycle)
        return self.scaler.fit_transform(cycle_interp.reshape(-1, 1)).flatten()

    # -------------------------- 2. 无监督聚类 --------------------------
    def extract_cycle_features(self, cycle_norm):
        """提取更丰富的波形特征"""
        # 1. 时域特征（8个）
        time_features = [
            np.mean(cycle_norm),          # 均值
            np.std(cycle_norm),           # 标准差
            np.max(cycle_norm),           # 最大值
            np.min(cycle_norm),           # 最小值
            np.max(cycle_norm) - np.min(cycle_norm),  # 峰峰值
            np.sqrt(np.mean(cycle_norm**2)),  # 均方根
            np.median(cycle_norm),        # 中位数
            np.sum(np.abs(np.diff(cycle_norm)))  # 一阶差分和（反映变化率）
        ]

        # 2. 频域特征（5个）- 修复警告
        # 确保nperseg不超过输入长度
        nperseg = min(256, len(cycle_norm))  # 关键修改
        freq, psd = welch(cycle_norm, fs=100, nperseg=nperseg)  # 增加参数
        top5_freq_idx = np.argsort(psd)[-5:]
        freq_features = psd[top5_freq_idx].tolist()

        # 3. 波形形态特征（2个）
        peaks, _ = find_peaks(cycle_norm)
        valleys, _ = find_peaks(-cycle_norm)
        shape_features = [
            len(peaks),  # 波峰数量
            len(valleys)  # 波谷数量
        ]

        return np.array(time_features + freq_features + shape_features)  # 共15个特征

    def cluster_cycles(self, max_cluster=10):
        """聚类并保存详细聚类过程数据"""
        try:
            # 检查周期数量是否足够
            if len(self.cycles) < 3:  # 至少需要3个样本才能聚类
                logger.error(f"{ERROR} 聚类失败：周期数量太少（{len(self.cycles)}个），至少需要3个")
                return False
                
            # 提取特征
            cycle_features = np.array([self.extract_cycle_features(c[2]) for c in self.cycles])
            cycle_features = self.scaler.fit_transform(cycle_features)
            
            # 保存原始特征
            pd.DataFrame(cycle_features).to_csv(
                os.path.join(self.subdirs["cluster"], "cycle_features.csv"), index=False)
            
            # 可选PCA降维
            if self.use_pca and self.pca_components < cycle_features.shape[1]:
                self.pca_model = PCA(n_components=self.pca_components)
                cycle_features = self.pca_model.fit_transform(cycle_features)
                pd.DataFrame(cycle_features).to_csv(
                    os.path.join(self.subdirs["cluster"], "pca_features.csv"), index=False)
                logger.info(f"{INFO} PCA降维完成：{self.pca_model.n_features_} → {self.pca_components}维，"
                           f"解释方差比：{sum(self.pca_model.explained_variance_ratio_):.3f}")
    
            # 聚类与轮廓系数计算
            # 调整最大聚类数不超过样本数-1
            max_possible_k = min(max_cluster, len(self.cycles) - 1)
            if max_possible_k < 2:
                logger.error(f"{ERROR} 聚类失败：样本数量不足，无法进行有效聚类")
                return False
                
            sil_scores = []
            k_range = range(2, max_possible_k + 1)  # 使用调整后的范围
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(cycle_features)
                sil_scores.append(silhouette_score(cycle_features, labels))
            
            # 保存轮廓系数变化
            plt.figure(figsize=(10, 4))
            plt.plot(k_range, sil_scores, 'o-', color="#3498db")
            plt.title("不同聚类数的轮廓系数", fontsize=14)
            plt.xlabel("聚类数K"), plt.ylabel("轮廓系数")
            plt.grid(alpha=0.3), plt.tight_layout()
            plt.savefig(os.path.join(self.subdirs["cluster"], "silhouette_scores.png"))
            plt.close()
            
            # 选择最优聚类
            best_k = k_range[np.argmax(sil_scores)]
            self.cluster_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            self.cluster_labels = self.cluster_model.fit_predict(cycle_features)
            
            # 确定正常簇
            cluster_counts = np.bincount(self.cluster_labels)
            self.normal_cluster_id = np.argmax(cluster_counts)
            
            # 划分结果
            self.cluster_normal = [
                (t, curr) for i, (t, curr, _) in enumerate(self.cycles)
                if self.cluster_labels[i] == self.normal_cluster_id
            ]
            self.cluster_abnormal = [
                (t, curr) for i, (t, curr, _) in enumerate(self.cycles)
                if self.cluster_labels[i] != self.normal_cluster_id
            ]
            
            # 保存聚类标签
            pd.DataFrame({
                "cycle_id": range(1, len(self.cycles)+1),
                "cluster_label": self.cluster_labels,
                "is_normal": [1 if l == self.normal_cluster_id else 0 for l in self.cluster_labels]
            }).to_csv(os.path.join(self.subdirs["cluster"], "cluster_labels.csv"), index=False)
            
            # 可视化聚类结果
            self.plot_cluster_result()
            
            logger.info(f"{SUCCESS} 聚类完成：最优K={best_k}，正常簇ID={self.normal_cluster_id}，"
                       f"正常周期{len(self.cluster_normal)}个，异常周期{len(self.cluster_abnormal)}个")
            return True
        except Exception as e:
            logger.error(f"{ERROR} 聚类失败：{str(e)}")
            return False

    def plot_cluster_result(self):
        """增强聚类可视化"""
        # 1. 所有周期的聚类分布（归一化波形）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 子图1：聚类波形叠加
        ax1.set_title("所有周期波形聚类结果", fontsize=14)
        ax1.set_xlabel("归一化时间"), ax1.set_ylabel("电流（归一化）")
        for i, (_, _, norm_curr) in enumerate(self.cycles):
            color = "#e74c3c" if self.cluster_labels[i] == self.normal_cluster_id else "#95a5a6"
            ax1.plot(np.linspace(0, 1, self.fixed_len), norm_curr, color=color, alpha=0.6, linewidth=1)
        
        # 子图2：聚类数量分布
        cluster_counts = np.bincount(self.cluster_labels)
        colors = ["#e74c3c" if i == self.normal_cluster_id else "#95a5a6" for i in range(len(cluster_counts))]
        ax2.bar(range(len(cluster_counts)), cluster_counts, color=colors)
        ax2.set_title("各聚类簇样本数量", fontsize=14)
        ax2.set_xlabel("聚类ID"), ax2.set_ylabel("周期数量")
        ax2.set_xticks(range(len(cluster_counts)))
        for i, v in enumerate(cluster_counts):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.subdirs["cluster"], "cluster_waveforms.png"))
        plt.close()
        
        # 2. 各类簇的平均波形对比
        unique_labels = np.unique(self.cluster_labels)
        plt.figure(figsize=(12, 6))
        for label in unique_labels:
            cluster_cycles = [c[2] for i, c in enumerate(self.cycles) if self.cluster_labels[i] == label]
            mean_cycle = np.mean(cluster_cycles, axis=0)
            plt.plot(np.linspace(0, 1, self.fixed_len), mean_cycle, 
                     linewidth=2, label=f"簇 {label}（{len(cluster_cycles)}个周期）")
        plt.title("各聚类簇的平均波形", fontsize=14)
        plt.xlabel("归一化时间"), plt.ylabel("归一化电流")
        plt.legend(), plt.grid(alpha=0.3), plt.tight_layout()
        plt.savefig(os.path.join(self.subdirs["cluster"], "cluster_mean_waveforms.png"))
        plt.close()

    # -------------------------- 3. 机器学习模型 --------------------------
    class WaveformDataset(Dataset):
        def __init__(self, cycles_norm, labels):
            self.cycles_norm = cycles_norm
            self.labels = labels

        def __len__(self):
            return len(self.cycles_norm)

        def __getitem__(self, idx):
            cycle = torch.FloatTensor(self.cycles_norm[idx])
            label = torch.LongTensor([self.labels[idx]])
            return cycle, label

    class TransformerClassifier(nn.Module):
        def __init__(self, input_len=200, d_model=64, nhead=4, num_layers=2):
            super().__init__()
            self.embedding = nn.Linear(1, d_model)
            self.pos_encoder = nn.Parameter(torch.randn(1, input_len, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=0.3, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model * input_len, 2)

        def forward(self, x):
            x = x.unsqueeze(2)  # [batch, len, 1]
            x = self.embedding(x) + self.pos_encoder  # [batch, len, d_model]
            x = self.transformer(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)
            return x

    def prepare_ml_data(self):
        """准备ML数据并保存数据集划分结果"""
        try:
            # 生成标签
            labels = np.array([
                0 if self.cluster_labels[i] == self.normal_cluster_id else 1
                for i in range(len(self.cycles))
            ])
            
            # 检查查类别分布
            unique, counts = np.unique(labels, return_counts=True)
            class_counts = dict(zip(unique, counts))
            logger.info(f"{INFO} 类别分布: {class_counts}")
            
            # 处理样本数量不平衡问题
            test_size = 0.3
            stratify = labels if all(v >= 2 for v in class_counts.values()) else None
            if stratify is None:
                logger.warning(f"{WARNING} 某些类别样本数少于2，将不使用分层抽样")
            
            # 划分数据集（根据类别分布决定是否使用分层抽样）
            X_train, X_test, y_train, y_test = train_test_split(
                np.array([c[2] for c in self.cycles]), 
                labels, 
                test_size=test_size, 
                random_state=42, 
                stratify=stratify  # 只有当所有类别都有足够样本时才使用分层抽样
            )
            
            # 保存数据集划分
            pd.DataFrame({
                "cycle_id": range(1, len(labels)+1),
                "dataset": ["train" if i in np.where(np.isin(labels, y_train))[0] else "test" 
                           for i in range(len(labels))],
                "label": labels
            }).to_csv(os.path.join(self.subdirs["ml"], "dataset_split.csv"), index=False)
            
            # 创建数据加载器
            self.train_dataset = self.WaveformDataset(X_train, y_train)
            self.test_dataset = self.WaveformDataset(X_test, y_test)
            self.infer_dataset = self.WaveformDataset(np.array([c[2] for c in self.cycles]), labels)
            
            self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=8, shuffle=False)
            self.infer_loader = DataLoader(self.infer_dataset, batch_size=8, shuffle=False)
            
            logger.info(f"{SUCCESS} ML数据准备完成：训练集{len(X_train)}个（正常{sum(y_train==0)}），"
                       f"测试集{len(X_test)}个（正常{sum(y_test==0)}）")
            return True
        except Exception as e:
            logger.error(f"{ERROR} ML数据准备失败：{str(e)}")
            return False

    def evaluate_model(self):
        """详细评估模型并保存混淆矩阵（修复标签不匹配问题）"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.squeeze().to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        # 修复：确保标签与实际存在的类别一致
        unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
        display_labels = ["正常", "异常"] if len(unique_labels) >= 2 else ["正常"]

        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
        plt.title("测试集混淆矩阵", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.subdirs["ml"], "confusion_matrix.png"))
        plt.close()

        # 同时修复学习率调度器警告
        return 100 * correct / total

    def train_transformer(self, epochs=20, lr=1e-3, patience=5):
        """增强模型训练过程，修复学习率调度器警告"""
        try:
            self.model = self.TransformerClassifier(input_len=self.fixed_len).to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

            # 修复：移除verbose参数以消除警告
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=3  # 移除verbose=True
            )  # 学习率调度器

            self.train_losses = []
            self.test_accuracies = []
            best_test_acc = 0.0
            early_stop_counter = 0  # 早停计数器

            for epoch in range(epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0.0
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.squeeze().to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item() * inputs.size(0)

                # 计算训练指标
                avg_train_loss = train_loss / len(self.train_loader.dataset)
                self.train_losses.append(avg_train_loss)

                # 测试阶段
                test_acc = self.evaluate_model()
                self.test_accuracies.append(test_acc)

                # 学习率调度
                self.scheduler.step(test_acc)

                # 早停检查
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    torch.save(self.model.state_dict(), os.path.join(self.subdirs["models"], "best_transformer.pth"))
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        logger.info(f"{INFO} 早停触发：在第{epoch+1}轮未提升")
                        break
                    
                logger.info(f"Epoch {epoch+1:2d}/{epochs} | 训练损失: {avg_train_loss:.4f} | 测试精度: {test_acc:.1f}%")

            # 保存训练曲线
            self.plot_training_curves()

            # 保存最终模型
            torch.save(self.model.state_dict(), os.path.join(self.subdirs["models"], "final_transformer.pth"))
            logger.info(f"{SUCCESS} 模型训练完成，最优精度：{best_test_acc:.1f}%")
            return True
        except Exception as e:
            logger.error(f"{ERROR} 模型训练失败：{str(e)}")
            return False

    def plot_training_curves(self):
        """绘制训练损失和测试精度曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        ax1.plot(range(1, len(self.train_losses)+1), self.train_losses, 'o-', color="#e74c3c")
        ax1.set_title("训练损失曲线", fontsize=14)
        ax1.set_xlabel("Epoch"), ax1.set_ylabel("损失值")
        ax1.grid(alpha=0.3)
        
        # 精度曲线
        ax2.plot(range(1, len(self.test_accuracies)+1), self.test_accuracies, 'o-', color="#3498db")
        ax2.set_title("测试精度曲线", fontsize=14)
        ax2.set_xlabel("Epoch"), ax2.set_ylabel("精度(%)")
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.subdirs["ml"], "training_curves.png"))
        plt.close()



    def infer_with_model(self):
        """推理并保存ML分类结果的详细对比"""
        try:
            # 加载最优模型
            self.model = self.TransformerClassifier(input_len=self.fixed_len).to(self.device)
            self.model.load_state_dict(torch.load(os.path.join(self.subdirs["models"], "best_transformer.pth")))
            self.model.eval()
            
            # 推理
            preds = []
            with torch.no_grad():
                for inputs, _ in self.infer_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    _, batch_preds = torch.max(outputs, 1)
                    preds.extend(batch_preds.cpu().numpy())
            
            # 划分结果
            self.ml_normal = [
                (t, curr) for i, (t, curr, _) in enumerate(self.cycles) if preds[i] == 0
            ]
            self.ml_abnormal = [
                (t, curr) for i, (t, curr, _) in enumerate(self.cycles) if preds[i] == 1
            ]
            
            # 保存ML分类标签
            pd.DataFrame({
                "cycle_id": range(1, len(self.cycles)+1),
                "cluster_label": self.cluster_labels,
                "ml_prediction": preds,
                "is_match": [
                    1 if (preds[i] == 0 and self.cluster_labels[i] == self.normal_cluster_id) or
                        (preds[i] == 1 and self.cluster_labels[i] != self.normal_cluster_id)
                    else 0 for i in range(len(self.cycles))
                ]
            }).to_csv(os.path.join(self.subdirs["ml"], "ml_predictions.csv"), index=False)
            
            # 可视化ML与聚类结果的对比
            self.plot_classification_comparison()
            
            logger.info(f"{SUCCESS} ML推理完成：正常{len(self.ml_normal)}个，异常{len(self.ml_abnormal)}个")
            return True
        except Exception as e:
            logger.error(f"{ERROR} ML推理失败：{str(e)}")
            return False

    # -------------------------- 4. 结果保存与可视化 --------------------------
    def save_results(self):
        """保存所有分类结果的详细数据与图形"""
        # 保存聚类分类结果
        self._save_cycle_group("cluster_normal", self.cluster_normal)
        self._save_cycle_group("cluster_abnormal", self.cluster_abnormal)
        
        # 保存ML分类结果
        self._save_cycle_group("ml_normal", self.ml_normal)
        self._save_cycle_group("ml_abnormal", self.ml_abnormal)
        
        # 保存正常周期的平均波形（用于后续参考）
        if self.cluster_normal:
            normal_cycles_norm = [c[2] for i, c in enumerate(self.cycles) 
                                if self.cluster_labels[i] == self.normal_cluster_id]
            normal_mean = np.mean(normal_cycles_norm, axis=0)
            pd.DataFrame({
                "normalized_time": np.linspace(0, 1, self.fixed_len),
                "mean_current": normal_mean
            }).to_csv(os.path.join(self.subdirs["cluster"], "normal_mean_waveform.csv"), index=False)
            
            plt.figure(figsize=(10, 4))
            plt.plot(np.linspace(0, 1, self.fixed_len), normal_mean, color="#e74c3c", linewidth=2)
            plt.title("正常周期的平均波形", fontsize=14)
            plt.xlabel("归一化时间"), plt.ylabel("归一化电流")
            plt.grid(alpha=0.3), plt.tight_layout()
            plt.savefig(os.path.join(self.subdirs["cluster"], "normal_mean_waveform.png"))
            plt.close()

    def _save_cycle_group(self, group_name, cycles):
        """保存分类后的周期数据与图形"""
        group_dir = os.path.join(self.subdirs["ml" if "ml" in group_name else "cluster"], group_name)
        os.makedirs(group_dir, exist_ok=True)
        
        # 保存单个周期
        for i, (cycle_time, cycle_current) in enumerate(cycles, 1):
            pd.DataFrame({
                "Time(S)": cycle_time,
                "Current(A)": cycle_current
            }).to_csv(os.path.join(group_dir, f"{group_name}_cycle_{i}.csv"), index=False)
            
            plt.figure(figsize=(8, 3))
            plt.plot(cycle_time, cycle_current, color="#e74c3c" if "normal" in group_name else "#95a5a6")
            plt.title(f"{group_name.replace('_', ' ')} Cycle {i}", fontsize=12)
            plt.xlabel("Time(S)"), plt.ylabel("Current(A)")
            plt.grid(alpha=0.3), plt.tight_layout()
            plt.savefig(os.path.join(group_dir, f"{group_name}_cycle_{i}.png"))
            plt.close()
        
        # 保存汇总图
        if cycles:
            plt.figure(figsize=(10, 4))
            plt.title(f"All {group_name.replace('_', ' ')} Cycles", fontsize=12)
            plt.xlabel("Normalized Time"), plt.ylabel("Current(A)")
            for i, (cycle_time, cycle_current) in enumerate(cycles[:10]):
                norm_time = np.linspace(0, 1, len(cycle_time))
                plt.plot(norm_time, cycle_current, alpha=0.7, label=f"Cycle {i+1}")
            plt.legend(), plt.grid(alpha=0.3), plt.tight_layout()
            plt.savefig(os.path.join(group_dir, f"all_{group_name}_cycles.png"))
            plt.close()
        
        logger.info(f"{SUCCESS} 已保存 {len(cycles)} 个 {group_name} 周期到 {group_dir}")

    def plot_classification_comparison(self):
        """增强版分类结果对比可视化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 聚类结果
        ax1.bar(["正常", "异常"], [len(self.cluster_normal), len(self.cluster_abnormal)],
                color=["#e74c3c", "#95a5a6"], width=0.6)
        ax1.set_title("无监督聚类分类结果", fontsize=14)
        ax1.set_ylabel("周期数量")
        for i, v in enumerate([len(self.cluster_normal), len(self.cluster_abnormal)]):
            ax1.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=12)
        
        # ML结果
        ax2.bar(["正常", "异常"], [len(self.ml_normal), len(self.ml_abnormal)],
                color=["#3498db", "#95a5a6"], width=0.6)
        ax2.set_title("Transformer分类结果", fontsize=14)
        ax2.set_ylabel("周期数量")
        for i, v in enumerate([len(self.ml_normal), len(self.ml_abnormal)]):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.subdirs["ml"], "classification_comparison.png"))
        plt.close()


# -------------------------- 调用示例 --------------------------
if __name__ == "__main__":
    # 配置参数
    CSV_PATH = "16-波形检测与分类\\knee-sensor\\内翻-0-1.csv"  # 替换为实际路径
    MAX_CLUSTER = 10
    EPOCHS = 50
    USE_PCA = False  # 是否启用PCA降维
    PCA_COMPONENTS = 5  # PCA主成分数量

    # 初始化处理器
    processor = WaveformClusterClassifier(
        csv_path=CSV_PATH,
        fixed_len=200,
        use_pca=USE_PCA,
        pca_components=PCA_COMPONENTS
    )

    # 执行完整流程
    try:
        # 步骤1：数据加载与预处理
        if not processor.load_csv_data(start_time=20, end_time=130):
            raise Exception("数据加载失败")
        if not processor.detect_peaks_valleys(peak_thr=0.2, valley_thr=0.2, min_dist=20):
            raise Exception("波峰波谷检测失败")
        if not processor.split_cycles_by_extremes():
            raise Exception("周期分割失败")

        # 步骤2：无监督聚类
        if not processor.cluster_cycles(max_cluster=MAX_CLUSTER):
            raise Exception("聚类失败")

        # 步骤3：机器学习模型
        if not processor.prepare_ml_data():
            raise Exception("ML数据准备失败")
        if not processor.train_transformer(epochs=EPOCHS, lr=1e-3):
            raise Exception("模型训练失败")
        if not processor.infer_with_model():
            raise Exception("ML推理失败")

        # 步骤4：保存所有结果
        processor.save_results()

        logger.info(f"\n{SUCCESS} 所有流程执行完成！结果已保存到 output_cluster_ml 目录")

    except Exception as e:
        logger.error(f"\n{ERROR} 执行出错：{str(e)}")
