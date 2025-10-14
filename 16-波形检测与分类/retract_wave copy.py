import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 解决Tkinter线程冲突
import matplotlib.pyplot as plt
import logging
from scipy.signal import find_peaks, welch
from scipy.spatial.distance import cdist
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
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("waveform_processing.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SUCCESS = "[SUCCESS]"
ERROR = "[ERROR]"
WARNING = "[WARNING]"
INFO = "[INFO]"

class WaveformClusterClassifier:
    def __init__(self, csv_path, output_dir="output_cluster_ml", 
                 fixed_len=200, use_pca=False, pca_components=5):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.fixed_len = fixed_len  # 波形序列长度（200）
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.normal_mean_cycle = None  # 正常簇平均波形（用于特征增强）
        
        self.subdirs = {
            "raw": os.path.join(output_dir, "0_raw_data"),
            "cycles": os.path.join(output_dir, "1_split_cycles"),
            "cluster": os.path.join(output_dir, "2_cluster_results"),
            "ml": os.path.join(output_dir, "3_ml_results"),
            "models": os.path.join(output_dir, "4_trained_models")
        }
        for dir_path in self.subdirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        self.time = None
        self.current = None
        self.cycles = []  # 存储 (原始时间, 原始电流, 归一化电流序列)
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.cluster_labels = None
        self.normal_cluster_id = None
        self.pca_model = None
        self.cluster_normal = []  # 聚类正常周期（原始时间+电流）
        self.cluster_abnormal = []  # 聚类异常周期
        self.ml_normal = []  # ML正常周期
        self.ml_abnormal = []  # ML异常周期
        self.train_losses = []
        self.test_accuracies = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"{INFO} 使用设备: {self.device}")

    # -------------------------- 1. 数据加载与预处理（确保波形序列维度一致） --------------------------
    def load_csv_data(self, start_time, end_time):
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                data_header_idx = next(i for i, line in enumerate(lines) 
                                      if line.strip().startswith("Time(S),Current(A)"))

            df = pd.read_csv(self.csv_path, skiprows=data_header_idx, header=0, dtype=np.float64)
            time_filter = (df["Time(S)"] >= start_time) & (df["Time(S)"] <= end_time)
            df_filtered = df[time_filter].copy()

            if df_filtered.empty:
                logger.warning(f"{WARNING} {start_time}-{end_time}秒范围内未找到数据")
                return False

            self.time = df_filtered["Time(S)"].values
            self.current = df_filtered["Current(A)"].values

            # 保存原始数据
            df_filtered.to_csv(os.path.join(self.subdirs["raw"], "raw_data_filtered.csv"), index=False)
            df.to_csv(os.path.join(self.subdirs["raw"], "raw_data_full.csv"), index=False)

            # 绘制原始波形
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(self.time, self.current, color="#3498db")
            ax.axvline(x=start_time, color='r', linestyle='--', label=f'{start_time}s')
            ax.axvline(x=end_time, color='g', linestyle='--', label=f'{end_time}s')
            ax.set_title(f"{start_time}-{end_time}秒原始电流波形数据", fontsize=14)
            ax.set_xlabel("时间(S)"), ax.set_ylabel("电流(A)")
            ax.legend(), ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.subdirs["raw"], "raw_waveform_filtered.png"))
            plt.close(fig)
            del fig, ax

            logger.info(f"{SUCCESS} 加载并过滤数据：{len(self.time)}个采样点，时间范围 {self.time[0]:.2f}~{self.time[-1]:.2f}S")
            return True
        except Exception as e:
            logger.error(f"{ERROR} 数据加载失败：{str(e)}")
            return False

    def detect_peaks_valleys(self, peak_thr=0.2, valley_thr=0.2, min_dist=20):
        try:
            current_max, current_min = np.max(self.current), np.min(self.current)
            current_range = current_max - current_min
            peak_threshold = current_min + current_range * peak_thr
            valleys_threshold = current_max - current_range * valley_thr
            
            self.peaks, _ = find_peaks(self.current, height=peak_threshold, distance=min_dist)
            self.valleys, _ = find_peaks(-self.current, height=-valleys_threshold, distance=min_dist)
            
            # 绘制波峰波谷
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(self.time, self.current, color="#3498db", label="原始波形")
            ax.scatter(self.time[self.peaks], self.current[self.peaks], 
                       color="#e74c3c", s=50, marker='^', label="波峰")
            ax.scatter(self.time[self.valleys], self.current[self.valleys], 
                       color="#2ecc71", s=50, marker='v', label="波谷")
            ax.set_title("波峰波谷检测结果", fontsize=14)
            ax.set_xlabel("时间(S)"), ax.set_ylabel("电流(A)")
            ax.legend(), ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.subdirs["raw"], "peaks_valleys_detection.png"))
            plt.close(fig)
            del fig, ax
            
            if len(self.peaks) < 3 or len(self.valleys) < 3:
                logger.warning(f"{WARNING} 波峰({len(self.peaks)}个)或波谷({len(self.valleys)}个)数量不足")
            else:
                logger.info(f"{SUCCESS} 检测到波峰{len(self.peaks)}个，波谷{len(self.valleys)}个")
            return True
        except Exception as e:
            logger.error(f"{ERROR} 波峰波谷检测失败：{str(e)}")
            return False

    def split_cycles_by_extremes(self):
        try:
            all_extremes = np.sort(np.concatenate([self.peaks, self.valleys]))
            cycle_start = 0
            peak_count = 0
            valley_count = 0
            cycle_idx = 1
            
            for idx in all_extremes:
                if idx in self.peaks:
                    peak_count += 1
                else:
                    valley_count += 1
                
                # 至少2个波峰+2个波谷才视为完整周期
                if peak_count >= 2 and valley_count >= 2:
                    cycle_time = self.time[cycle_start:idx+1]
                    cycle_current = self.current[cycle_start:idx+1]
                    
                    # 过滤过短周期
                    if len(cycle_time) < 10 or len(cycle_current) < 10:
                        logger.warning(f"{WARNING} 跳过过短周期（长度：{len(cycle_time)}）")
                        cycle_start = idx + 1
                        peak_count = 0
                        valley_count = 0
                        continue
                    
                    # 确保时间和电流长度一致
                    if len(cycle_time) != len(cycle_current):
                        min_len = min(len(cycle_time), len(cycle_current))
                        cycle_time = cycle_time[:min_len]
                        cycle_current = cycle_current[:min_len]
                        logger.warning(f"{WARNING} 周期{cycle_idx}时间和电流长度不匹配，已截断到{min_len}个点")
                    
                    # 归一化波形序列（固定为200维）
                    cycle_current_norm = self._normalize_cycle(cycle_current)
                    self.cycles.append((cycle_time, cycle_current, cycle_current_norm))
                    
                    # 保存周期数据
                    raw_df = pd.DataFrame({
                        "Time(S)": cycle_time,
                        "Current(A)": cycle_current
                    })
                    raw_df.to_csv(os.path.join(self.subdirs["cycles"], f"cycle_{cycle_idx}_raw.csv"), index=False)
                    
                    norm_df = pd.DataFrame({
                        "Normalized_Time": np.linspace(0, 1, self.fixed_len),
                        "Normalized_Current": cycle_current_norm
                    })
                    norm_df.to_csv(os.path.join(self.subdirs["cycles"], f"cycle_{cycle_idx}_normalized.csv"), index=False)
                    
                    # 绘制单个周期
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(cycle_time, cycle_current, color="#3498db")
                    ax.set_title(f"分割的周期 #{cycle_idx}", fontsize=12)
                    ax.set_xlabel("时间(S)"), ax.set_ylabel("电流(A)")
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.subdirs["cycles"], f"cycle_{cycle_idx}.png"))
                    plt.close(fig)
                    del fig, ax
                    
                    # 重置周期起始点和计数
                    cycle_start = idx + 1
                    peak_count = 0
                    valley_count = 0
                    cycle_idx += 1
            
            if not self.cycles:
                logger.error(f"{ERROR} 未检测到有效周期")
                return False
            
            # 绘制所有归一化周期
            fig, ax = plt.subplots(figsize=(12, 6))
            for i, (_, _, norm_curr) in enumerate(self.cycles):
                ax.plot(np.linspace(0, 1, self.fixed_len), norm_curr, 
                         alpha=0.6, linewidth=1, label=f"周期 {i+1}" if i < 10 else "")
            ax.set_title("所有周期的归一化波形（200维）", fontsize=14)
            ax.set_xlabel("归一化时间"), ax.set_ylabel("归一化电流")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left'), ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.subdirs["cycles"], "all_cycles_normalized.png"))
            plt.close(fig)
            del fig, ax
            
            logger.info(f"{SUCCESS} 分割完成：共{len(self.cycles)}个完整周期，每个周期归一化为{self.fixed_len}维")
            return True
        except Exception as e:
            logger.error(f"{ERROR} 周期分割失败：{str(e)}")
            return False

    def _normalize_cycle(self, cycle):
        """将任意长度的周期电流归一化为固定200维序列"""
        # 线性插值到固定长度
        norm_x = np.linspace(0, 1, self.fixed_len)
        raw_x = np.linspace(0, 1, len(cycle))
        cycle_interp = np.interp(norm_x, raw_x, cycle)
        # 标准化（均值为0，标准差为1）
        return self.scaler.fit_transform(cycle_interp.reshape(-1, 1)).flatten()

    # -------------------------- 2. 无监督聚类（严格筛选正常波形） --------------------------
    def extract_cycle_features(self, cycle_norm):
        """从200维归一化波形中提取18维特征（用于聚类）"""
        # 1. 时域特征（8个）
        time_features = [
            np.mean(cycle_norm), np.std(cycle_norm), np.max(cycle_norm), np.min(cycle_norm),
            np.max(cycle_norm) - np.min(cycle_norm), np.sqrt(np.mean(cycle_norm**2)),  # 均方根
            np.median(cycle_norm), np.sum(np.abs(np.diff(cycle_norm)))  # 一阶差分和（变化率）
        ]

        # 2. 频域特征（5个）
        nperseg = min(256, len(cycle_norm))
        freq, psd = welch(cycle_norm, fs=100, nperseg=nperseg)
        # 确保频域特征为5个（不足补0）
        if len(psd) < 5:
            freq_features = np.pad(psd, (0, 5 - len(psd)), mode='constant').tolist()
        else:
            top5_freq_idx = np.argsort(psd)[-5:]  # 取功率最大的5个频率
            freq_features = psd[top5_freq_idx].tolist()

        # 3. 形态特征（5个）
        peaks, _ = find_peaks(cycle_norm, distance=10)  # 波峰检测（最小间距10）
        valleys, _ = find_peaks(-cycle_norm, distance=10)  # 波谷检测
        curvature = np.abs(np.diff(np.diff(cycle_norm)))  # 二阶差分（曲率，反映突变）
        third = len(cycle_norm) // 3
        shape_features = [
            len(peaks), len(valleys), 
            np.max(curvature) if len(curvature) > 0 else 0,
            np.sum(cycle_norm[:third]**2) / np.sum(cycle_norm**2) if np.sum(cycle_norm**2) != 0 else 0  # 前期能量占比
        ]
        
        # 新增：与正常簇平均波形的余弦相似度（聚类后更新）
        if self.normal_mean_cycle is not None:
            cos_sim = np.dot(cycle_norm, self.normal_mean_cycle) / (
                np.linalg.norm(cycle_norm) * np.linalg.norm(self.normal_mean_cycle)
                if (np.linalg.norm(cycle_norm) * np.linalg.norm(self.normal_mean_cycle)) != 0 else 1
            )
        else:
            cos_sim = 0.0
        shape_features.append(cos_sim)

        # 强制特征长度为18
        assert len(time_features + freq_features + shape_features) == 18, "特征长度必须为18"
        return np.array(time_features + freq_features + shape_features, dtype=np.float64)

    def cluster_cycles(self, max_cluster=8, sil_threshold=0.01, distance_percentile=60):
        """
        放宽聚类逻辑：
        1. 轮廓系数阈值降至0.01（允许更多有效聚类）
        2. 正常簇为前60%最紧密的簇（大幅放宽紧密性要求）
        3. 候选簇样本数≥1（不再过滤单样本正常簇）
        4. 核心样本保留前100%（不排除任何正常簇样本）
        """
        try:
            if len(self.cycles) < 2:
                logger.error(f"{ERROR} 聚类失败：周期数量太少（{len(self.cycles)}个），至少需要2个")
                return False
                
            # 提取特征并标准化（用于聚类）
            cycle_features = np.array([self.extract_cycle_features(c[2]) for c in self.cycles], dtype=np.float64)
            cycle_features = self.scaler.fit_transform(cycle_features)
            pd.DataFrame(cycle_features).to_csv(
                os.path.join(self.subdirs["cluster"], "cycle_features.csv"), index=False)
            logger.info(f"{INFO} 聚类特征数组形状：{cycle_features.shape}（样本数×18维特征）")
            
            # PCA降维（可选，若用则关闭，避免特征损失导致聚类偏差）
            if self.use_pca and self.pca_components < cycle_features.shape[1]:
                self.pca_model = PCA(n_components=self.pca_components)
                cycle_features = self.pca_model.fit_transform(cycle_features)
                pd.DataFrame(cycle_features).to_csv(
                    os.path.join(self.subdirs["cluster"], "pca_features.csv"), index=False)
                logger.info(f"{INFO} PCA降维完成：{self.pca_model.n_features_} → {self.pca_components}维，解释方差比：{sum(self.pca_model.explained_variance_ratio_):.3f}")
    
            # 1. 筛选最优聚类数K（优先选小K，避免簇过度细分）
            max_possible_k = min(max_cluster, len(self.cycles) - 1)
            if max_possible_k < 2:
                logger.warning(f"{WARNING} 样本数量不足，强制聚类数K=2")
                best_k = 2
            else:
                sil_scores = []
                k_range = range(2, max_possible_k + 1)
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(cycle_features)
                    sil_scores.append(silhouette_score(cycle_features, labels))
                
                # 保存轮廓系数图
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(k_range, sil_scores, 'o-', color="#3498db")
                ax.axhline(y=sil_threshold, color='r', linestyle='--', label=f'轮廓系数阈值={sil_threshold}')
                ax.set_title("不同聚类数的轮廓系数（放宽筛选）", fontsize=14)
                ax.set_xlabel("聚类数K"), ax.set_ylabel("轮廓系数")
                ax.legend(), ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.subdirs["cluster"], "silhouette_scores.png"))
                plt.close(fig)
                del fig, ax
                
                # 选择最优K：优先选“轮廓系数≥阈值的最小K”（避免簇过度细分）
                valid_k = [k for k, sil in zip(k_range, sil_scores) if sil >= sil_threshold]
                if valid_k:
                    best_k = min(valid_k)  # 关键：选最小K，减少簇细分，让正常簇样本数更多
                else:
                    logger.warning(f"{WARNING} 无轮廓系数≥{sil_threshold}的聚类，选择轮廓系数最大的K={k_range[np.argmax(sil_scores)]}")
                    best_k = k_range[np.argmax(sil_scores)]
            
            # 2. 训练KMeans模型（小K值，减少簇细分）
            self.cluster_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            self.cluster_labels = self.cluster_model.fit_predict(cycle_features)
            cluster_centers = self.cluster_model.cluster_centers_
            cluster_counts = np.bincount(self.cluster_labels)
            logger.info(f"{INFO} 聚类结果：K={best_k}，各簇样本数：{cluster_counts}")
    
            # 3. 计算各簇紧密性（平均距离越小越紧密）
            cluster_distances = []
            for label in np.unique(self.cluster_labels):
                cluster_samples = cycle_features[self.cluster_labels == label]
                avg_dist = np.mean(cdist(cluster_samples, [cluster_centers[label]], metric='euclidean'))
                cluster_distances.append(avg_dist)
            distance_threshold = np.percentile(cluster_distances, distance_percentile)  # 前60%最紧密
            
            # 4. 放宽正常簇筛选：样本数≥1 + 前60%最紧密（核心修改）
            candidate_clusters = [
                label for label in range(best_k) 
                if cluster_counts[label] >= 1 and cluster_distances[label] <= distance_threshold
            ]
            logger.info(f"{INFO} 候选正常簇（前{distance_percentile}%最紧密+样本数≥1）：{candidate_clusters}")
            
            # 确定最终正常簇（候选簇中样本数最多的，若没有则选“紧密性最小的簇”）
            if candidate_clusters:
                candidate_sizes = [cluster_counts[label] for label in candidate_clusters]
                self.normal_cluster_id = candidate_clusters[np.argmax(candidate_sizes)]
            else:
                logger.warning(f"{WARNING} 无候选簇，选择最紧密的簇（平均距离最小）")
                self.normal_cluster_id = np.argmin(cluster_distances)  # 选最紧密的簇（更可能是正常簇）
            
            # 5. 核心样本筛选：保留100%（不排除任何样本，彻底放宽）
            normal_sample_idx = np.where(self.cluster_labels == self.normal_cluster_id)[0]
            core_sample_idx = normal_sample_idx  # 直接保留所有正常簇样本，不筛选
            logger.info(f"{INFO} 正常簇（ID={self.normal_cluster_id}）样本数：{len(core_sample_idx)}")
    
            # 6. 划分聚类结果
            self.cluster_normal = [self.cycles[i][:2] for i in core_sample_idx]
            self.cluster_abnormal = [
                self.cycles[i][:2] for i in range(len(self.cycles)) 
                if i not in core_sample_idx
            ]
    
            # 保存正常簇平均波形
            normal_cycles_norm = [self.cycles[i][2] for i in core_sample_idx]
            self.normal_mean_cycle = np.mean(normal_cycles_norm, axis=0) if len(normal_cycles_norm) > 0 else np.zeros(self.fixed_len)
    
            # 保存聚类标签
            is_normal = np.zeros(len(self.cycles), dtype=int)
            is_normal[core_sample_idx] = 1
            cluster_result = pd.DataFrame({
                "cycle_id": range(1, len(self.cycles)+1),
                "cluster_label": self.cluster_labels,
                "is_normal": is_normal,
                "distance_to_center": np.concatenate([
                    [np.mean(cdist([cycle_features[i]], [cluster_centers[self.normal_cluster_id]], metric='euclidean')) for i in core_sample_idx],
                    [np.nan]*(len(self.cycles)-len(core_sample_idx))
                ])
            })
            cluster_result.to_csv(os.path.join(self.subdirs["cluster"], "cluster_labels.csv"), index=False)
            
            # 绘制聚类结果
            self.plot_cluster_result(core_sample_idx, normal_sample_idx)
            
            logger.info(f"{SUCCESS} 放宽聚类完成：最优K={best_k}，正常簇ID={self.normal_cluster_id}")
            logger.info(f"          | 正常周期{len(self.cluster_normal)}个，异常周期{len(self.cluster_abnormal)}个")
            logger.info(f"          | 聚类参数：轮廓系数阈值={sil_threshold}，簇紧密性阈值={distance_threshold:.4f}")
            return True
        except Exception as e:
            logger.error(f"{ERROR} 聚类失败：{str(e)}")
            return False

    def plot_cluster_result(self, core_sample_idx, normal_sample_idx):
        """绘制聚类结果可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1：所有周期聚类分布（核心正常标红，边缘正常标橙，异常标灰）
        ax1.set_title("所有周期波形聚类结果", fontsize=14)
        ax1.set_xlabel("归一化时间"), ax1.set_ylabel("归一化电流")
        for i, (_, _, norm_curr) in enumerate(self.cycles):
            if i in core_sample_idx:
                color = "#e74c3c"  # 核心正常：红色
            elif i in normal_sample_idx:
                color = "#f39c12"  # 边缘正常：橙色
            else:
                color = "#95a5a6"  # 异常：灰色
            ax1.plot(np.linspace(0, 1, self.fixed_len), norm_curr, color=color, alpha=0.6, linewidth=1)
        ax1.plot(np.linspace(0, 1, self.fixed_len), self.normal_mean_cycle, color="#27ae60", linewidth=3, label="正常平均波形")
        ax1.legend(), ax1.grid(alpha=0.3)
        
        # 子图2：各簇样本数量
        cluster_counts = np.bincount(self.cluster_labels)
        colors = ["#e74c3c" if label == self.normal_cluster_id else "#95a5a6" for label in range(len(cluster_counts))]
        ax2.bar(range(len(cluster_counts)), cluster_counts, color=colors)
        ax2.set_title("各聚类簇样本数量（红色为正常簇）", fontsize=14)
        ax2.set_xlabel("聚类ID"), ax2.set_ylabel("周期数量")
        ax2.set_xticks(range(len(cluster_counts)))
        for i, v in enumerate(cluster_counts):
            ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=12)
        
        # 子图3：正常簇核心vs边缘样本
        core_norm = [self.cycles[i][2] for i in core_sample_idx]
        edge_norm = [self.cycles[i][2] for i in normal_sample_idx if i not in core_sample_idx]
        ax3.plot(np.linspace(0, 1, self.fixed_len), np.mean(core_norm, axis=0), color="#e74c3c", linewidth=2, label="核心样本平均")
        if edge_norm:
            ax3.plot(np.linspace(0, 1, self.fixed_len), np.mean(edge_norm, axis=0), color="#f39c12", linewidth=2, label="边缘样本平均")
        ax3.set_title("正常簇核心样本 vs 边缘样本", fontsize=14)
        ax3.set_xlabel("归一化时间"), ax3.set_ylabel("归一化电流")
        ax3.legend(), ax3.grid(alpha=0.3)
        
        # 子图4：各簇紧密性（平均距离）
        cluster_distances = []
        for label in np.unique(self.cluster_labels):
            cluster_samples = np.array([self.cycles[i][2] for i in range(len(self.cycles)) if self.cluster_labels[i] == label])
            avg_dist = np.mean(cdist(cluster_samples, [np.mean(cluster_samples, axis=0)], metric='euclidean'))
            cluster_distances.append(avg_dist)
        colors_dist = ["#e74c3c" if label == self.normal_cluster_id else "#95a5a6" for label in range(len(cluster_distances))]
        ax4.bar(range(len(cluster_distances)), cluster_distances, color=colors_dist)
        ax4.set_title("各簇平均距离（越小越紧密）", fontsize=14)
        ax4.set_xlabel("聚类ID"), ax4.set_ylabel("平均欧氏距离")
        ax4.set_xticks(range(len(cluster_distances)))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.subdirs["cluster"], "cluster_waveforms.png"))
        plt.close(fig)
        del fig, ax1, ax2, ax3, ax4

    # -------------------------- 3. 机器学习模型（基于200维波形序列，准确分类） --------------------------
    class WaveformDataset(Dataset):
        """数据集：输入为200维归一化波形序列，标签为0（异常）/1（正常）"""
        def __init__(self, cycles_norm, labels):
            # 确保输入是二维数组（样本数×200维）
            assert len(cycles_norm.shape) == 2 and cycles_norm.shape[1] == 200, \
                f"波形序列必须是(n_samples, 200)，实际形状：{cycles_norm.shape}"
            self.cycles_norm = cycles_norm.astype(np.float32)  # 转换为float32（适配PyTorch）
            self.labels = labels.astype(np.int64)  # 标签为int64

        def __len__(self):
            return len(self.cycles_norm)

        def __getitem__(self, idx):
            # 返回（波形序列，标签）张量
            cycle = torch.FloatTensor(self.cycles_norm[idx])
            label = torch.LongTensor([self.labels[idx]])
            return cycle, label

    class TransformerClassifier(nn.Module):
        """Transformer模型：输入200维波形序列，输出正常/异常分类"""
        def __init__(self, input_len=200, d_model=64, nhead=4, num_layers=2):
            super().__init__()
            self.embedding = nn.Linear(1, d_model)  # 1维波形→d_model维嵌入
            self.pos_encoder = nn.Parameter(torch.randn(1, input_len, d_model))  # 位置编码
            # Transformer编码器（增强正则化）
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=0.5, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # 分类头
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(d_model * input_len, 2)  # 输出2类（正常/异常）
            )

        def forward(self, x):
            # x: [batch_size, input_len] → [batch_size, input_len, 1]
            x = x.unsqueeze(2)
            # 嵌入+位置编码：[batch_size, input_len, d_model]
            x = self.embedding(x) + self.pos_encoder
            # Transformer编码
            x = self.transformer(x)
            # 展平：[batch_size, d_model*input_len]
            x = x.reshape(x.size(0), -1)
            # 分类输出
            x = self.fc(x)
            return x

    def prepare_ml_data(self):
        """准备ML数据：输入为200维波形序列，标签基于聚类结果"""
        try:
            # 1. 生成标签（1=正常，0=异常）
            core_sample_idx = [i for i, (t,c) in enumerate(self.cluster_normal) for _ in [i]]
            labels = np.array([1 if i in core_sample_idx else 0 for i in range(len(self.cycles))], dtype=np.int64)
            
            # 检查类别分布
            unique, counts = np.unique(labels, return_counts=True)
            class_counts = dict(zip(unique, counts))
            logger.info(f"{INFO} ML数据类别分布: {class_counts}（0=异常，1=正常）")
            if len(unique) < 2:
                logger.error(f"{ERROR} ML数据类别不足（仅{len(unique)}类），无法训练分类模型")
                return False
            
            # 2. 提取200维归一化波形序列（输入特征）
            cycles_norm = np.array([self.cycles[i][2] for i in range(len(self.cycles))], dtype=np.float32)
            logger.info(f"{INFO} ML输入波形序列形状：{cycles_norm.shape}（样本数×200维）")
            
            # 3. 划分训练集/测试集（分层抽样，确保类别分布一致）
            stratify = labels if all(v >= 2 for v in class_counts.values()) else None
            if stratify is None:
                logger.warning(f"{WARNING} 某类样本数<2，禁用分层抽样")
            
            X_train, X_test, y_train, y_test = train_test_split(
                cycles_norm, labels, test_size=0.3, random_state=42, stratify=stratify
            )
            
            # 4. 保存数据集划分结果
            dataset_result = pd.DataFrame({
                "cycle_id": range(1, len(labels)+1),
                "dataset": ["train" if i in np.where(np.isin(labels, y_train))[0] else "test" 
                           for i in range(len(labels))],
                "label": labels,
                "is_normal": labels
            })
            dataset_result.to_csv(os.path.join(self.subdirs["ml"], "dataset_split.csv"), index=False)
            
            # 5. 创建数据集和数据加载器
            self.train_dataset = self.WaveformDataset(X_train, y_train)
            self.test_dataset = self.WaveformDataset(X_test, y_test)
            self.infer_dataset = self.WaveformDataset(cycles_norm, labels)  # 全量数据推理
            
            self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)  # 小批量适配少样本
            self.test_loader = DataLoader(self.test_dataset, batch_size=4, shuffle=False)
            self.infer_loader = DataLoader(self.infer_dataset, batch_size=4, shuffle=False)
            
            logger.info(f"{SUCCESS} ML数据准备完成：")
            logger.info(f"          | 训练集：{len(X_train)}个样本（正常{sum(y_train==1)}，异常{sum(y_train==0)}）")
            logger.info(f"          | 测试集：{len(X_test)}个样本（正常{sum(y_test==1)}，异常{sum(y_test==0)}）")
            return True
        except Exception as e:
            logger.error(f"{ERROR} ML数据准备失败：{str(e)}")
            return False

    def evaluate_model(self):
        """评估模型：返回准确率、预测分布、混淆矩阵"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():  # 禁用梯度计算
            for inputs, labels in self.test_loader:
                # 转移到设备（CPU/GPU）
                inputs, labels = inputs.to(self.device), labels.squeeze().to(self.device)
                # 模型推理
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)  # 取概率最大的类别

                # 统计结果
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        accuracy = 100 * correct / total if total > 0 else 0.0
        test_normal_pred = sum(1 for p in all_preds if p == 1)
        test_abnormal_pred = sum(1 for p in all_preds if p == 0)

        # 绘制混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["异常", "正常"])
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(f"测试集混淆矩阵（准确率：{accuracy:.1f}%）", fontsize=14)
        # 添加数值标签
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.subdirs["ml"], "confusion_matrix.png"))
        plt.close(fig)
        del fig, ax

        return accuracy, test_normal_pred, test_abnormal_pred

    def train_transformer(self, epochs=30, lr=1e-4, patience=5, weight_decay=1e-4):
        """训练Transformer模型：使用类别平衡损失，解决样本不平衡"""
        try:
            # 初始化模型
            self.model = self.TransformerClassifier(input_len=self.fixed_len).to(self.device)
            logger.info(f"{INFO} Transformer模型结构：输入{self.fixed_len}维 → 输出2类")
            
            # 1. 计算类别权重（解决样本不平衡）
            train_labels = self.train_dataset.labels
            class_counts = np.bincount(train_labels)
            class_weights = torch.FloatTensor(len(train_labels) / (2 * class_counts)).to(self.device)
            logger.info(f"{INFO} 类别权重：异常类={class_weights[0]:.2f}，正常类={class_weights[1]:.2f}")
            
            # 2. 定义损失函数和优化器
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)  # 类别平衡损失
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)  # 学习率衰减

            # 3. 训练过程
            self.train_losses = []
            self.test_accuracies = []
            best_test_acc = 0.0
            early_stop_counter = 0

            for epoch in range(epochs):
                self.model.train()  # 训练模式
                train_loss = 0.0
                train_normal_pred = 0
                train_abnormal_pred = 0
                train_total = 0

                for inputs, labels in self.train_loader:
                    # 转移到设备
                    inputs, labels = inputs.to(self.device), labels.squeeze().to(self.device)

                    # 梯度清零
                    self.optimizer.zero_grad()
                    # 模型推理
                    outputs = self.model(inputs)
                    # 计算损失
                    loss = self.criterion(outputs, labels)
                    # 反向传播
                    loss.backward()
                    # 优化器更新
                    self.optimizer.step()

                    # 统计训练结果
                    train_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    train_normal_pred += (preds == 1).sum().item()
                    train_abnormal_pred += (preds == 0).sum().item()
                    train_total += labels.size(0)

                # 计算训练损失
                avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
                self.train_losses.append(avg_train_loss)
                
                # 测试集评估
                test_acc, test_normal_pred, test_abnormal_pred = self.evaluate_model()
                self.test_accuracies.append(test_acc)

                # 打印训练日志
                logger.info(f"Epoch {epoch+1:2d}/{epochs} | 训练损失: {avg_train_loss:.4f}")
                logger.info(f"          | 训练预测：正常{train_normal_pred}个，异常{train_abnormal_pred}个（总计{train_total}个）")
                logger.info(f"          | 测试精度: {test_acc:.1f}% | 测试预测：正常{test_normal_pred}个，异常{test_abnormal_pred}个")

                # 学习率衰减
                self.scheduler.step(test_acc)

                # 保存最优模型
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    torch.save(self.model.state_dict(), os.path.join(self.subdirs["models"], "best_transformer.pth"))
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    # 早停（防止过拟合）
                    if early_stop_counter >= patience:
                        logger.info(f"{INFO} 早停触发：连续{patience}轮测试精度未提升")
                        break

            # 保存最终模型和训练曲线
            torch.save(self.model.state_dict(), os.path.join(self.subdirs["models"], "final_transformer.pth"))
            self.plot_training_curves()
            
            logger.info(f"{SUCCESS} 模型训练完成：最优测试精度={best_test_acc:.1f}%")
            return True
        except Exception as e:
            logger.error(f"{ERROR} 模型训练失败：{str(e)}")
            return False

    def plot_training_curves(self):
        """绘制训练损失和测试精度曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 训练损失曲线
        ax1.plot(range(1, len(self.train_losses)+1), self.train_losses, 'o-', color="#e74c3c")
        ax1.set_title("训练损失曲线（类别平衡损失）", fontsize=14)
        ax1.set_xlabel("Epoch"), ax1.set_ylabel("损失值")
        ax1.grid(alpha=0.3)
        
        # 测试精度曲线
        ax2.plot(range(1, len(self.test_accuracies)+1), self.test_accuracies, 'o-', color="#3498db")
        ax2.set_title("测试精度曲线", fontsize=14)
        ax2.set_xlabel("Epoch"), ax2.set_ylabel("精度(%)")
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.subdirs["ml"], "training_curves.png"))
        plt.close(fig)
        del fig, ax1, ax2

    def infer_with_model(self, normal_prob_threshold=0.85):
        """模型推理：严格判定正常波形（概率≥阈值）"""
        try:
            # 加载最优模型
            self.model = self.TransformerClassifier(input_len=self.fixed_len).to(self.device)
            self.model.load_state_dict(torch.load(os.path.join(self.subdirs["models"], "best_transformer.pth")))
            self.model.eval()
            
            # 推理结果存储
            preds = []
            normal_probs = []  # 正常类概率
            all_cycle_ids = range(1, len(self.cycles)+1)

            with torch.no_grad():
                for inputs, _ in self.infer_loader:
                    inputs = inputs.to(self.device)
                    # 模型推理
                    outputs = self.model(inputs)
                    # 计算正常类概率（softmax后取第1类）
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    normal_probs.extend(probs)
                    # 严格判定：概率≥阈值为正常
                    batch_preds = np.where(probs >= normal_prob_threshold, 1, 0)
                    preds.extend(batch_preds)

            # 打印概率分布（关键：确认异常样本识别情况）
            logger.info(f"\n{INFO} ML推理概率分布：")
            logger.info(f"          | 正常类概率范围：[{min(normal_probs):.3f}, {max(normal_probs):.3f}]")
            logger.info(f"          | 正常类概率均值：{np.mean(normal_probs):.3f}")
            logger.info(f"          | 概率≥{normal_prob_threshold}（正常）：{sum(1 for p in normal_probs if p >= normal_prob_threshold)}个")
            logger.info(f"          | 概率<{normal_prob_threshold}（异常）：{sum(1 for p in normal_probs if p < normal_prob_threshold)}个")

            # 划分ML结果（正常/异常周期）
            core_sample_idx = [i for i, (t,c) in enumerate(self.cluster_normal) for _ in [i]]
            self.ml_normal = [
                (self.cycles[i][0], self.cycles[i][1]) 
                for i in range(len(self.cycles)) if preds[i] == 1
            ]
            self.ml_abnormal = [
                (self.cycles[i][0], self.cycles[i][1]) 
                for i in range(len(self.cycles)) if preds[i] == 0
            ]
            
            # 保存推理结果
            infer_result = pd.DataFrame({
                "cycle_id": all_cycle_ids,
                "cluster_is_normal": [1 if i in core_sample_idx else 0 for i in range(len(self.cycles))],
                "ml_normal_prob": normal_probs,
                "ml_is_normal": preds,
                "is_consistent": [
                    1 if ((1 if i in core_sample_idx else 0) == preds[i]) else 0 
                    for i in range(len(self.cycles))
                ]
            })
            infer_result.to_csv(os.path.join(self.subdirs["ml"], "ml_predictions.csv"), index=False)
            
            # 绘制分类结果对比
            self.plot_classification_comparison()
            
            logger.info(f"{SUCCESS} ML严格推理完成：")
            logger.info(f"          | 正常周期：{len(self.ml_normal)}个（概率≥{normal_prob_threshold}）")
            logger.info(f"          | 异常周期：{len(self.ml_abnormal)}个（概率<{normal_prob_threshold}）")
            logger.info(f"          | 与聚类结果一致性：{sum(infer_result['is_consistent'])/len(infer_result)*100:.1f}%")
            return True
        except Exception as e:
            logger.error(f"{ERROR} ML推理失败：{str(e)}")
            return False

    # -------------------------- 4. 结果保存与可视化 --------------------------
    def save_results(self):
        """保存所有周期结果（聚类+ML）"""
        self._save_cycle_group("cluster_normal", self.cluster_normal, "聚类正常周期")
        self._save_cycle_group("cluster_abnormal", self.cluster_abnormal, "聚类异常周期")
        self._save_cycle_group("ml_normal", self.ml_normal, "ML正常周期")
        self._save_cycle_group("ml_abnormal", self.ml_abnormal, "ML异常周期")
        
        # 保存正常波形模板（聚类平均波形）
        if self.cluster_normal:
            normal_cycles_norm = [self.cycles[i][2] for i in [j for j, (t,c) in enumerate(self.cluster_normal)]]
            normal_mean = np.mean(normal_cycles_norm, axis=0)
            normal_template = pd.DataFrame({
                "normalized_time": np.linspace(0, 1, self.fixed_len),
                "normalized_current": normal_mean
            })
            normal_template.to_csv(os.path.join(self.subdirs["cluster"], "normal_waveform_template.csv"), index=False)
            
            # 绘制正常波形模板
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(np.linspace(0, 1, self.fixed_len), normal_mean, color="#e74c3c", linewidth=2)
            ax.fill_between(np.linspace(0, 1, self.fixed_len), 
                           normal_mean - np.std(normal_cycles_norm, axis=0), 
                           normal_mean + np.std(normal_cycles_norm, axis=0), 
                           alpha=0.3, color="#e74c3c", label="±1标准差范围")
            ax.set_title("正常波形模板（聚类核心样本平均）", fontsize=14)
            ax.set_xlabel("归一化时间"), ax.set_ylabel("归一化电流")
            ax.legend(), ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.subdirs["cluster"], "normal_waveform_template.png"))
            plt.close(fig)
            del fig, ax

    def _save_cycle_group(self, group_name, cycles, group_desc):
        """保存指定组的周期数据和波形图"""
        group_dir = os.path.join(self.subdirs["ml" if "ml" in group_name else "cluster"], group_name)
        os.makedirs(group_dir, exist_ok=True)
        
        if not cycles:
            logger.info(f"{INFO} 无{group_desc}可保存")
            return
        
        # 保存每个周期
        for i, (cycle_time, cycle_current) in enumerate(cycles, 1):
            # 保存CSV
            cycle_df = pd.DataFrame({
                "Time(S)": cycle_time,
                "Current(A)": cycle_current
            })
            cycle_df.to_csv(os.path.join(group_dir, f"{group_name}_cycle_{i}.csv"), index=False)
            
            # 绘制波形图
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(cycle_time, cycle_current, color="#e74c3c" if "normal" in group_name else "#95a5a6")
            ax.set_title(f"{group_desc} #{i}", fontsize=12)
            ax.set_xlabel("Time(S)"), ax.set_ylabel("Current(A)")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(group_dir, f"{group_name}_cycle_{i}.png"))
            plt.close(fig)
            del fig, ax
        
        # 绘制组内所有周期对比
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"所有{group_desc}波形", fontsize=12)
        ax.set_xlabel("Normalized Time"), ax.set_ylabel("Current(A)")
        for i, (cycle_time, cycle_current) in enumerate(cycles[:10]):  # 最多显示10个
            norm_time = np.linspace(0, 1, len(cycle_time))
            ax.plot(norm_time, cycle_current, alpha=0.7, label=f"Cycle {i+1}")
        ax.legend(), ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(group_dir, f"all_{group_name}_cycles.png"))
        plt.close(fig)
        del fig, ax
        
        logger.info(f"{SUCCESS} 已保存 {len(cycles)} 个{group_desc}到 {group_dir}")

    def plot_classification_comparison(self):
        """绘制聚类与ML分类结果对比"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1：聚类结果
        cluster_counts = [len(self.cluster_abnormal), len(self.cluster_normal)]
        ax1.bar(["异常", "正常"], cluster_counts, color=["#95a5a6", "#e74c3c"], width=0.6)
        ax1.set_title("严格化无监督聚类结果", fontsize=14)
        ax1.set_ylabel("周期数量")
        for i, v in enumerate(cluster_counts):
            ax1.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=12)
        
        # 子图2：ML结果
        ml_counts = [len(self.ml_abnormal), len(self.ml_normal)]
        ax2.bar(["异常", "正常"], ml_counts, color=["#95a5a6", "#3498db"], width=0.6)
        ax2.set_title("Transformer分类结果（概率≥0.85）", fontsize=14)
        ax2.set_ylabel("周期数量")
        for i, v in enumerate(ml_counts):
            ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.subdirs["ml"], "classification_comparison.png"))
        plt.close(fig)
        del fig, ax1, ax2


# -------------------------- 调用示例（确保准确识别正常波形） --------------------------
if __name__ == "__main__":
    # 配置参数（关键参数：确保严格筛选正常波形）
    CSV_PATH = "16-波形检测与分类\\knee-sensor\\内翻-0-1.csv"  # 替换为实际路径
    FIXED_LEN = 200  # 固定波形序列长度
    MAX_CLUSTER = 10  # 聚类数范围（避免过多聚类）
    EPOCHS = 30  # 模型训练轮次（适配少样本）
    USE_PCA = False  # 禁用PCA（18维特征无需降维）
    PCA_COMPONENTS = 5

    # 严格化参数（核心：控制正常波形筛选严格度）
    SIL_THRESHOLD = 0.05  # 轮廓系数阈值（≥0.1视为有效聚类）
    DISTANCE_PERCENTILE = 50  # 正常簇为前30%最紧密簇
    NORMAL_PROB_THRESHOLD = 0.6  # ML正常概率阈值（≥0.85才视为正常）
    WEIGHT_DECAY = 1e-4  # 模型权重衰减（防止过拟合）

    # 初始化处理器
    processor = WaveformClusterClassifier(
        csv_path=CSV_PATH,
        fixed_len=FIXED_LEN,
        use_pca=USE_PCA,
        pca_components=PCA_COMPONENTS
    )

    # 执行完整流程（数据加载→聚类→ML训练→推理→保存）
    try:
        # 步骤1：数据加载与周期分割
        logger.info("\n" + "="*50 + " 步骤1：数据加载与周期分割 " + "="*50)
        if not processor.load_csv_data(start_time=0, end_time=130):
            raise Exception("数据加载失败")
        if not processor.detect_peaks_valleys(peak_thr=0.2, valley_thr=0.2, min_dist=20):
            raise Exception("波峰波谷检测失败")
        if not processor.split_cycles_by_extremes():
            raise Exception("周期分割失败")

        # 步骤2：严格化无监督聚类（筛选正常波形）
        logger.info("\n" + "="*50 + " 步骤2：严格化无监督聚类 " + "="*50)
        if not processor.cluster_cycles(
            max_cluster=MAX_CLUSTER, 
            sil_threshold=SIL_THRESHOLD, 
            distance_percentile=DISTANCE_PERCENTILE
        ):
            raise Exception("聚类失败")

        # 步骤3：ML数据准备与模型训练
        logger.info("\n" + "="*50 + " 步骤3：ML模型训练 " + "="*50)
        if not processor.prepare_ml_data():
            raise Exception("ML数据准备失败")
        if not processor.train_transformer(
            epochs=EPOCHS, 
            lr=1e-4,  # 小学习率适配少样本
            weight_decay=WEIGHT_DECAY
        ):
            raise Exception("模型训练失败")

        # 步骤4：ML严格推理（识别正常波形）
        logger.info("\n" + "="*50 + " 步骤4：ML严格推理 " + "="*50)
        if not processor.infer_with_model(
            normal_prob_threshold=NORMAL_PROB_THRESHOLD
        ):
            raise Exception("ML推理失败")

        # 步骤5：保存所有结果
        logger.info("\n" + "="*50 + " 步骤5：结果保存 " + "="*50)
        processor.save_results()

        logger.info(f"\n{SUCCESS} 所有流程执行完成！结果已保存到 {processor.output_dir} 目录")
        logger.info(f"核心结果：")
        logger.info(f"  - 聚类：正常周期{len(processor.cluster_normal)}个，异常周期{len(processor.cluster_abnormal)}个")
        logger.info(f"  - ML：正常周期{len(processor.ml_normal)}个，异常周期{len(processor.ml_abnormal)}个")

    except Exception as e:
        logger.error(f"\n{ERROR} 执行出错：{str(e)}")