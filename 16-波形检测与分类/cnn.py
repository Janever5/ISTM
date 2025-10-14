import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # 解决无GUI环境绘图问题
import matplotlib.pyplot as plt
import logging
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# -------------------------- 全局配置与日志 --------------------------
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

# 日志配置（同时输出到文件和控制台）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dominant_waveform.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 输出目录
OUTPUT_DIR = "dominant_waveform_results"
SUB_DIRS = {
    "raw": os.path.join(OUTPUT_DIR, "0_raw_data"),
    "cycles": os.path.join(OUTPUT_DIR, "1_split_cycles"),
    "features": os.path.join(OUTPUT_DIR, "2_extracted_features"),
    "dominant": os.path.join(OUTPUT_DIR, "3_dominant_results"),
    "normal_abnormal": os.path.join(OUTPUT_DIR, "4_normal_abnormal"),
    "normal_data": os.path.join(OUTPUT_DIR, "4_normal_abnormal", "normal_data"),
    "normal_plots": os.path.join(OUTPUT_DIR, "4_normal_abnormal", "normal_plots"),
    "abnormal_data": os.path.join(OUTPUT_DIR, "4_normal_abnormal", "abnormal_data"),
    "abnormal_plots": os.path.join(OUTPUT_DIR, "4_normal_abnormal", "abnormal_plots"),
    "total_waveform": os.path.join(OUTPUT_DIR, "5_total_waveform"),
    "plots": os.path.join(OUTPUT_DIR, "6_visualizations")
}
for dir_path in SUB_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)


# -------------------------- 1. 数据加载 --------------------------
def load_waveform_csv(csv_path, start_time=80, end_time=130):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            header_idx = next(i for i, line in enumerate(lines) 
                             if line.strip().startswith("Time(S),Current(A)"))
        
        # 读取全量数据
        df_full = pd.read_csv(csv_path, skiprows=header_idx, header=0, dtype=np.float64)
        # 按时间范围过滤有效数据
        time_filter = (df_full["Time(S)"] >= start_time) & (df_full["Time(S)"] <= end_time)
        df_filtered = df_full[time_filter].copy()
        
        if df_filtered.empty:
            logger.error(f"时间范围{start_time}-{end_time}秒内无数据")
            return None, None, None
        
        # 保存过滤后和全量原始数据
        df_filtered.to_csv(os.path.join(SUB_DIRS["raw"], "filtered_raw_data.csv"), index=False)
        df_full.to_csv(os.path.join(SUB_DIRS["raw"], "full_raw_data.csv"), index=False)
        logger.info(f"[SUCCESS] 加载数据：{len(df_filtered)}个采样点，时间范围{df_filtered['Time(S)'].min():.2f}~{df_filtered['Time(S)'].max():.2f}S")
        
        return df_filtered["Time(S)"].values, df_filtered["Current(A)"].values, df_full
    except Exception as e:
        logger.error(f"数据加载失败：{str(e)}")
        return None, None, None


# -------------------------- 2. 周期分割（核心修改：简化波峰波谷图表，仅描点） --------------------------
def split_cycles_by_peaks(time, current, fixed_len=200, peak_thr=0.1, valley_thr=0.1, min_dist=10):
    try:
        # 1. 检测波峰波谷（保留波峰波谷的幅值信息，用于图表标注）
        current_max, current_min = np.max(current), np.min(current)
        current_range = current_max - current_min
        peak_height = current_min + current_range * peak_thr
        valley_height = current_max - current_range * valley_thr
        
        # 检测波峰（返回峰值信息）
        peaks, peak_props = find_peaks(current, height=peak_height, distance=min_dist)
        peak_heights = peak_props['peak_heights']  # 波峰的电流幅值
        # 检测波谷（返回谷值信息，注意valley_heights是负的，需取反）
        valleys, valley_props = find_peaks(-current, height=-valley_height, distance=min_dist)
        valley_heights = -valley_props['peak_heights']  # 波谷的电流幅值（还原为正值）
        
        if len(peaks) < 1 or len(valleys) < 2:
            logger.error(f"波峰({len(peaks)}个)或波谷({len(valleys)}个)数量不足，无法按'第一个波峰到第二个波谷'分割周期")
            return None, None, None
        
        # -------------------------- 核心修改：简化波峰波谷图，仅描点 --------------------------
        fig, ax = plt.subplots(figsize=(14, 6))
        # 1. 绘制原始波形（基础层）
        ax.plot(time, current, color="#3498db", linewidth=1.2, label="原始波形")
        # 2. 描波峰点（红色三角，放大尺寸，标注幅值）
        ax.scatter(time[peaks], peak_heights, color="#e74c3c", s=80, marker='^', 
                   label="波峰", zorder=5)  # zorder确保波峰点在最上层
        # 3. 描波谷点（绿色倒三角，放大尺寸，标注幅值）
        ax.scatter(time[valleys], valley_heights, color="#2ecc71", s=80, marker='v', 
                   label="波谷", zorder=5)  # zorder确保波谷点在最上层
        
        # 图表简化配置（无多余元素，仅保留必要信息）
        ax.set_title("波峰波谷识别结果（仅标记波峰波谷点）", fontsize=16)
        ax.set_xlabel("时间(S)", fontsize=14), ax.set_ylabel("电流(A)", fontsize=14)
        ax.legend(fontsize=12), ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
        # 调整坐标轴范围，避免波峰波谷点贴近边缘
        ax.set_xlim(time.min() - 1, time.max() + 1)
        ax.set_ylim(current_min - 0.5, current_max + 0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(SUB_DIRS["plots"], "peaks_valleys_marking.png"), dpi=120)  # 提高dpi让描点更清晰
        plt.close(fig)
        logger.info(f"[SUCCESS] 波峰波谷描点图已保存：{os.path.join(SUB_DIRS['plots'], 'peaks_valleys_marking.png')}")
        # ------------------------------------------------------------------------------------------

        # 2. 按"第一个波峰→第二个波谷"分割周期（原有逻辑保留）
        raw_cycles = []  # 原始周期（时间数组+电流数组）
        norm_cycles = []  # 归一化周期
        cycle_time_ranges = []  # 每个周期的时间范围 (start_time, end_time)
        scaler = StandardScaler()
        
        # 合并并排序所有极值点，保留其类型信息
        peak_info = [(p, 'peak') for p in peaks]
        valley_info = [(v, 'valley') for v in valleys]
        all_extremes = sorted(peak_info + valley_info, key=lambda x: x[0])
        
        current_pos = 0  # 当前处理位置
        while current_pos < len(all_extremes):
            # 寻找当前位置后的第一个波峰
            first_peak_idx = None
            for i in range(current_pos, len(all_extremes)):
                if all_extremes[i][1] == 'peak':
                    first_peak_idx = i
                    break
            
            if first_peak_idx is None:
                logger.warning("未找到更多波峰，停止周期分割")
                break
            
            # 从第一个波峰之后寻找第二个波谷
            valley_count = 0
            second_valley_idx = None
            for i in range(first_peak_idx + 1, len(all_extremes)):
                if all_extremes[i][1] == 'valley':
                    valley_count += 1
                    if valley_count == 2:  # 找到第二个波谷
                        second_valley_idx = i
                        break
            
            if second_valley_idx is None:
                logger.warning("在第一个波峰后未找到足够的波谷，停止周期分割")
                break
            
            # 提取周期：从第一个波峰到第二个波谷（包含端点）
            cycle_start_idx = all_extremes[first_peak_idx][0]
            cycle_end_idx = all_extremes[second_valley_idx][0]
            
            # 提取周期数据
            cycle_time = time[cycle_start_idx:cycle_end_idx+1]
            cycle_current = current[cycle_start_idx:cycle_end_idx+1]
            cycle_time_range = (cycle_time[0], cycle_time[-1])
            
            # 过滤过短周期
            if len(cycle_time) < 10:
                logger.warning(f"跳过过短周期（长度{len(cycle_time)}）")
                current_pos = second_valley_idx + 1
                continue
            
            # 周期归一化
            norm_x = np.linspace(0, 1, fixed_len)
            raw_x = np.linspace(0, 1, len(cycle_current))
            cycle_interp = np.interp(norm_x, raw_x, cycle_current)
            cycle_norm = scaler.fit_transform(cycle_interp.reshape(-1, 1)).flatten()
            
            # 保存周期数据
            raw_cycles.append((cycle_time, cycle_current))
            norm_cycles.append(cycle_norm)
            cycle_time_ranges.append(cycle_time_range)
            
            # 更新当前位置，继续寻找下一个周期
            current_pos = second_valley_idx + 1
            logger.debug(f"已分割周期 {len(raw_cycles)}: 从索引{cycle_start_idx}到{cycle_end_idx}")

        if not norm_cycles:
            logger.error(f"未分割到有效周期")
            return None, None, None
        
        # 保存所有周期基础数据（原有逻辑保留）
        norm_cycles = np.array(norm_cycles)
        pd.DataFrame(norm_cycles).to_csv(os.path.join(SUB_DIRS["cycles"], "normalized_cycles.csv"), 
                                        index=False, header=[f"point_{i}" for i in range(fixed_len)])
        for i, (cycle_time, cycle_current) in enumerate(raw_cycles, 1):
            pd.DataFrame({
                "Time(S)": cycle_time,
                "Current(A)": cycle_current
            }).to_csv(os.path.join(SUB_DIRS["cycles"], f"raw_cycle_{i}.csv"), index=False)
        
        # 可视化所有归一化周期（原有逻辑保留）
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, cycle_norm in enumerate(norm_cycles):
            ax.plot(np.linspace(0, 1, fixed_len), cycle_norm, alpha=0.6, linewidth=1, 
                    label=f"周期{i+1}" if i < 10 else "")
        ax.set_title(f"所有归一化周期（共{len(norm_cycles)}个，长度{fixed_len}）", fontsize=14)
        ax.set_xlabel("归一化时间"), ax.set_ylabel("归一化电流")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left'), ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SUB_DIRS["plots"], "all_normalized_cycles.png"))
        plt.close(fig)
        
        # 可视化典型周期的波峰波谷标记（原有逻辑保留）
        if len(raw_cycles) > 0:
            plot_count = min(3, len(raw_cycles))
            fig, axes = plt.subplots(plot_count, 1, figsize=(12, 4*plot_count))
            if plot_count == 1:
                axes = [axes]
            
            for i in range(plot_count):
                cycle_time, cycle_current = raw_cycles[i]
                cycle_indices = np.where((time >= cycle_time[0]) & (time <= cycle_time[-1]))[0]
                cycle_peaks = [p for p in peaks if p in cycle_indices]
                cycle_valleys = [v for v in valleys if v in cycle_indices]
                
                axes[i].plot(cycle_time, cycle_current, color="#3498db", label="周期波形")
                axes[i].scatter(time[cycle_peaks], current[cycle_peaks], color="#e74c3c", s=50, marker='^', label="波峰")
                axes[i].scatter(time[cycle_valleys], current[cycle_valleys], color="#2ecc71", s=50, marker='v', label="波谷")
                axes[i].axvline(x=time[cycle_peaks[0]] if cycle_peaks else cycle_time[0], 
                                color="#f39c12", linestyle='--', label="周期开始")
                axes[i].axvline(x=time[cycle_valleys[1]] if len(cycle_valleys)>=2 else cycle_time[-1], 
                                color="#9b59b6", linestyle='--', label="周期结束")
                axes[i].set_title(f"周期 {i+1} 波形（波峰-波谷标记）", fontsize=12)
                axes[i].set_xlabel("时间(S)"), axes[i].set_ylabel("电流(A)")
                axes[i].legend(), axes[i].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(SUB_DIRS["plots"], "cycle_with_peak_valley_marks.png"))
            plt.close(fig)
        
        logger.info(f"[SUCCESS] 周期分割完成：共{len(norm_cycles)}个有效周期（基于第一个波峰到第二个波谷）")
        return norm_cycles, raw_cycles, cycle_time_ranges
    except Exception as e:
        logger.error(f"周期分割失败：{str(e)}")
        return None, None, None


# -------------------------- 3. CNN特征提取器 --------------------------
class WaveformFeatureExtractor(nn.Module):
    def __init__(self, input_len=200, feature_dim=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return x


# -------------------------- 4. 辅助函数：绘制单个波形图表 --------------------------
def plot_single_waveform(cycle_time, cycle_current, save_path, title, is_normal=True):
    try:
        fig, ax = plt.subplots(figsize=(8, 3))
        color = "#2ecc71" if is_normal else "#e74c3c"
        ax.plot(cycle_time, cycle_current, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time(S)", fontsize=10), ax.set_ylabel("Current(A)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(cycle_time.min() - 0.1, cycle_time.max() + 0.1)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        return True
    except Exception as e:
        logger.warning(f"[WARNING] 绘制波形图表失败：{str(e)}")
        return False


# -------------------------- 5. 主导波形识别 --------------------------
def find_dominant_waveforms(cycles_norm, raw_cycles, cycle_time_ranges, min_cluster=2, max_cluster=8, threshold=0.4):
    try:
        n_cycles, fixed_len = cycles_norm.shape
        logger.info(f"[INFO] 开始主导波形识别：{n_cycles}个周期，特征提取维度{fixed_len}")
        
        # 1. CNN特征提取
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = WaveformFeatureExtractor(input_len=fixed_len).to(device)
        with torch.no_grad():
            cycles_tensor = torch.FloatTensor(cycles_norm).to(device)
            features = extractor(cycles_tensor).cpu().numpy()
        
        pd.DataFrame(features).to_csv(os.path.join(SUB_DIRS["features"], "cnn_features.csv"), 
                                      index=False, header=[f"feat_{i}" for i in range(64)])
        logger.info(f"[SUCCESS] CNN特征提取完成：shape={features.shape}")

        # 2. 最优聚类数选择
        best_k = 2
        best_sil_score = -1
        sil_scores = []
        for k in range(min_cluster, max_cluster + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            if len(np.unique(cluster_labels)) > 1:
                sil_score = silhouette_score(features, cluster_labels)
                sil_scores.append(sil_score)
                if sil_score > best_sil_score:
                    best_sil_score = sil_score
                    best_k = k
            else:
                sil_scores.append(-1)
                logger.warning(f"[WARNING] 聚类数K={k}时仅生成1个簇")
        
        # 可视化轮廓系数
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(min_cluster, max_cluster + 1), sil_scores, 'o-', color="#3498db", linewidth=2)
        ax.axvline(x=best_k, color="#e74c3c", linestyle='--', 
                   label=f"最优K={best_k}（轮廓系数={best_sil_score:.3f}）")
        ax.set_title("不同聚类数的轮廓系数", fontsize=14)
        ax.set_xlabel("聚类数K"), ax.set_ylabel("轮廓系数")
        ax.legend(), ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SUB_DIRS["plots"], "silhouette_scores.png"))
        plt.close(fig)

        # 3. 最终聚类与占比统计
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        final_labels = kmeans_final.fit_predict(features)
        cluster_counts = np.bincount(final_labels)
        cluster_ratios = cluster_counts / n_cycles
        
        # 保存聚类基础结果
        cluster_result_df = pd.DataFrame({
            "cycle_id": range(1, n_cycles + 1),
            "cluster_label": final_labels,
            "cluster_ratio": [cluster_ratios[label] for label in final_labels],
            "start_time": [cycle_time_ranges[i][0] for i in range(n_cycles)],
            "end_time": [cycle_time_ranges[i][1] for i in range(n_cycles)]
        })
        cluster_result_df.to_csv(os.path.join(SUB_DIRS["dominant"], "cluster_results.csv"), index=False)

        # 4. 筛选正常（主导）/异常（非主导）周期
        dominant_cluster_labels = [i for i, ratio in enumerate(cluster_ratios) if ratio >= threshold]
        if not dominant_cluster_labels:
            dominant_cluster_labels = [np.argmax(cluster_ratios)]
            logger.warning(f"[WARNING] 无占比≥{threshold}的簇，选择占比最高簇为正常（占比{cluster_ratios[dominant_cluster_labels[0]]:.2f}）")
        
        # 分离正常/异常周期的索引
        normal_indices = [i for i, label in enumerate(final_labels) if label in dominant_cluster_labels]
        abnormal_indices = [i for i, label in enumerate(final_labels) if label not in dominant_cluster_labels]
        logger.info(f"[INFO] 分类结果：正常周期{len(normal_indices)}个，异常周期{len(abnormal_indices)}个")

        # 5. 保存正常波形（数据+图表）
        if normal_indices:
            for seq, idx in enumerate(normal_indices, 1):
                cycle_id = idx + 1
                cycle_time, cycle_current = raw_cycles[idx]
                cycle_norm = cycles_norm[idx]
                start_time, end_time = cycle_time_ranges[idx]
                
                # 保存正常波形数据
                raw_save_path = os.path.join(SUB_DIRS["normal_data"], f"normal_cycle_{seq}_id{cycle_id}.csv")
                pd.DataFrame({
                    "Time(S)": cycle_time,
                    "Current(A)": cycle_current
                }).to_csv(raw_save_path, index=False)
                
                norm_save_path = os.path.join(SUB_DIRS["normal_data"], f"normal_cycle_{seq}_id{cycle_id}_normalized.csv")
                pd.DataFrame(cycle_norm).T.to_csv(
                    norm_save_path, index=False, header=[f"point_{i}" for i in range(fixed_len)]
                )
                
                # 绘制正常波形图表
                plot_save_path = os.path.join(SUB_DIRS["normal_plots"], f"normal_cycle_{seq}_id{cycle_id}.png")
                plot_title = f"正常周期 #{seq}（ID：{cycle_id}，时间：{start_time:.2f}~{end_time:.2f}S）"
                plot_single_waveform(cycle_time, cycle_current, plot_save_path, plot_title, is_normal=True)
            
            # 绘制正常周期汇总图
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_count = min(len(normal_indices), 20)
            for seq in range(plot_count):
                idx = normal_indices[seq]
                cycle_time, cycle_current = raw_cycles[idx]
                norm_time = np.linspace(0, 1, len(cycle_time))
                ax.plot(norm_time, cycle_current, color="#2ecc71", alpha=0.4, linewidth=1,
                        label=f"正常周期{seq+1}" if seq < 10 else "")
            # 正常周期平均波形
            normal_avg = np.mean(cycles_norm[normal_indices], axis=0)
            ax.plot(np.linspace(0, 1, fixed_len), normal_avg, color="#27ae60", linewidth=2.5,
                    label=f"正常平均波形（共{len(normal_indices)}个周期）")
            ax.set_title("正常波形汇总（前20个+平均波形）", fontsize=14)
            ax.set_xlabel("归一化时间"), ax.set_ylabel("电流(A)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left'), ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(SUB_DIRS["normal_plots"], "normal_cycles_summary.png"))
            plt.close(fig)
        else:
            logger.warning(f"[WARNING] 无正常周期可保存")

        # 6. 保存异常波形（数据+图表）
        if abnormal_indices:
            for seq, idx in enumerate(abnormal_indices, 1):
                cycle_id = idx + 1
                cycle_time, cycle_current = raw_cycles[idx]
                cycle_norm = cycles_norm[idx]
                start_time, end_time = cycle_time_ranges[idx]
                
                # 保存异常波形数据
                raw_save_path = os.path.join(SUB_DIRS["abnormal_data"], f"abnormal_cycle_{seq}_id{cycle_id}.csv")
                pd.DataFrame({
                    "Time(S)": cycle_time,
                    "Current(A)": cycle_current
                }).to_csv(raw_save_path, index=False)
                
                norm_save_path = os.path.join(SUB_DIRS["abnormal_data"], f"abnormal_cycle_{seq}_id{cycle_id}_normalized.csv")
                pd.DataFrame(cycle_norm).T.to_csv(
                    norm_save_path, index=False, header=[f"point_{i}" for i in range(fixed_len)]
                )
                
                # 绘制异常波形图表
                plot_save_path = os.path.join(SUB_DIRS["abnormal_plots"], f"abnormal_cycle_{seq}_id{cycle_id}.png")
                plot_title = f"异常周期 #{seq}（ID：{cycle_id}，时间：{start_time:.2f}~{end_time:.2f}S）"
                plot_single_waveform(cycle_time, cycle_current, plot_save_path, plot_title, is_normal=False)
            
            # 绘制异常周期汇总图
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_count = min(len(abnormal_indices), 20)
            for seq in range(plot_count):
                idx = abnormal_indices[seq]
                cycle_time, cycle_current = raw_cycles[idx]
                norm_time = np.linspace(0, 1, len(cycle_time))
                ax.plot(norm_time, cycle_current, color="#e74c3c", alpha=0.4, linewidth=1,
                        label=f"异常周期{seq+1}" if seq < 10 else "")
            # 异常周期平均波形
            if len(abnormal_indices) > 1:
                abnormal_avg = np.mean(cycles_norm[abnormal_indices], axis=0)
                ax.plot(np.linspace(0, 1, fixed_len), abnormal_avg, color="#c0392b", linewidth=2.5,
                        label=f"异常平均波形（共{len(abnormal_indices)}个周期）")
            ax.set_title("异常波形汇总（前20个+平均波形）", fontsize=14)
            ax.set_xlabel("归一化时间"), ax.set_ylabel("电流(A)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left'), ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(SUB_DIRS["abnormal_plots"], "abnormal_cycles_summary.png"))
            plt.close(fig)
        else:
            logger.warning(f"[WARNING] 无异常周期可保存")

        # 7. 主导波形可视化
        dominant_results = {
            "dominant_cluster_labels": dominant_cluster_labels,
            "dominant_ratios": [cluster_ratios[label] for label in dominant_cluster_labels],
            "dominant_cycles_idx": [np.where(final_labels == label)[0] for label in dominant_cluster_labels],
            "dominant_cycles_norm": [cycles_norm[np.where(final_labels == label)[0]] for label in dominant_cluster_labels],
            "dominant_cycles_raw": [[raw_cycles[i] for i in np.where(final_labels == label)[0]] for label in dominant_cluster_labels],
            "normal_indices": normal_indices,
            "abnormal_indices": abnormal_indices,
            "cycle_time_ranges": cycle_time_ranges
        }

        fig, axes = plt.subplots(len(dominant_cluster_labels), 1, figsize=(12, 5 * len(dominant_cluster_labels)))
        if len(dominant_cluster_labels) == 1:
            axes = [axes]
        for i, (ax, label, ratio, cycles_norm_cluster) in enumerate(zip(axes, dominant_cluster_labels, dominant_results["dominant_ratios"], dominant_results["dominant_cycles_norm"])):
            for cycle_norm in cycles_norm_cluster:
                ax.plot(np.linspace(0, 1, fixed_len), cycle_norm, color="#95a5a6", alpha=0.3, linewidth=1)
            mean_cycle = np.mean(cycles_norm_cluster, axis=0)
            ax.plot(np.linspace(0, 1, fixed_len), mean_cycle, color="#e74c3c", linewidth=2, label=f"平均波形（占比{ratio:.2%}）")
            std_cycle = np.std(cycles_norm_cluster, axis=0)
            ax.fill_between(np.linspace(0, 1, fixed_len), mean_cycle - std_cycle, mean_cycle + std_cycle,
                            color="#e74c3c", alpha=0.2, label="±1标准差")
            ax.set_title(f"主导波形簇 {label}（共{len(cycles_norm_cluster)}个周期）", fontsize=14)
            ax.set_xlabel("归一化时间"), ax.set_ylabel("归一化电流")
            ax.legend(), ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SUB_DIRS["plots"], "dominant_waveforms.png"))
        plt.close(fig)

        logger.info(f"[SUCCESS] 主导波形识别与正常/异常分类完成！")
        return dominant_results
    except Exception as e:
        logger.error(f"主导波形识别失败：{str(e)}")
        return None


# -------------------------- 6. 总波形标注 --------------------------
def plot_total_waveform_with_annotation(full_raw_df, dominant_results, start_time=80, end_time=130):
    try:
        # 筛选总波形中目标时间范围的数据
        total_data = full_raw_df[full_raw_df["Time(S)"].between(start_time, end_time)]
        total_time = total_data["Time(S)"].values
        total_current = total_data["Current(A)"].values
        
        # 提取正常/异常周期的时间范围
        cycle_time_ranges = dominant_results["cycle_time_ranges"]
        normal_indices = dominant_results["normal_indices"]
        abnormal_indices = dominant_results["abnormal_indices"]
        normal_time_ranges = [cycle_time_ranges[i] for i in normal_indices]
        abnormal_time_ranges = [cycle_time_ranges[i] for i in abnormal_indices]
        
        # 绘制总波形标注图
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(total_time, total_current, color="#3498db", linewidth=1.2, label="原始总波形")
        # 标注正常段
        for (s, e) in normal_time_ranges:
            ax.axvspan(s, e, alpha=0.3, color="#2ecc71", 
                       label="正常波形段" if (s, e) == normal_time_ranges[0] else "")
        # 标注异常段
        for (s, e) in abnormal_time_ranges:
            ax.axvspan(s, e, alpha=0.3, color="#e74c3c", 
                       label="异常波形段" if (s, e) == abnormal_time_ranges[0] else "")
        
        ax.set_title(f"{start_time}-{end_time}秒总波形（正常/异常段标注）", fontsize=16)
        ax.set_xlabel("时间(S)", fontsize=12), ax.set_ylabel("电流(A)", fontsize=12)
        ax.set_xticks(np.linspace(start_time, end_time, 11))
        ax.legend(fontsize=10), ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(SUB_DIRS["total_waveform"], "total_waveform_annotation.png"), dpi=100)
        plt.close(fig)
        
        # 保存总波形标注数据
        annotation_df = pd.DataFrame({
            "type": ["normal"]*len(normal_time_ranges) + ["abnormal"]*len(abnormal_time_ranges),
            "start_time": [r[0] for r in normal_time_ranges] + [r[0] for r in abnormal_time_ranges],
            "end_time": [r[1] for r in normal_time_ranges] + [r[1] for r in abnormal_time_ranges],
            "duration": [r[1]-r[0] for r in normal_time_ranges] + [r[1]-r[0] for r in abnormal_time_ranges]
        })
        annotation_df.to_csv(os.path.join(SUB_DIRS["total_waveform"], "total_waveform_annotation.csv"), index=False)
        
        logger.info(f"[SUCCESS] 总波形正常/异常标注完成！")
        return True
    except Exception as e:
        logger.error(f"总波形标注失败：{str(e)}")
        return False


# -------------------------- 7. 主函数 --------------------------
def main(csv_path, fixed_len=200, start_time=80, end_time=130, dominant_threshold=0.4):
    logger.info("="*60)
    logger.info("开始波形主导模式识别+正常/异常分类流程")
    logger.info("="*60)
    
    # 步骤1：加载数据
    time, current, full_raw_df = load_waveform_csv(csv_path, start_time, end_time)
    if time is None or current is None or full_raw_df is None:
        logger.error("流程终止：数据加载失败")
        return
    
    # 步骤2：分割周期（波峰波谷描点图在此步骤生成）
    cycles_norm, raw_cycles, cycle_time_ranges = split_cycles_by_peaks(
        time, current, fixed_len
    )
    if cycles_norm is None or raw_cycles is None:
        logger.error("流程终止：周期分割失败")
        return
    
    # 步骤3：主导波形识别+正常/异常分类
    dominant_results = find_dominant_waveforms(
        cycles_norm, raw_cycles, cycle_time_ranges,
        threshold=dominant_threshold
    )
    if dominant_results is None:
        logger.error("流程终止：主导波形识别失败")
        return
    
    # 步骤4：总波形正常/异常标注
    plot_total_waveform_with_annotation(full_raw_df, dominant_results, start_time, end_time)
    
    # 输出最终结果汇总（突出波峰波谷描点图路径）
    normal_count = len(dominant_results["normal_indices"])
    abnormal_count = len(dominant_results["abnormal_indices"])
    total_count = normal_count + abnormal_count
    logger.info("="*60)
    logger.info("流程完成！最终结果汇总：")
    logger.info(f"1. 周期统计：")
    logger.info(f"   - 总周期数：{total_count}个")
    logger.info(f"   - 正常周期：{normal_count}个（{normal_count/total_count*100:.1f}%）")
    logger.info(f"   - 异常周期：{abnormal_count}个（{abnormal_count/total_count*100:.1f}%）")
    logger.info(f"2. 关键图表路径：")
    logger.info(f"   - 波峰波谷描点图：{os.path.join(SUB_DIRS['plots'], 'peaks_valleys_marking.png')}")
    logger.info(f"   - 总波形标注图：{os.path.join(SUB_DIRS['total_waveform'], 'total_waveform_annotation.png')}")
    logger.info(f"3. 数据文件路径：")
    logger.info(f"   - 正常波形数据：{SUB_DIRS['normal_data']}")
    logger.info(f"   - 异常波形数据：{SUB_DIRS['abnormal_data']}")
    logger.info("="*60)


# -------------------------- 8. 调用示例 --------------------------
if __name__ == "__main__":
    # 配置参数（确保CSV路径正确，建议使用绝对路径或当前目录相对路径）
    CSV_PATH = "16-波形检测与分类\\knee-sensor\\内翻-0-1.csv"
    FIXED_LEN = 200
    START_TIME = 70
    END_TIME = 130
    DOMINANT_THRESHOLD = 0.7   
    main(
        csv_path=CSV_PATH,
        fixed_len=FIXED_LEN,
        start_time=START_TIME,
        end_time=END_TIME,
        dominant_threshold=DOMINANT_THRESHOLD
    )