import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 全局配置与日志 --------------------------
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dominant_waveform.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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


# -------------------------- 1. 时域物理特征提取函数（不变） --------------------------
def extract_waveform_physical_features(raw_cycles, time, current, peak_thr=0.15, valley_thr=0.15, min_dist=15):
    physical_features = []
    scaler = MinMaxScaler()

    for cycle_idx, (cycle_time, cycle_current) in enumerate(raw_cycles):
        cycle_start = np.where(time >= cycle_time[0])[0][0]
        cycle_end = np.where(time <= cycle_time[-1])[0][-1]
        cycle_current_slice = current[cycle_start:cycle_end+1]
        cycle_time_slice = time[cycle_start:cycle_end+1]

        current_max = np.max(cycle_current_slice)
        current_min = np.min(cycle_current_slice)
        current_range = current_max - current_min
        peak_height = current_min + current_range * peak_thr
        valley_height = current_max - current_range * valley_thr

        peaks, peak_props = find_peaks(cycle_current_slice, height=peak_height, distance=min_dist)
        valleys, valley_props = find_peaks(-cycle_current_slice, height=-valley_height, distance=min_dist)
        peak_heights = peak_props.get('peak_heights', np.array([]))
        valley_heights = -valley_props.get('peak_heights', np.array([]))

        num_peaks = len(peaks)
        max_peak = np.max(peak_heights) if num_peaks > 0 else 0.0
        avg_peak = np.mean(peak_heights) if num_peaks > 0 else 0.0

        num_valleys = len(valleys)
        min_valley = np.min(valley_heights) if num_valleys > 0 else 0.0
        avg_valley = np.mean(valley_heights) if num_valleys > 0 else 0.0

        peak_valley_dist = 0.0
        if num_peaks > 0 and num_valleys > 0:
            first_peak_time = cycle_time_slice[peaks[0]]
            nearest_valley_idx = np.argmin(np.abs(valleys - peaks[0]))
            nearest_valley_time = cycle_time_slice[valleys[nearest_valley_idx]]
            peak_valley_dist = np.abs(first_peak_time - nearest_valley_time)

        peak_peak_dist = 0.0
        if num_peaks >= 2:
            peak_times = cycle_time_slice[peaks]
            peak_peak_dist = np.mean(np.diff(peak_times))

        current_var = np.var(cycle_current_slice)
        current_diff = np.diff(cycle_current_slice)
        max_diff = np.max(np.abs(current_diff)) if len(current_diff) > 0 else 0.0

        # 新增1：波峰-波谷幅值差（反映单个周期内电流的最大波动范围，正常波形该值稳定）
        peak_valley_amp_diff = max_peak - min_valley if (num_peaks > 0 and num_valleys > 0) else 0.0
        # 新增2：波峰占空比（波峰持续时间/周期总时间，异常波形可能出现波峰过短/过长）
        cycle_total_duration = cycle_time[-1] - cycle_time[0]  # 周期总时间
        peak_duration = 0.0
        if num_peaks > 0 and cycle_total_duration > 0:
            # 简单计算：波峰前后0.1倍周期长度的时间作为波峰持续时间（可根据数据调整）
            peak_half_window = int(len(cycle_current_slice) * 0.1)  # 波峰窗口长度
            for peak_idx in peaks:
                start = max(0, peak_idx - peak_half_window)
                end = min(len(cycle_current_slice)-1, peak_idx + peak_half_window)
                peak_duration += cycle_time_slice[end] - cycle_time_slice[start]
            peak_duty_cycle = peak_duration / cycle_total_duration  # 波峰占空比

        # 更新特征列表（新增2个特征，共12维）
        cycle_features = [
            num_peaks, max_peak, avg_peak,
            num_valleys, min_valley, avg_valley,
            peak_valley_dist, peak_peak_dist,
            current_var, max_diff,
            peak_valley_amp_diff,  # 新增
            peak_duty_cycle        # 新增
        ]
        
        physical_features.append(cycle_features)

    physical_features = scaler.fit_transform(physical_features)
    feat_names = [
        "num_peaks", "max_peak", "avg_peak",
        "num_valleys", "min_valley", "avg_valley",
        "peak_valley_dist", "peak_peak_dist",
        "current_var", "max_diff",
        "peak_valley_amp_diff", "peak_duty_cycle"
    ]
    pd.DataFrame(physical_features, columns=feat_names).to_csv(
        os.path.join(SUB_DIRS["features"], "physical_features.csv"), index=False
    )
    logger.info(f"[SUCCESS] 时域物理特征提取完成：{len(physical_features)}个周期，{len(feat_names)}个特征")
    return physical_features


# -------------------------- 2. CNN特征提取器（不变） --------------------------
class WaveformFeatureExtractor(nn.Module):
    def __init__(self, input_len=200, cnn_feature_dim=64, physical_feature_dim=12):  # 物理特征维度改为12
        super().__init__()
        # 新增：多尺度卷积（3个不同卷积核，分别捕捉波峰波谷的细/中/粗粒度特征）
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),  # 小核：捕捉细粒度（如波峰尖点）
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 下采样，保留关键特征
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),  # 中核：捕捉中粒度（如波峰-波谷过渡）
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),  # 大核：捕捉粗粒度（如整个周期形态）
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 融合多尺度特征后的后续网络
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(16*3, 64, kernel_size=3, stride=2, padding=1),  # 3个分支特征拼接（16*3=48）
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.2)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # 特征融合分支（物理特征+CNN特征）
        self.fusion_branch = nn.Sequential(
            nn.Linear(cnn_feature_dim + physical_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, cnn_feature_dim)
        )

    def forward(self, x_waveform, x_physical):
        # 多尺度卷积分支
        x1 = self.conv1(x_waveform.unsqueeze(1))  # (batch, 16, L/2)
        x2 = self.conv2(x_waveform.unsqueeze(1))  # (batch, 16, L/2)
        x3 = self.conv3(x_waveform.unsqueeze(1))  # (batch, 16, L/2)
        # 拼接多尺度特征（通道维度拼接）
        x_cnn = torch.cat([x1, x2, x3], dim=1)  # (batch, 48, L/2)
        x_cnn = self.fusion_conv(x_cnn)         # (batch, 64, L/4)
        x_cnn = self.global_pool(x_cnn).squeeze(-1)  # (batch, 64)
        # 融合物理特征
        x_fused = torch.cat([x_cnn, x_physical], dim=1)
        x_fused = self.fusion_branch(x_fused)
        return x_fused


# -------------------------- 3. 数据加载（不变） --------------------------
def load_waveform_csv(csv_path, start_time=80, end_time=130):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("Time(S),Current(A)"))
        
        df_full = pd.read_csv(csv_path, skiprows=header_idx, header=0, dtype=np.float64)
        time_filter = (df_full["Time(S)"] >= start_time) & (df_full["Time(S)"] <= end_time)
        df_filtered = df_full[time_filter].copy()
        
        if df_filtered.empty:
            logger.error(f"时间范围{start_time}-{end_time}秒内无数据")
            return None, None, None
        
        df_filtered.to_csv(os.path.join(SUB_DIRS["raw"], "filtered_raw_data.csv"), index=False)
        df_full.to_csv(os.path.join(SUB_DIRS["raw"], "full_raw_data.csv"), index=False)
        logger.info(f"[SUCCESS] 加载数据：{len(df_filtered)}个采样点，时间范围{df_filtered['Time(S)'].min():.2f}~{df_filtered['Time(S)'].max():.2f}S")
        
        return df_filtered["Time(S)"].values, df_filtered["Current(A)"].values, df_full
    except Exception as e:
        logger.error(f"数据加载失败：{str(e)}")
        return None, None, None


# -------------------------- 4. 周期分割（修改为：波谷开始→两波峰→波谷结束） --------------------------
def split_cycles_by_peaks(time, current, fixed_len=200, peak_thr=0.05, valley_thr=0.05, min_dist=10):
    try:
        current_max, current_min = np.max(current), np.min(current)
        current_range = current_max - current_min
        peak_height = current_min + current_range * peak_thr  # 波峰检测阈值（相对电流范围的比例）
        valley_height = current_max - current_range * valley_thr  # 波谷检测阈值

        # 检测波峰和波谷（波谷通过检测负电流的波峰实现）
        peaks, peak_props = find_peaks(current, height=peak_height, distance=min_dist)
        valleys, valley_props = find_peaks(-current, height=-valley_height, distance=min_dist)
        peak_heights = peak_props['peak_heights']
        valley_heights = -valley_props['peak_heights']

        # 校验：至少需要2个波谷（起点和终点）和2个波峰才能形成一个完整周期
        if len(peaks) < 2 or len(valleys) < 2:
            logger.error(f"波峰({len(peaks)}个)或波谷({len(valleys)}个)数量不足，无法分割周期（需至少2个波峰和2个波谷）")
            return None, None, None

        # 绘制波峰波谷描点图（辅助验证分割逻辑）
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(time, current, color="#3498db", linewidth=1.2, label="原始波形")
        ax.scatter(time[peaks], peak_heights, color="#e74c3c", s=80, marker='^', label="波峰", zorder=5)
        ax.scatter(time[valleys], valley_heights, color="#2ecc71", s=80, marker='v', label="波谷", zorder=5)
        ax.set_title("波峰波谷识别结果（周期分割：波谷→两波峰→波谷）", fontsize=16)
        ax.set_xlabel("时间(S)", fontsize=14), ax.set_ylabel("电流(A)", fontsize=14)
        ax.legend(fontsize=12), ax.grid(alpha=0.3)
        ax.set_xlim(time.min()-1, time.max()+1), ax.set_ylim(current_min-0.5, current_max+0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(SUB_DIRS["plots"], "peaks_valleys_marking.png"), dpi=120)
        plt.close(fig)

        # 周期分割核心逻辑
        raw_cycles = []  # 存储原始周期（时间+电流）
        norm_cycles = []  # 存储归一化后的周期
        cycle_time_ranges = []  # 存储每个周期的时间范围（start_time, end_time）
        scaler = StandardScaler()  # 归一化器

        # 合并波峰波谷并按时间排序（格式：(索引, 类型)，类型为'peak'或'valley'）
        peak_info = [(p, 'peak') for p in peaks]
        valley_info = [(v, 'valley') for v in valleys]
        all_extremes = sorted(peak_info + valley_info, key=lambda x: x[0])  # 按索引（时间）排序
        current_pos = 0  # 当前处理位置（在all_extremes中的索引）

        while current_pos < len(all_extremes):
            # 步骤1：找周期的起始波谷（第一个波谷）
            first_valley_idx = None
            for i in range(current_pos, len(all_extremes)):
                if all_extremes[i][1] == 'valley':
                    first_valley_idx = i
                    break
            if first_valley_idx is None:
                logger.warning("未找到更多波谷，停止分割")
                break  # 没有波谷，无法开始新周期
            
            # 步骤2：从起始波谷后找两个波峰
            peak_count = 0
            second_peak_idx = None
            for i in range(first_valley_idx + 1, len(all_extremes)):
                if all_extremes[i][1] == 'peak':
                    peak_count += 1
                    if peak_count == 2:  # 找到第二个波峰
                        second_peak_idx = i
                        break
            if second_peak_idx is None:
                logger.warning("起始波谷后未找到2个波峰，停止分割")
                break  # 波峰数量不足，无法形成周期
            
            # 步骤3：从第二个波峰后找周期的结束波谷（该波谷将作为下一个周期的起点）
            end_valley_idx = None
            for i in range(second_peak_idx + 1, len(all_extremes)):
                if all_extremes[i][1] == 'valley':
                    end_valley_idx = i
                    break
            if end_valley_idx is None:
                logger.warning("第二个波峰后未找到结束波谷，停止分割")
                break  # 没有结束波谷，周期不完整
            
            # 确定周期的起止索引（原始数据中的索引）
            cycle_start_idx = all_extremes[first_valley_idx][0]  # 起始波谷的索引
            cycle_end_idx = all_extremes[end_valley_idx][0]      # 结束波谷的索引

            # 提取该周期的时间和电流数据
            cycle_time = time[cycle_start_idx:cycle_end_idx + 1]  # 包含结束波谷
            cycle_current = current[cycle_start_idx:cycle_end_idx + 1]
            cycle_time_range = (cycle_time[0], cycle_time[-1])  # 周期的时间范围

            # 过滤过短周期（避免噪声干扰）
            if len(cycle_time) < 10:
                logger.warning(f"跳过过短周期（长度{len(cycle_time)}，时间范围{cycle_time_range}）")
                current_pos = end_valley_idx + 1  # 从结束波谷的下一个点继续
                continue
            
            # 周期归一化（固定长度为fixed_len，便于后续特征提取）
            norm_x = np.linspace(0, 1, fixed_len)  # 归一化时间轴（0到1）
            raw_x = np.linspace(0, 1, len(cycle_current))  # 原始周期的相对时间
            cycle_interp = np.interp(norm_x, raw_x, cycle_current)  # 插值到固定长度
            cycle_norm = scaler.fit_transform(cycle_interp.reshape(-1, 1)).flatten()  # 标准化

            # 保存周期数据
            raw_cycles.append((cycle_time, cycle_current))
            norm_cycles.append(cycle_norm)
            cycle_time_ranges.append(cycle_time_range)

            # 更新当前位置：下一个周期从当前结束波谷开始（实现“结束波谷=新周期起点”）
            current_pos = end_valley_idx  # 注意：这里不+1，因为结束波谷是下一个周期的起点
            logger.debug(f"已分割周期 {len(raw_cycles)}: 索引{cycle_start_idx}~{cycle_end_idx}，时间范围{cycle_time_range}")

        # 校验是否分割到有效周期
        if not norm_cycles:
            logger.error(f"未分割到有效周期")
            return None, None, None

        # 保存周期数据到文件
        norm_cycles = np.array(norm_cycles)
        pd.DataFrame(norm_cycles).to_csv(
            os.path.join(SUB_DIRS["cycles"], "normalized_cycles.csv"),
            index=False, header=[f"point_{i}" for i in range(fixed_len)]
        )
        for i, (cycle_time, cycle_current) in enumerate(raw_cycles, 1):
            pd.DataFrame({
                "Time(S)": cycle_time,
                "Current(A)": cycle_current
            }).to_csv(os.path.join(SUB_DIRS["cycles"], f"raw_cycle_{i}.csv"), index=False)

        logger.info(f"[SUCCESS] 周期分割完成：共{len(norm_cycles)}个有效周期（每个周期：波谷→2个波峰→波谷）")
        return norm_cycles, raw_cycles, cycle_time_ranges

    except Exception as e:
        logger.error(f"周期分割失败：{str(e)}")
        return None, None, None



# -------------------------- 5. 辅助函数：绘制单个波形（不变） --------------------------
def plot_single_waveform(cycle_time, cycle_current, save_path, title, is_normal=True):
    try:
        fig, ax = plt.subplots(figsize=(8, 3))
        color = "#2ecc71" if is_normal else "#e74c3c"
        ax.plot(cycle_time, cycle_current, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time(S)"), ax.set_ylabel("Current(A)")
        ax.grid(alpha=0.3), ax.set_xlim(cycle_time.min()-0.1, cycle_time.max()+0.1)
        plt.tight_layout(), plt.savefig(save_path, dpi=100), plt.close(fig)
        return True
    except Exception as e:
        logger.warning(f"[WARNING] 绘制波形失败：{str(e)}")
        return False


# -------------------------- 6. 修复：主导波形识别（聚类数范围+异常处理） --------------------------
def find_dominant_waveforms(cycles_norm, raw_cycles, time, current, cycle_time_ranges, 
                            min_cluster=2, max_cluster=5, threshold=0.4, fixed_len=200):
    try:
        n_cycles = len(cycles_norm)
        logger.info(f"[INFO] 开始主导波形识别：{n_cycles}个周期，波形长度{fixed_len}")
        
        # 安全校验：调整聚类数范围（必须满足 2 ≤ K ≤ n_cycles-1）
        if n_cycles < 2:
            logger.error(f"周期数过少（{n_cycles}个），无法进行聚类")
            return None
        # 修正最大聚类数：不超过“周期数-1”，且不小于最小聚类数
        max_cluster = min(max_cluster, n_cycles - 1)
        min_cluster = max(min_cluster, 2)
        if min_cluster > max_cluster:
            min_cluster = 2
            max_cluster = min(5, n_cycles - 1)  # 若仍不满足，强制设为2~5（或周期数-1）
        logger.info(f"[INFO] 修正聚类数范围：K ∈ [{min_cluster}, {max_cluster}]（周期数={n_cycles}）")

        # 步骤1：提取时域物理特征
        physical_features = extract_waveform_physical_features(
            raw_cycles=raw_cycles,
            time=time,
            current=current,
            peak_thr=0.15,
            valley_thr=0.15,
            min_dist=15
        )

        # 步骤2：CNN特征提取（融合物理特征）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = WaveformFeatureExtractor(
            input_len=fixed_len,
            cnn_feature_dim=64,
            physical_feature_dim=physical_features.shape[1]
        ).to(device)

        cycles_tensor = torch.FloatTensor(cycles_norm).to(device)
        physical_tensor = torch.FloatTensor(physical_features).to(device)

        with torch.no_grad():
            fused_features = extractor(cycles_tensor, physical_tensor).cpu().numpy()

        # 保存融合特征
        pd.DataFrame(fused_features).to_csv(
            os.path.join(SUB_DIRS["features"], "fused_features.csv"),
            index=False, header=[f"fused_feat_{i}" for i in range(64)]
        )
        logger.info(f"[SUCCESS] 融合特征提取完成：shape={fused_features.shape}（CNN特征+物理特征）")

        # 步骤3：最优聚类数选择（基于修正后的K范围）
        best_k = 2
        best_sil_score = -1
        sil_scores = []
        valid_k_list = []  # 记录有效K值（避免后续索引越界）
        for k in range(min_cluster, max_cluster + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
                cluster_labels = kmeans.fit_predict(fused_features)
                # 仅当簇数=K时才计算轮廓系数（避免所有样本聚为1个簇）
                if len(np.unique(cluster_labels)) == k:
                    sil_score = silhouette_score(fused_features, cluster_labels)
                    sil_scores.append(sil_score)
                    valid_k_list.append(k)
                    if sil_score > best_sil_score:
                        best_sil_score = sil_score
                        best_k = k
                else:
                    sil_scores.append(-1)
                    valid_k_list.append(k)
                    logger.warning(f"[WARNING] 聚类数K={k}时，实际簇数={len(np.unique(cluster_labels))}（≠K），跳过轮廓系数计算")
            except Exception as e:
                sil_scores.append(-1)
                valid_k_list.append(k)
                logger.warning(f"[WARNING] 聚类数K={k}计算失败：{str(e)}")

        # 可视化轮廓系数（仅显示有效K值）
        if valid_k_list and sil_scores:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(valid_k_list, sil_scores, 'o-', color="#3498db", linewidth=2)
            ax.axvline(x=best_k, color="#e74c3c", linestyle='--', 
                       label=f"最优K={best_k}（轮廓系数={best_sil_score:.3f}）")
            ax.set_title(f"不同聚类数的轮廓系数（K ∈ [{min_cluster}, {max_cluster}]）", fontsize=14)
            ax.set_xlabel("聚类数K"), ax.set_ylabel("轮廓系数")
            ax.set_xticks(valid_k_list)  # 仅显示有效K值
            ax.legend(), ax.grid(alpha=0.3)
            plt.tight_layout(), plt.savefig(os.path.join(SUB_DIRS["plots"], "silhouette_scores.png")), plt.close(fig)
        else:
            logger.warning(f"[WARNING] 无有效聚类数，默认选择K=2")
            best_k = 2

        # 步骤4：最终聚类与占比统计
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        final_labels = kmeans_final.fit_predict(fused_features)
        cluster_counts = np.bincount(final_labels)
        cluster_ratios = cluster_counts / n_cycles
        
        # 保存聚类结果
        cluster_result_df = pd.DataFrame({
            "cycle_id": range(1, n_cycles + 1),
            "cluster_label": final_labels,
            "cluster_ratio": [cluster_ratios[label] for label in final_labels],
            "start_time": [cycle_time_ranges[i][0] for i in range(n_cycles)],
            "end_time": [cycle_time_ranges[i][1] for i in range(n_cycles)]
        })
        cluster_result_df.to_csv(os.path.join(SUB_DIRS["dominant"], "cluster_results.csv"), index=False)

        # 步骤5：筛选正常（主导）/异常（非主导）周期
        dominant_cluster_labels = [i for i, ratio in enumerate(cluster_ratios) if ratio >= threshold]
        if not dominant_cluster_labels:
            dominant_cluster_labels = [np.argmax(cluster_ratios)]
            logger.warning(f"[WARNING] 无占比≥{threshold}的簇，选择占比最高簇为正常（占比{cluster_ratios[dominant_cluster_labels[0]]:.2f}）")
        
        # 分离正常/异常索引
        normal_indices = [i for i, label in enumerate(final_labels) if label in dominant_cluster_labels]
        abnormal_indices = [i for i, label in enumerate(final_labels) if label not in dominant_cluster_labels]
        logger.info(f"[INFO] 分类结果：正常周期{len(normal_indices)}个，异常周期{len(abnormal_indices)}个")

        # 步骤6：保存正常波形
        if normal_indices:
            for seq, idx in enumerate(normal_indices, 1):
                cycle_id = idx + 1
                cycle_time, cycle_current = raw_cycles[idx]
                cycle_norm = cycles_norm[idx]
                start_time, end_time = cycle_time_ranges[idx]

                raw_save_path = os.path.join(SUB_DIRS["normal_data"], f"normal_cycle_{seq}_id{cycle_id}.csv")
                pd.DataFrame({"Time(S)": cycle_time, "Current(A)": cycle_current}).to_csv(raw_save_path, index=False)
                norm_save_path = os.path.join(SUB_DIRS["normal_data"], f"normal_cycle_{seq}_id{cycle_id}_normalized.csv")
                pd.DataFrame(cycle_norm).T.to_csv(norm_save_path, index=False, header=[f"point_{i}" for i in range(fixed_len)])

                plot_save_path = os.path.join(SUB_DIRS["normal_plots"], f"normal_cycle_{seq}_id{cycle_id}.png")
                plot_title = f"正常周期 #{seq}（ID：{cycle_id}，时间：{start_time:.2f}~{end_time:.2f}S）"
                plot_single_waveform(cycle_time, cycle_current, plot_save_path, plot_title, is_normal=True)

            # 正常周期汇总图
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_count = min(len(normal_indices), 20)
            for seq in range(plot_count):
                idx = normal_indices[seq]
                cycle_time, cycle_current = raw_cycles[idx]
                norm_time = np.linspace(0, 1, len(cycle_time))
                ax.plot(norm_time, cycle_current, color="#2ecc71", alpha=0.4, linewidth=1,
                        label=f"正常周期{seq+1}" if seq < 10 else "")
            normal_avg = np.mean(cycles_norm[normal_indices], axis=0)
            ax.plot(np.linspace(0, 1, fixed_len), normal_avg, color="#27ae60", linewidth=2.5,
                    label=f"正常平均波形（共{len(normal_indices)}个周期）")
            ax.set_title("正常波形汇总（前20个+平均波形）", fontsize=14)
            ax.set_xlabel("归一化时间"), ax.set_ylabel("电流(A)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left'), ax.grid(alpha=0.3)
            plt.tight_layout(), plt.savefig(os.path.join(SUB_DIRS["normal_plots"], "normal_cycles_summary.png")), plt.close(fig)
        else:
            logger.warning(f"[WARNING] 无正常周期可保存")

        # 步骤7：保存异常波形
        if abnormal_indices:
            for seq, idx in enumerate(abnormal_indices, 1):
                cycle_id = idx + 1
                cycle_time, cycle_current = raw_cycles[idx]
                cycle_norm = cycles_norm[idx]
                start_time, end_time = cycle_time_ranges[idx]

                raw_save_path = os.path.join(SUB_DIRS["abnormal_data"], f"abnormal_cycle_{seq}_id{cycle_id}.csv")
                pd.DataFrame({"Time(S)": cycle_time, "Current(A)": cycle_current}).to_csv(raw_save_path, index=False)
                norm_save_path = os.path.join(SUB_DIRS["abnormal_data"], f"abnormal_cycle_{seq}_id{cycle_id}_normalized.csv")
                pd.DataFrame(cycle_norm).T.to_csv(norm_save_path, index=False, header=[f"point_{i}" for i in range(fixed_len)])

                plot_save_path = os.path.join(SUB_DIRS["abnormal_plots"], f"abnormal_cycle_{seq}_id{cycle_id}.png")
                plot_title = f"异常周期 #{seq}（ID：{cycle_id}，时间：{start_time:.2f}~{end_time:.2f}S）"
                plot_single_waveform(cycle_time, cycle_current, plot_save_path, plot_title, is_normal=False)

            # 异常周期汇总图
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_count = min(len(abnormal_indices), 20)
            for seq in range(plot_count):
                idx = abnormal_indices[seq]
                cycle_time, cycle_current = raw_cycles[idx]
                norm_time = np.linspace(0, 1, len(cycle_time))
                ax.plot(norm_time, cycle_current, color="#e74c3c", alpha=0.4, linewidth=1,
                        label=f"异常周期{seq+1}" if seq < 10 else "")
            if len(abnormal_indices) > 1:
                abnormal_avg = np.mean(cycles_norm[abnormal_indices], axis=0)
                ax.plot(np.linspace(0, 1, fixed_len), abnormal_avg, color="#c0392b", linewidth=2.5,
                        label=f"异常平均波形（共{len(abnormal_indices)}个周期）")
            ax.set_title("异常波形汇总（前20个+平均波形）", fontsize=14)
            ax.set_xlabel("归一化时间"), ax.set_ylabel("电流(A)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left'), ax.grid(alpha=0.3)
            plt.tight_layout(), plt.savefig(os.path.join(SUB_DIRS["abnormal_plots"], "abnormal_cycles_summary.png")), plt.close(fig)
        else:
            logger.warning(f"[WARNING] 无异常周期可保存")

        # 步骤8：主导波形可视化
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
        plt.tight_layout(), plt.savefig(os.path.join(SUB_DIRS["plots"], "dominant_waveforms.png")), plt.close(fig)

        logger.info(f"[SUCCESS] 主导波形识别与正常/异常分类完成！（基于融合特征）")
        return dominant_results
    except Exception as e:
        logger.error(f"主导波形识别失败：{str(e)}")
        return None


# -------------------------- 7. 总波形标注（不变） --------------------------
def plot_total_waveform_with_annotation(full_raw_df, dominant_results, start_time=80, end_time=130):
    try:
        total_data = full_raw_df[full_raw_df["Time(S)"].between(start_time, end_time)]
        total_time = total_data["Time(S)"].values
        total_current = total_data["Current(A)"].values
        
        cycle_time_ranges = dominant_results["cycle_time_ranges"]
        normal_indices = dominant_results["normal_indices"]
        abnormal_indices = dominant_results["abnormal_indices"]
        normal_time_ranges = [cycle_time_ranges[i] for i in normal_indices]
        abnormal_time_ranges = [cycle_time_ranges[i] for i in abnormal_indices]
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(total_time, total_current, color="#3498db", linewidth=1.2, label="原始总波形")
        for (s, e) in normal_time_ranges:
            ax.axvspan(s, e, alpha=0.3, color="#2ecc71", label="正常波形段" if (s, e) == normal_time_ranges[0] else "")
        for (s, e) in abnormal_time_ranges:
            ax.axvspan(s, e, alpha=0.3, color="#e74c3c", label="异常波形段" if (s, e) == abnormal_time_ranges[0] else "")
        
        ax.set_title(f"{start_time}-{end_time}秒总波形（正常/异常段标注）", fontsize=16)
        ax.set_xlabel("时间(S)", fontsize=12), ax.set_ylabel("电流(A)", fontsize=12)
        ax.set_xticks(np.linspace(start_time, end_time, 11))
        ax.legend(fontsize=10), ax.grid(alpha=0.3)
        
        plt.tight_layout(), plt.savefig(os.path.join(SUB_DIRS["total_waveform"], "total_waveform_annotation.png"), dpi=100), plt.close(fig)
        
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


# -------------------------- 8. 主函数（不变） --------------------------
def main(csv_path, fixed_len=200, start_time=80, end_time=130, dominant_threshold=0.4, min_dist=10):
    logger.info("="*60)
    logger.info("开始波形主导模式识别+正常/异常分类流程（融合特征版）")
    logger.info("="*60)
    
    # 步骤1：加载数据
    time, current, full_raw_df = load_waveform_csv(csv_path, start_time, end_time)
    if time is None or current is None or full_raw_df is None:
        logger.error("流程终止：数据加载失败")
        return
    
    # 步骤2：分割周期
    cycles_norm, raw_cycles, cycle_time_ranges = split_cycles_by_peaks(
        time, current, fixed_len, min_dist=min_dist,
    )
    if cycles_norm is None or raw_cycles is None:
        logger.error("流程终止：周期分割失败")
        return
    
    # 步骤3：主导波形识别
    dominant_results = find_dominant_waveforms(
        cycles_norm=cycles_norm,
        raw_cycles=raw_cycles,
        time=time,
        current=current,
        cycle_time_ranges=cycle_time_ranges,
        min_cluster=2,
        max_cluster=5,  # 初始设为5（后续会自动修正）
        threshold=dominant_threshold,
        fixed_len=fixed_len
    )
    if dominant_results is None:
        logger.error("流程终止：主导波形识别失败")
        return
    
    # 步骤4：总波形标注
    plot_total_waveform_with_annotation(full_raw_df, dominant_results, start_time, end_time)
    
    # 输出结果汇总
    normal_count = len(dominant_results["normal_indices"])
    abnormal_count = len(dominant_results["abnormal_indices"])
    total_count = normal_count + abnormal_count
    logger.info("="*60)
    logger.info("流程完成！最终结果汇总：")
    logger.info(f"1. 周期统计：")
    logger.info(f"   - 总周期数：{total_count}个")
    logger.info(f"   - 正常周期：{normal_count}个（{normal_count/total_count*100:.1f}%）")
    logger.info(f"   - 异常周期：{abnormal_count}个（{abnormal_count/total_count*100:.1f}%）")
    logger.info(f"2. 关键特征文件：")
    logger.info(f"   - 物理特征：{os.path.join(SUB_DIRS['features'], 'physical_features.csv')}")
    logger.info(f"   - 融合特征：{os.path.join(SUB_DIRS['features'], 'fused_features.csv')}")
    logger.info(f"3. 图表路径：")
    logger.info(f"   - 波峰波谷图：{os.path.join(SUB_DIRS['plots'], 'peaks_valleys_marking.png')}")
    logger.info(f"   - 总波形标注图：{os.path.join(SUB_DIRS['total_waveform'], 'total_waveform_annotation.png')}")
    logger.info("="*60)


# -------------------------- 9. 调用示例（不变） --------------------------
if __name__ == "__main__":
    CSV_PATH = "16-波形检测与分类\\knee-sensor\\内翻-0-1.csv"
    FIXED_LEN = 200
    START_TIME = 40
    END_TIME = 130
    DOMINANT_THRESHOLD = 0.5
    MIN_DIST = 10
    
    main(
        csv_path=CSV_PATH,
        fixed_len=FIXED_LEN,
        start_time=START_TIME,
        end_time=END_TIME,
        dominant_threshold=DOMINANT_THRESHOLD,
        min_dist=MIN_DIST,
    )