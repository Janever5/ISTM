import numpy as np
import pandas as pd
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from scipy.signal import correlate, find_peaks, medfilt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 【关键可调参数】—— 在此处修改，无需改动函数内部！
# ==============================================================================
# ！！！修改为你的CSV文件路径（避免空格/特殊字符）
CSV_FILE_PATH = r"16-波形检测与分类\knee-sensor\内翻-90-1.csv"

# 1. 数据预处理参数
START_TIME = 0.1          # 分析起始时间（s）
END_TIME = 180.0          # 分析结束时间（s）
SIGMA_THRESHOLD = 3       # 异常值过滤：3σ原则（可调1-5）
MEDFILT_KERNEL = 5        # 中值滤波核大小（奇数，可调3-7）

# 2. 周期估计参数
AUTOCORR_PROMINENCE = 0.2 # 自相关峰值突出度（越小越灵敏，可调0.1-0.5）
MIN_LAG_SEC = 0.5         # 自相关最小滞后时间（s，避免0滞后干扰）

# 3. 伪标签生成参数
PSEUDO_PEAK_HEIGHT = 0.03 # 信号峰值高度（相对于标准差，越小越灵敏，可调0.01-0.1）
PSEUDO_PEAK_DISTANCE = 0.3 # 信号峰值最小距离（相对于最小周期，可调0.2-0.5）

# 4. CNN训练参数
CNN_SEQ_LEN_RATIO = 1.0   # 序列长度=平均周期×该比例（可调0.8-1.5，解决序列过长）
MAX_SEQ_LEN = 200         # 最大序列长度（限制过长序列，可调100-300）
CNN_BATCH_SIZE = 16       # 批次大小（可调8-32）
CNN_LR = 3e-4             # 学习率（可调1e-4-5e-4，解决过拟合）
CNN_WEIGHT_DECAY = 1e-5    # 权重衰减（可调1e-6-1e-4）
CNN_EPOCHS = 100          # 最大轮次
CNN_EARLY_STOP = 8        # 早停耐心值（可调5-10，避免提前停止）

# 5. 周期提取参数
PERIOD_THRESHOLD_PERCENTILE = 40 # 动态阈值分位数（越小越宽松，可调30-60）
MIN_CYCLE_RATIO = 0.5     # 最小周期=估计最小周期×该比例（可调0.3-0.8）
MAX_CYCLE_RATIO = 1.5     # 最大周期=估计最大周期×该比例（可调1.2-2.0）
CYCLE_PEAK_REQUIRED = False # 是否要求周期内有峰值（False：允许平滑周期，解决无峰值问题）
CYCLE_PEAK_HEIGHT = 0.02  # 周期内峰值高度（可调0.01-0.05，仅当CYCLE_PEAK_REQUIRED=True时生效）

# ==============================================================================
# 1. 全局配置
# ==============================================================================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "cnn_normal_waveform_detection"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
SUB_DIRS = {
    "0_raw": os.path.join(OUTPUT_DIR, "0_raw_data"),
    "1_preprocessed": os.path.join(OUTPUT_DIR, "1_preprocessed_data"),
    "2_period_est": os.path.join(OUTPUT_DIR, "2_period_estimation"),
    "3_pseudo_labels": os.path.join(OUTPUT_DIR, "3_pseudo_labels"),
    "4_model": os.path.join(OUTPUT_DIR, "4_trained_model"),
    "5_normal_cycles": os.path.join(OUTPUT_DIR, "5_normal_cycles"),
    "6_visualizations": os.path.join(OUTPUT_DIR, "6_visualizations")
}
for d in SUB_DIRS.values():
    os.makedirs(d, exist_ok=True)

# 日志增强：打印概率分布、周期数量等关键信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(OUTPUT_DIR, "detection_log.log")), logging.StreamHandler()]
)
logger = logging.getLogger()


# ==============================================================================
# 2. 数据加载与预处理（不变，已修复维度问题）
# ==============================================================================
def load_and_preprocess_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        header_idx = next(i for i, line in enumerate(lines) if "Time(S)" in line and "Current(A)" in line)
        df = pd.read_csv(file_path, skiprows=header_idx)
    except Exception as e:
        logger.warning(f"标准表头读取失败，兼容处理：{str(e)}")
        df = pd.read_csv(file_path)
        df.columns = ["Time(S)", "Current(A)"] + list(df.columns[2:]) if len(df.columns) > 2 else ["Time(S)", "Current(A)"]

    # 分步筛选，确保维度一致
    time = pd.to_numeric(df["Time(S)"], errors='coerce').values.astype(np.float32)
    signal = pd.to_numeric(df["Current(A)"], errors='coerce').values.astype(np.float32)
    
    # 1. 过滤NaN
    nan_mask = ~np.isnan(time) & ~np.isnan(signal)
    time_nan = time[nan_mask]
    signal_nan = signal[nan_mask]
    
    # 2. 过滤异常值
    signal_mean = np.mean(signal_nan)
    signal_std = np.std(signal_nan)
    sigma_mask = np.abs(signal_nan - signal_mean) <= SIGMA_THRESHOLD * signal_std
    time_sigma = time_nan[sigma_mask]
    signal_sigma = signal_nan[sigma_mask]
    
    # 3. 时间范围过滤
    time_range_mask = (time_sigma >= START_TIME) & (time_sigma <= END_TIME)
    time_final = time_sigma[time_range_mask]
    signal_final = signal_sigma[time_range_mask]
    
    # 4. 去噪
    signal_denoised = medfilt(signal_final, kernel_size=MEDFILT_KERNEL)

    # 保存与日志
    pd.DataFrame({
        "Time(S)": time_final,
        "Raw_Current(A)": signal_final,
        "Denoised_Current(A)": signal_denoised
    }).to_csv(os.path.join(SUB_DIRS["1_preprocessed"], "preprocessed_data.csv"), index=False)
    logger.info(f"预处理完成：{len(time_final)}个采样点（原始{len(time)}个，过滤{len(time)-len(time_final)}个）")
    assert len(time_final) == len(signal_denoised), "时间与信号维度不匹配！"
    return time_final, signal_denoised


# ==============================================================================
# 3. 周期估计优化（更灵敏的峰值检测）
# ==============================================================================
def estimate_period_range(time, signal):
    dt = np.mean(np.diff(time))
    signal_norm = signal - np.mean(signal)

    # 自相关分析（跳过最小滞后）
    autocorr = correlate(signal_norm, signal_norm, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # 正滞后部分
    lags = np.arange(len(autocorr)) * dt
    # 过滤最小滞后（避免0滞后干扰）
    lag_mask = lags >= MIN_LAG_SEC
    autocorr = autocorr[lag_mask]
    lags = lags[lag_mask]

    # 检测自相关峰值（降低突出度，更灵敏）
    peaks, _ = find_peaks(
        autocorr,
        distance=int(MIN_LAG_SEC / dt),  # 峰值最小距离
        prominence=np.std(autocorr) * AUTOCORR_PROMINENCE  # 可调突出度
    )

    # 确定周期范围（兼容无峰值情况）
    if len(peaks) == 0:
        logger.warning("未检测到自相关峰值，使用经验周期范围（0.5-5s）")
        min_period_sec = 0.5
        max_period_sec = 5.0
    else:
        peak_periods = lags[peaks[:5]]  # 取前5个峰值，更鲁棒
        min_period_sec = np.min(peak_periods) * 0.7  # 放宽下限
        max_period_sec = np.max(peak_periods) * 1.3  # 放宽上限
    logger.info(f"周期范围估计：{min_period_sec:.2f}s ~ {max_period_sec:.2f}s（采样间隔：{dt:.4f}s）")

    # 可视化自相关结果（便于调试）
    plt.figure(figsize=(12, 4))
    plt.plot(lags, autocorr, color='purple', label='自相关曲线')
    plt.scatter(lags[peaks], autocorr[peaks], color='red', s=50, label='自相关峰值')
    plt.axvspan(min_period_sec, max_period_sec, color='red', alpha=0.2, label='估计周期范围')
    plt.xlabel('滞后时间 (s)')
    plt.ylabel('自相关系数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SUB_DIRS["2_period_est"], "autocorrelation.png"), dpi=150)
    plt.close()

    return min_period_sec, max_period_sec, dt


# ==============================================================================
# 4. 伪标签生成优化（增加周期区域，解决正负样本失衡）
# ==============================================================================
def generate_precise_pseudo_labels(time, signal, min_period_sec, max_period_sec, dt):
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    min_period_samples = int(min_period_sec / dt)

    # 1. 检测信号峰值（降低高度阈值，更灵敏）
    peaks, _ = find_peaks(
        signal,
        height=signal_mean + PSEUDO_PEAK_HEIGHT * signal_std,  # 可调高度
        distance=int(PSEUDO_PEAK_DISTANCE * min_period_samples)  # 可调距离
    )
    logger.info(f"信号峰值检测：共{len(peaks)}个峰值（阈值：{signal_mean + PSEUDO_PEAK_HEIGHT * signal_std:.6f}）")

    # 2. 基于峰值生成伪标签（增加周期覆盖）
    pseudo_labels = np.zeros_like(signal)
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            # 放宽周期长度限制
            cycle_duration = (end_idx - start_idx) * dt
            if (min_period_sec * 0.5) <= cycle_duration <= (max_period_sec * 1.5):
                pseudo_labels[start_idx:end_idx] = 1
    else:
        # 无峰值时：滑动窗口（增加步长密度）
        window_samples = int((min_period_sec + max_period_sec) / 2 / dt)
        step = int(window_samples * 0.3)  # 步长=30%窗口，增加覆盖
        for i in range(0, len(signal) - window_samples, step):
            pseudo_labels[i:i + window_samples] = 1

    # 统计伪标签比例（避免正负失衡）
    pseudo_ratio = np.mean(pseudo_labels)
    logger.info(f"伪标签生成：周期区域占比{ pseudo_ratio:.2%}（避免<10%导致失衡）")
    if pseudo_ratio < 0.1:
        logger.warning("伪标签周期区域过少（<10%），可能导致模型偏向非周期预测！")

    # 保存与可视化
    pd.DataFrame({
        "Time(S)": time,
        "Denoised_Current(A)": signal,
        "Pseudo_Label(1=周期)": pseudo_labels
    }).to_csv(os.path.join(SUB_DIRS["3_pseudo_labels"], "precise_pseudo_labels.csv"), index=False)

    plt.figure(figsize=(15, 5))
    plt.plot(time, signal, color='steelblue', label='去噪后信号')
    plt.scatter(time[peaks], signal[peaks], color='orange', s=30, label='信号峰值')
    plt.fill_between(time, signal.min(), signal.max(), where=(pseudo_labels == 1),
                     color='green', alpha=0.2, label='伪标签周期区域')
    plt.xlabel('时间 (s)')
    plt.ylabel('电流 (A)')
    plt.title(f'伪标签（周期区域占比：{pseudo_ratio:.2%}）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SUB_DIRS["3_pseudo_labels"], "pseudo_labels_visual.png"), dpi=150)
    plt.close()

    return pseudo_labels


# ==============================================================================
# 5. CNN模型与训练优化（解决序列过长、过拟合）
# ==============================================================================
class MultiScalePeriodCNN(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        # 简化模型，避免过拟合
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, padding=2)  # 减少通道数
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Conv1d(32, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.1)  # 降低 dropout 比例

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x)).squeeze(1)
        return x


class WaveformDataset(Dataset):
    def __init__(self, signal, labels, seq_len):
        self.signal = signal
        self.labels = labels
        self.seq_len = seq_len
        self.valid_indices = [i for i in range(len(signal) - seq_len + 1)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.seq_len
        x = torch.FloatTensor(self.signal[start:end]).unsqueeze(0)
        y = torch.FloatTensor(self.labels[start:end])
        return x, y


def train_multi_scale_cnn(signal, labels, min_period_sec, max_period_sec, dt):
    # 优化序列长度（解决795过长问题）
    avg_period_samples = int((min_period_sec + max_period_sec) / 2 / dt)
    seq_len = int(avg_period_samples * CNN_SEQ_LEN_RATIO)
    seq_len = min(seq_len, MAX_SEQ_LEN)  # 限制最大长度
    seq_len = max(seq_len, 32)  # 限制最小长度
    logger.info(f"CNN序列长度：{seq_len}个采样点（平均周期：{avg_period_samples}个采样点）")

    # 数据归一化
    scaler = StandardScaler()
    signal_norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    np.save(os.path.join(SUB_DIRS["4_model"], "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(SUB_DIRS["4_model"], "scaler_std.npy"), scaler.scale_)

    # 构建数据集（确保有足够样本）
    if len(signal_norm) - seq_len + 1 < 10:
        logger.error(f"样本数量过少（{len(signal_norm) - seq_len + 1} < 10），无法训练！")
        raise ValueError("样本不足，请扩大时间范围或减小序列长度")
    dataset = WaveformDataset(signal_norm, labels, seq_len)
    dataloader = DataLoader(dataset, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=0)

    # 模型初始化（简化模型，避免过拟合）
    model = MultiScalePeriodCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 优化损失函数（动态调整正样本权重）
    pos_ratio = np.mean(labels)
    if pos_ratio < 0.01:
        pos_weight = torch.tensor([10.0], device=device)
    else:
        pos_weight = torch.tensor([(1 - pos_ratio) / pos_ratio], device=device)
    criterion = nn.BCELoss(weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)

    # 训练循环（优化早停）
    best_loss = float('inf')
    early_stop_cnt = 0
    for epoch in range(CNN_EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        # 每5轮打印一次日志（更频繁观察）
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{CNN_EPOCHS} | 平均损失：{avg_loss:.6f}（正样本权重：{pos_weight.item():.2f}）")
        
        # 早停逻辑（放宽耐心值）
        if avg_loss < best_loss - 1e-6:  # 允许微小波动
            best_loss = avg_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), os.path.join(SUB_DIRS["4_model"], "best_period_cnn.pth"))
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= CNN_EARLY_STOP:
                logger.info(f"早停触发（{CNN_EARLY_STOP}轮无改善），最佳损失：{best_loss:.6f}")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(SUB_DIRS["4_model"], "best_period_cnn.pth"), weights_only=True))
    logger.info("CNN模型训练完成")
    return model, scaler, seq_len


# ==============================================================================
# 6. 周期提取优化（核心修复：解决无周期问题）
# ==============================================================================
def extract_and_save_normal_cycles(time, signal, probs, min_period_sec, max_period_sec, dt):
    # 1. 修复动态阈值计算（处理probs全为0的情况）
    non_zero_probs = probs[probs > 1e-6]  # 允许微小非零值
    if len(non_zero_probs) == 0:
        logger.warning("CNN预测概率全为0，使用固定阈值0.3（宽松判定）")
        threshold = 0.3
    else:
        threshold = np.percentile(non_zero_probs, PERIOD_THRESHOLD_PERCENTILE)
        threshold = max(0.1, min(0.7, threshold))  # 限制阈值在0.1-0.7之间
    predictions = (probs > threshold).astype(int)
    logger.info(f"周期判定：预测概率范围[{probs.min():.6f}, {probs.max():.6f}] | 动态阈值：{threshold:.4f}")

    # 2. 调整周期长度范围（更宽松）
    min_cycle_samples = int(min_period_sec * MIN_CYCLE_RATIO / dt)
    max_cycle_samples = int(max_period_sec * MAX_CYCLE_RATIO / dt)
    min_cycle_samples = max(min_cycle_samples, 3)  # 最小3个采样点
    max_cycle_samples = max(max_cycle_samples, min_cycle_samples + 5)  # 确保max>min
    logger.info(f"周期长度范围：{min_cycle_samples} ~ {max_cycle_samples}个采样点（{min_cycle_samples*dt:.2f} ~ {max_cycle_samples*dt:.2f}s）")

    # 3. 提取候选周期
    candidate_cycles = []
    in_cycle = False
    cycle_start = 0
    for i, val in enumerate(predictions):
        if val == 1 and not in_cycle:
            cycle_start = i
            in_cycle = True
        elif val == 0 and in_cycle:
            cycle_end = i
            cycle_length = cycle_end - cycle_start
            if min_cycle_samples <= cycle_length <= max_cycle_samples:
                candidate_cycles.append((cycle_start, cycle_end))
            in_cycle = False
    # 处理最后一个周期
    if in_cycle:
        cycle_end = len(predictions)
        cycle_length = cycle_end - cycle_start
        if min_cycle_samples <= cycle_length <= max_cycle_samples:
            candidate_cycles.append((cycle_start, cycle_end))
    logger.info(f"候选周期数量：{len(candidate_cycles)}个（未过滤前）")

    # 4. 过滤无效周期（可选：是否要求峰值）
    normal_cycles = []
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    for (s, e) in candidate_cycles:
        cycle_signal = signal[s:e]
        # 条件1：周期内信号有波动（避免平直线）
        cycle_std = np.std(cycle_signal)
        if cycle_std < 1e-6:
            continue
        # 条件2：可选峰值要求
        if CYCLE_PEAK_REQUIRED:
            cycle_peaks, _ = find_peaks(
                cycle_signal,
                height=signal_mean + CYCLE_PEAK_HEIGHT * signal_std
            )
            if len(cycle_peaks) == 0:
                continue
        # 满足条件则保留
        normal_cycles.append((s, e))

    # 5. 保存正常周期
    if len(normal_cycles) == 0:
        logger.warning("未提取到有效正常周期！建议：1.降低PERIOD_THRESHOLD_PERCENTILE；2.设置CYCLE_PEAK_REQUIRED=False")
        # 保存预测概率图，便于调试
        plt.figure(figsize=(12, 3))
        plt.plot(time, probs, color='purple', label='CNN预测概率')
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'阈值：{threshold:.4f}')
        plt.xlabel('时间 (s)')
        plt.ylabel('周期概率')
        plt.title('CNN预测概率与阈值（无有效周期）')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(SUB_DIRS["6_visualizations"], "prob_debug.png"), dpi=150)
        plt.close()
        return normal_cycles, predictions

    # 保存每个正常周期
    logger.info(f"有效正常周期：{len(normal_cycles)}个，开始存储...")
    for cycle_idx, (s, e) in enumerate(normal_cycles, 1):
        cycle_time = time[s:e]
        cycle_signal = signal[s:e]
        cycle_duration = (e - s) * dt

        # CSV文件
        cycle_df = pd.DataFrame({
            "Time(S)": cycle_time,
            "Current(A)": cycle_signal,
            "Cycle_ID": cycle_idx,
            "Duration(S)": cycle_duration,
            "Sample_Count": len(cycle_signal)
        })
        cycle_df.to_csv(os.path.join(SUB_DIRS["5_normal_cycles"], f"normal_cycle_{cycle_idx}.csv"), index=False)

        # 波形图
        plt.figure(figsize=(8, 4))
        plt.plot(cycle_time, cycle_signal, color='green', linewidth=1.5)
        # 标注峰值（若有）
        if CYCLE_PEAK_REQUIRED:
            cycle_peaks, _ = find_peaks(cycle_signal, height=signal_mean + CYCLE_PEAK_HEIGHT * signal_std)
            if len(cycle_peaks) > 0:
                plt.scatter(cycle_time[cycle_peaks], cycle_signal[cycle_peaks], color='red', s=40, label='周期峰值')
        plt.xlabel('时间 (s)')
        plt.ylabel('电流 (A)')
        plt.title(f'正常周期 {cycle_idx} | 时长：{cycle_duration:.2f}s | 采样点：{len(cycle_signal)}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SUB_DIRS["5_normal_cycles"], f"normal_cycle_{cycle_idx}.png"), dpi=150)
        plt.close()

    # 周期汇总
    cycle_summary = pd.DataFrame({
        "Cycle_ID": range(1, len(normal_cycles) + 1),
        "Start_Time(S)": [time[s] for s, e in normal_cycles],
        "End_Time(S)": [time[e] for s, e in normal_cycles],
        "Duration(S)": [(e - s) * dt for s, e in normal_cycles],
        "Sample_Count": [e - s for s, e in normal_cycles]
    })
    cycle_summary.to_csv(os.path.join(SUB_DIRS["5_normal_cycles"], "normal_cycles_summary.csv"), index=False)
    logger.info(f"正常周期存储完成：{SUB_DIRS['5_normal_cycles']}")

    return normal_cycles, predictions


# ==============================================================================
# 7. 预测与可视化（不变）
# ==============================================================================
def predict_and_visualize(model, scaler, signal, time, seq_len, min_period_sec, max_period_sec, dt):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 数据归一化
    signal_norm = scaler.transform(signal.reshape(-1, 1)).flatten()

    # 滑动窗口预测（增加步长密度，避免漏检）
    probs = np.zeros_like(signal_norm)
    counts = np.zeros_like(signal_norm)
    step = max(1, seq_len // 4)  # 步长=25%序列长度，更密集

    with torch.no_grad():
        for i in range(0, len(signal_norm) - seq_len + 1, step):
            x = signal_norm[i:i+seq_len].reshape(1, 1, -1)
            x_tensor = torch.FloatTensor(x).to(device)
            pred = model(x_tensor).cpu().numpy().flatten()
            # 累加概率（避免边缘误差）
            probs[i:i+seq_len] += pred
            counts[i:i+seq_len] += 1

    # 概率平均（处理计数为0的情况）
    counts[counts == 0] = 1
    probs = probs / counts

    # 提取正常周期
    normal_cycles, predictions = extract_and_save_normal_cycles(
        time, signal, probs, min_period_sec, max_period_sec, dt
    )

    # 整体可视化
    plt.figure(figsize=(18, 10))
    # 子图1：原始信号+正常周期
    plt.subplot(4, 1, 1)
    plt.plot(time, signal, color='steelblue', label='去噪后信号', linewidth=1.0)
    for i, (s, e) in enumerate(normal_cycles):
        label = "正常周期" if i == 0 else ""
        plt.axvspan(time[s], time[e], color='green', alpha=0.2, label=label)
    plt.xlabel('时间 (s)')
    plt.ylabel('电流 (A)')
    plt.title('原始信号与正常周期')
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图2：CNN预测概率
    plt.subplot(4, 1, 2)
    plt.plot(time, probs, color='purple', label='CNN预测概率', linewidth=1.0)
    threshold = np.percentile(probs[probs > 1e-6], PERIOD_THRESHOLD_PERCENTILE) if len(probs[probs > 1e-6]) > 0 else 0.3
    threshold = max(0.1, min(0.7, threshold))
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'判定阈值：{threshold:.4f}')
    plt.xlabel('时间 (s)')
    plt.ylabel('周期概率')
    plt.title('CNN预测概率曲线')
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图3：二值化结果
    plt.subplot(4, 1, 3)
    plt.plot(time, predictions, color='orange', label='二值化结果（1=周期）', linewidth=1.0)
    for i, (s, e) in enumerate(normal_cycles):
        plt.text((time[s]+time[e])/2, 1.1, f'周期{i+1}', ha='center', va='bottom', color='green', fontweight='bold')
    plt.xlabel('时间 (s)')
    plt.ylabel('二值化结果')
    plt.title('周期二值化结果')
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图4：候选周期筛选过程（调试用）
    plt.subplot(4, 1, 4)
    plt.plot(time, signal, color='gray', alpha=0.5, label='原始信号（淡化）')
    # 绘制候选周期（红色虚线）
    candidate_cycles = []
    in_cycle = False
    cycle_start = 0
    min_cycle_samples = int(min_period_sec * MIN_CYCLE_RATIO / dt)
    max_cycle_samples = int(max_period_sec * MAX_CYCLE_RATIO / dt)
    for i, val in enumerate(predictions):
        if val == 1 and not in_cycle:
            cycle_start = i
            in_cycle = True
        elif val == 0 and in_cycle:
            cycle_end = i
            if min_cycle_samples <= (cycle_end - cycle_start) <= max_cycle_samples:
                candidate_cycles.append((cycle_start, cycle_end))
                plt.axvspan(time[cycle_start], time[cycle_end], color='yellow', alpha=0.2, label='候选周期' if len(candidate_cycles)==1 else "")
            in_cycle = True if (i - cycle_start) < min_cycle_samples else False  # 未达最小长度不终止
    # 绘制有效周期（绿色实线）
    for i, (s, e) in enumerate(normal_cycles):
        plt.axvspan(time[s], time[e], color='green', alpha=0.3, label='有效正常周期' if i==0 else "")
    plt.xlabel('时间 (s)')
    plt.ylabel('电流 (A)')
    plt.title('候选周期→有效周期筛选过程')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SUB_DIRS["6_visualizations"], "final_detection_result.png"), dpi=150)
    plt.close()
    logger.info(f"可视化完成：{SUB_DIRS['6_visualizations']}")

    return normal_cycles


# ==============================================================================
# 8. 主函数（简化，调用优化后的模块）
# ==============================================================================
def main(file_path):
    logger.info("="*60)
    logger.info("开始CNN正常波形自动识别流程（优化版）")
    logger.info("="*60)
    # 打印可调参数，便于调试
    logger.info("当前可调参数：")
    logger.info(f"- 分析时间范围：{START_TIME}~{END_TIME}s")
    logger.info(f"- 周期判定阈值分位数：{PERIOD_THRESHOLD_PERCENTILE}%（越小越宽松）")
    logger.info(f"- 要求周期内有峰值：{CYCLE_PEAK_REQUIRED}（False=允许平滑周期）")
    logger.info(f"- CNN序列长度：平均周期×{CNN_SEQ_LEN_RATIO}（最大{MAX_SEQ_LEN}个采样点）")

    try:
        # 1. 数据加载与预处理
        time, signal = load_and_preprocess_data(file_path)

        # 2. 周期范围估计
        min_period_sec, max_period_sec, dt = estimate_period_range(time, signal)

        # 3. 生成伪标签
        pseudo_labels = generate_precise_pseudo_labels(time, signal, min_period_sec, max_period_sec, dt)

        # 4. 训练CNN
        model, scaler, seq_len = train_multi_scale_cnn(signal, pseudo_labels, min_period_sec, max_period_sec, dt)

        # 5. 预测与存储正常周期
        normal_cycles = predict_and_visualize(model, scaler, signal, time, seq_len, min_period_sec, max_period_sec, dt)

        # 结果汇总
        logger.info("="*60)
        logger.info("流程完成！最终结果：")
        logger.info(f"1. 有效采样点：{len(time)}个")
        logger.info(f"2. 估计周期范围：{min_period_sec:.2f}~{max_period_sec:.2f}s")
        logger.info(f"3. 识别正常周期：{len(normal_cycles)}个")
        logger.info(f"4. 结果路径：{OUTPUT_DIR}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"流程失败：{str(e)}", exc_info=True)


# ==============================================================================
# 9. 执行入口（修改CSV路径）
# ==============================================================================
if __name__ == "__main__":
    
    
    if not os.path.exists(CSV_FILE_PATH):
        logger.error(f"文件不存在：{CSV_FILE_PATH}，请检查路径！")
    else:
        main(CSV_FILE_PATH)