import numpy as np
import pandas as pd
import os
import shutil  # 用于清空目录
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# -------------------------- 全局配置与日志 --------------------------
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

# 清空并重建输出目录
OUTPUT_DIR = "cnn_waveform_results"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"🧹 Cleared existing output directory: {OUTPUT_DIR}")

SUB_DIRS = {
    "raw": os.path.join(OUTPUT_DIR, "0_raw_data"),
    "windows": os.path.join(OUTPUT_DIR, "1_sliding_windows"),
    "model": os.path.join(OUTPUT_DIR, "2_model"),
    "results": os.path.join(OUTPUT_DIR, "3_detection_results"),
    "normal": os.path.join(OUTPUT_DIR, "4_normal_segments"),
    "abnormal": os.path.join(OUTPUT_DIR, "5_abnormal_segments"),
    "plots": os.path.join(OUTPUT_DIR, "6_visualizations")
}
for dir_path in SUB_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cnn_waveform_detection.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 打印输出目录
logger.info("="*60)
logger.info("📁 Output directory structure:")
for name, path in SUB_DIRS.items():
    logger.info(f"  {name:15s} -> {os.path.abspath(path)}")
logger.info("="*60)


# -------------------------- 数据加载（略，同前）--------------------------
def load_waveform_data(file_path, start_time=None, end_time=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        header_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Time(S),Current(A)"):
                header_index = i
                break
        if header_index is None:
            for i, line in enumerate(lines):
                if "Time(S)" in line and "Current(A)" in line:
                    header_index = i
                    break
        if header_index is None:
            logger.error("Header 'Time(S),Current(A)' not found.")
            return None, None
        
        df = pd.read_csv(file_path, skiprows=header_index, encoding='utf-8')
        if "Time(S)" not in df.columns or "Current(A)" not in df.columns:
            logger.error(f"Columns missing. Got: {list(df.columns)}")
            return None, None
        
        time = pd.to_numeric(df["Time(S)"], errors='coerce').values.astype(np.float32)
        signal = pd.to_numeric(df["Current(A)"], errors='coerce').values.astype(np.float32)
        valid_mask = ~np.isnan(time) & ~np.isnan(signal)
        time, signal = time[valid_mask], signal[valid_mask]
        if len(time) == 0:
            return None, None
        
        if start_time is not None and end_time is not None:
            mask = (time >= start_time) & (time <= end_time)
            time, signal = time[mask], signal[mask]
        
        pd.DataFrame({'Time(S)': time, 'Current(A)': signal}).to_csv(
            os.path.join(SUB_DIRS["raw"], "original_data.csv"), index=False
        )
        
        plt.figure(figsize=(15, 4))
        plt.plot(time, signal, color='steelblue', linewidth=0.9)
        plt.title("原始波形")
        plt.xlabel("时间 (s)"); plt.ylabel("电流 (A)"); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SUB_DIRS["plots"], "01_original_waveform.png"), dpi=150)
        plt.close()
        
        logger.info(f"✅ Loaded {len(time)} points, range [{time.min():.2f}, {time.max():.2f}]s")
        return time, signal
    except Exception as e:
        logger.error(f"❌ Load failed: {e}")
        return None, None


# -------------------------- 滑动窗口（略）--------------------------
def generate_sliding_windows(time, signal, window_size=50, step_size=25):
    if len(signal) < window_size:
        window_size = max(10, len(signal) // 2)
    scaler = StandardScaler()
    signal_norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    
    windows, window_times, window_indices = [], [], []
    n_windows = (len(signal) - window_size) // step_size + 1
    for i in range(n_windows):
        start, end = i * step_size, i * step_size + window_size
        windows.append(signal_norm[start:end])
        window_times.append((time[start], time[end-1]))
        window_indices.append((start, end))
    
    windows = np.array(windows, dtype=np.float32)
    pd.DataFrame(windows).to_csv(os.path.join(SUB_DIRS["windows"], "normalized_windows.csv"), index=False)
    meta_df = pd.DataFrame({
        "window_id": range(len(windows)),
        "start_time": [t[0] for t in window_times],
        "end_time": [t[1] for t in window_times],
        "start_index": [idx[0] for idx in window_indices],
        "end_index": [idx[1] for idx in window_indices]
    })
    meta_df.to_csv(os.path.join(SUB_DIRS["windows"], "window_metadata.csv"), index=False)
    
    logger.info(f"✅ Generated {len(windows)} windows (size={window_size}, step={step_size})")
    return windows, window_times, window_indices, window_size, scaler


# -------------------------- 数据集类 --------------------------
class WaveformDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx].astype(np.float32)).unsqueeze(0)


# -------------------------- 改进自编码器（略）--------------------------
class WaveformAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.LeakyReLU(0.2), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(input_size // 4)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(128, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(64, 32, 5, padding=2), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, 7, padding=3), nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if x.size(-1) != self.input_size:
            x = nn.functional.interpolate(x, size=self.input_size, mode='linear', align_corners=False)
        return x


# -------------------------- 训练（略）--------------------------
def train_autoencoder(windows, window_size, epochs=100, batch_size=16):
    dataset = WaveformDataset(windows)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveformAutoencoder(window_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()
    
    logger.info(f"🚀 Training on {device} for {epochs} epochs")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss/len(dataloader):.6f}")
        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(SUB_DIRS["model"], "autoencoder.pth"))
    logger.info("✅ Model saved")
    return model


# -------------------------- 异常检测（略）--------------------------
def detect_anomalies(model, windows, window_times, window_indices, time, signal, threshold_percentile=90):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    dataset = WaveformDataset(windows)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon = model(batch)
            error = torch.mean((batch - recon)**2, dim=(1,2)).cpu().numpy()
            errors.extend(error)
    
    errors = np.array(errors)
    threshold = np.percentile(errors, threshold_percentile)
    predictions = (errors > threshold).astype(int)
    
    pd.DataFrame({
        "window_id": range(len(windows)),
        "start_time": [t[0] for t in window_times],
        "end_time": [t[1] for t in window_times],
        "reconstruction_error": errors,
        "is_abnormal": predictions
    }).to_csv(os.path.join(SUB_DIRS["results"], "detection_results.csv"), index=False)
    
    logger.info(f"✅ Detection done | Threshold: {threshold:.6f} | Normal: {np.sum(predictions==0)}, Abnormal: {np.sum(predictions==1)}")
    visualize_detection(time, signal, window_times, predictions, errors, threshold)
    extract_segments_with_plots(time, signal, window_indices, predictions, errors)  # ← 关键修改
    return predictions, errors


# -------------------------- 可视化（略）--------------------------
def visualize_detection(time, signal, window_times, predictions, errors, threshold):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax1.plot(time, signal, color='steelblue', linewidth=0.9)
    for i, (s, e) in enumerate(window_times):
        ax1.axvspan(s, e, color='red' if predictions[i] else 'green', alpha=0.2)
    ax1.set_ylabel('Current (A)'); ax1.grid(alpha=0.3)
    ax1.set_title('Detected Anomalies (Red=Abnormal)')
    
    centers = [(s+e)/2 for s,e in window_times]
    ax2.plot(centers, errors, 'o-', color='purple', markersize=4)
    ax2.axhline(threshold, color='red', linestyle='--')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('MSE'); ax2.grid(alpha=0.3)
    ax2.set_title('Reconstruction Error')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SUB_DIRS["plots"], "02_detection_results.png"), dpi=150)
    plt.close()


# -------------------------- ✨ 新增：每个片段单独绘图 --------------------------
def extract_segments_with_plots(time, signal, window_indices, predictions, errors):
    normal_dir = SUB_DIRS["normal"]
    abnormal_dir = SUB_DIRS["abnormal"]
    
    normal_count = abnormal_count = 0
    
    for i, (start_idx, end_idx) in enumerate(window_indices):
        seg_t = time[start_idx:end_idx]
        seg_s = signal[start_idx:end_idx]
        err = errors[i]
        
        # 保存数据
        df = pd.DataFrame({'Time(S)': seg_t, 'Current(A)': seg_s})
        if predictions[i] == 0:  # Normal
            base_name = f"normal_{normal_count:03d}_err{err:.4f}"
            df.to_csv(os.path.join(normal_dir, base_name + ".csv"), index=False)
            normal_count += 1
        else:  # Abnormal
            base_name = f"abnormal_{abnormal_count:03d}_err{err:.4f}"
            df.to_csv(os.path.join(abnormal_dir, base_name + ".csv"), index=False)
            abnormal_count += 1
        
        # ✅ 保存独立图像
        plt.figure(figsize=(8, 3))
        plt.plot(seg_t, seg_s, color='steelblue')
        plt.title(f"{'Normal' if predictions[i]==0 else 'Abnormal'} Segment - Error: {err:.4f}")
        plt.xlabel("Time (s)"); plt.ylabel("Current (A)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(normal_dir if predictions[i]==0 else abnormal_dir, base_name + ".png"), dpi=120)
        plt.close()
    
    logger.info(f"✅ Saved {normal_count} normal and {abnormal_count} abnormal segments (with plots)")


# -------------------------- 主函数 --------------------------
def main(file_path, window_size=50, step_size=25, epochs=100,
         start_time=0.1, end_time=180.0, threshold_percentile=90):
    logger.info("🎯 Starting waveform anomaly detection...")
    time, signal = load_waveform_data(file_path, start_time, end_time)
    if time is None: return
    
    windows, window_times, window_indices, ws, _ = generate_sliding_windows(
        time, signal, window_size, step_size
    )
    
    model = train_autoencoder(windows, ws, epochs, batch_size=16)
    detect_anomalies(model, windows, window_times, window_indices, time, signal, threshold_percentile)
    
    logger.info("🎉 Done! Check output folders for results.")


# -------------------------- 执行 --------------------------
if __name__ == "__main__":
    FILE_PATH = r"16-波形检测与分类\knee-sensor\内翻-0-1.csv"
    main(
        file_path=FILE_PATH,
        window_size=50,
        step_size=25,
        epochs=100,
        start_time=80,
        end_time=100,
        threshold_percentile=90
    )