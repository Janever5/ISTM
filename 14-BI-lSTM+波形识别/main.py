import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
# 生成模拟波形数据
def generate_waveform(cycle_length, num_cycles):
    t = np.linspace(0, 2 * np.pi * num_-cycles, cycle_length * num_cycles)
    waveform = np.sin(t)
    return waveform

# 在指定位置添加异常值
def add_anomalies(waveform, start, end):
    waveform[start:end] = waveform[start:end] + 0.3
    return waveform

# 划分数据
def split_data(waveform):
    train_data = waveform[:len(waveform)//2]
    test_data = waveform[len(waveform)//2:]
    return train_data, test_data

# 创建滑动窗口数据
def create_window_data(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)

# 创建模拟数据
waveform = generate_waveform(150, 8)
waveform = add_anomalies(waveform, len(waveform)//2, len(waveform)//2 + len(waveform)//4)
train_data, test_data = split_data(waveform)

# 填充和转换数据
train_windows = create_window_data(train_data, 50)
test_windows = create_window_data(test_data, 50)

train_windows_padded = train_windows.reshape(-1, 50, 1)
test_windows_padded = test_windows.reshape(-1, 50, 1)

# 创建Bi-LSTM模型
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(50, 1)),
    Bidirectional(LSTM(32, return_sequences=True)),
    Dense(1)
])

# 使用Adam优化器
optimizer = Adam(learning_rate=0.001)

# 编译模型
model.compile(loss='mean_squared_error', optimizer=optimizer)

# 训练模型
model.fit(train_windows_padded, train_windows_padded, epochs=50, verbose=1)

# 预测
predictions = model.predict(test_windows_padded)

plt.figure(figsize=(12, 6))

# 连接训练数据和测试数据
combined_data = np.concatenate((train_data, test_data))

plt.plot(combined_data, label='Combined Waveform')
plt.plot(np.arange(len(train_data) + 49, len(combined_data)), predictions[:, -1], label='Predictions', linestyle='--')

# 计算预测误差
errors = np.abs(predictions[:, -1].flatten() - test_data[49:])

# 使用阈值0.1来标记异常
for i, error in enumerate(errors):
    if error > 0.1:
        plt.axvspan(i + len(train_data), i + len(train_data) + 1, color='red', alpha=0.5)

plt.axvline(len(train_data), color='gray', linestyle='--', linewidth=1, label='Train/Test Split')
plt.legend()
# plt.show()
plt.savefig('waveform_plot2.png')  # 保存图像而不是显示它
