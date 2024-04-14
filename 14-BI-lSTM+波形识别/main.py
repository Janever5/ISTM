import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# 生成模拟波形数据
def generate_waveform(cycle_length, num_cycles):
    t = np.linspace(0, 2 * np.pi * num_cycles, cycle_length * num_cycles)
    waveform = np.sin(t)
    return waveform

# 在指定位置添加异常值
def add_anomalies(waveform, start, end):
    waveform[start:end] = 1.2
    return waveform

# 创建模拟数据
waveform = generate_waveform(150, 8)
waveform = add_anomalies(waveform, len(waveform)//2, len(waveform)//2 + len(waveform)//4)

# 划分数据
train_data = waveform[:len(waveform)//2-1]
test_data = waveform[len(waveform)//2-1:]

# 填充和转换数据
train_data_padded = train_data.reshape(1, -1, 1)

# 创建Bi-LSTM模型
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(None, 1)),
    Bidirectional(LSTM(32, return_sequences=True)),
    Dense(1)
])

# 使用Adam优化器
optimizer = Adam(learning_rate=0.001)

# 编译模型
model.compile(loss='mean_squared_error', optimizer=optimizer)

# 训练模型
model.fit(train_data_padded, train_data_padded, epochs=5, verbose=1)

# 预测
predictions = model.predict(train_data_padded)

plt.figure(figsize=(12, 6))

# 绘制合并的数据
combined_data = np.concatenate((train_data, test_data))

plt.plot(combined_data, label='Combined Waveform')
plt.plot(np.arange(len(train_data), len(train_data) + len(predictions.flatten())), predictions.flatten(), label='Predictions', linestyle='--')

# 计算预测误差
errors = np.abs(predictions.flatten() - test_data[:len(predictions.flatten())])

# 使用阈值0.1来标记异常
for i, error in enumerate(errors):
    if error > 0.1:
        plt.axvspan(i + len(train_data), i + len(train_data) + 1, color='red', alpha=0.5)

plt.axvline(len(train_data), color='gray', linestyle='--', linewidth=1, label='Train/Test Split')
plt.legend()

# 保存图像
plt.savefig('waveform_plot.png')
