import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# 生成模拟波形数据
def generate_waveform(cycle_length, num_cycles):
    t = np.linspace(0, 2 * np.pi * num_cycles, cycle_length * num_cycles)
    waveform = np.sin(t)
    return waveform

# 在指定位置添加异常值
def add_anomalies(waveform, start, end):
    waveform[start:end] =  waveform[start:end] +0.3
    return waveform

# 划分数据
def split_data(waveform):
    train_data = waveform[:len(waveform)//2]
    test_data = waveform[len(waveform)//2:]
    return train_data, test_data

# 创建模拟数据
waveform = generate_waveform(150, 8)
# 在后50%的数据中添加异常值
waveform = add_anomalies(waveform, len(waveform)//2, len(waveform)//2 + len(waveform)//4)
train_data, test_data = split_data(waveform)

# 填充和转换数据
train_data_padded = train_data.reshape(1, -1, 1)
test_data_padded = test_data.reshape(1, -1, 1)

# 创建Bi-LSTM模型
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(None, 1)),
    Bidirectional(LSTM(32, return_sequences=True)),
    Dense(1)  # 输出层只有一个单元
])

# 使用RMSprop优化器
optimizer = Adam(learning_rate=0.001)

# 编译模型
model.compile(loss='mean_squared_error', optimizer=optimizer)

# 训练模型
model.fit(train_data_padded, train_data_padded, epochs=50, verbose=1)

# 预测
predictions = model.predict(test_data_padded)

plt.figure(figsize=(12, 6))

# 连接训练数据和测试数据
combined_data = np.concatenate((train_data, test_data))

plt.plot(combined_data, label='Combined Waveform')
plt.plot(np.arange(len(train_data), len(combined_data)), predictions.flatten(), label='Predictions', linestyle='--')

# 计算预测误差
errors = np.abs(predictions.flatten() - test_data)

# 使用阈值0.5来标记异常
for i, error in enumerate(errors):
    if error > 0.1:
        plt.axvspan(i + len(train_data), i + len(train_data) + 1, color='red', alpha=0.5)

plt.axvline(len(train_data), color='gray', linestyle='--', linewidth=1, label='Train/Test Split')  # 添加训练和测试数据的分界线
plt.legend()
# plt.show()
plt.savefig('waveform_plot.png')  # 保存图像而不是显示它
