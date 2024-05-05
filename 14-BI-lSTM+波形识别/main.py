import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# 读取txt文件并生成波形数据
def read_waveform_from_txt(file_path):
    data = np.genfromtxt(file_path, delimiter='\t', skip_header=0, filling_values=np.nan)
    return data[:, 1]  # 只返回第二列的数据，即波形数据

# 创建滑动窗口数据
def create_window_data(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)

# 读取txt文件作为模拟波形数据
waveform = read_waveform_from_txt('data/elbow.txt')  # 替换成你的txt文件路径

# 划分训练集和测试集
# 划分训练集和测试集
train_data = waveform[100*5:100*35]  # 前30个数据作为训练集*
test_data = waveform[100*35:100*65]  # 后30个数据作为测试集

# 填充和转换数据
window_size = 50  # 窗口大小
train_windows = create_window_data(train_data, window_size)
test_windows = create_window_data(test_data, window_size)

train_windows_padded = train_windows.reshape(-1, window_size, 1)
test_windows_padded = test_windows.reshape(-1, window_size, 1)

print("Train windows shape:", train_windows_padded.shape)
print("Test windows shape:", test_windows_padded.shape)

# 修改目标数据
train_targets = np.roll(train_windows_padded, -1, axis=1)
train_targets[:, -1, :] = train_windows_padded[:, -1, :]  # 最后一个时间步的数据保持不变


# 创建Bi-LSTM模型
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(window_size, 1)),
    Bidirectional(LSTM(32, return_sequences=True)),
    Dense(1)
])
# 使用Adam优化器
optimizer = Adam(learning_rate=0.001)
# 编译模型
model.compile(loss='mean_squared_error', optimizer=optimizer)

# 训练模型
model.fit(train_windows_padded, train_targets, epochs=5, verbose=1)
# 预测
predictions = model.predict(train_windows_padded)

# 获取预测数据的最后一个时间步的值
predicted_values = predictions[:, -1]
predicted_values = predicted_values.flatten()

# 计算预测结果的长度
predicted_length = len(train_data) + len(test_data) - window_size + 1

plt.figure(figsize=(12, 6))

# 连接训练数据和测试数据
combined_data = np.concatenate((train_data, test_data))

# 绘制原始数据
plt.plot(combined_data[:predicted_length], label='Combined Waveform')

# 绘制预测数据
plt.plot(np.arange(len(train_data), predicted_length), predicted_values, label='Predictions', linestyle='--')

# 计算预测误差
errors = np.abs(predictions[:, -1].flatten() - test_data[window_size - 1:])

# 使用阈值0.1来标记异常
for i, error in enumerate(errors):
    if error > 0.1:
        plt.axvspan(i + len(train_data), i + len(train_data) + 1, color='red', alpha=0.5)

plt.axvline(len(train_data), color='gray', linestyle='--', linewidth=1, label='Train/Test Split')
plt.legend()
plt.savefig('waveform_plot2.png')  # 保存图像而不是显示它
