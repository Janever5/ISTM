import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.signal import find_peaks

# 假设我们有以下每组的波形数据：
groups_data = [
    # 每组数据为一个NumPy数组，形如：np.array([[1, 2], [2, 3], [3, 2], [4, 3], ...])
]

# 设定一个阈值来决定异常占比
anomaly_threshold = 0.05  # 5%

# 标记数据的函数
def mark_anomaly_based_on_threshold(waveform_group, cycle_length):
    cycles = len(waveform_group) // cycle_length
    anomalies = 0
    
    for i in range(cycles):
        cycle_start = i * cycle_length
        cycle_end = (i + 1) * cycle_length
        cycle_data = waveform_group[cycle_start:cycle_end]
        
        # 假设我们有一个函数 identify_anomaly_cycle 检查单个周期是否异常
        # 这里我们接收该函数的返回值
        if identify_anomaly_cycle(cycle_data):
            anomalies += 1
    
    anomaly_ratio = anomalies / cycles
    return anomaly_ratio > anomaly_threshold
    
def identify_anomaly_cycle(cycle_data):
    # 举例，这里根据某个规则来识别异常周期，实际应用需要根据实际情况来写规则
    # 简单例子: 如果周期中的最大值与最小值的差异超过某个阈值，则标记为异常
    threshold = 5
    return max(cycle_data[:, 1]) - min(cycle_data[:, 1]) > threshold

# 定义一个周期识别函数来获取周期长度
def detect_cycle_length(waveform):
    # 使用信号处理算法来检测周期
    peaks, _ = find_peaks(waveform[:, 1])  # 假设第二列是我们关心的特征
    return np.diff(peaks).mean()  # 取平均值作为周期长度

labels = []
padded_waveforms = []
max_cycle_length = 0  # 假设所有组数据中最长的周期

for waveform_group in groups_data:
    cycle_length = int(detect_cycle_length(waveform_group))
    label = mark_anomaly_based_on_threshold(waveform_group, cycle_length)
    labels.append(1 if label else 0)  # 1为异常，0为正常
    
    max_cycle_length = max(max_cycle_length, cycle_length)  # 更新最长周期长度，用于后续的数据填充
    
    # 我们将所有组数据填充（或截断）到相同的长度
    padded_waveform = pad_sequences([waveform_group], maxlen=max_cycle_length, padding='post', truncating='post', value=0)[0]
    padded_waveforms.append(padded_waveform)

data_padded = np.array(padded_waveforms)
labels = to_categorical(labels)  # 将标签转换成one-hot编码

# 现在我们分割数据集
X_train, X_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, random_state=42)

# 创建Bi-LSTM模型
model = Sequential([
    Bidirectional(LSTM(64, input_shape=(max_cycle_length, 2))),
    Dense(2, activation='softmax')  # 用于分类的输出层
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 预测
predictions = model.predict(X_test)