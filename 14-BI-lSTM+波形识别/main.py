import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from scipy.signal import find_peaks


# 生成一些模拟波形数据
def generate_waveform_data(num_samples, cycle_length):
    groups_data = []
    for _ in range(num_samples):
        waveform = np.array([[i, np.sin(i) + random.choice([-1, 0, 1]) * random.random()] for i in np.linspace(0, 2 * np.pi, cycle_length)])
        groups_data.append(waveform)
    return groups_data

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
   peaks, _ = find_peaks(waveform[:, 1]) # 假设第二列是我们关心的特征
   return np.diff(peaks).mean() # 取平均值作为周期长度

# 修改后的周期检测函数
def detect_cycle_length(waveform):
    peaks, _ = find_peaks(waveform[:, 1])
    if len(peaks) > 1:
        return int(np.diff(peaks).mean())
    else:
        return len(waveform)  # 如果没有足够的峰值被发现，返回整个波形长度作为周期长度

# 创建模拟数据
groups_data = generate_waveform_data(num_samples=100, cycle_length=50)

# 初始化标签和填充波形数据的列表
labels = []
padded_waveforms = []
max_cycle_length = max(len(waveform_group) for waveform_group in groups_data)  # 记录最长周期长度

for waveform_group in groups_data:
    cycle_length = detect_cycle_length(waveform_group)
    label = mark_anomaly_based_on_threshold(waveform_group, cycle_length)
    labels.append(1 if label else 0)

    padded_waveform = pad_sequences([waveform_group], maxlen=max_cycle_length, padding='post', truncating='post', dtype='float', value=0.0)[0]
    padded_waveforms.append(padded_waveform)

# 填充后的数据转换为NumPy数组
data_padded = np.array(padded_waveforms)

# 将标签转换成one-hot编码
labels = to_categorical(labels,num_classes=2)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, random_state=42)

# 创建Bi-LSTM模型
model = Sequential([
    Bidirectional(LSTM(64, input_shape=(max_cycle_length, data_padded.shape[2]))),
    Dense(2, activation='softmax')  # 用于分类的输出层
])

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 预测
predictions = model.predict(X_test)


predicted_class=np.argmax(predictions, axis=-1)
loss,accuracy = model.evaluate(X_test,y_test,verbose=False)
print(f"Test Accuacy:{accuracy*100:.2f}%")