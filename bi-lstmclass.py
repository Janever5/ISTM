import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, TimeDistributed, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from normalclass import assign_category,read_waveform_from_txt,create_newdata,create_features_and_labels

# 假设您已经有了以下函数：
# read_waveform_from_txt(file_path) - 读取波形数据
# create_features_and_labels(waveform, window_size, normal_indices, anomaly_indices) - 创建特征和标签
# assign_category(file_name) - 根据文件名分配类别

def build_bilstm_model(window_size, num_classes):
    model = Sequential()
    # 使用较少的单元数的Bi-LSTM层
    model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(window_size, 1)))
    model.add(Dropout(0.5))
    # 再次使用较少单元数的Bi-LSTM层
    model.add(Bidirectional(LSTM(10)))
    model.add(Dropout(0.5))
    # 使用较少单元数的全连接层
    model.add(Dense(num_classes, activation='softmax'))
    return model

def process_files_and_train(folder_path, window_size, normal_indices, anomaly_indices):
    all_features = []
    all_labels = []
    category_mapping = {}  # 存储类别与索引的映射

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            category = assign_category(file_name)
            if category not in category_mapping:
                category_mapping[category] = len(category_mapping)
            file_path = os.path.join(folder_path, file_name)
            waveform = read_waveform_from_txt(file_path)
            X, y = create_features_and_labels(waveform, window_size, normal_indices, anomaly_indices)
            
            all_features.append(X)
            all_labels.append(y + category_mapping[category])

    # 将所有特征和标签合并
    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

    # 动态确定类别数量
    num_classes = max(all_labels) + 1  # +1 是因为索引是从0开始的

    # 构建模型
    model = build_bilstm_model(window_size, num_classes)

    # 将标签转换为one-hot编码
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test, num_classes=num_classes)

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train_onehot, epochs=5, batch_size=32, validation_data=(X_test, y_test_onehot))

    return model, category_mapping

# 预测函数
def predict_category_and_name_bilstm(model, file_path, window_size):
    waveform = read_waveform_from_txt(file_path)
    new_features = create_newdata(waveform, window_size)
    # 重塑数据以匹配模型输入
    new_features = new_features.reshape((1, new_features.shape[0], 1))
    predictions = model.predict(new_features)
    predicted_category_id = np.argmax(predictions, axis=1)[0]
    return predicted_category_id

# 示例使用
folder_path = './Dataset_Folders/Training_Set'
window_size = 800
normal_indices = list(range(0, 3000))
anomaly_indices = list(range(3000, 4000))

# 处理文件，训练模型
model, category_mapping = process_files_and_train(folder_path, window_size, normal_indices, anomaly_indices)
# 预测测试文件的类别
test_file = './Dataset_Folders/Test_Set/filtered_swallow_noisy_9.txt'  # 测试文件路径
predicted_category_id = predict_category_and_name_bilstm(model, test_file, window_size)
predicted_category_name = [k for k, v in category_mapping.items() if v == predicted_category_id][0]
print(f"预测结果: 测试文件属于 '{predicted_category_name}' 类别.")