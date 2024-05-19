import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Function to create features and labels
def create_newdata(waveform, window_size):
    features = []
    labels = []
    for i in range(len(waveform) - window_size + 1):
        window = waveform[i:i + window_size]
        features.append(window)
    return np.array(features)
# Function to read waveform data from a txt file
def read_waveform_from_txt(file_path):
    data = np.genfromtxt(file_path, delimiter='\t', skip_header=0, filling_values=np.nan)
    return data[:, 1]  # Return only the second column, i.e., waveform data

def create_features_and_labels(waveform, window_size, normal_indices, anomaly_indices):
    features = []
    labels = []
    for i in range(len(waveform) - window_size + 1):
        window = waveform[i:i + window_size]

        if i + window_size - 1 in normal_indices:
            features.append(window)
            labels.append(1)  # 0 represents normal
        elif i + window_size - 1 in anomaly_indices:
            features.append(window)
            labels.append(-1)  # 1 represents anomaly
    return np.array(features), np.array(labels)  # 确保返回的是ndarray类型

# 新增函数：根据文件名自动分配类别
def assign_category(file_name):
    if 'finger' in file_name:
        return 1  # finger类别
    elif 'elbow' in file_name:
        return 2  # elbow类别
    elif 'knee' in file_name:
        return 3  # knee类别
    elif 'swallow' in file_name:
        return 4  # knee类别
    else:
        return 0 # 其他类别

def process_files_and_train(folder_path, window_size, normal_indices, anomaly_indices):
    all_features = []
    all_labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            category = assign_category(file_name)
            if category == 0:  # 跳过其他未知类别
                continue
            file_path = os.path.join(folder_path, file_name)
            waveform = read_waveform_from_txt(file_path)
            X, y = create_features_and_labels(waveform, window_size, normal_indices, anomaly_indices)
            
            # 明确确保y即使只有一个元素，也是一个np.array
            if isinstance(y, np.ndarray):
                all_labels.extend(y * category)  # 如果y已经是np.array，则直接乘以category扩展
            else:
                raise ValueError("Expected y to be a numpy array but got", type(y))  # 确保调试时发现问题

            all_features.extend(X)
    
    # 确保all_labels中的每个元素都是至少一维的numpy数组
    all_labels = [np.array([label]) if not isinstance(label, np.ndarray) else label for label in all_labels]
    X_all = np.concatenate(all_features)
    y_all = np.concatenate(all_labels)  # 应该能成功拼接了，因为确保了all_labels中的元素至少为一维
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_all.reshape(-1, window_size), y_all, test_size=0.2, random_state=42)
    
    # 训练模型
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')  # 或者使用'saga'，根据Scikit-learn版本和数据量选择
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def predict_category_and_name(model, file_path, window_size):
    waveform = read_waveform_from_txt(file_path)
    new_features = create_newdata(waveform, window_size)
    predictions = model.predict(new_features.reshape(-1, window_size))
    # 直接取第一个预测值作为预测类别ID，因为每个文件被视为一个实例
    predicted_category_id = predictions[0]
    return predicted_category_id

# 示例使用
folder_path = './Dataset_Folders/Training_Set'  # 指定数据文件夹路径
window_size = 100
normal_indices = list(range(0, 3000))
anomaly_indices = list(range(3000, 4000))

model, X_test, y_test = process_files_and_train(folder_path, window_size, normal_indices, anomaly_indices)
test_accuracy = model.score(X_test.reshape(-1, window_size), y_test)
print(f"测试集准确率: {test_accuracy}")

# 预测某个测试文件属于哪一类
test_file = './Dataset_Folders/Test_Set/filtered_swallow_noisy_9.txt'  # 测试文件路径
predicted_category_id = predict_category_and_name(model, test_file, window_size)
category_names = {1: 'finger', 2: 'elbow', 3: 'knee',4:'swallow',0: 'other'}
predicted_category = category_names.get(predicted_category_id, 'unknown')
print(f"预测结果: 测试文件属于 '{predicted_category}' 类别.")