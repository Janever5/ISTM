import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Function to create features and labels
def create_newdata(waveform, window_size):
    features = []
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
            labels.append(1)  # 1 represents normal
        elif i + window_size - 1 in anomaly_indices:
            features.append(window)
            labels.append(-1)  # -1 represents anomaly
    return np.array(features), np.array(labels)

def assign_category(file_name):
    if 'finger' in file_name:
        return 1  # finger category
    elif 'elbow' in file_name:
        return 2  # elbow category
    elif 'knee' in file_name:
        return 3  # knee category
    elif 'swallow' in file_name:
        return 4  # swallow category
    elif 'heart' in file_name:
        return 5 # heartbeat category
    else:
        return 0  # other category

def process_files_and_train(folder_path, window_size, normal_indices, anomaly_indices):
    all_features = []
    all_labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            category = assign_category(file_name)
            if category == 0:  # Skip other unknown categories
                continue
            file_path = os.path.join(folder_path, file_name)
            waveform = read_waveform_from_txt(file_path)
            X, y = create_features_and_labels(waveform, window_size, normal_indices, anomaly_indices)
            if isinstance(y, np.ndarray):
                all_labels.extend(y * category)
            else:
                raise ValueError("Expected y to be a numpy array but got", type(y))
            all_features.extend(X)
    
    X_all = np.array(all_features)
    y_all = np.array(all_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=20, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def predict_category_and_name(model, file_path, window_size):
    waveform = read_waveform_from_txt(file_path)
    new_features = create_newdata(waveform, window_size)
    predictions = model.predict(new_features.reshape(-1, window_size))
    predicted_category_id = predictions[0]
    return predicted_category_id

# Generate and plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, category_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=list(category_names.keys()))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=category_names.values(), yticklabels=category_names.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Example usage
folder_path = './Dataset_Folders/Training_Set'
window_size = 1000
normal_indices = list(range(0, 3000))
anomaly_indices = list(range(3000, 4000))

model, X_train, X_test, y_train, y_test = process_files_and_train(folder_path, window_size, normal_indices, anomaly_indices)

# Predict on training data to generate confusion matrix
y_train_pred = model.predict(X_train)
category_names = {1: 'finger', 2: 'elbow', 3: 'knee', 4: 'swallow', 5: 'heart'}

# Plot confusion matrix for training data
# plot_confusion_matrix(y_train, y_train_pred, category_names, title='Training Confusion Matrix')

# Predict on test data to generate confusion matrix
y_test_pred = model.predict(X_test)

# Plot confusion matrix for test data
plot_confusion_matrix(y_test, y_test_pred, category_names, title='Test Confusion Matrix')

# Test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {test_accuracy}")

# Predict category of a test file
test_file = './Dataset_Folders/Test_Set/filtered_swallow_noisy_9.txt'
predicted_category_id = predict_category_and_name(model, test_file, window_size)
predicted_category = category_names.get(predicted_category_id, 'unknown')
print(f"Predicted category: {predicted_category}")
