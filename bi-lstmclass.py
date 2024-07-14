import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from normalclass import assign_category, read_waveform_from_txt, create_newdata, create_features_and_labels

from tensorflow.keras.optimizers import Adam

def build_bilstm_model(window_size, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(8, return_sequences=True), input_shape=(window_size, 1)))
    model.add(Dropout(0.1)) # 试试0.1，0.2
    model.add(Bidirectional(LSTM(8))) # 修改层数
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def process_files_and_train(folder_path, window_size, normal_indices, anomaly_indices):
    all_features = []
    all_labels = []
    category_mapping = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            category = assign_category(file_name)
            if category == 0: 
                continue
            if category not in category_mapping:
                category_mapping[category] = len(category_mapping)+1
            file_path = os.path.join(folder_path, file_name)
            waveform = read_waveform_from_txt(file_path)
            X, y = create_features_and_labels(waveform, window_size, normal_indices, anomaly_indices)
            all_features.append(X)
            all_labels.append(y  * category)

    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

    num_classes = max(all_labels) + 1

    model = build_bilstm_model(window_size, num_classes)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test, num_classes=num_classes)

    # 实例化Adam优化器并设置学习率
    optimizer_bz = Adam(learning_rate=0.0001)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer_bz, 
                  metrics=['accuracy'])

    history = model.fit(X_train, 
                        y_train_onehot, 
                        epochs=50,
                        batch_size=128, 
                        validation_data=(X_test, y_test_onehot))
    # batch size: [16,32,64,128,256]

    plot_training_history(history,'training_history')
    save_training_history(history, 'training_history.npz')
    # 保存为文本文件
    with open('training_history.txt', 'w') as f:
        f.write('epoch,accuracy,val_accuracy,loss,val_loss\n')
        for epoch in range(len(history.history['accuracy'])):
            f.write(f"{epoch + 1},{history.history['accuracy'][epoch]},{history.history['val_accuracy'][epoch]},"
                    f"{history.history['loss'][epoch]},{history.history['val_loss'][epoch]}\n")

    return model, category_mapping, X_train, y_train, X_test, y_test

def save_training_history(history, file_path):
    np.savez(file_path, 
             accuracy=history.history['accuracy'], 
             val_accuracy=history.history['val_accuracy'],
             loss=history.history['loss'],
             val_loss=history.history['val_loss'])

def plot_training_history(history, save_path=None):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if save_path:
        plt.savefig(save_path + '_accuracy.png')  # Save accuracy plot
    plt.close()

    # Plot training & validation loss values
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if save_path:
        plt.savefig(save_path + '_loss.png')  # Save loss plot
    plt.close()


def predict_category_and_name_bilstm(model, file_path, window_size):
    waveform = read_waveform_from_txt(file_path)
    new_features = create_newdata(waveform, window_size)

    # Assuming new_features shape is (201,)
    new_features = new_features.reshape((1, new_features.shape[0], 1))
    predictions = model.predict(new_features)
    predicted_category_id = np.argmax(predictions, axis=1)[0]
    return predicted_category_id


def plot_confusion_matrix(y_true, y_pred, category_mapping, title='Confusion Matrix', save_path=None):
    category_names = {v: k for k, v in category_mapping.items()}
    cm = confusion_matrix(y_true, y_pred, labels=list(category_names.keys()))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=category_names.values(), yticklabels=category_names.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    if save_path:
        plt.savefig(save_path + '.png')
    plt.close()

def plot_confusion_matrix_bz(y_true, y_pred, category_mapping, title='Confusion Matrix', save_path=None):
    category_names = {v: k for k, v in category_mapping.items()}
    cm = confusion_matrix(y_true, y_pred, labels=list(category_names.keys()))
    
    # 使用"Purples"颜色映射，从白色到深紫色
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=category_names.values(), yticklabels=category_names.values())
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path + '.png')
    plt.close()

# 示例调用函数（假设你有y_true, y_pred, 和 category_mapping）
# plot_confusion_matrix(y_true, y_pred, category_mapping, save_path='./confusion_matrix')


if __name__ == '__main__':

    folder_path = './Dataset_Folders/Training_Set'
    window_size = 800
    normal_indices = list(range(0, 3000))
    anomaly_indices = list(range(3000, 4000))    
    model, category_mapping, X_train, y_train, X_test, y_test = process_files_and_train(folder_path, window_size, normal_indices, anomaly_indices)    
    y_train_pred = model.predict(X_train)
    y_train_pred_classes = np.argmax(y_train_pred, axis=1)    
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)    
    plot_confusion_matrix_bz(y_train, y_train_pred_classes, category_mapping, title='Training Confusion Matrix',save_path='confusion_matrix_train')
    plot_confusion_matrix_bz(y_test, y_test_pred_classes, category_mapping, title='Test Confusion Matrix',save_path='confusion_matrix_test')    
    test_accuracy = accuracy_score(y_test, y_test_pred_classes)
    print(f"Test accuracy: {test_accuracy}")    
    # test_file = './Dataset_Folders/Test_Set/filtered_swallow_noisy_9.txt'
    # predicted_category_id = predict_category_and_name_bilstm(model, test_file, window_size)
    # predicted_category_name = [k for k, v in category_mapping.items() if v == predicted_category_id][0]
    # print(f"预测结果: 测试文件属于 '{predicted_category_name}' 类别.")
