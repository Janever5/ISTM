import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Function to read waveform data from a txt file
def read_waveform_from_txt(file_path):
    data = np.genfromtxt(file_path, delimiter='\t', skip_header=0, filling_values=np.nan)
    return data[:, 1]  # Return only the second column, i.e., waveform data

# Function to create features and labels
def create_features_and_labels(waveform, window_size, normal_indices, anomaly_indices):
    features = []
    labels = []
    for i in range(len(waveform) - window_size + 1):
        window = waveform[i:i + window_size]

        if i + window_size - 1 in normal_indices:
            features.append(window)
            labels.append(0)  # 0 represents normal
        elif i + window_size - 1 in anomaly_indices:
            features.append(window)
            labels.append(1)  # 1 represents anomaly
    return np.array(features), np.array(labels)

# Function to create features and labels
def create_newdata(waveform, window_size):
    features = []
    labels = []
    for i in range(len(waveform) - window_size + 1):
        window = waveform[i:i + window_size]
        features.append(window)
    return np.array(features)

# Read waveform data
waveform = read_waveform_from_txt('data/nodding.txt')

# Define window size and known time periods of normal and anomaly data
window_size = 100
normal_indices = list(range(2000, 3000))  # Time period of 10-30s
anomaly_indices = list(range(3000, 9000))  # Time period of 40-50s

# Create features and labels
X, y = create_features_and_labels(waveform, window_size, normal_indices, anomaly_indices)

# Create and train the model (Logistic Regression)
model = LogisticRegression()
model.fit(X.reshape(-1, window_size), y)

# Predict normal and anomaly segments in the new_data
new_data = waveform[1000:8000]  # Time period of 10-60s
new_features = create_newdata(new_data, window_size)
predictions = model.predict(new_features.reshape(-1, window_size))

# Find the positions of predicted anomalies
anomalies = [i for i, pred in enumerate(predictions) if pred == 1]

print(anomalies)

# Plot the new_data waveform and mark the predicted anomalies
plt.plot(new_data)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Waveform Data with Predicted Anomalies')
for anomaly in anomalies:
    if anomaly < 2000:
       continue
    plt.axvspan(anomaly , (anomaly + 1) , color='red', alpha=0.3)
plt.show()
