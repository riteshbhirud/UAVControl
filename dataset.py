import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.signal import butter, lfilter

data_dir = 'project'
labels = {'up': 0, 'down': 1, 'right': 2, 'left': 3, 'front': 4, 'back': 5}
feature_columns = ['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z']


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def normalize_test_data(test_data):
    # return (test_data - train_min) / (train_max - train_min)
    # Minimum value for each column (feature)
    min_values = np.min(test_data, axis=0)
    # Maximum value for each column (feature)
    max_values = np.max(test_data, axis=0)
    normalized_data = (test_data - min_values) / (max_values - min_values)
    return normalized_data


def apply_lowpass_filter(data, cutoff=3, fs=50):
    # Create an array to store filtered results
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):      # Loop over each column (feature)
        filtered_data[:, i] = butter_lowpass_filter(
            data[:, i], cutoff=cutoff, fs=fs)
    return filtered_data


def normalize_min_max(data, feature_colums):
    normalized_data = data.copy()
    for col in feature_columns:
        min_val = data[col].min()
        max_val = data[col].max()
        normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    return normalized_data


all_data = []

for sub_dir, label in labels.items():
    path = os.path.join(data_dir, sub_dir)
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path, header=None)
            flattened_data = apply_lowpass_filter(normalize_test_data(
                df.to_numpy())).flatten()[:1116]  # Flatten the data
            # Append as a tuple (data, label)
            all_data.append((flattened_data, label))

        elif filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path, usecols=[1, 2, 3, 4, 5, 6])
            flattened_data = apply_lowpass_filter(normalize_test_data(
                df.to_numpy())).flatten()  # Flatten the data
            # Append as a tuple (data, label)
            all_data.append((flattened_data, label))

flattened_df = pd.DataFrame(all_data, columns=['data', 'label'])

X = np.stack(flattened_df['data'].values)  # Stack arrays into 2D array
print("X:", X.shape)
y = flattened_df['label'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

model_name_pairs = [
    ("SVM", svm.SVC(kernel='linear')),
    ("Random Forest", RandomForestClassifier(max_depth=30)),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5))
]

for name, model in model_name_pairs:
    print("===" + name + "===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.2f}")
    print()

    conf_matrix = confusion_matrix(y_val, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues")
    plt.title('Confusion Matrix for ' + name)
    plt.xlabel('pred')
    plt.ylabel('actual')
    plt.show()

with open('svm_project.pkl', 'wb') as f:
    svm_model = model_name_pairs[0][1]
    pickle.dump(svm_model, f)
