import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

data_dir = 'project'
labels = {'up': 0, 'down': 1, 'right': 2, 'left': 3, 'front': 4, 'back': 5}
# labels = {'up': 0, 'down': 1, 'right': 2, 'left': 3}

all_data = []

for sub_dir, label in labels.items():
    path = os.path.join(data_dir, sub_dir)
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path, header=None)
            flattened_data = df.to_numpy().flatten()[:1116]  # Flatten the data
            # Append as a tuple (data, label)
            all_data.append((flattened_data, label))

        elif filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path, usecols=[1, 2, 3, 4, 5, 6])
            flattened_data = df.to_numpy().flatten()  # Flatten the data
            # Append as a tuple (data, label)
            all_data.append((flattened_data, label))

flattened_df = pd.DataFrame(all_data, columns=['data', 'label'])

X = np.stack(flattened_df['data'].values)  # Stack arrays into 2D array
print("X:", X.shape)
y = flattened_df['label'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# classifier = svm.SVC(kernel='linear')
# classifier.fit(X_train, y_train)
classifier = RandomForestClassifier(max_depth=30)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

with open('rfc_project.pkl', 'wb') as f:
    pickle.dump(classifier, f)


conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues")
plt.title('train')
plt.xlabel('pred')
plt.ylabel('actual')
plt.show()
