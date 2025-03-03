# Cleaned Neural Network Notebook (nn.ipynb)

import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load Dataset
data_dir = "dataset/audio"
labels = []
features = []

# Extract Features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

for file in os.listdir(data_dir):
    if file.endswith(".wav"):
        labels.append(file.split("-")[2])  # Assuming label is in filename
        features.append(extract_features(os.path.join(data_dir, file)))

# Encode Labels
le = LabelEncoder()
labels = le.fit_transform(labels)
y = to_categorical(labels)
X = np.array(features)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save Model
model.save("models/emotion_model.h5")

# Save Label Encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
