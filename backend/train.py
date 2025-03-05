import os
import numpy as np
import librosa
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import glob

# Path to the dataset
base_path = r"C:\Users\gaura\.cache\kagglehub\datasets\uwrfkaggler\ravdess-emotional-speech-audio\versions\1"

# Emotion mapping (assuming filenames contain emotion codes like "01", "02", etc.)
emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

# Feature extraction functions
def feature_chromagram(waveform, sample_rate):
    stft_spectrogram=np.abs(librosa.stft(waveform))
    chromagram=np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T, axis=0)
    return chromagram

def feature_melspectrogram(waveform, sample_rate):
    melspectrogram=np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000).T, axis=0)
    return melspectrogram

def feature_mfcc(waveform, sample_rate):
    mfc_coefficients=np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40).T, axis=0) 
    return mfc_coefficients

def get_features(file):
    with soundfile.SoundFile(file) as audio:
        waveform = audio.read(dtype="float32")
        sample_rate = audio.samplerate
        chromagram = feature_chromagram(waveform, sample_rate)
        melspectrogram = feature_melspectrogram(waveform, sample_rate)
        mfc_coefficients = feature_mfcc(waveform, sample_rate)

        feature_matrix = np.hstack((chromagram, melspectrogram, mfc_coefficients))
        return feature_matrix

# Emotion mapping for RAVDESS dataset
emotions ={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

def load_data():
    X, y = [], []
    count = 0
    for file in glob.glob(f"{base_path}\\actor_*\\*.wav"):
        if count > 1430:
            break
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        try:
            features = get_features(file)
        except:
            continue
        X.append(features)
        y.append(emotion)
        count += 1
    return np.array(X), np.array(y)

features, emotions = load_data()

# Encode Labels
le = LabelEncoder()
emotions = le.fit_transform(emotions)
X = features

# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, emotions, test_size=0.2, random_state=42)

# Build MLP Classifier
model = MLPClassifier(
    activation='logistic', 
    solver='adam', 
    alpha=0.001, 
    hidden_layer_sizes=(300,), 
    max_iter=1000, 
    random_state=69
)

# Train Model
model.fit(X_train, y_train)

# Evaluate Model
print(f'MLP Model\'s accuracy on training set is {100*model.score(X_train, y_train):.2f}%')
print(f'MLP Model\'s accuracy on test set is {100*model.score(X_test, y_test):.2f}%')

# Save the model, scaler, and label encoder
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)
