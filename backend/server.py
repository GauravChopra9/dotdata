from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import pickle
import librosa
import shutil
import boto3
import os

app = FastAPI()

# Load model, scaler, and label encoder
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
LE_PATH = "models/label_encoder.pkl"

# Load the model, scaler, and label encoder
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(LE_PATH, 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Cloudflare R2 Configuration
CLOUDFLARE_R2_BUCKET = "dotdata"
CLOUDFLARE_ACCESS_KEY = "c92b5aec55a682f18a67da06680d789f"
CLOUDFLARE_SECRET_KEY = "1ca84ae58abd29a0a8f4e54d7e4b4f0f3f661e2a1c2643ecb1847a48ff95cb71"
CLOUDFLARE_ENDPOINT = "https://4b2961a49618759564f8badb5af3ed30.r2.cloudflarestorage.com"

s3 = boto3.client("s3", endpoint_url=CLOUDFLARE_ENDPOINT, 
                  aws_access_key_id=CLOUDFLARE_ACCESS_KEY, 
                  aws_secret_access_key=CLOUDFLARE_SECRET_KEY)

# Feature extraction functions (from training script)
def feature_chromagram(waveform, sample_rate):
    stft_spectrogram = np.abs(librosa.stft(waveform))
    chromagram = np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T, axis=0)
    return chromagram

def feature_melspectrogram(waveform, sample_rate):
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000).T, axis=0)
    return melspectrogram

def feature_mfcc(waveform, sample_rate):
    mfc_coefficients = np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40).T, axis=0) 
    return mfc_coefficients

# Preprocess function
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Extract chromagram, melspectrogram, and MFCC features
    chromagram = feature_chromagram(y, sr)
    melspectrogram = feature_melspectrogram(y, sr)
    mfccs = feature_mfcc(y, sr)
    
    # Combine features into a single feature vector
    features = np.hstack((chromagram, melspectrogram, mfccs))
    
    return np.expand_dims(features, axis=0)

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    file_location = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract features and preprocess
    features = extract_features(file_location)
    scaled_features = scaler.transform(features)  # Apply the scaler

    # Make prediction
    prediction = model.predict(scaled_features)
    emotion = label_encoder.inverse_transform([prediction])[0]  # Convert to emotion label
    
    # Upload to Cloudflare R2
    s3.upload_file(file_location, CLOUDFLARE_R2_BUCKET, file.filename)
    os.remove(file_location)

    return {"filename": file.filename, "emotion": emotion}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
