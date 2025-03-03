# Backend - FastAPI Implementation (backend/main.py)

from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import tensorflow as tf
import librosa
import shutil
import boto3
import os
import streamlit as st
import requests

app = FastAPI()

# Load model
MODEL_PATH = "models/emotion_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Cloudflare R2 Configuration
CLOUDFLARE_R2_BUCKET = "your-bucket-name"
CLOUDFLARE_ACCESS_KEY = "your-access-key"
CLOUDFLARE_SECRET_KEY = "your-secret-key"
CLOUDFLARE_ENDPOINT = "your-endpoint"

s3 = boto3.client("s3", endpoint_url=CLOUDFLARE_ENDPOINT, 
                  aws_access_key_id=CLOUDFLARE_ACCESS_KEY, 
                  aws_secret_access_key=CLOUDFLARE_SECRET_KEY)

# Preprocess function
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs, axis=0)

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    file_location = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    features = extract_features(file_location)
    prediction = model.predict(features)
    emotion = np.argmax(prediction)
    
    # Upload to Cloudflare R2
    s3.upload_file(file_location, CLOUDFLARE_R2_BUCKET, file.filename)
    os.remove(file_location)

    return {"filename": file.filename, "emotion": int(emotion)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)