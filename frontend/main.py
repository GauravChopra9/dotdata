import streamlit as st
import requests
import io
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000/upload/"

# Streamlit UI
st.title("Emotion Detection from Audio")
st.write("Upload an audio file to detect the emotion in the speech.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# When the user uploads a file
if uploaded_file is not None:
    # Display the file name
    st.write(f"File uploaded: {uploaded_file.name}")

    # Convert the uploaded file to bytes and send to the FastAPI backend
    file_bytes = uploaded_file.read()

    # Send to the backend for prediction
    response = requests.post(BACKEND_URL, files={"file": (uploaded_file.name, io.BytesIO(file_bytes), uploaded_file.type)})

    # Check if the response was successful
    if response.status_code == 200:
        result = response.json()
        emotion = result["emotion"]
        
        # Display emotion result
        st.write(f"Predicted Emotion: {emotion}")

        # Load the audio file to display a waveform or MFCCs (for visualization)
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)

        # Display waveform using librosa
        st.subheader("Audio Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set(title="Audio Waveform")
        st.pyplot(fig)

        # Display MFCC feature visualization
        st.subheader("MFCC Features")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(mfccs, x_axis="time", ax=ax)
        ax.set(title="MFCC Features")
        st.pyplot(fig)

    else:
        st.error("Error in predicting emotion. Please try again.")

