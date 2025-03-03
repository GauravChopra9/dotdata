# Frontend - Streamlit App (frontend/app.py)

def main():
    st.title("Emotion Recognition from Audio")
    st.write("Upload an audio file and get the predicted emotion.")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        response = requests.post("http://localhost:8000/upload/", 
                                 files={"file": open("temp_audio.wav", "rb")})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Emotion: {result['emotion']}")
        else:
            st.error("Error processing the file.")

if __name__ == "__main__":
    main()