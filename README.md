# **Speech Emotion Recognition with FastAPI, Streamlit, and Cloudflare R2**

## **Overview**
This repository contains a neural network model for **speech emotion recognition** that:  
- Processes audio files to classify emotions  
- Exposes the model via a **FastAPI** backend  
- Stores and manages files using **Cloudflare R2**  
- Provides an interactive **Streamlit** frontend for user interaction  

## **Tech Stack**
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Storage:** Cloudflare R2  
- **Machine Learning:** Neural network for emotion recognition  

## **Getting Started**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
```

### **Install dependencies**
```bash
pip install -r requirements.txt
```

### **Set Up Cloudfare R2**
```bash
R2_ACCESS_KEY=<your-access-key>
R2_SECRET_KEY=<your-secret-key>
R2_BUCKET_NAME=<your-bucket-name>
```

### **Run the backend**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### **Run the frontend**
```bash
streamlit run frontend/app.py
```

### **Usage**
- Upload an audio file through the Streamlit frontend
- The file is processed and stored in Cloudflare R2
- The FastAPI backend analyzes the file and predicts the emotion using the ML model
- The predicted emotion is displayed in the frontend

### **Future Improvements**
- Improve model accuracy with larger datasets
- Add support for real-time audio analysis
- Deploy the system on a cloud platform

