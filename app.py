import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Allow CORS (Fix for Power BI access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (Adjust as needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# ✅ Load model & tokenizer
model_name = "JCKipkemboi/hate_speech_detector_bert"  # Replace with your actual model repo
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class TextInput(BaseModel):
    text: str

# ✅ API Endpoint for Predictions
@app.post("/predict")
def predict(input: TextInput):
    """API Endpoint to classify text as Hate Speech or Not Hate Speech"""
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    label = "Hate Speech" if prediction == 1 else "Not Hate Speech"
    return {"text": input.text, "prediction": label}

# ✅ Streamlit UI (For Interactive Testing)
st.set_page_config(page_title="Hate Speech Detector", page_icon="⚠️", layout="centered")
st.title("🛑 Hate Speech Detector")
st.write("Enter a sentence below to check if it's **Hate Speech** or **Not Hate Speech**.")

# User Input
text = st.text_area("🔤 Enter your text:", height=100)

# Predict Function
def streamlit_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
    
    label = "Hate Speech" if prediction == 1 else "Not Hate Speech"
    confidence = probabilities[0][prediction].item() * 100
    return label, confidence

# Predict Button with Loading
if st.button("🚀 Predict"):
    if text.strip():
        with st.spinner("🔄 Processing... Please wait"):
            label, confidence = streamlit_predict(text)
        st.success(f"✅ **Prediction:** {label}")
        st.write(f"📊 **Confidence:** {confidence:.2f}%")
    else:
        st.warning("⚠️ Please enter some text before predicting.")

# Footer
st.markdown(
    """
    ---
    **Note:** This tool is a prototype and may not always be accurate.
    """,
    unsafe_allow_html=True,
)

# ✅ Run Streamlit only when executed directly
if __name__ == "__main__":
    st.write("Ready to classify text.")
