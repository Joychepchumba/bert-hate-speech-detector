import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load Model & Tokenizer
model_name = "JCKipkemboi/hate_speech_detector"  
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Hate Speech" if prediction == 1 else "Not Hate Speech"

# Streamlit UI
st.title("Hate Speech Detector")
st.write("Enter a sentence to check if it's hate speech.")

user_input = st.text_area("Enter text:")
if st.button("Predict"):
    result = classify_text(user_input)
    st.write(f"**Prediction:** {result}")
