import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ✅ Load model & tokenizer
model_name = "JCKipkemboi/hate_speech_detector_bert"  # Replace with your actual model repo
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Streamlit UI
st.set_page_config(page_title="Hate Speech Detector", layout="wide")
st.title("Hate Speech Detector App")
st.write("Enter a sentence below to check if it's Hate Speech or Not.")

# ✅ Text input
user_input = st.text_area("Enter text:")

if st.button("Predict"):
    if user_input.strip():
        # Tokenization
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()

        # ✅ Display results
        label = "Hate Speech" if prediction == 1 else "Not Hate Speech"
        st.subheader(f"Prediction: {label}")

    else:
        st.warning("Please enter some text before predicting.")

# ✅ Run Streamlit only when executed directly
if __name__ == "__main__":
    st.write("Ready to classify text.")
