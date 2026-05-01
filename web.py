import streamlit as st
import joblib
from PIL import Image
import easyocr
import numpy as np
import cv2
import re

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\u0980-\u09FF\s]", "", text)  # Only Bangla text
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# -----------------------------
# UI Layout
# -----------------------------
st.title("📰 Multimodal Fake News Detection System")
st.write("Enter news text OR upload an image to detect fake news.")

# Input Section
news_input = st.text_area("📝 Enter News Text")
uploaded_image = st.file_uploader("📸 Upload News Image", type=["png", "jpg", "jpeg"])

image_text = ""

# -----------------------------
# Image Processing & OCR
# -----------------------------
if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.info("🔍 Extracting Bangla text from image...")

        # Convert PIL → numpy
        img_np = np.array(image)

        # Preprocess image (grayscale + threshold)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Initialize OCR
        reader = easyocr.Reader(['bn'])
        results = reader.readtext(thresh)

        # Filter OCR results
        cleaned_results = []
        for bbox, text, conf in results:
            if len(text) > 3 and conf > 0.5:
                (p1, p2, p3, p4) = bbox
                width = abs(p2[0] - p1[0])
                height = abs(p4[1] - p1[1])
                if width > 80 and height > 20:
                    cleaned_results.append(text)

        image_text = " ".join(cleaned_results)
        st.text_area("📌 Extracted Bangla Text", image_text)

    except Exception as e:
        st.error(f"Error reading image: {e}")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    final_text = news_input.strip() if news_input.strip() else image_text.strip()

    if not final_text:
        st.warning("⚠️ Please enter text or upload an image first.")
    else:
        # Clean text before prediction
        final_text = clean_text(final_text)
        
        # Vectorize
        transformed = vectorizer.transform([final_text])
        
        # Predict
        prediction = model.predict(transformed)

        # Dataset mapping: 1 = Authentic-48K.csv (Real), 0 = Fake-1K.csv (Fake)
        if prediction[0] == 1:
            st.success("✅ Real News (Authentic)")
        else:
            st.error("❌ Fake News")