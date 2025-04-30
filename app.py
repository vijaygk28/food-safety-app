import streamlit as st
import os
import tempfile
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import pytesseract
from groq import Groq
from dotenv import load_dotenv
import cv2
import numpy as np
import json

# Load environment variables (if required)
load_dotenv()

# Set Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if API key is available
if not groq_api_key:
    st.error("❗ GROQ API Key is missing. Please set it in your .env file.")
    st.stop()

# Check if the secrets are loaded correctly
st.write(st.secrets)

# Google Cloud Vision API Credentials from Streamlit secrets
google_credentials_json = st.secrets["GOOGLE_CLOUD"]["GOOGLE_APPLICATION_CREDENTIALS_JSON"]

# Write the credentials to a temporary file for the Vision API client
with open("/tmp/google-credentials.json", "w") as json_file:
    json_file.write(google_credentials_json)

# Load the credentials from the temporary file
credentials = service_account.Credentials.from_service_account_file("/tmp/google-credentials.json")

# Create the Vision API client
client = vision.ImageAnnotatorClient(credentials=credentials)

# Basic Image OCR (fast, without preprocessing)
def basic_ocr(image_path):
    img = Image.open(image_path).convert('RGB')
    text = pytesseract.image_to_string(img)
    return text

# Advanced Image Preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding
    threshold_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Save the preprocessed image over the original
    cv2.imwrite(image_path, threshold_img)

# Enhanced OCR (after preprocessing)
def enhanced_ocr(image_path):
    preprocess_image(image_path)
    img = Image.open(image_path).convert('RGB')
    text = pytesseract.image_to_string(img)
    return text

# Assess food safety using Groq Llama3
def assess_food_safety(text, years, months):
    prompt = f"""
You are a senior pediatric nutritionist.

Analyze the following food product's nutrition content for a child aged {years} years and {months} months:

Food Product Content:
---
{text}
---

Rules:
- Under 2 years: No added sugars, low sodium (<200mg/day), avoid preservatives (BHA, BHT), artificial colors, caffeine, honey.
- 2-5 years: Limit sodium, added sugars, saturated fat; avoid high caffeine, artificial additives.
- Always highlight allergens like peanuts, soy, gluten, dairy.

Provide:
1. Safety Verdict: ("Safe" / "Caution" / "Not Safe")
2. Key Positives
3. Key Concerns
4. Final Advice (simple, warm, non-technical).

DO NOT copy the entire text back. Summarize nicely.
"""

    try:
        client = Groq(api_key=groq_api_key)

        chat_response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return chat_response.choices[0].message.content.strip()

    except Exception as e:
        return f"An error occurred during analysis: {str(e)}"

# Google Cloud Vision OCR
def perform_ocr_with_vision(image):
    """Function to perform OCR using Google Cloud Vision API"""
    content = image.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts

# Main Streamlit App
def app():
    st.title("🍼 Food Safety Assessment for Kids")

    uploaded_image = st.file_uploader(
        "Upload an image with nutrition label details", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        with st.spinner('🔍 Extracting text from image...'):
            extracted_text = ""
            temp_file_path = ""

            try:
                # Save uploaded image to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(uploaded_image.read())
                    temp_file_path = tmp_file.name

                # Step 1: Try basic OCR first
                extracted_text = basic_ocr(temp_file_path)

                # Step 2: If text is too small, fallback to enhanced OCR
                if len(extracted_text.strip()) < 30:
                    st.info("🔄 Low confidence OCR detected. Retrying with enhanced image processing...")
                    extracted_text = enhanced_ocr(temp_file_path)

                # If basic OCR fails, try Vision API
                if len(extracted_text.strip()) < 30:
                    st.info("🔄 Basic OCR not confident. Retrying with Google Vision API...")
                    texts = perform_ocr_with_vision(uploaded_image)
                    if texts:
                        extracted_text = texts[0].description

            except Exception as e:
                st.error(f"❗ Error during text extraction: {str(e)}")

            finally:
                # Clean up temp file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        st.subheader("📄 Extracted Text:")
        st.write(extracted_text if extracted_text else "No text could be extracted.")

        # Child's Age Inputs
        child_years = st.number_input("Child's Age (Years)", min_value=0, max_value=10, value=2)
        child_months = st.number_input("Child's Age (Months)", min_value=0, max_value=11, value=6)

        if st.button("🔍 Assess Food Safety"):
            if extracted_text:
                with st.spinner('🤖 Analyzing food safety...'):
                    verdict = assess_food_safety(extracted_text, child_years, child_months)

                st.subheader("🛡️ Food Safety Verdict:")
                st.markdown(f"""
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #eee;">
                    <p style="font-size:18px;">{verdict}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❗ No text was extracted to assess. Please upload a valid image.")

if __name__ == "__main__":
    app()
