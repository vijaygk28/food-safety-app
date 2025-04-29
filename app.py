# food_safety_assessment_app.py

import streamlit as st
import os
import tempfile
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
import io
import json
from google.cloud import vision
from google.oauth2 import service_account

# Load env if needed (for local testing)
load_dotenv()

# Load Groq API key
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❗ GROQ API Key is missing. Please set it in Streamlit Secrets.")
    st.stop()

# Google Vision Credentials from secrets
creds_dict = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
credentials = service_account.Credentials.from_service_account_info(creds_dict)

# OCR with Google Vision
def extract_text_google_vision(image_path):
    client = vision.ImageAnnotatorClient(credentials=credentials)
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else "No text detected."

# Assess food safety with Groq + LLaMA3
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
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred during analysis: {str(e)}"

# Main App
def app():
    st.title("🍼 Food Safety Assessment for Kids")

    uploaded_image = st.file_uploader(
        "Upload an image of the nutrition label", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Extracting text using Google Vision OCR..."):
            temp_file_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_image.read())
                    temp_file_path = tmp.name

                extracted_text = extract_text_google_vision(temp_file_path)
            except Exception as e:
                st.error(f"❗ Error during OCR: {str(e)}")
                extracted_text = ""
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        st.subheader("📄 Extracted Text:")
        st.write(extracted_text if extracted_text else "No text extracted.")

        child_years = st.number_input("Child's Age (Years)", min_value=0, max_value=10, value=2)
        child_months = st.number_input("Child's Age (Months)", min_value=0, max_value=11, value=6)

        if st.button("🔍 Assess Food Safety"):
            if extracted_text:
                with st.spinner("🤖 Analyzing nutrition..."):
                    verdict = assess_food_safety(extracted_text, child_years, child_months)
                st.subheader("🛡️ Food Safety Verdict:")
                st.markdown(f"""
                <div style="background-color:#f9f9f9; padding:20px; border-radius:10px; border:1px solid #eee;">
                    <p style="font-size:18px;">{verdict}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❗ No text found for assessment.")

if __name__ == "__main__":
    app()
