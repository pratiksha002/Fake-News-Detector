import streamlit as st
import joblib
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load model + vectorizer
model = joblib.load("../models/fake_news_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

def predict_news(news_text):
    cleaned = clean_text(news_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = float(proba[pred])

    label = "REAL ✅" if pred == 1 else "FAKE ❌"
    return label, confidence

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write("Paste a news article or headline below and click **Predict**.")

news_input = st.text_area("Enter news text:", height=200)

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        label, confidence = predict_news(news_input)
        st.subheader("Prediction Result:")
        st.write(f"### {label}")
        st.write(f"Confidence: **{confidence*100:.2f}%**")
