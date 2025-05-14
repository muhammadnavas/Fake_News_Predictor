import streamlit as st
import joblib

# Page config
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Enter a **news article or headline**, choose a model, and find out whether it's **real** or **fake**.")

# Model options
model_option = st.selectbox(
    "📊 Choose a model:",
    ("Naive Bayes", "Logistic Regression", "Random Forest", "CatBoost")
)

model_files = {
    "Naive Bayes": "model_naive_bayes.pkl",
    "Logistic Regression": "model_logistic_regression.pkl",
    "Random Forest": "model_random_forest.pkl",
    "CatBoost": "model_catboost.pkl"
}

# Load model and vectorizer
try:
    model = joblib.load(model_files[model_option])
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Input
input_text = st.text_area("✍️ Enter the news text here:")

# Predict button
if st.button("🔍 Predict"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        input_vector = vectorizer.transform([input_text.lower()])
        prediction = model.predict(input_vector)[0]

        try:
            confidence = model.predict_proba(input_vector)[0][prediction]
        except AttributeError:
            confidence = 1.0

        if prediction == 1:
            st.success(f"✅ This news is **REAL** (Confidence: {confidence:.2f})")
        else:
            st.error(f"❌ This news is **FAKE** (Confidence: {confidence:.2f})")
