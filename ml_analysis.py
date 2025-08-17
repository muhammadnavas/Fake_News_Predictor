import joblib
import numpy as np
from typing import Dict
import streamlit as st

# Load all ML models
@st.cache_resource
def load_all_models():
    """Load all available ML models and vectorizer"""
    models = {}
    model_files = {
        "Naive Bayes": "model_naive_bayes.pkl",
        "Logistic Regression": "model_logistic_regression.pkl", 
        "Random Forest": "model_random_forest.pkl",
        "CatBoost": "model_catboost.pkl"
    }
    
    try:
        vectorizer = joblib.load("vectorizer.pkl")
    except FileNotFoundError:
        st.error("❌ Vectorizer not found. Please run train_models.py first.")
        return None, None
    
    for model_name, file_path in model_files.items():
        try:
            models[model_name] = joblib.load(file_path)
        except FileNotFoundError:
            st.warning(f"⚠️ {model_name} model not found")
    
    return models, vectorizer

def analyze_with_all_models(text: str, models: Dict, vectorizer) -> Dict:
    """Analyze text with all available ML models"""
    results = {}
    input_vector = vectorizer.transform([text.lower()])
    
    for model_name, model in models.items():
        try:
            prediction = model.predict(input_vector)[0]
            
            try:
                probabilities = model.predict_proba(input_vector)[0]
                confidence = max(probabilities)
                if len(probabilities) > 1:
                    fake_prob = probabilities[0]
                    real_prob = probabilities[1]
                else:
                    fake_prob = 1 - probabilities[0] if prediction == 1 else probabilities[0]
                    real_prob = probabilities[0] if prediction == 1 else 1 - probabilities[0]
            except AttributeError:
                confidence = 0.8
                fake_prob = 0.0 if prediction == 1 else 1.0
                real_prob = 1.0 if prediction == 1 else 0.0
            
            results[model_name] = {
                "prediction": "REAL" if prediction == 1 else "FAKE",
                "confidence": confidence,
                "fake_probability": fake_prob,
                "real_probability": real_prob,
                "raw_prediction": prediction
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    return results