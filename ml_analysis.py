import joblib
import numpy as np
import os
from typing import Dict, Optional, Tuple
import streamlit as st

@st.cache_resource
def load_all_models() -> Tuple[Optional[Dict], Optional[object]]:
    """Load all available ML models and vectorizer from models folder"""
    models = {}
    models_dir = "models"
    
    model_files = {
        "Naive Bayes": "model_naive_bayes.pkl",
        "Logistic Regression": "model_logistic_regression.pkl", 
        "Random Forest": "model_random_forest.pkl",
        "CatBoost": "model_catboost.pkl"
    }
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        st.error(f"❌ Models folder '{models_dir}' not found. Please run train_model.py first.")
        return None, None
    
    # Load vectorizer
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    try:
        vectorizer = joblib.load(vectorizer_path)
        st.success("✅ Vectorizer loaded successfully")
    except FileNotFoundError:
        st.error(f"❌ Vectorizer not found at {vectorizer_path}. Please run train_model.py first.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading vectorizer: {str(e)}")
        return None, None
    
    # Load all models
    loaded_count = 0
    for model_name, file_name in model_files.items():
        model_path = os.path.join(models_dir, file_name)
        try:
            models[model_name] = joblib.load(model_path)
            loaded_count += 1
        except FileNotFoundError:
            st.warning(f"⚠️ {model_name} model not found at {model_path}")
        except Exception as e:
            st.warning(f"⚠️ Error loading {model_name}: {str(e)}")
    
    if loaded_count > 0:
        st.success(f"✅ Successfully loaded {loaded_count} out of {len(model_files)} models")
    else:
        st.error("❌ No models could be loaded. Please check your models folder and run train_model.py")
        return None, None
    
    return models, vectorizer

def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    if not text:
        return ""
    text = text.lower().strip()
    # Basic cleaning - remove extra spaces
    text = ' '.join(text.split())
    return text

def analyze_with_all_models(text: str, models: Dict, vectorizer) -> Dict:
    """Analyze text with all available ML models"""
    if not text.strip():
        return {}
    
    results = {}
    
    try:
        # Preprocess and vectorize the input text
        processed_text = clean_text(text)
        input_vector = vectorizer.transform([processed_text])
    except Exception as e:
        st.error(f"❌ Error processing text: {str(e)}")
        return {}
    
    for model_name, model in models.items():
        try:
            # Make prediction
            prediction = model.predict(input_vector)[0]
            
            # Get probability scores if available
            try:
                probabilities = model.predict_proba(input_vector)[0]
                confidence = max(probabilities)
                
                if len(probabilities) >= 2:
                    fake_prob = probabilities[0]  # Class 0 = Fake
                    real_prob = probabilities[1]  # Class 1 = Real
                else:
                    # Handle single probability case
                    fake_prob = 1 - probabilities[0] if prediction == 1 else probabilities[0]
                    real_prob = probabilities[0] if prediction == 1 else 1 - probabilities[0]
                    
            except AttributeError:
                # Model doesn't support predict_proba
                confidence = 0.8  # Default confidence
                fake_prob = 0.0 if prediction == 1 else 1.0
                real_prob = 1.0 if prediction == 1 else 0.0
            except Exception as e:
                # Fallback for any other probability calculation errors
                confidence = 0.5
                fake_prob = 0.5
                real_prob = 0.5
            
            results[model_name] = {
                "prediction": "REAL" if prediction == 1 else "FAKE",
                "confidence": round(confidence, 4),
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "raw_prediction": int(prediction),
                "status": "success"
            }
            
        except Exception as e:
            results[model_name] = {
                "error": str(e),
                "status": "error"
            }
    
    return results

def get_ensemble_prediction(results: Dict) -> Dict:
    """Get ensemble prediction from all successful model results"""
    if not results:
        return {}
    
    successful_results = {k: v for k, v in results.items() 
                         if v.get("status") == "success"}
    
    if not successful_results:
        return {"error": "No successful predictions from any model"}
    
    # Calculate average probabilities
    avg_fake_prob = np.mean([r["fake_probability"] for r in successful_results.values()])
    avg_real_prob = np.mean([r["real_probability"] for r in successful_results.values()])
    
    # Majority voting
    fake_votes = sum(1 for r in successful_results.values() if r["prediction"] == "FAKE")
    real_votes = sum(1 for r in successful_results.values() if r["prediction"] == "REAL")
    
    ensemble_prediction = "FAKE" if fake_votes > real_votes else "REAL"
    confidence = max(avg_fake_prob, avg_real_prob)
    
    return {
        "prediction": ensemble_prediction,
        "confidence": round(confidence, 4),
        "fake_probability": round(avg_fake_prob, 4),
        "real_probability": round(avg_real_prob, 4),
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "total_models": len(successful_results),
        "voting_consensus": round((max(fake_votes, real_votes) / len(successful_results)) * 100, 1)
    }