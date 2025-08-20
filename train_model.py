import pandas as pd
import string
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from catboost import CatBoostClassifier
import numpy as np

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove punctuation but keep spaces
    text = ''.join([c if c not in string.punctuation else ' ' for c in text])
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

def create_models_folder():
    """Create models folder if it doesn't exist"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"📁 Created {models_dir} folder")
    return models_dir

def load_and_prepare_data():
    """Load and prepare the training data"""
    print("📊 Loading data...")
    
    # Check if data files exist
    if not os.path.exists("Fake.csv") or not os.path.exists("True.csv"):
        print("❌ Error: Fake.csv and True.csv files not found!")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        return None, None, None, None
    
    # Load data
    fake_df = pd.read_csv("Fake.csv")
    real_df = pd.read_csv("True.csv")
    
    # Add labels
    fake_df["label"] = 0  # Fake news
    real_df["label"] = 1  # Real news
    
    # Combine datasets
    df = pd.concat([fake_df, real_df], ignore_index=True)
    
    # Select relevant columns and clean
    df = df[["title", "text", "label"]].dropna()
    df["combined"] = df["title"].astype(str) + " " + df["text"].astype(str)
    df["combined"] = df["combined"].apply(clean_text)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Total articles: {len(df)}")
    print(f"   Real news: {len(df[df['label'] == 1])}")
    print(f"   Fake news: {len(df[df['label'] == 0])}")
    
    return df["combined"], df["label"], df, None

def create_vectorizer_and_split(X, y, models_dir):
    """Create TF-IDF vectorizer and split data"""
    print("\n🔄 Creating vectorizer and splitting data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        min_df=2,
        max_features=10000,
        ngram_range=(1, 2)
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save vectorizer in models folder
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"💾 Vectorizer saved as: {vectorizer_path}")
    
    print(f"   Training samples: {X_train_vec.shape[0]}")
    print(f"   Test samples: {X_test_vec.shape[0]}")
    print(f"   Features: {X_train_vec.shape[1]}")
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

def apply_smote(X_train_vec, y_train):
    """Apply SMOTE for class balancing"""
    print("\n⚖️ Applying SMOTE for class balancing...")
    
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_vec, y_train)
    
    print(f"   Original training samples: {X_train_vec.shape[0]}")
    print(f"   Balanced training samples: {X_train_bal.shape[0]}")
    
    return X_train_bal, y_train_bal

def train_and_evaluate_models(X_train_bal, y_train_bal, X_test_vec, y_test, models_dir):
    """Train and evaluate all models"""
    print("\n🤖 Training models...")
    
    models = {
        "naive_bayes": MultinomialNB(alpha=0.1),
        "logistic_regression": LogisticRegression(
            max_iter=1000, 
            C=1.0, 
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            verbose=0,
            random_state=42
        )
    }
    
    model_results = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training {name.replace('_', ' ').title()}...")
        
        try:
            # Train model
            model.fit(X_train_bal, y_train_bal)
            
            # Make predictions
            y_pred = model.predict(X_test_vec)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model in models folder
            model_filename = os.path.join(models_dir, f"model_{name}.pkl")
            joblib.dump(model, model_filename)
            
            # Store results
            model_results[name] = {
                "model": model,
                "accuracy": accuracy,
                "predictions": y_pred
            }
            
            print(f"   ✅ {name} trained successfully!")
            print(f"   📈 Accuracy: {accuracy:.4f}")
            print(f"   💾 Saved as: {model_filename}")
            
            # Print detailed classification report
            print(f"\n📊 Classification Report for {name.replace('_', ' ').title()}:")
            print(classification_report(y_test, y_pred, 
                                      target_names=['Fake', 'Real']))
            
        except Exception as e:
            print(f"   ❌ Error training {name}: {e}")
            continue
    
    return model_results

def save_training_summary(model_results, models_dir):
    """Save training summary to a text file"""
    summary_path = os.path.join(models_dir, "training_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("FAKE NEWS DETECTION - TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        if model_results:
            f.write(f"Successfully trained {len(model_results)} models:\n\n")
            for name, results in model_results.items():
                f.write(f"• {name.replace('_', ' ').title()}: {results['accuracy']:.4f} accuracy\n")
            
            # Find best model
            best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
            f.write(f"\nBest performing model: {best_model[0].replace('_', ' ').title()}\n")
            f.write(f"Best accuracy: {best_model[1]['accuracy']:.4f}\n")
        else:
            f.write("No models were successfully trained!\n")
        
        f.write(f"\nTraining completed at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📄 Training summary saved as: {summary_path}")

def main():
    """Main function to orchestrate the training process"""
    print("🚀 Starting Fake News Detection Model Training")
    print("=" * 50)
    
    # Create models folder
    models_dir = create_models_folder()
    
    # Load and prepare data
    X, y, df, _ = load_and_prepare_data()
    if X is None:
        return
    
    # Create vectorizer and split data
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = create_vectorizer_and_split(X, y, models_dir)
    
    # Apply SMOTE
    X_train_bal, y_train_bal = apply_smote(X_train_vec, y_train)
    
    # Train and evaluate models
    model_results = train_and_evaluate_models(X_train_bal, y_train_bal, X_test_vec, y_test, models_dir)
    
    # Save training summary
    save_training_summary(model_results, models_dir)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TRAINING SUMMARY")
    print("=" * 50)
    
    if model_results:
        print(f"✅ Successfully trained {len(model_results)} models:")
        for name, results in model_results.items():
            print(f"   • {name.replace('_', ' ').title()}: {results['accuracy']:.4f} accuracy")
        
        # Find best model
        best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best performing model: {best_model[0].replace('_', ' ').title()}")
        print(f"   📈 Best accuracy: {best_model[1]['accuracy']:.4f}")
        
        print(f"\n📁 All models and vectorizer saved in: {models_dir}/")
        print("   Files created:")
        print("   • vectorizer.pkl")
        for name in model_results.keys():
            print(f"   • model_{name}.pkl")
        print("   • training_summary.txt")
        
    else:
        print("❌ No models were successfully trained!")
    
    print("\n🎉 Training complete! You can now run the Streamlit app.")
    print("   Command: streamlit run app.py")
    print("   Make sure to update your app.py to load models from the 'models' folder")

if __name__ == "__main__":
    main()