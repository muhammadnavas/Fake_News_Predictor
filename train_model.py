import pandas as pd
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

# Load and label data
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")
fake_df["label"] = 0
real_df["label"] = 1
df = pd.concat([fake_df, real_df])
df = df[["title", "text", "label"]].dropna()
df["combined"] = df["title"] + " " + df["text"]

# Clean text
def clean_text(text):
    text = text.lower()
    return ''.join([c for c in text if c not in string.punctuation])

df["combined"] = df["combined"].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(df["combined"], df["label"], test_size=0.2, random_state=42)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_vec, y_train)

# Save vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

# Train models
models = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "catboost": CatBoostClassifier(verbose=0)
}

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    joblib.dump(model, f"model_{name}.pkl")
    y_pred = model.predict(X_test_vec)
    print(f"\n📊 {name.upper()}:\n", classification_report(y_test, y_pred))

print("✅ All models and vectorizer saved.")
