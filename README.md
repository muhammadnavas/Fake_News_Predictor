# Fake News Predictor

A multi-source, RAG-enhanced fake news detection system using ML models, retrieval-augmented generation, and AI-powered analysis. Combines news API verification, knowledge base retrieval, and advanced analytics in a Streamlit web app.

## Features
- **Multi-API News Verification**: Checks news existence across NewsAPI, GNews, CurrentsAPI, ContextualWeb, and Google Fact Check.
- **RAG Knowledge Base**: Ingests facts from datasets and APIs, supports semantic retrieval (ChromaDB + SentenceTransformers, TF-IDF fallback).
- **ML Model Analysis**: Uses multiple trained models (Naive Bayes, Logistic Regression, Random Forest, CatBoost) for prediction.
- **AI Assessment**: Integrates Gemini AI for deep analysis.
- **Content Validation**: Ensures only news-like content is analyzed.
- **Bulk Dataset Ingestion**: Supports streaming ingestion from large CSVs (True.csv, Fake.csv).

## Setup
1. **Clone the repository**
   ```sh
   git clone https://github.com/muhammadnavas/Fake_News_Predictor.git
   cd Fake_News_Predictor
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Configure API keys**
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your keys:
     ```toml
     NEWSAPI_KEY = "your_newsapi_key"
     GNEWS_KEY = "your_gnews_key"
     CURRENTS_KEY = "your_currents_key"
     CONTEXTUALWEB_KEY = "your_contextualweb_key"
     GOOGLE_FACTCHECK_API_KEY = "your_google_factcheck_key"
     GEMINI_API_KEY = "your_gemini_key"
     ```
   - Or set keys in `.env` (see `.env.example`).
4. **Run the app**
   ```sh
   streamlit run app.py
   ```

## Usage
- Enter a news headline or article in the input box.
- The app validates content and runs analysis using selected methods (API, RAG, ML, AI).
- View results in tabs: Verification, RAG Analysis, ML Models, AI Assessment, Summary.
- Use the sidebar to ingest datasets and manage the knowledge base.

## Troubleshooting
- **No news fetched**: Ensure API keys are set and match expected names in both secrets and code.
- **Analysis blocked**: Input must be news-like (headline or article, not personal/casual text).
- **Module import errors**: Run `pip install -r requirements.txt` to install all dependencies.
- **Secrets parse error**: All values in `.streamlit/secrets.toml` must be quoted strings.

## Security
- Never commit real API keys to version control. `.gitignore` excludes `.env` and `.streamlit/secrets.toml` by default.
- Rotate keys if accidentally exposed.

## Project Structure
- `app.py` — Main Streamlit app
- `rag_system.py`, `rag_pipeline.py` — RAG logic and pipeline
- `ml_analysis.py` — ML model loading and analysis
- `content_detector.py` — Content validation
- `fetch_news.py` — News API integration
- `models/` — Pretrained ML models
- `chroma_db/` — ChromaDB persistent storage
- `fact_database.json` — Knowledge base facts

## License
MIT (or specify your license)

## Repository
- **GitHub**: https://github.com/muhammadnavas/Fake_News_Predictor.git
- **Owner**: muhammadnavas (MrHidey)
- **Branch**: main

## Credits
- Built by MrHidey (muhammadnavas) and contributors
- Uses open-source libraries: Streamlit, scikit-learn, chromadb, sentence-transformers, Google Generative AI, etc.

---
For questions or contributions, open an issue or pull request at the GitHub repository.
