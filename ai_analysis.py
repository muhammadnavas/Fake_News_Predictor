import os
from typing import List, Dict
import google.generativeai as genai


try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


def get_gemini_key() -> str:
    """Fetch Gemini API key safely from env or Streamlit secrets."""

    key = os.getenv("GEMINI_API") or os.getenv("GEMINI_API_KEY")
    
    if not key:
        try:
            import streamlit as st

            key = st.secrets.get("GEMINI_API", None) or st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            key = None

    if not key:

        return None

    return key.strip()

def get_newsapi_key() -> str:
    """Fetch NewsAPI key safely from env or Streamlit secrets."""

    key = os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI")
    
    if not key:
        try:
            import streamlit as st

            key = st.secrets.get("NEWSAPI_KEY", None) or st.secrets.get("NEWS_API_KEY", None) or st.secrets.get("NEWSAPI", None)
        except Exception:
            key = None

    if not key:
        return None

    return key.strip()


GEMINI_KEY = get_gemini_key()
NEWSAPI_KEY = get_newsapi_key()

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        model = None
else:
    model = None


def check_api_keys():
    """Check which API keys are available and return status."""
    status = {
        "gemini_available": GEMINI_KEY is not None and model is not None,
        "newsapi_available": NEWSAPI_KEY is not None,
        "gemini_key": GEMINI_KEY is not None,
        "newsapi_key": NEWSAPI_KEY is not None
    }
    return status


def rag_enhanced_gemini_analysis(news_text: str, relevant_facts: List[Dict]) -> str:
    """Enhance Gemini analysis with RAG-retrieved facts."""
    if not model:
        return "⚠️ Gemini analysis unavailable. Please configure GEMINI_API_KEY in Streamlit secrets."

    facts_context = ""
    if relevant_facts:
        facts_context = "\n\n**RETRIEVED KNOWLEDGE BASE FACTS:**\n"
        for i, fact in enumerate(relevant_facts[:3], 1):
            sources = ", ".join(fact.get("sources", []))
            similarity = fact.get("similarity", 0.0)
            facts_context += (
                f"{i}. {fact.get('content','[No content]')} "
                f"(Similarity: {similarity:.2f})\n"
                f"   Sources: {sources}\n"
            )

    enhanced_prompt = f"""
As an expert fact-checker and news analyst with access to a knowledge base, 
analyze this text:

TEXT: "{news_text}"
{facts_context}

Note: Do not classify a text as FAKE simply because you cannot verify exact dates or specific numbers. Classify as FAKE only if there are clear signs of fabrication, extreme bias, conspiracy theories, or illogical inconsistencies.

Provide exactly one short paragraph summarizing your analysis of credibility, contradictions with established facts, and red flags.
Then, on a new line, provide the final result in this exact format:
**RESULT:** [REAL or FAKE] (Confidence: [High/Medium/Low])
"""

    try:
        response = model.generate_content(enhanced_prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error in RAG-enhanced analysis: {str(e)}"


def standard_gemini_analysis(news_text: str) -> str:
    """Standard Gemini analysis without RAG enhancement."""
    if not model:
        return "⚠️ Gemini analysis unavailable. Please configure GEMINI_API_KEY in Streamlit secrets."

    prompt = f"""
As an expert fact-checker and news analyst, analyze this text:

TEXT: "{news_text}"

Note: Do not classify a text as FAKE simply because you cannot verify exact dates or specific numbers. Classify as FAKE only if there are clear signs of fabrication, extreme bias, conspiracy theories, or illogical inconsistencies.

Provide exactly one short paragraph summarizing your analysis of credibility, red flags, and consistency.
Then, on a new line, provide the final result in this exact format:
**RESULT:** [REAL or FAKE] (Confidence: [High/Medium/Low])
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error in Gemini analysis: {str(e)}"


def validate_api_keys():
    """Validate and print all API key statuses for debugging."""
    print("[*] API Key Validation:")
    

    gemini_api = os.getenv("GEMINI_API")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    newsapi_key = os.getenv("NEWSAPI_KEY")
    
    print(f"GEMINI_API: {'[OK] Found' if gemini_api else '[--] Missing'}")
    print(f"GEMINI_API_KEY: {'[OK] Found' if gemini_api_key else '[--] Missing'}")
    print(f"NEWSAPI_KEY: {'[OK] Found' if newsapi_key else '[--] Missing'}")
    

    try:
        import streamlit as st
        gemini_secret = st.secrets.get("GEMINI_API", None) or st.secrets.get("GEMINI_API_KEY", None)
        newsapi_secret = st.secrets.get("NEWSAPI_KEY", None)
        print(f"Streamlit Gemini Secret: {'[OK] Found' if gemini_secret else '[--] Missing'}")
        print(f"Streamlit NewsAPI Secret: {'[OK] Found' if newsapi_secret else '[--] Missing'}")
    except:
        print("Streamlit secrets: Not available (running locally)")
    
    print(f"Final GEMINI_KEY: {'[OK] Available' if GEMINI_KEY else '[--] Missing'}")
    print(f"Final NEWSAPI_KEY: {'[OK] Available' if NEWSAPI_KEY else '[--] Missing'}")
    print(f"Gemini Model: {'[OK] Initialized' if model else '[--] Failed'}")


def display_api_status():
    """Display API key status in Streamlit sidebar or main area."""
    try:
        import streamlit as st
        
        st.subheader("🔧 API Configuration Status")
        

        st.code(f"""
Debug Info:
GEMINI_API (env): {os.getenv('GEMINI_API', 'Not found')}
GEMINI_API_KEY (env): {os.getenv('GEMINI_API_KEY', 'Not found')}
NEWSAPI_KEY (env): {os.getenv('NEWSAPI_KEY', 'Not found')}
Final GEMINI_KEY: {'Available' if GEMINI_KEY else 'Missing'}
Final NEWSAPI_KEY: {'Available' if NEWSAPI_KEY else 'Missing'}
Model initialized: {'Yes' if model else 'No'}
        """)
        
        status = check_api_keys()
        
        if status["gemini_available"]:
            st.success("✅ Gemini API: Connected and Ready")
        elif status["gemini_key"]:
            st.warning("⚠️ Gemini API: Key found but model initialization failed")
        else:
            st.error("❌ Gemini API: Not configured")
            st.info("💡 Make sure GEMINI_API is set in your .env file or Streamlit secrets")
        
        if status["newsapi_available"]:
            st.success("✅ NewsAPI: Connected")
        else:
            st.error("❌ NewsAPI: Not configured")
            st.info("💡 Make sure NEWSAPI_KEY is set in your .env file or Streamlit secrets")
        

        with st.expander("📋 How to set up API keys"):
            st.markdown("""
            **For Streamlit Cloud:**
            ```toml
            GEMINI_API = "your_gemini_api_key_here"
            NEWSAPI_KEY = "your_newsapi_key_here"
            ```
            
            **For local development (.env file):**
            ```
            GEMINI_API=your_gemini_api_key_here
            NEWSAPI_KEY=your_newsapi_key_here
            ```
            """)
            
    except ImportError:

        pass


validate_api_keys()