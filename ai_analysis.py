import os
from typing import List, Dict
import google.generativeai as genai

# Try dotenv first (local dev), then Streamlit secrets (cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # if dotenv not available, ignore

# --- API Key Handling ---
def get_gemini_key() -> str:
    """Fetch Gemini API key safely from env or Streamlit secrets."""
    # Try multiple environment variable names for flexibility
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API")
    
    if not key:
        try:
            import streamlit as st
            # Try both possible secret names
            key = st.secrets.get("GEMINI_API_KEY", None) or st.secrets.get("GEMINI_API", None)
        except Exception:
            key = None

    if not key:
        # Don't print here as it may cause issues in Streamlit
        return None

    return key.strip()

def get_newsapi_key() -> str:
    """Fetch NewsAPI key safely from env or Streamlit secrets."""
    # Try multiple environment variable names for flexibility
    key = os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI")
    
    if not key:
        try:
            import streamlit as st
            # Try multiple possible secret names
            key = st.secrets.get("NEWSAPI_KEY", None) or st.secrets.get("NEWS_API_KEY", None) or st.secrets.get("NEWSAPI", None)
        except Exception:
            key = None

    if not key:
        return None

    return key.strip()

# --- Initialize Gemini ---
GEMINI_KEY = get_gemini_key()
NEWSAPI_KEY = get_newsapi_key()

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        model = None
else:
    model = None

# --- Helper function to check API availability ---
def check_api_keys():
    """Check which API keys are available and return status."""
    status = {
        "gemini_available": GEMINI_KEY is not None and model is not None,
        "newsapi_available": NEWSAPI_KEY is not None,
        "gemini_key": GEMINI_KEY is not None,
        "newsapi_key": NEWSAPI_KEY is not None
    }
    return status

# --- RAG-enhanced Gemini analysis ---
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
provide a comprehensive analysis of this text:

TEXT: "{news_text}"
{facts_context}

Analyze the following aspects:

1. **CLASSIFICATION**: Is this REAL or FAKE news? Provide your confidence level (High/Medium/Low).

2. **KNOWLEDGE BASE ANALYSIS**: 
- How does this text relate to the retrieved facts from our knowledge base?
- Are there any contradictions with established facts?
- What supporting or conflicting evidence do we have?

3. **CREDIBILITY INDICATORS**:
- Source reliability signals
- Writing style and tone analysis
- Presence of sensational or biased language
- Factual consistency and logical coherence

4. **RED FLAGS**: Identify any warning signs of misinformation:
- Emotional manipulation techniques
- Unsupported claims or statistics
- Conspiracy theory elements
- Clickbait characteristics

5. **VERIFICATION SUGGESTIONS**: How could this be independently verified?

6. **OVERALL ASSESSMENT**: Provide a final verdict with reasoning, 
considering both the text analysis and knowledge base facts.

Format your response clearly with headers and bullet points.
"""

    try:
        response = model.generate_content(enhanced_prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error in RAG-enhanced analysis: {str(e)}"

# --- Standard Gemini analysis ---
def standard_gemini_analysis(news_text: str) -> str:
    """Standard Gemini analysis without RAG enhancement."""
    if not model:
        return "⚠️ Gemini analysis unavailable. Please configure GEMINI_API_KEY in Streamlit secrets."

    prompt = f"""
As an expert fact-checker and news analyst, provide a comprehensive analysis of this text:

TEXT: "{news_text}"

Analyze the following aspects:

1. **CLASSIFICATION**: Is this REAL or FAKE news? Provide your confidence level (High/Medium/Low).

2. **CREDIBILITY INDICATORS**:
- Source reliability signals
- Writing style and tone analysis
- Presence of sensational or biased language
- Factual consistency and logical coherence

3. **RED FLAGS**: Identify any warning signs of misinformation:
- Emotional manipulation techniques
- Unsupported claims or statistics
- Conspiracy theory elements
- Clickbait characteristics

4. **VERIFICATION SUGGESTIONS**: How could this be independently verified?

5. **OVERALL ASSESSMENT**: Provide a final verdict with reasoning.

Format your response clearly with headers and bullet points.
"""

    try:
        response = model.generate_content(enhanced_prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error in Gemini analysis: {str(e)}"

# --- Function to display API status in Streamlit ---
def display_api_status():
    """Display API key status in Streamlit sidebar or main area."""
    try:
        import streamlit as st
        
        st.subheader("🔧 API Configuration Status")
        
        status = check_api_keys()
        
        if status["gemini_available"]:
            st.success("✅ Gemini API: Connected and Ready")
        elif status["gemini_key"]:
            st.warning("⚠️ Gemini API: Key found but model initialization failed")
        else:
            st.error("❌ Gemini API: Not configured")
            st.info("💡 Add GEMINI_API_KEY to your Streamlit secrets")
        
        if status["newsapi_available"]:
            st.success("✅ NewsAPI: Connected")
        else:
            st.error("❌ NewsAPI: Not configured")
            st.info("💡 Add NEWSAPI_KEY to your Streamlit secrets")
        
        # Instructions for setting up secrets
        with st.expander("📋 How to set up API keys in Streamlit Cloud"):
            st.markdown("""
            **Step 1:** Go to your Streamlit Cloud app dashboard
            
            **Step 2:** Click on your app, then go to Settings
            
            **Step 3:** In the Secrets section, add:
            ```toml
            GEMINI_API_KEY = "your_gemini_api_key_here"
            NEWSAPI_KEY = "your_newsapi_key_here"
            ```
            
            **Step 4:** Save and restart your app
            
            **For local development, create a .env file:**
            ```
            GEMINI_API_KEY=your_gemini_api_key_here
            NEWSAPI_KEY=your_newsapi_key_here
            ```
            """)
            
    except ImportError:
        # If streamlit is not available, just pass
        pass