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
    key = os.getenv("GEMINI_API")  # Use GEMINI_API_KEY for consistency
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API", None)
        except Exception:
            key = None

    if not key:
        print("⚠️ GEMINI API key not found. Gemini analysis will be disabled.")
        return None

    return key.strip()


# --- Initialize Gemini ---
GEMINI_KEY = get_gemini_key()
if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        print(f"⚠️ Failed to initialize Gemini: {e}")
        model = None
else:
    model = None


# --- RAG-enhanced Gemini analysis ---
def rag_enhanced_gemini_analysis(news_text: str, relevant_facts: List[Dict]) -> str:
    """Enhance Gemini analysis with RAG-retrieved facts."""
    if not model:
        return "⚠️ Gemini analysis unavailable (no API key configured)."

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
        return "⚠️ Gemini analysis unavailable (no API key configured)."

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
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error in Gemini analysis: {str(e)}"
