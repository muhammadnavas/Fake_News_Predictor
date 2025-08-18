import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

# Ensure correct key is loaded
GEMINI = os.getenv("GEMINI")
if not GEMINI:
    raise ValueError("❌ GEMINI not found in .env")

# Strip hidden spaces/newlines
GEMINI = GEMINI.strip()

# Initialize Gemini client
genai.configure(api_key=GEMINI)
model = genai.GenerativeModel("gemini-2.0-flash")

# RAG-enhanced Gemini analysis
def rag_enhanced_gemini_analysis(news_text: str, relevant_facts: List[Dict]) -> str:
    """Enhance Gemini analysis with RAG-retrieved facts"""
    
    facts_context = ""
    if relevant_facts:
        facts_context = "\n\n**RETRIEVED KNOWLEDGE BASE FACTS:**\n"
        for i, fact in enumerate(relevant_facts[:3], 1):
            facts_context += f"{i}. {fact['content']} (Similarity: {fact['similarity']:.2f})\n"
            facts_context += f"   Sources: {', '.join(fact['sources'])}\n"
    
    enhanced_prompt = f"""
    As an expert fact-checker and news analyst with access to a knowledge base, provide a comprehensive analysis of this text:
    
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
    
    6. **OVERALL ASSESSMENT**: Provide a final verdict with reasoning, considering both the text analysis and knowledge base facts.
    
    Format your response clearly with headers and bullet points where appropriate.
    """
    
    try:
        response = model.generate_content(enhanced_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error in RAG-enhanced analysis: {e}"


def standard_gemini_analysis(news_text: str) -> str:
    """Standard Gemini analysis without RAG enhancement"""
    enhanced_prompt = f"""
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
    
    Format your response clearly with headers and bullet points where appropriate.
    """
    
    try:
        response = model.generate_content(enhanced_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error in Gemini analysis: {e}"
