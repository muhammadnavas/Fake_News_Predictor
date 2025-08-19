import os
import streamlit as st

# Try dotenv first (local dev), then Streamlit secrets (cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_api_key(key_names):
    """Get API key from multiple possible sources and names."""
    if isinstance(key_names, str):
        key_names = [key_names]
    
    # Try environment variables
    for key_name in key_names:
        key = os.getenv(key_name)
        if key:
            return key.strip()
    
    # Try Streamlit secrets
    try:
        for key_name in key_names:
            key = st.secrets.get(key_name, None)
            if key:
                return key.strip()
    except Exception:
        pass
    
    return None

def check_required_api_keys():
    """Check all required API keys and return status."""
    # Define all possible names for each API key
    api_key_mappings = {
        "GEMINI": ["GEMINI_API_KEY", "GEMINI_API"],
        "NEWSAPI": ["NEWSAPI_KEY", "NEWS_API_KEY", "NEWSAPI"],
        "GNEWS": ["GNEWS_KEY", "GNEWS_API_KEY"],
        "CURRENTS": ["CURRENTS_KEY", "CURRENTS_API_KEY"],
        "CONTEXTUAL_WEB": ["ContextualWeb_KEY", "CONTEXTUAL_WEB_KEY"],
        "GOOGLE_FACTCHECK": ["GOOGLE_FACTCHECK", "GOOGLE_FACTCHECK_KEY"]
    }
    
    results = {}
    missing_keys = []
    
    for service, possible_names in api_key_mappings.items():
        key = get_api_key(possible_names)
        results[service] = key is not None
        if not key and service in ["GEMINI", "NEWSAPI"]:  # Only these are required
            missing_keys.extend(possible_names)
    
    return results, missing_keys

def display_api_status():
    """Display comprehensive API key status."""
    st.subheader("🔧 API Configuration Status")
    
    results, missing_keys = check_required_api_keys()
    
    # Show status for each service
    services = {
        "GEMINI": "Gemini AI Analysis",
        "NEWSAPI": "News API",
        "GNEWS": "GNews API (Optional)",
        "CURRENTS": "Currents API (Optional)",
        "CONTEXTUAL_WEB": "ContextualWeb API (Optional)",
        "GOOGLE_FACTCHECK": "Google Fact Check API (Optional)"
    }
    
    for service, description in services.items():
        if results.get(service, False):
            st.success(f"✅ {description}: Connected")
        else:
            if service in ["GEMINI", "NEWSAPI"]:
                st.error(f"❌ {description}: Missing (Required)")
            else:
                st.warning(f"⚠️ {description}: Missing (Optional)")
    
    # Show debugging info
    with st.expander("🔍 Debug Information"):
        st.code(f"""
Environment Variables Found:
GEMINI_API: {os.getenv('GEMINI_API', 'Not found')}
GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY', 'Not found')}
NEWSAPI_KEY: {os.getenv('NEWSAPI_KEY', 'Not found')}
NEWS_API_KEY: {os.getenv('NEWS_API_KEY', 'Not found')}

Streamlit Secrets Status: {_check_streamlit_secrets()}
        """)
    
    # Instructions
    with st.expander("📋 Setup Instructions"):
        st.markdown("""
        **For Streamlit Cloud (Recommended variable names):**
        ```toml
        GEMINI_API_KEY = "your_gemini_key"
        NEWSAPI_KEY = "your_newsapi_key"
        ```
        
        **For local development (.env file):**
        ```env
        GEMINI_API_KEY=your_gemini_key
        NEWSAPI_KEY=your_newsapi_key
        ```
        
        **Alternative names also supported:**
        - Gemini: GEMINI_API or GEMINI_API_KEY
        - NewsAPI: NEWSAPI_KEY, NEWS_API_KEY, or NEWSAPI
        """)
    
    return len(missing_keys) == 0

def _check_streamlit_secrets():
    """Check if Streamlit secrets are available."""
    try:
        return "Available" if hasattr(st, 'secrets') else "Not available"
    except:
        return "Not available"

# Export the key values for backward compatibility
GEMINI_KEY = get_api_key(["GEMINI_API_KEY", "GEMINI_API"])
NEWSAPI_KEY = get_api_key(["NEWSAPI_KEY", "NEWS_API_KEY", "NEWSAPI"])

# Print status for debugging
if __name__ == "__main__":
    results, missing = check_required_api_keys()
    print("API Key Status:")
    for service, status in results.items():
        print(f"{service}: {'✅' if status else '❌'}")
    
    if missing:
        print(f"\n⚠️ Missing required keys: {missing}")
    else:
        print("\n✅ All required API keys found!")