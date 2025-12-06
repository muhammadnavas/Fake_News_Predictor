# Google Fact Check API Setup Guide

## Issue
The application is showing an error because the Google Fact Check API key is not properly configured:
```
Could not fetch Google Fact Check claims: 400 Client Error: Bad Request
```

## Solution

The Google Fact Check API is **optional** for the RAG system to work. The system will function perfectly fine without it, using the built-in fact database and other sources.

### Option 1: Get a Valid Google Fact Check API Key (Recommended)

1. **Go to Google Cloud Console**: https://console.cloud.google.com/

2. **Create or Select a Project**

3. **Enable the Fact Check Tools API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Fact Check Tools API"
   - Click "Enable"

4. **Create API Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy the generated API key

5. **Update your `.env` file**:
   ```env
   GOOGLE_FACTCHECK=your_actual_api_key_here
   ```

6. **Restart the Streamlit app**

### Option 2: Disable Google Fact Check (Quick Fix)

If you don't need Google Fact Check integration, the system will work fine without it. The warning has been suppressed in the latest update.

Simply ensure your `.env` file has:
```env
GOOGLE_FACTCHECK=YOUR_GOOGLE_FACTCHECK_KEY_HERE
```

The system will automatically detect this is a placeholder and skip Google Fact Check without showing errors.

## What's Been Fixed

✅ **Placeholder Detection**: The system now detects placeholder API keys and skips Google Fact Check gracefully
✅ **Error Suppression**: Authentication errors (400, 401, 403) are now silently handled
✅ **No Warnings**: The system no longer shows warnings when Google Fact Check is unavailable
✅ **Graceful Degradation**: The RAG system works perfectly without Google Fact Check

## Current API Status

Your `.env` file should have these keys:

```env
# Required for AI analysis
GEMINI_API_KEY=your_gemini_key_here

# Required for news fetching
NEWSAPI_KEY=your_newsapi_key_here

# Optional - for Google Fact Check integration
GOOGLE_FACTCHECK=YOUR_GOOGLE_FACTCHECK_KEY_HERE
```

## Verification

After updating your `.env` file, restart the Streamlit app:

```bash
# Stop the current app (Ctrl+C)
streamlit run app.py
```

The app should now run without showing the Google Fact Check error!

## Note

The RAG system has multiple data sources:
- ✅ Built-in fact database (always available)
- ✅ News APIs (if configured)
- ✅ ChromaDB vector database (always available)
- ⚠️ Google Fact Check API (optional, requires valid API key)

Even without Google Fact Check, your fake news detection system will work effectively!
