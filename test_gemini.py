import os
import google.generativeai as genai
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

def run_tests():
    # Load API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API")
    
    if not api_key:
        print("[--] Error: GEMINI_API_KEY not found in environment variables or .env file.")
        print("Please add GEMINI_API_KEY=your_key to your .env file.")
        return

    # Configure Gemini
    genai.configure(api_key=api_key)

    # List of known models that typically have a free tier on Google AI Studio
    free_models = [
        "gemini-2.5-flash",       # Referenced in your app
        "gemini-2.0-flash",       # Newest stable flash
        "gemini-2.0-flash-lite-preview-02-05", # Fast, lightweight
        "gemini-1.5-flash",       # Standard 1.5 flash
        "gemini-1.5-flash-8b",    # 1.5 flash 8B parameters
        "gemini-1.5-pro",         # Pro model (has free tier limits)
    ]

    test_prompt = "Explain why the sky is blue in exactly one short sentence."

    print("[*] Starting Gemini API Model Tests...\n")
    print(f"Using API Key: {api_key[:5]}...{api_key[-4:] if len(api_key) > 9 else ''}")
    print("-" * 60)

    for model_name in free_models:
        print(f"Testing Model: {model_name}")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(test_prompt)
            print("[OK] Success!")
            print(f"Response: {response.text.strip()}")
        except Exception as e:
            print("[--] Failed!")
            print(f"Error: {e}")
        print("-" * 60)

    print("\n[*] Testing Complete!")

if __name__ == "__main__":
    run_tests()
