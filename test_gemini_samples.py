import os
from dotenv import load_dotenv
load_dotenv(override=True)

import google.generativeai as genai
from ai_analysis import standard_gemini_analysis

samples = [
    "The sky is blue and water is wet.",
    "Aliens landed in Times Square yesterday and abducted the mayor.",
    "The Federal Reserve announced a 0.25% interest rate hike today to combat inflation.",
    "Drinking bleach cures all known viral infections, according to a new Facebook post.",
    "Apple Inc. released their quarterly earnings report, beating analyst expectations by 5%.",
    "The Cockroach Janta Party has demanded ₹1 crore compensation for NEET paper leak-linked suicide cases. In an open letter to Prime Minister Modi, CJP founder Abhijeet Dipke noted that 11 NEET aspirants died by suicide in the last one week, with five of them in the last 48 hours."
]

for i, sample in enumerate(samples):
    print(f"\n--- Sample {i+1} ---")
    print(f"Text: {sample}")
    try:
        res = standard_gemini_analysis(sample)
        print(f"Result: {res}")
    except Exception as e:
        print(f"Error: {e}")
