# content_detector.py
import re
import nltk
import hashlib
import streamlit as st
from typing import Dict, Tuple, List

# Download required NLTK data (safe to ignore if already present)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception:
    pass


def detect_content_type(text: str) -> Dict:
    """
    Detect if the input text is likely to be news content.
    Returns a dictionary with content type analysis.
    """
    if not text or len(text.strip()) < 10:
        return {
            "is_news": False,
            "content_type": "insufficient_text",
            "confidence": 0.0,
            "reason": "Text too short to analyze",
            "suggestions": ["Please enter at least 10 characters of news content"],
            "text_hash": "",
            "validation_status": "invalid"
        }

    text = text.strip()
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]  # Short hash for tracking

    # Initialize scoring
    news_score = 0
    reasons = []
    content_type = "unknown"

    # 1. Personal statements (negative)
    personal_patterns = [
        r'\b(my name is|i am|i\'m|hello|hi there|dear)\b',
        r'\b(i think|i believe|in my opinion|personally)\b',
        r'\b(sincerely|regards|thank you|thanks|cheers)\b',
        r'\b(how are you|nice to meet|pleased to meet)\b'
    ]
    personal_matches = sum(len(re.findall(p, text.lower())) for p in personal_patterns)
    if personal_matches > 0:
        news_score -= personal_matches * 15
        reasons.append(f"Contains {personal_matches} personal statement patterns")
        content_type = "personal_statement"

    # 2. Casual text
    casual_patterns = [
        r'^\s*(hello|hi|hey|yo)\s*[,!.]?\s*$',
        r'^\s*(ok|okay|yes|no|maybe)\s*[,!.]?\s*$',
        r'^\s*(thanks?|thx|ty)\s*[,!.]?\s*$',
        r'\b(lol|omg|wtf|btw|imo|imho)\b'
    ]
    casual_matches = sum(1 for p in casual_patterns if re.search(p, text.lower()))
    if casual_matches > 0 or (len(text.split()) < 5 and any(w in text.lower() for w in ['hello', 'hi', 'hey', 'thanks', 'ok', 'yes', 'no'])):
        news_score -= 25
        reasons.append("Appears to be casual conversation")
        content_type = "casual_text"

    # 3. Placeholder/test text
    test_patterns = [
        r'\b(test|testing|hello world|sample text|dummy)\b',
        r'\b(lorem ipsum|placeholder|example text)\b',
        r'^\s*(a|an|the)\s+\w+\s+(is|are|was|were)\s+\w+\s*[.!]?\s*$',
        r'^\s*this\s+is\s+(a|an)?\s*\w+\s*[.!]?\s*$'
    ]
    test_matches = sum(1 for p in test_patterns if re.search(p, text.lower()))
    if test_matches > 0:
        news_score -= 30
        reasons.append("Contains test or placeholder patterns")
        content_type = "test_text"

    # 4. News-related terms
    news_patterns = [
        r'\b(breaking|urgent|alert|developing|exclusive)\b',
        r'\b(reported|according to|sources say|officials|spokesperson|authorities)\b',
        r'\b(yesterday|today|this morning|last night|earlier|recently)\b',
        r'\b(government|police|court|parliament|congress|minister|president|mayor)\b',
        r'\b(announced|declared|confirmed|denied|stated|revealed|disclosed)\b',
        r'\b(investigation|incident|accident|crisis|emergency|outbreak)\b',
        r'\b(arrested|charged|convicted|sentenced|acquitted)\b',
        r'\b(killed|injured|died|wounded|hospitalized)\b',
        r'\b(billion|million|percent|dollars|euros|pounds)\b',
        r'\b(company|corporation|industry|market|economy|financial)\b'
    ]
    news_pattern_count = sum(len(re.findall(p, text.lower())) for p in news_patterns)
    if news_pattern_count > 0:
        news_score += news_pattern_count * 8
        reasons.append(f"Contains {news_pattern_count} news-related terms")
        if content_type == "unknown":
            content_type = "news_like"

    # 5. Proper nouns
    proper_noun_count = 0
    try:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        proper_nouns = [w for w, pos in pos_tags if pos in ['NNP', 'NNPS']]
        proper_noun_count = len(proper_nouns)
        if proper_noun_count >= 2:
            news_score += min(proper_noun_count * 4, 20)
            reasons.append(f"Contains {proper_noun_count} proper nouns")
    except Exception:
        # Fallback capitalization check
        words = text.split()
        capitalized = [w for w in words if w and w[0].isupper()]
        proper_noun_count = len(capitalized)
        if proper_noun_count >= 2:
            news_score += min(proper_noun_count * 3, 15)
            reasons.append(f"Contains {proper_noun_count} capitalized words")

    # 6. Length and structure
    word_count = len(text.split())
    sentence_count = len([s for s in text.split('.') if s.strip()])
    if word_count >= 50:
        news_score += 20
        reasons.append("Good length for news content")
    elif word_count >= 20:
        news_score += 12
        reasons.append("Adequate length for news content")
    elif word_count >= 10:
        news_score += 5
        reasons.append("Moderate length content")
    else:
        news_score -= 10
        reasons.append("Very short content")

    # 7. Questions
    question_count = text.count('?')
    if question_count > 2:
        news_score -= 12
        reasons.append("Contains multiple questions")
        if content_type == "unknown":
            content_type = "question_text"
    elif question_count == 1:
        news_score -= 5
        reasons.append("Contains a question")

    # 8. Formal structure
    if sentence_count >= 3:
        news_score += 8
        reasons.append("Multi-sentence structure")
    elif sentence_count >= 2:
        news_score += 4
        reasons.append("Basic sentence structure")

    # 9. Quotes
    quote_count = sum(len(re.findall(p, text)) for p in [r'"[^"]{10,}"', r"'[^']{10,}'"])
    if quote_count > 0:
        news_score += quote_count * 6
        reasons.append(f"Contains {quote_count} quotations")

    # 10. Time/date
    time_patterns = [
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\b',
        r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b'
    ]
    time_matches = sum(len(re.findall(p, text.lower())) for p in time_patterns)
    if time_matches > 0:
        news_score += min(time_matches * 5, 15)
        reasons.append(f"Contains {time_matches} time/date references")

    # Final score
    news_score = max(0, min(100, news_score))
    confidence = news_score / 100

    is_news = (
        news_score >= 35 and
        news_pattern_count > 0 and
        personal_matches == 0 and
        casual_matches == 0 and
        test_matches == 0 and
        word_count >= 8
    )

    if content_type == "unknown":
        content_type = "potential_news" if is_news else "non_news"

    # Suggestions
    suggestions = []
    if not is_news:
        suggestions = [
            "Enter actual news headlines or articles for analysis",
            "Examples: 'Breaking: Government announces new policy affecting citizens'",
            "Include specific details like dates, locations, and official sources"
        ]
        if content_type == "personal_statement":
            suggestions.append("Avoid personal introductions or opinions")
        elif content_type == "casual_text":
            suggestions.append("Avoid casual greetings or short responses")
        elif content_type == "test_text":
            suggestions.append("Avoid test phrases or placeholder text")
        elif content_type == "question_text":
            suggestions.append("Focus on factual statements rather than questions")
        elif word_count < 10:
            suggestions.append("Provide more detailed text for accurate analysis")

    return {
        "is_news": is_news,
        "content_type": content_type,
        "confidence": confidence,
        "news_score": news_score,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "proper_noun_count": proper_noun_count,
        "news_pattern_count": news_pattern_count,
        "personal_matches": personal_matches,
        "reasons": reasons,
        "suggestions": suggestions,
        "text_hash": text_hash,
        "validation_status": "valid" if is_news else "invalid"
    }


def validate_news_content(text: str) -> Tuple[bool, str, List[str]]:
    analysis = detect_content_type(text)
    if analysis["is_news"]:
        return True, f"✅ Content appears to be news-like (confidence: {analysis['confidence']:.1%})", []
    else:
        return False, f"⚠️ Not News Content - Detected as: {analysis['content_type'].replace('_', ' ').title()}", analysis["suggestions"]


def add_content_validation_to_streamlit(text_input: str, force_clear_state: bool = False) -> bool:
    if force_clear_state:
        reset_content_validation_state()

    if not text_input or not text_input.strip():
        return True

    current_hash = hashlib.md5(text_input.encode()).hexdigest()[:8]
    validation_key = f"content_validation_{current_hash}"

    if validation_key not in st.session_state or force_clear_state:
        is_valid, message, suggestions = validate_news_content(text_input)
        st.session_state[validation_key] = {
            "is_valid": is_valid,
            "message": message,
            "suggestions": suggestions,
            "text_hash": current_hash
        }

    result = st.session_state[validation_key]
    if not result["is_valid"]:
        st.warning(result["message"])
        if result["suggestions"]:
            st.info("**Suggestions:**")
            for s in result["suggestions"]:
                st.write(f"• {s}")
        return False
    else:
        st.success(result["message"])
        return True


def get_detailed_content_analysis(text: str) -> Dict:
    analysis = detect_content_type(text)
    analysis["text_preview"] = text[:100] + "..." if len(text) > 100 else text
    analysis["character_count"] = len(text)
    analysis["confidence_level"] = (
        "High" if analysis["confidence"] >= 0.7 else
        "Medium" if analysis["confidence"] >= 0.4 else
        "Low"
    )
    analysis["pattern_summary"] = {
        "news_indicators": analysis["news_pattern_count"],
        "personal_indicators": analysis["personal_matches"],
        "structural_quality": (
            "Good" if analysis["sentence_count"] >= 3 else
            "Fair" if analysis["sentence_count"] >= 2 else
            "Poor"
        )
    }
    return analysis


def reset_content_validation_state():
    for key in list(st.session_state.keys()):
        if key.startswith('content_validation_'):
            del st.session_state[key]


def is_content_news_suitable(text: str, bypass_validation: bool = False) -> Tuple[bool, Dict]:
    if bypass_validation:
        return True, {"bypassed": True, "warning": "Validation bypassed by user"}
    analysis = detect_content_type(text)
    return analysis["is_news"], analysis


def get_content_validation_message(text: str) -> str:
    analysis = detect_content_type(text)
    if analysis["is_news"]:
        return f"✅ **Valid News Content** (Confidence: {analysis['confidence']:.1%})"
    else:
        return f"⚠️ **Not News Content** - Detected as: {analysis['content_type'].replace('_', ' ').title()}"


if __name__ == "__main__":
    test_cases = [
        "Breaking: Government announces new economic policy affecting millions",
        "Hello, how are you today?",
        "President Biden met with world leaders to discuss climate change initiatives",
        "Lorem ipsum dolor sit amet",
        "Police reported a major traffic accident on Highway 101 this morning",
        "I think this is a good idea",
        "Local authorities confirm three people were injured in the incident",
        "My name is John and I am testing this system",
        "Test message",
        "Officials announced yesterday that the new healthcare policy will take effect next month, according to government spokesperson Sarah Johnson."
    ]
    print("Content Type Detection Test Results:")
    print("=" * 80)
    for i, text in enumerate(test_cases, 1):
        a = detect_content_type(text)
        print(f"\n{i}. Text: {text}")
        print(f"   Is News: {a['is_news']}")
        print(f"   Type: {a['content_type']}")
        print(f"   Confidence: {a['confidence']:.1%}")
        print(f"   Score: {a['news_score']}/100")
        print(f"   News patterns: {a['news_pattern_count']}")
        print(f"   Personal matches: {a['personal_matches']}")
        print(f"   Word count: {a['word_count']}")
        print(f"   Reasons: {'; '.join(a['reasons'][:3])}")
