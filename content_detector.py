# content_detector.py
import re
import nltk
from typing import Dict, Tuple, List, Optional
import streamlit as st
import hashlib

# Download required NLTK data (run once)
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

def detect_content_type(text: str) -> Dict:
    """
    Detect if the input text is likely to be news content
    Returns a dictionary with content type analysis
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
    max_score = 100
    reasons = []
    content_type = "unknown"
    
    # 1. Check for personal statements (negative indicators)
    personal_patterns = [
        r'\b(my name is|i am|i\'m|hello|hi there|dear)\b',
        r'\b(i think|i believe|in my opinion|personally)\b',
        r'\b(sincerely|regards|thank you|thanks|cheers)\b',
        r'\b(how are you|nice to meet|pleased to meet)\b'
    ]
    
    personal_matches = 0
    for pattern in personal_patterns:
        matches = len(re.findall(pattern, text.lower()))
        if matches > 0:
            personal_matches += matches
            news_score -= matches * 15
    
    if personal_matches > 0:
        reasons.append(f"Contains {personal_matches} personal statement patterns")
        content_type = "personal_statement"
    
    # 2. Check for greeting/casual text patterns
    casual_patterns = [
        r'^\s*(hello|hi|hey|yo)\s*[,!.]?\s*$',
        r'^\s*(ok|okay|yes|no|maybe)\s*[,!.]?\s*$',
        r'^\s*(thanks?|thx|ty)\s*[,!.]?\s*$',
        r'\b(lol|omg|wtf|btw|imo|imho)\b'
    ]
    
    casual_matches = 0
    for pattern in casual_patterns:
        if re.search(pattern, text.lower()):
            casual_matches += 1
            news_score -= 25
    
    if casual_matches > 0 or (len(text.split()) < 5 and any(word in text.lower() for word in ['hello', 'hi', 'hey', 'thanks', 'ok', 'yes', 'no'])):
        reasons.append("Appears to be casual conversation")
        content_type = "casual_text"
    
    # 3. Check for test/placeholder content
    test_patterns = [
        r'\b(test|testing|hello world|sample text|dummy)\b',
        r'\b(lorem ipsum|placeholder|example text)\b',
        r'^\s*(a|an|the)\s+\w+\s+(is|are|was|were)\s+\w+\s*[.!]?\s*$',  # Simple statements
        r'^\s*this\s+is\s+(a|an)?\s*\w+\s*[.!]?\s*$'
    ]
    
    test_matches = 0
    for pattern in test_patterns:
        if re.search(pattern, text.lower()):
            test_matches += 1
            news_score -= 30
    
    if test_matches > 0:
        reasons.append("Contains test or placeholder patterns")
        content_type = "test_text"
    
    # 4. Check for news-like patterns (positive indicators)
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
    
    news_pattern_count = 0
    for pattern in news_patterns:
        matches = len(re.findall(pattern, text.lower()))
        if matches > 0:
            news_score += matches * 8
            news_pattern_count += matches
    
    if news_pattern_count > 0:
        reasons.append(f"Contains {news_pattern_count} news-related terms")
        if content_type == "unknown":
            content_type = "news_like"
    
    # 5. Check for proper nouns (locations, organizations, people)
    proper_noun_count = 0
    try:
        import nltk
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        proper_nouns = [word for word, pos in pos_tags if pos in ['NNP', 'NNPS']]
        proper_noun_count = len(proper_nouns)
        
        if proper_noun_count >= 2:
            news_score += min(proper_noun_count * 4, 20)  # Cap at 20 points
            reasons.append(f"Contains {proper_noun_count} proper nouns")
    except Exception:
        # Fallback: simple capitalization check
        import string
        words = text.split()
        capitalized_words = [w for w in words if w and w[0].isupper() and len(w) > 2 and w not in string.punctuation]
        proper_noun_count = len(capitalized_words)
        if proper_noun_count >= 2:
            news_score += min(proper_noun_count * 3, 15)
            reasons.append(f"Contains {proper_noun_count} capitalized words")
    
    # 6. Check text length and structure
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
    
    # 7. Check for question patterns (usually not news)
    question_count = text.count('?')
    if question_count > 2:
        news_score -= 12
        reasons.append("Contains multiple questions")
        if content_type == "unknown":
            content_type = "question_text"
    elif question_count == 1:
        news_score -= 5
        reasons.append("Contains a question")
    
    # 8. Check for formal news structure
    if sentence_count >= 3:
        news_score += 8
        reasons.append("Multi-sentence structure")
    elif sentence_count >= 2:
        news_score += 4
        reasons.append("Basic sentence structure")
    
    # 9. Check for quotes (common in news)
    quote_patterns = [r'"[^"]{10,}"', r"'[^']{10,}'"]
    quote_count = sum(len(re.findall(pattern, text)) for pattern in quote_patterns)
    if quote_count > 0:
        news_score += quote_count * 6
        reasons.append(f"Contains {quote_count} quotations")
    
    # 10. Check for time/date references
    time_patterns = [
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\b',
        r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b'
    ]
    
    time_matches = sum(len(re.findall(pattern, text.lower())) for pattern in time_patterns)
    if time_matches > 0:
        news_score += min(time_matches * 5, 15)
        reasons.append(f"Contains {time_matches} time/date references")
    
    # Calculate final confidence and determination
    news_score = max(0, min(100, news_score))
    confidence = news_score / 100
    
    # Determine if it's news based on multiple factors
    is_news = (
        news_score >= 35 and  # Base threshold
        news_pattern_count > 0 and  # Must have news terms
        personal_matches == 0 and  # No personal statements
        casual_matches == 0 and  # No casual patterns
        test_matches == 0 and  # No test patterns
        word_count >= 8  # Minimum word count
    )
    
    # Adjust content type based on final determination
    if content_type == "unknown":
        if is_news:
            content_type = "potential_news"
        else:
            content_type = "non_news"
    
    # Generate suggestions based on analysis
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
    
    validation_status = "valid" if is_news else "invalid"
    
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
        "validation_status": validation_status
    }


def validate_news_content(text: str) -> Tuple[bool, str, List[str]]:
    """
    Validate if content is suitable for news analysis
    Returns: (is_valid, message, suggestions)
    """
    analysis = detect_content_type(text)
    
    if analysis["is_news"]:
        return True, f"✅ Content appears to be news-like (confidence: {analysis['confidence']:.1%})", []
    else:
        message = f"⚠️ This doesn't appear to be news content. Detected as: {analysis['content_type'].replace('_', ' ').title()}"
        return False, message, analysis["suggestions"]


def add_content_validation_to_streamlit(text_input: str, force_clear_state: bool = False) -> bool:
    """
    Add content validation to Streamlit interface with state management
    Returns True if content is valid for analysis, False otherwise
    """
    if force_clear_state:
        # Clear any previous validation state
        for key in list(st.session_state.keys()):
            if key.startswith('content_validation_'):
                del st.session_state[key]
    
    if not text_input or not text_input.strip():
        return True  # Don't validate empty input
    
    # Generate a hash for the current text to track changes
    current_hash = hashlib.md5(text_input.encode()).hexdigest()[:8]
    
    # Check if we've already validated this exact text
    validation_key = f"content_validation_{current_hash}"
    
    if validation_key not in st.session_state or force_clear_state:
        # Perform fresh validation
        is_valid, message, suggestions = validate_news_content(text_input)
        st.session_state[validation_key] = {
            "is_valid": is_valid,
            "message": message,
            "suggestions": suggestions,
            "text_hash": current_hash
        }
    
    # Get validation results
    validation_result = st.session_state[validation_key]
    
    if not validation_result["is_valid"]:
        st.warning(validation_result["message"])
        if validation_result["suggestions"]:
            st.info("**Suggestions:**")
            for suggestion in validation_result["suggestions"]:
                st.write(f"• {suggestion}")
        return False
    else:
        st.success(validation_result["message"])
        return True


def get_detailed_content_analysis(text: str) -> Dict:
    """
    Get detailed analysis for debugging/reporting purposes
    """
    analysis = detect_content_type(text)
    
    # Additional analysis
    analysis["text_preview"] = text[:100] + "..." if len(text) > 100 else text
    analysis["character_count"] = len(text)
    
    # Categorize confidence levels
    if analysis["confidence"] >= 0.7:
        analysis["confidence_level"] = "High"
    elif analysis["confidence"] >= 0.4:
        analysis["confidence_level"] = "Medium"
    else:
        analysis["confidence_level"] = "Low"
    
    # Add pattern analysis summary
    analysis["pattern_summary"] = {
        "news_indicators": analysis["news_pattern_count"],
        "personal_indicators": analysis["personal_matches"],
        "structural_quality": "Good" if analysis["sentence_count"] >= 3 else "Fair" if analysis["sentence_count"] >= 2 else "Poor"
    }
    
    return analysis


def reset_content_validation_state():
    """
    Reset all content validation state in Streamlit session
    """
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith('content_validation_')]
    for key in keys_to_remove:
        del st.session_state[key]


def is_content_news_suitable(text: str, bypass_validation: bool = False) -> Tuple[bool, Dict]:
    """
    Check if content is suitable for news analysis
    Returns: (is_suitable, analysis_dict)
    """
    if bypass_validation:
        return True, {"bypassed": True, "warning": "Validation bypassed by user"}
    
    analysis = detect_content_type(text)
    return analysis["is_news"], analysis


def get_content_validation_message(text: str) -> str:
    """
    Get a user-friendly validation message for the content
    """
    analysis = detect_content_type(text)
    
    if analysis["is_news"]:
        return f"✅ **Valid News Content** (Confidence: {analysis['confidence']:.1%})"
    else:
        content_type_friendly = analysis['content_type'].replace('_', ' ').title()
        return f"⚠️ **Not News Content** - Detected as: {content_type_friendly}"


# Example usage and testing
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
        analysis = detect_content_type(text)
        print(f"\n{i}. Text: '{text}'")
        print(f"   Is News: {analysis['is_news']}")
        print(f"   Type: {analysis['content_type']}")
        print(f"   Confidence: {analysis['confidence']:.1%}")
        print(f"   Score: {analysis['news_score']}/100")
        print(f"   News patterns: {analysis['news_pattern_count']}")
        print(f"   Personal matches: {analysis['personal_matches']}")
        print(f"   Word count: {analysis['word_count']}")
        print(f"   Reasons: {'; '.join(analysis['reasons'][:3])}")  # First 3 reasons