import os
from datetime import datetime

import streamlit as st
if not st.session_state.get("_page_configured", False):
    try:
        st.set_page_config(
            page_title="🔍 Fake News Predictor",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass
    st.session_state["_page_configured"] = True


import numpy as np
import pandas as pd
from dotenv import load_dotenv

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import nltk

def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

ensure_nltk_resources()

from fetch_news import get_all_news, comprehensive_news_check
from ml_analysis import load_all_models, analyze_with_all_models
from ai_analysis import (
    rag_enhanced_gemini_analysis,
    standard_gemini_analysis,
)
from content_detector import (
    add_content_validation_to_streamlit,
    detect_content_type,
    get_detailed_content_analysis,
    reset_content_validation_state,
    is_content_news_suitable,
    get_content_validation_message,
    validate_news_content,
)


load_dotenv()


def _get_secret(name: str):
    try:
        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        return None

def safe_columns(spec, gap="small"):
    try:
        return st.columns(spec, gap=gap)
    except TypeError:
        return st.columns(spec)

GEMINI_API_KEY = _get_secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
NEWSAPI_KEY    = _get_secret("NEWSAPI_KEY")    or os.getenv("NEWSAPI_KEY")
GNEWS_KEY      = _get_secret("GNEWS_KEY")      or os.getenv("GNEWS_KEY")
CURRENTS_KEY   = _get_secret("CURRENTS_KEY")   or os.getenv("CURRENTS_KEY")

required_keys = {"GEMINI_API_KEY": GEMINI_API_KEY, "NEWSAPI_KEY": NEWSAPI_KEY}
missing_keys = [key for key, value in required_keys.items() if not value]
if missing_keys:
    st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
    st.info("Please set the required API keys in your .env file")
    st.stop()


st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
        /* Reduce padding specifically for the main title */
        h1 {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
            padding-bottom: 0rem !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Fake News Predictor")
st.markdown("**Multi-API News Verification** • **ML-Powered Assessment** • **AI-Powered Assessment**")
st.markdown("---")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📡 Latest News")
    topic = st.text_input("Enter topic for news:", value="technology")

    if st.button("🔄 Fetch Latest News"):
        with st.spinner("Fetching from multiple sources..."):
            try:
                articles = get_all_news(keyword=topic, max_articles=30)
                if articles:
                    st.success(f"Found {len(articles)} articles from multiple sources")
                    st.session_state.articles = articles[:10]
                else:
                    st.warning("No articles found")
            except Exception as e:
                st.error(f"Error: {e}")

    if "articles" in st.session_state:
        st.markdown("### 📰 Recent Articles")
        for i, article in enumerate(st.session_state.articles):
            title = (article.get("title") or "Untitled").strip()
            short_title = (title[:40] + "...") if len(title) > 40 else title
            with st.expander(short_title):
                st.write(f"**Source:** {article.get('source', 'Unknown')}")
                desc = (article.get("description") or "N/A").strip()
                st.write(f"**Description:** {desc[:100]}{'...' if len(desc) > 100 else ''}")
                if st.button("Use Article", key=f"use_{i}"):
                    content = title
                    if article.get("description"):
                        content += " " + article["description"]
                    st.session_state.selected_text = content
                    reset_content_validation_state()
                    st.rerun()

    st.markdown("---")
    st.markdown(
        f"""
**API Status:**
- NewsAPI: {'✅' if NEWSAPI_KEY else '❌'}
- GNews: {'✅' if GNEWS_KEY else '❌'}
- CurrentsAPI: {'✅' if CURRENTS_KEY else '❌'}
- Gemini AI: {'✅' if GEMINI_API_KEY else '❌'}
"""
    )


# ── Load ML Models ────────────────────────────────────────────────────────────
models, vectorizer = load_all_models()
if models is None or vectorizer is None:
    st.error("❌ Could not load models or vectorizer. Please check 'models/' folder.")


# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("📝 Enter News Text")
st.info(
    "💡 **Tip**: Enter actual news headlines or articles for accurate analysis. "
    "Personal statements or casual text may not be analyzed correctly."
)

if "selected_text" not in st.session_state:
    st.session_state.selected_text = ""

if st.button("🔄 Clear State", help="Clear validation cache for fresh analysis"):
    reset_content_validation_state()
    st.session_state.selected_text = ""
    st.rerun()

input_text = st.text_area(
    "Paste news headline or article text:",
    value=st.session_state.selected_text,
    key="selected_text",
    height=120,
    placeholder="Example: 'Breaking: Government announces new policy...' or 'Local authorities report incident...'",
)


# ── Content Validation ────────────────────────────────────────────────────────
content_analysis = None
analysis_allowed = True

if input_text and input_text.strip():
    content_analysis = detect_content_type(input_text)
    with st.container():
        if content_analysis.get("is_news"):
            st.success(
                f"✅ **Content Validation**: Appears to be news content "
                f"(Confidence: {content_analysis.get('confidence', 0.0):.1%})"
            )
            analysis_allowed = True
        else:
            st.warning("⚠️ **Content Validation**: This doesn't appear to be news content")
            detected_type = (content_analysis.get("content_type") or "unknown").replace("_", " ").title()
            st.error(f"**Detected as**: {detected_type}")
            analysis_allowed = False


# ── Options ───────────────────────────────────────────────────────────────────
check_existence = True
use_gemini = True
advanced_analysis = True


# ── Analyse Button ────────────────────────────────────────────────────────────
if st.button("🚀 Analyse News", type="primary", use_container_width=True):
    if not (input_text and input_text.strip()):
        st.warning("⚠️ Please enter text to analyze")
    elif not analysis_allowed:
        st.error(
            "❌ **Analysis blocked**: Content doesn't appear to be news. "
            "Please provide an actual news headline or article text."
        )
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🔍 Verification", "🤖 ML Models", "🧠 AI Assessment", "📊 Summary"]
        )

        # ── Tab 1: News Verification ──────────────────────────────────────────
        with tab1:
            st.subheader("Multi-Source News Verification")
            verification_results = None
            if check_existence:
                with st.spinner("Checking across NewsAPI, GNews, and CurrentsAPI..."):
                    try:
                        verification_results = comprehensive_news_check(input_text)
                    except Exception as e:
                        st.error(f"Verification error: {e}")

                if verification_results:
                    col_a, col_b, col_c = safe_columns(3)
                    with col_a:
                        st.metric("Sources Found", len(verification_results.get("sources_found", [])))
                    with col_b:
                        st.metric("Total Matches", verification_results.get("total_matches", 0))
                    with col_c:
                        st.metric(
                            "Confidence Score",
                            f"{verification_results.get('confidence_score', 0.0):.1f}%"
                        )

                    st.subheader("API Search Results")
                    for api_name, result in verification_results.get("search_summary", {}).items():
                        if result.get("found"):
                            st.success(f"✅ **{api_name}**: Found {result.get('count', 0)} matching articles")
                        elif result.get("error"):
                            st.error(f"❌ **{api_name}**: {result.get('error')}")
                        else:
                            st.info(f"ℹ️ **{api_name}**: No matches found")

                    if verification_results.get("matched_articles"):
                        st.subheader("📰 Matched Articles")
                        for art in verification_results["matched_articles"][:5]:
                            with st.expander(art.get("title", "Article")[:80]):
                                st.write(f"**Source:** {art.get('source', 'N/A')}")
                                if art.get("url"):
                                    st.write(f"**URL:** {art['url']}")
                                if art.get("publishedAt"):
                                    st.write(f"**Published:** {art['publishedAt']}")
                else:
                    st.info("No verification results to display.")

        # ── Tab 2: ML Models ──────────────────────────────────────────────────
        with tab2:
            st.subheader("🤖 Machine Learning Model Analysis")
            if models:
                try:
                    with st.spinner("Analyzing with all ML models..."):
                        ml_results = analyze_with_all_models(input_text, models, vectorizer)

                    if advanced_analysis and ml_results:
                        model_names, predictions, fake_probs, real_probs = [], [], [], []
                        for model_name, result in ml_results.items():
                            if "error" not in result:
                                model_names.append(model_name)
                                predictions.append(result["prediction"])
                                fake_probs.append(result["fake_probability"])
                                real_probs.append(result["real_probability"])

                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=("Model Predictions", "Fake vs Real Probabilities"),
                            specs=[[{"type": "bar"}], [{"type": "bar"}]],
                        )
                        colors = ["red" if p == "FAKE" else "green" for p in predictions]
                        fig.add_trace(
                            go.Bar(
                                x=model_names,
                                y=[1] * len(model_names),
                                marker_color=colors,
                                text=predictions,
                                textposition="inside",
                                name="Predictions",
                            ),
                            row=1, col=1,
                        )
                        fig.add_trace(
                            go.Bar(x=model_names, y=fake_probs, name="Fake Probability", marker_color="red"),
                            row=2, col=1,
                        )
                        fig.add_trace(
                            go.Bar(x=model_names, y=real_probs, name="Real Probability", marker_color="green"),
                            row=2, col=1,
                        )
                        fig.update_layout(height=700, showlegend=True, title_text="Comprehensive ML Model Analysis")
                        st.plotly_chart(fig, use_container_width=True)

                    for model_name, result in ml_results.items():
                        if "error" not in result:
                            if result["prediction"] == "REAL":
                                st.success(
                                    f"✅ **{model_name}**: {result['prediction']} "
                                    f"(Confidence: {result['confidence']:.2%})"
                                )
                            else:
                                st.error(
                                    f"❌ **{model_name}**: {result['prediction']} "
                                    f"(Confidence: {result['confidence']:.2%})"
                                )
                        else:
                            st.warning(f"⚠️ {model_name}: {result.get('error')}")
                except Exception as e:
                    st.error(f"ML analysis failed: {e}")
            else:
                st.error("No ML models loaded.")

        # ── Tab 3: Gemini AI ──────────────────────────────────────────────────
        with tab3:
            st.subheader("AI-Powered Assessment")
            gemini_analysis = None
            if use_gemini:
                try:
                    with st.spinner("Getting Gemini AI analysis..."):
                        gemini_analysis = standard_gemini_analysis(input_text)

                    st.markdown("### 🤖 Gemini AI Detailed Analysis")
                    st.markdown(gemini_analysis)

                    import re
                    match = re.search(r'RESULT:[\s\*]*(REAL|FAKE)', gemini_analysis or "", re.IGNORECASE)
                    if match:
                        verdict = match.group(1).upper()
                        if verdict == "REAL":
                            st.success("🎯 **Gemini Assessment: REAL NEWS**")
                        else:
                            st.error("🎯 **Gemini Assessment: FAKE NEWS**")
                    else:
                        st.warning("🎯 **Gemini Assessment: INCONCLUSIVE**")
                except Exception as e:
                    st.error(f"Gemini API error: {e}")
            else:
                st.info("Enable Gemini AI Analysis to view AI assessment.")

        # ── Tab 4: Summary ────────────────────────────────────────────────────
        with tab4:
            st.subheader("📊 Summary Report")
            if content_analysis:
                if content_analysis.get("is_news"):
                    st.success(
                        f"✅ Detected as News Content (Confidence: {content_analysis.get('confidence', 0.0):.1%})"
                    )
                else:
                    st.error(
                        f"❌ Not News Content (Detected as: "
                        f"{(content_analysis.get('content_type') or 'unknown').replace('_', ' ').title()})"
                    )

            summary_rows = []

            # News Verification
            if check_existence and "verification_results" in locals() and verification_results:
                if verification_results.get("sources_found"):
                    summary_rows.append([
                        "News Verification", "VERIFIED",
                        f"{len(verification_results['sources_found'])} sources, "
                        f"{verification_results.get('confidence_score', 0.0):.1f}% confidence",
                    ])
                else:
                    summary_rows.append(["News Verification", "NOT FOUND", "No matches across APIs"])

            # ML Models
            if models and "ml_results" in locals():
                valid_results = [r for r in ml_results.values() if "error" not in r]
                real_votes = sum(1 for r in valid_results if r["prediction"] == "REAL")
                fake_votes = len(valid_results) - real_votes
                avg_conf = float(np.mean([r["confidence"] for r in valid_results])) if valid_results else 0.0

                if real_votes > fake_votes:
                    summary_rows.append([
                        "ML Models Consensus", "REAL",
                        f"{real_votes}/{real_votes + fake_votes} models, Avg confidence: {avg_conf:.2%}",
                    ])
                elif fake_votes > real_votes:
                    summary_rows.append([
                        "ML Models Consensus", "FAKE",
                        f"{fake_votes}/{real_votes + fake_votes} models, Avg confidence: {avg_conf:.2%}",
                    ])
                else:
                    summary_rows.append([
                        "ML Models Consensus", "SPLIT",
                        f"Equal votes, Avg confidence: {avg_conf:.2%}",
                    ])

            # Gemini
            if use_gemini and "gemini_analysis" in locals() and gemini_analysis is not None:
                import re
                match = re.search(r'RESULT:[\s\*]*(REAL|FAKE)', gemini_analysis or "", re.IGNORECASE)
                if match:
                    verdict = match.group(1).upper()
                    if verdict == "REAL":
                        summary_rows.append(["Gemini AI", "REAL", "AI analysis indicates authentic news"])
                    else:
                        summary_rows.append(["Gemini AI", "FAKE", "AI analysis indicates fake news"])
                else:
                    summary_rows.append(["Gemini AI", "UNCERTAIN", "AI analysis inconclusive"])

            if summary_rows:
                df_summary = pd.DataFrame(summary_rows, columns=["Method", "Result", "Details"])
                st.table(df_summary)

                st.markdown("### 🎯 Final Assessment")
                if input_text.strip() and content_analysis and not content_analysis.get("is_news"):
                    st.error(
                        "⚠️ **ANALYSIS LIMITATION**: Input was not identified as news content. "
                        "Results may not be meaningful for fake news detection."
                    )

                real_indicators = sum(
                    1 for row in summary_rows
                    if ("REAL" in row[1]) or ("VERIFIED" in row[1])
                )
                fake_indicators = sum(
                    1 for row in summary_rows
                    if ("FAKE" in row[1])
                )

                if real_indicators > fake_indicators:
                    st.success(f"**LIKELY AUTHENTIC NEWS** ({real_indicators}/{len(summary_rows)} positive indicators)")
                    st.markdown(
                        "✅ **Recommendation**: This appears to be legitimate news based on multiple verification methods."
                    )
                elif fake_indicators > real_indicators:
                    st.error(f"**LIKELY FAKE NEWS** ({fake_indicators}/{len(summary_rows)} negative indicators)")
                    st.markdown(
                        "❌ **Recommendation**: This appears to be fake or misleading news. "
                        "Cross-reference with trusted sources before sharing."
                    )
                else:
                    st.warning("**INCONCLUSIVE** - Mixed or insufficient evidence")
                    st.markdown(
                        "⚠️ **Recommendation**: Insufficient evidence to make a definitive determination. "
                        "Seek additional verification from trusted sources."
                    )
            else:
                st.info("No analysis results to display. Please run the analysis first.")


# ── System Info ───────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔧 System Information"):
    st.markdown("**Content Validation Info:**")
    if input_text and input_text.strip():
        try:
            detailed = get_detailed_content_analysis(input_text)
        except Exception:
            detailed = {
                "character_count": len(input_text),
                "word_count": len(input_text.split()),
                "sentence_count": input_text.count("."),
                "news_score": 0,
            }

        col1, col2, col3, col4 = safe_columns(4)
        with col1:
            st.metric("Characters", detailed.get("character_count", 0))
        with col2:
            st.metric("Words", detailed.get("word_count", 0))
        with col3:
            st.metric("Sentences", detailed.get("sentence_count", 0))
        with col4:
            st.metric("News Score", f"{detailed.get('news_score', 0)}/100")
    else:
        col1, col2, col3, col4 = safe_columns(4)
        with col1:
            st.metric("Characters", 0)
        with col2:
            st.metric("Words", 0)
        with col3:
            st.metric("Sentences", 0)
        with col4:
            st.metric("News Score", "0/100")

    if st.button("🧹 Clear Validation Cache"):
        reset_content_validation_state()
        st.toast("✅ Validation cache cleared!")
        st.rerun()

    st.markdown("---")
    model_list = ", ".join(models.keys()) if models else "None"
    st.markdown(
        f"""
**Available ML Models:** {model_list}

**API Status:**
- NewsAPI: {'✅' if NEWSAPI_KEY else '❌'}
- GNews: {'✅' if GNEWS_KEY else '❌'}
- CurrentsAPI: {'✅' if CURRENTS_KEY else '❌'}
- Gemini AI: {'✅' if GEMINI_API_KEY else '❌'}
"""
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p><b>🔍 Multi-Source Fake News Detection System</b></p>
        <p><small>Combining ML Models, AI Analysis, and Multi-API Verification • Always verify important news independently</small></p>
        <p><small><em>⚠️ Content validation ensures meaningful analysis of news content only</em></small></p>
    </div>
    """,
    unsafe_allow_html=True,
)
