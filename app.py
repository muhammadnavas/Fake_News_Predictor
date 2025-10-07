import os
from datetime import datetime

import streamlit as st  # MUST be imported before any module that uses Streamlit

# ----------------------------------------------------------------------------------
# IMPORTANT: set_page_config MUST be the very first Streamlit command.
# Some imported modules (e.g., rag_system) call st.toast() during import/initialization.
# To avoid the "set_page_config can only be called once" error, we configure the page
# immediately and guard against accidental re-calls.
# ----------------------------------------------------------------------------------
if not st.session_state.get("_page_configured", False):
    try:
        st.set_page_config(
            page_title="🔍 RAG-Enhanced Fake News Predictor",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        # If already configured (e.g., in multipage context), ignore silently.
        pass
    st.session_state["_page_configured"] = True

# Defer heavier imports until after page config
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Plotly (px is currently unused, but keeping import in case you add charts later)
import plotly.express as px  # noqa: F401
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── NLTK: download only if missing ──────────────────────────────────────────────
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
                # Don't hard-fail the app if NLTK can't download (e.g., no internet)
                pass

ensure_nltk_resources()

# ── Import custom modules (ensure these files exist in your project) ────────────
from fetch_news import get_all_news, comprehensive_news_check  # noqa: E402

from rag_system import (  # noqa: E402
    RAGKnowledgeBase,
    analyze_against_facts,
    calculate_kb_confidence,
    comprehensive_news_check_with_rag,
    check_rag_health,
)

# New advanced RAG pipeline (higher-level ingestion + QA)
from rag_pipeline import get_or_create_rag_pipeline  # noqa: E402

from ml_analysis import load_all_models, analyze_with_all_models  # noqa: E402

# IMPORTANT: Avoid name collisions — import Gemini analysis only from ai_analysis
from ai_analysis import (  # noqa: E402
    rag_enhanced_gemini_analysis,
    standard_gemini_analysis,
)

from content_detector import (  # noqa: E402
    add_content_validation_to_streamlit,  # (optional helper, not required below)
    detect_content_type,
    get_detailed_content_analysis,
    reset_content_validation_state,
    is_content_news_suitable,
    get_content_validation_message,
    validate_news_content,
)

# (Page config already set above; section retained for readability/documentation.)

# ── Load environment variables and validate critical keys ───────────────────────
load_dotenv()

# Prefer Streamlit secrets if present; fallback to environment variables
def _get_secret(name: str):
    try:
        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        return None

GEMINI_API_KEY = _get_secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
NEWSAPI_KEY = _get_secret("NEWSAPI_KEY") or os.getenv("NEWSAPI_KEY")
GNEWS_KEY = _get_secret("GNEWS_KEY") or os.getenv("GNEWS_KEY")
CURRENTS_KEY = _get_secret("CURRENTS_KEY") or os.getenv("CURRENTS_KEY")

required_keys = {"GEMINI_API_KEY": GEMINI_API_KEY, "NEWSAPI_KEY": NEWSAPI_KEY}
missing_keys = [key for key, value in required_keys.items() if not value]
if missing_keys:
    st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
    st.info("Please set the required API keys in your .env file")
    st.stop()

# ── Title / Header ──────────────────────────────────────────────────────────────
st.title("🔍 RAG-Enhanced Fake News Predictor")
st.markdown("**Multi-API News Verification** • **RAG Technology** • **ML-Powered Assessment** • **AI-Powered Assessment**")
st.markdown("---")

# ── Initialize RAG system quietly with robust error reporting ───────────────────
def initialize_rag():
    """Initialize RAG system with comprehensive error handling (no Streamlit calls here)."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    f = io.StringIO()
    try:
        with redirect_stdout(f), redirect_stderr(f):
            rag = RAGKnowledgeBase()
        return rag
    except Exception as e:
        # Bubble up a clean error; caller handles UI
        raise RuntimeError(f"RAG init failed: {e}")

with st.spinner("Initializing system..."):
    try:
        rag_system = initialize_rag()
        # Initialize higher-level pipeline (wraps rag_system internally if needed)
        rag_pipeline = get_or_create_rag_pipeline()
    except Exception as e:
        st.error(str(e))
        st.stop()

# ── Sidebar: News Fetching + Knowledge Base Management ──────────────────────────
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

    # Display fetched articles
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

    # Knowledge Base Management
    st.header("🧠 Knowledge Base")
    fact_count = 0
    try:
        # Safely access fact_database if provided by your RAG implementation
        fact_count = len(getattr(rag_system, "fact_database", []))
    except Exception:
        pass
    st.markdown(f"**Facts in DB:** {fact_count}")

    # Bulk ingestion controls
    with st.expander("📥 Bulk Ingest Datasets (True/False News)"):
        st.caption("Ingest a limited number of rows from large datasets to expand RAG knowledge base. Uses chunking.")
        ingest_limit = st.slider("Rows per file (preview limit)", 50, 500, 150, step=50)
        if st.button("🚀 Ingest Default Datasets"):
            try:
                with st.spinner("Ingesting datasets into RAG knowledge base..."):
                    added = rag_pipeline.bulk_ingest_default_datasets(per_file_limit=ingest_limit)
                if added > 0:
                    st.success(f"✅ Ingested {added} fact chunks from datasets")
                    st.toast("Knowledge base expanded!")
                    st.rerun()
                else:
                    st.info("No new facts added (possible duplicates or empty rows)")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    with st.expander("➕ Add New Fact"):
        new_fact_content = st.text_area("Fact content:")
        new_fact_category = st.selectbox(
            "Category:",
            ["science", "health", "technology", "politics", "environment", "other"],
        )
        new_fact_sources = st.text_input("Sources (comma-separated):")

        if st.button("Add Fact"):
            if new_fact_content.strip():
                sources_list = [s.strip() for s in new_fact_sources.split(",") if s.strip()]
                try:
                    rag_system.add_fact(new_fact_content.strip(), new_fact_category, True, sources_list)
                    st.toast("Fact added to knowledge base!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add fact: {e}")
            else:
                st.warning("Please enter fact content before adding.")

# ── Load ML models ──────────────────────────────────────────────────────────────
models, vectorizer = load_all_models()
if models is None or vectorizer is None:
    st.error("❌ Could not load models or vectorizer. Please check 'models/' folder.")


# ── Input area with content validation ──────────────────────────────────────────
st.subheader("📝 Enter News Text")
st.info(
    "💡 **Tip**: Enter actual news headlines or articles for accurate analysis. "
    "Personal statements or casual text may not be analyzed correctly."
)

# Ensure input is bound to session_state
if "selected_text" not in st.session_state:
    st.session_state.selected_text = ""

# Clear validation state button
if st.button("🔄 Clear State", help="Clear validation cache for fresh analysis"):
    reset_content_validation_state()
    st.session_state.selected_text = ""  # Clear input text
    st.rerun()  # <-- Use this instead of experimental_rerun

# Input text area, bound to session_state
input_text = st.text_area(
    "Paste news headline or article text:",
    value=st.session_state.selected_text,
    key="selected_text",  # bind directly to session_state
    height=120,
    placeholder="Example: 'Breaking: Government announces new policy...' or 'Local authorities report incident...'",
)


# Real-time content validation
content_analysis = None
analysis_allowed = True

if input_text and input_text.strip():
    # Fresh validation each time input changes
    content_analysis = detect_content_type(input_text)
    validation_container = st.container()
    with validation_container:
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

# ── Analysis options (added missing Force Analysis) ─────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    check_existence = st.checkbox("🔍 Multi-API Verification", value=True)
with c2:
    use_rag = st.checkbox("🧠 RAG Analysis", value=True)
with c3:
    use_gemini = st.checkbox("🤖 Gemini AI Analysis", value=True)
with c4:
    advanced_analysis = st.checkbox("📊 Advanced Analytics", value=True)

# ── Main Analysis Button ────────────────────────────────────────────────────────
if st.button("🚀 RAG-Enhanced Analysis", type="primary", use_container_width=True):
    if not (input_text and input_text.strip()):
        st.warning("⚠️ Please enter text to analyze")
    elif not analysis_allowed:
        st.error(
            "❌ **Analysis blocked**: Content doesn't appear to be news. "
            "Please provide an actual news headline or article text."
        )
    else:

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🔍 Verification", "🧠 RAG Analysis", "🤖 ML Models", "🧠 AI Assessment", "📊 Summary"]
        )

        # ── Tab 1: Multi-Source Verification ────────────────────────────────────
        with tab1:
            st.subheader("Multi-Source News Verification")

            verification_results = None
            if check_existence:
                with st.spinner("Checking across NewsAPI, GNews, and CurrentsAPI..."):
                    try:
                        if use_rag:
                            verification_results = comprehensive_news_check_with_rag(input_text, rag_system)
                        else:
                            verification_results = comprehensive_news_check(input_text)
                    except Exception as e:
                        st.error(f"Verification error: {e}")
                        verification_results = None

                if verification_results:
                    col_a, col_b, col_c = st.columns(3)
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
                else:
                    st.info("No verification results to display.")

        # ── Tab 2: RAG Analysis ─────────────────────────────────────────────────
        with tab2:
            st.subheader("🧠 RAG Knowledge Base Analysis")
            if use_rag:
                try:
                    with st.spinner("Retrieving relevant facts from knowledge base..."):
                        relevant_facts = rag_system.retrieve_relevant_facts(input_text, top_k=5)
                        fact_analysis = analyze_against_facts(input_text, relevant_facts)
                        kb_confidence = calculate_kb_confidence(relevant_facts, fact_analysis)

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Relevant Facts", len(relevant_facts))
                    with col_b:
                        st.metric("KB Confidence", f"{kb_confidence:.1f}%")
                    with col_c:
                        consistency = (fact_analysis.get("overall_consistency") or "neutral").title()
                        st.metric("Consistency", consistency)

                    if relevant_facts:
                        st.subheader("📚 Retrieved Knowledge Base Facts")
                        for i, fact in enumerate(relevant_facts, 1):
                            sim = float(fact.get("similarity", 0.0))
                            sim_color = "green" if sim > 0.7 else ("orange" if sim > 0.4 else "red")
                            sources = fact.get("sources") or []
                            sources_txt = ", ".join(sources) if sources else "N/A"
                            st.markdown(
                                f"**Fact {i}** (Similarity: :{sim_color}[{sim:.3f}])\n"
                                f"- **Content:** {fact.get('content','')}\n"
                                f"- **Category:** {fact.get('category','other')}\n"
                                f"- **Sources:** {sources_txt}\n"
                            )

                    if fact_analysis.get("confirmations"):
                        st.success("✅ **Supporting Facts Found:**")
                        for conf in fact_analysis["confirmations"]:
                            st.write(f"- {conf.get('fact','')} (Similarity: {conf.get('similarity',0.0):.3f})")

                    if fact_analysis.get("contradictions"):
                        st.error("❌ **Contradictory Facts Found:**")
                        for contra in fact_analysis["contradictions"]:
                            st.write(f"- {contra.get('fact','')} (Similarity: {contra.get('similarity',0.0):.3f})")

                    if not fact_analysis.get("confirmations") and not fact_analysis.get("contradictions"):
                        st.info("ℹ️ No strong matches found in knowledge base")

                    # --- New: Direct RAG QA (question answering over ingested corpus) ---
                    st.markdown("---")
                    st.markdown("### ❓ Ask a Question (RAG QA)")
                    qa_query = st.text_input("Enter a claim or question to verify against knowledge base:", key="rag_qa_query")
                    col_qa1, col_qa2, col_qa3 = st.columns([2,1,1])
                    with col_qa1:
                        k_ctx = st.number_input("Top K Contexts", min_value=1, max_value=15, value=5)
                    with col_qa2:
                        use_gemini_aug = st.checkbox("Gemini Augmentation", value=False, help="Refine answer with Gemini if API key present")
                    with col_qa3:
                        run_qa = st.button("🔎 Run QA", key="run_rag_qa")

                    if run_qa and qa_query.strip():
                        try:
                            with st.spinner("Running RAG QA pipeline..."):
                                qa_answer = rag_pipeline.generate_answer(qa_query.strip(), k=int(k_ctx), use_gemini=use_gemini_aug)
                            st.success("**Answer:**")
                            st.write(qa_answer.answer)
                            st.caption(f"Confidence: {qa_answer.confidence*100:.1f}% | Mode: {qa_answer.mode}")
                            with st.expander("🔍 Contexts Used"):
                                for c in qa_answer.contexts:
                                    st.markdown(f"- ({c.score:.3f}) {c.content[:250]}{'...' if len(c.content)>250 else ''}")
                            with st.expander("🧠 Reasoning Trace"):
                                st.code(qa_answer.reasoning, language="text")
                        except Exception as e:
                            st.error(f"RAG QA failed: {e}")
                except Exception as e:
                    st.error(f"RAG analysis failed: {e}")
            else:
                st.info("Enable RAG Analysis to view knowledge base results.")

        # ── Tab 3: ML Model Analysis ────────────────────────────────────────────
        with tab3:
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

                        # Predictions bar chart (colored by label)
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

                        # Probabilities bars
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

                    # Per-model textual results
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

        # ── Tab 4: AI-Powered Assessment (Gemini) ───────────────────────────────
        with tab4:
            st.subheader("AI-Powered Assessment")
            gemini_analysis = None
            if use_gemini:
                try:
                    with st.spinner("Getting RAG-enhanced Gemini AI analysis..."):
                        if use_rag:
                            relevant_facts_for_ai = rag_system.retrieve_relevant_facts(input_text, top_k=3)
                            gemini_analysis = rag_enhanced_gemini_analysis(input_text, relevant_facts_for_ai)
                        else:
                            gemini_analysis = standard_gemini_analysis(input_text)

                    st.markdown("### 🤖 Gemini AI Detailed Analysis")
                    st.markdown(gemini_analysis)

                    upper = (gemini_analysis or "").upper()
                    if "REAL" in upper:
                        st.success("🎯 **Gemini Assessment: REAL NEWS**")
                    elif "FAKE" in upper:
                        st.error("🎯 **Gemini Assessment: FAKE NEWS**")
                    else:
                        st.warning("🎯 **Gemini Assessment: INCONCLUSIVE**")
                except Exception as e:
                    st.error(f"Gemini API error: {e}")
            else:
                st.info("Enable Gemini AI Analysis to view AI assessment.")

        # ── Tab 5: Summary ──────────────────────────────────────────────────────
        with tab5:
            st.subheader("📊 Summary Report")
            if content_analysis:
                if content_analysis.get("is_news"):
                    st.success(
                        f"✅ Detected as News Content (Confidence: {content_analysis.get('confidence',0.0):.1%})"
                    )
                else:
                    st.error(
                        f"❌ Not News Content (Detected as: "
                        f"{(content_analysis.get('content_type') or 'unknown').replace('_',' ').title()})"
                    )

            summary_rows = []

            # Verification summary
            if check_existence and "verification_results" in locals() and verification_results:
                if verification_results.get("sources_found"):
                    summary_rows.append([
                        "News Verification",
                        "VERIFIED",
                        f"{len(verification_results['sources_found'])} sources, "
                        f"{verification_results.get('confidence_score',0.0):.1f}% confidence",
                    ])
                else:
                    summary_rows.append(["News Verification", "NOT FOUND", "No matches across APIs"])

            # RAG summary
            if use_rag and "kb_confidence" in locals():
                consistency_val = (fact_analysis.get("overall_consistency") or "neutral").lower()
                if consistency_val == "consistent":
                    summary_rows.append(["RAG Knowledge Base", "CONSISTENT",
                                         f"KB Confidence: {kb_confidence:.1f}%, {len(relevant_facts)} facts"])
                elif consistency_val == "contradictory":
                    summary_rows.append(["RAG Knowledge Base", "CONTRADICTORY",
                                         f"KB Confidence: {kb_confidence:.1f}%, conflicts found"])
                else:
                    summary_rows.append(["RAG Knowledge Base", "NEUTRAL",
                                         f"KB Confidence: {kb_confidence:.1f}%, no strong matches"])

            # ML summary
            if models and "ml_results" in locals():
                valid_results = [r for r in ml_results.values() if "error" not in r]
                real_votes = sum(1 for r in valid_results if r["prediction"] == "REAL")
                fake_votes = len(valid_results) - real_votes
                avg_conf = float(np.mean([r["confidence"] for r in valid_results])) if valid_results else 0.0

                if real_votes > fake_votes:
                    summary_rows.append(["ML Models Consensus", "REAL",
                                         f"{real_votes}/{real_votes + fake_votes} models, Avg confidence: {avg_conf:.2%}"])
                elif fake_votes > real_votes:
                    summary_rows.append(["ML Models Consensus", "FAKE",
                                         f"{fake_votes}/{real_votes + fake_votes} models, Avg confidence: {avg_conf:.2%}"])
                else:
                    summary_rows.append(["ML Models Consensus", "SPLIT",
                                         f"Equal votes, Avg confidence: {avg_conf:.2%}"])

            # Gemini summary
            if use_gemini and "gemini_analysis" in locals() and gemini_analysis is not None:
                up = gemini_analysis.upper()
                if "REAL" in up:
                    summary_rows.append(["Gemini AI", "REAL", "AI analysis indicates authentic news"])
                elif "FAKE" in up:
                    summary_rows.append(["Gemini AI", "FAKE", "AI analysis indicates fake news"])
                else:
                    summary_rows.append(["Gemini AI", "UNCERTAIN", "AI analysis inconclusive"])

            if summary_rows:
                df_summary = pd.DataFrame(summary_rows, columns=["Method", "Result", "Details"])
                st.table(df_summary)

                st.markdown("### 🎯 RAG-Enhanced Final Assessment")
                if input_text.strip() and content_analysis and not content_analysis.get("is_news"):
                    st.error(
                        "⚠️ **ANALYSIS LIMITATION**: Input was not identified as news content. "
                        "Results may not be meaningful for fake news detection."
                    )

                real_indicators = sum(
                    1 for row in summary_rows
                    if ("REAL" in row[1]) or ("CONSISTENT" in row[1]) or ("VERIFIED" in row[1])
                )
                fake_indicators = sum(
                    1 for row in summary_rows
                    if ("FAKE" in row[1]) or ("CONTRADICTORY" in row[1])
                )

                if real_indicators > fake_indicators:
                    st.success(f"**LIKELY AUTHENTIC NEWS** ({real_indicators}/{len(summary_rows)} positive indicators)")
                    st.markdown(
                        "✅ **Recommendation**: This appears to be legitimate news based on multiple "
                        "verification methods including knowledge base analysis."
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

                # RAG-specific insights
                if use_rag and "relevant_facts" in locals() and relevant_facts:
                    st.markdown("### 🧠 Knowledge Base Insights")
                    if fact_analysis.get("confirmations"):
                        st.success(f"✅ Found {len(fact_analysis['confirmations'])} supporting facts in knowledge base")
                    if fact_analysis.get("contradictions"):
                        st.error(f"❌ Found {len(fact_analysis['contradictions'])} contradictory facts in knowledge base")

                    st.markdown("**Key Knowledge Base Matches:**")
                    for fact in relevant_facts[:3]:
                        sim = float(fact.get("similarity", 0.0))
                        emoji = "🎯" if sim > 0.7 else ("📍" if sim > 0.4 else "📌")
                        st.markdown(f"{emoji} {fact.get('content','')[:100]}... (Similarity: {sim:.3f})")
            else:
                st.info("No analysis results to display. Please run the analysis first.")

# ── System Information / Health ─────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔧 System Information"):
    # RAG Health Check
    try:
        rag_health = check_rag_health(rag_system)
    except Exception as e:
        rag_health = {"health_check": f"Failed: {e}"}

    st.markdown("**RAG System Health Check:**")
    for component, status in (rag_health or {}).items():
        st.markdown(f"- {component}: {status}")

    # Content validation statistics
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

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", detailed.get("character_count", 0))
        with col2:
            st.metric("Words", detailed.get("word_count", 0))
        with col3:
            st.metric("Sentences", detailed.get("sentence_count", 0))
        with col4:
            st.metric("News Score", f"{detailed.get('news_score', 0)}/100")

        st.markdown("**Validation Cache:**")
        validation_keys = [k for k in st.session_state.keys() if k.startswith("content_validation_")]
        st.write(f"• Cached validations: {len(validation_keys)}")
        if content_analysis:
            st.write(f"• Current text hash: {content_analysis.get('text_hash', 'n/a')}")
            st.write(f"• Validation status: {content_analysis.get('validation_status', 'n/a')}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", 0)
        with col2:
            st.metric("Words", 0)
        with col3:
            st.metric("Sentences", 0)
        with col4:
            st.metric("News Score", "0/100")

    # Reset buttons
    rb1, rb2 = st.columns(2)
    with rb1:
        if st.button("🔄 Reset RAG System"):
            try:
                # Clear ChromaDB if available
                if hasattr(rag_system, "chroma_client") and getattr(rag_system, "chroma_client"):
                    try:
                        rag_system.chroma_client.delete_collection("news_facts")
                    except Exception:
                        pass
                # Clear Streamlit resource cache and re-run
                try:
                    st.cache_resource.clear()
                except Exception:
                    pass
                st.toast("✅ RAG system reset successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")

    with rb2:
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

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p><b>🔍 RAG-Enhanced Multi-Source Fake News Detection System</b></p>
        <p><small>Combining RAG Technology, ML Models, AI Analysis, and Multi-API Verification • Always verify important news independently</small></p>
        <p><small><em>⚠️ Content validation ensures meaningful analysis of news content only</em></small></p>
    </div>
    """,
    unsafe_allow_html=True,
)
