import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv

# Import custom modules
from fetch_news import get_all_news, comprehensive_news_check
from rag_system import RAGKnowledgeBase, analyze_against_facts, calculate_kb_confidence, comprehensive_news_check_with_rag, rag_enhanced_gemini_analysis, standard_gemini_analysis, check_rag_health
from ml_analysis import load_all_models, analyze_with_all_models
from ai_analysis import rag_enhanced_gemini_analysis, standard_gemini_analysis
from content_detector import (
    add_content_validation_to_streamlit, 
    detect_content_type, 
    get_detailed_content_analysis,
    reset_content_validation_state,
    is_content_news_suitable,
    get_content_validation_message,
    validate_news_content
)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_KEY = os.getenv("GNEWS_KEY")
CURRENTS_KEY = os.getenv("CURRENTS_KEY")

# Validate API keys
required_keys = {"GEMINI_API_KEY": GEMINI_API_KEY, "NEWSAPI_KEY": NEWSAPI_KEY}
missing_keys = [key for key, value in required_keys.items() if not value]

if missing_keys:
    st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
    st.info("Please set the required API keys in your .env file")
    st.stop()

# Streamlit App Configuration
st.set_page_config(
    page_title="🔍 RAG-Enhanced Fake News Detector", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 RAG-Enhanced Fake News Predictor")
st.markdown("**Multi-API News Verification** • **RAG Technology** • **AI-Powered Assessment**")
st.markdown("---")

# Initialize RAG system with error handling (suppressing notifications)
def initialize_rag():
    """Initialize RAG system with comprehensive error handling"""
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    try:
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            rag = RAGKnowledgeBase()
        return rag
    except Exception as e:
        # ❌ Don't call st.error here
        # just raise or return None
        raise RuntimeError(f"RAG init failed: {e}")

# Initialize RAG system quietly
with st.spinner("Initializing system..."):
    rag_system = initialize_rag()

# Sidebar for News Fetching and Knowledge Base Management
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
    if 'articles' in st.session_state:
        st.markdown("### 📰 Recent Articles")
        for i, article in enumerate(st.session_state.articles):
            with st.expander(f"{article['title'][:40]}..."):
                st.write(f"**Source:** {article['source']}")
                st.write(f"**Description:** {article.get('description', 'N/A')[:100]}...")
                if st.button(f"Use Article", key=f"use_{i}"):
                    content = article['title']
                    if article.get('description'):
                        content += " " + article['description']
                    st.session_state.selected_text = content
                    # Clear validation state when new text is selected
                    reset_content_validation_state()
                    st.rerun()
    
    st.markdown("---")
    
    # Knowledge Base Management
    st.header("🧠 Knowledge Base")
    st.markdown(f"**Facts in DB:** {len(rag_system.fact_database)}")
    
    with st.expander("➕ Add New Fact"):
        new_fact_content = st.text_area("Fact content:")
        new_fact_category = st.selectbox("Category:", 
            ["science", "health", "technology", "politics", "environment", "other"])
        new_fact_sources = st.text_input("Sources (comma-separated):")
        
        if st.button("Add Fact"):
            if new_fact_content:
                sources_list = [s.strip() for s in new_fact_sources.split(",") if s.strip()]
                rag_system.add_fact(new_fact_content, new_fact_category, True, sources_list)
                st.toast("Fact added to knowledge base!")
                st.rerun()

# Main Content Area
models, vectorizer = load_all_models()

if models is None:
    st.stop()

# Input area with content validation
st.subheader("📝 Enter News Text")
st.info("💡 **Tip**: Enter actual news headlines or articles for accurate analysis. Personal statements or casual text may not be analyzed correctly.")

# Add a button to clear validation state
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Clear State", help="Clear validation cache for fresh analysis"):
        reset_content_validation_state()
        st.rerun()

input_text = st.text_area(
    "Paste news headline or article text:",
    value=st.session_state.get('selected_text', ''),
    height=120,
    placeholder="Example: 'Breaking: Government announces new policy...' or 'Local authorities report incident...'"
)

# Real-time content validation with improved state management
content_analysis = None
analysis_allowed = True
force_analysis = False

if input_text and input_text.strip():
    # Get fresh analysis
    content_analysis = detect_content_type(input_text)
    
    # Clear message display area
    validation_container = st.container()
    
    with validation_container:
        # Show content validation status
        if content_analysis["is_news"]:
            st.success(f"✅ **Content Validation**: Appears to be news content (Confidence: {content_analysis['confidence']:.1%})")
            analysis_allowed = True
        else:
            st.warning(f"⚠️ **Content Validation**: This doesn't appear to be news content")
            st.error(f"**Detected as**: {content_analysis['content_type'].replace('_', ' ').title()}")
            
            # Show bypass option immediately
            force_analysis = st.checkbox(
                "🔓 **Force Analysis** (Analyze anyway - results may be inaccurate)", 
                help="Check this to analyze non-news content. Results may not be meaningful.",
                key=f"force_analysis_{content_analysis['text_hash']}"
            )
            analysis_allowed = force_analysis
            
            if content_analysis["suggestions"]:
                with st.expander("💡 **Suggestions for Better Analysis**"):
                    for suggestion in content_analysis["suggestions"]:
                        st.write(f"• {suggestion}")
            
            # Show detailed analysis in expander
            with st.expander("🔍 **Detailed Content Analysis**"):
                detailed_analysis = get_detailed_content_analysis(input_text)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Word Count", detailed_analysis["word_count"])
                with col2:
                    st.metric("News Score", f"{detailed_analysis['news_score']}/100")
                with col3:
                    st.metric("Confidence", f"{detailed_analysis['confidence']:.1%}")
                with col4:
                    st.metric("Proper Nouns", detailed_analysis["proper_noun_count"])
                
                st.write("**Analysis Reasons:**")
                for reason in detailed_analysis["reasons"]:
                    st.write(f"• {reason}")
                
                # Pattern summary
                pattern_summary = detailed_analysis["pattern_summary"]
                st.write("**Pattern Analysis:**")
                st.write(f"• News indicators: {pattern_summary['news_indicators']}")
                st.write(f"• Personal indicators: {pattern_summary['personal_indicators']}")
                st.write(f"• Structural quality: {pattern_summary['structural_quality']}")

# Analysis options
col1, col2, col3, col4 = st.columns(4)
with col1:
    check_existence = st.checkbox("🔍 Multi-API Verification", value=True)
with col2:
    use_rag = st.checkbox("🧠 RAG Analysis", value=True)
with col3:
    use_gemini = st.checkbox("🤖 Gemini AI Analysis", value=True)
with col4:
    advanced_analysis = st.checkbox("📊 Advanced Analytics", value=True)

# Main Analysis Button with improved validation
if st.button("🚀 RAG-Enhanced Analysis", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("⚠️ Please enter text to analyze")
    elif not analysis_allowed:
        st.error("❌ **Analysis blocked**: Content doesn't appear to be news. Check 'Force Analysis' to proceed anyway.")
    else:
        # Show warning if forcing analysis on non-news content
        if force_analysis and content_analysis and not content_analysis["is_news"]:
            st.warning("⚠️ **Analyzing non-news content**: Results may not be accurate or meaningful.")
            st.info(f"**Detected Content Type**: {content_analysis['content_type'].replace('_', ' ').title()}")
        
        # Create tabs for organized results
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Verification", "🧠 RAG Analysis", "🤖 ML Models", "🧠 AI Assessment", "📊 Summary"])
        
        with tab1:
            st.subheader("Multi-Source News Verification")
            
            if check_existence:
                with st.spinner("Checking across NewsAPI, GNews, and CurrentsAPI..."):
                    if use_rag:
                        verification_results = comprehensive_news_check_with_rag(input_text, rag_system)
                    else:
                        verification_results = comprehensive_news_check(input_text)
                
                # Display verification summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sources Found", len(verification_results["sources_found"]))
                with col2:
                    st.metric("Total Matches", verification_results["total_matches"])
                with col3:
                    st.metric("Confidence Score", f"{verification_results['confidence_score']:.1f}%")
                
                # API search summary
                st.subheader("API Search Results")
                for api_name, result in verification_results["search_summary"].items():
                    if result.get("found"):
                        st.success(f"✅ **{api_name}**: Found {result['count']} matching articles")
                    elif result.get("error"):
                        st.error(f"❌ **{api_name}**: {result['error']}")
                    else:
                        st.info(f"ℹ️ **{api_name}**: No matches found")
        
        with tab2:
            st.subheader("🧠 RAG Knowledge Base Analysis")
            
            if use_rag:
                with st.spinner("Retrieving relevant facts from knowledge base..."):
                    relevant_facts = rag_system.retrieve_relevant_facts(input_text, top_k=5)
                    fact_analysis = analyze_against_facts(input_text, relevant_facts)
                    kb_confidence = calculate_kb_confidence(relevant_facts, fact_analysis)
                
                # Display RAG results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Relevant Facts", len(relevant_facts))
                with col2:
                    st.metric("KB Confidence", f"{kb_confidence:.1f}%")
                with col3:
                    consistency = fact_analysis["overall_consistency"].title()
                    st.metric("Consistency", consistency)
                
                # Show relevant facts
                if relevant_facts:
                    st.subheader("📚 Retrieved Knowledge Base Facts")
                    for i, fact in enumerate(relevant_facts, 1):
                        similarity_color = "green" if fact["similarity"] > 0.7 else "orange" if fact["similarity"] > 0.4 else "red"
                        st.markdown(f"""
                        **Fact {i}** (Similarity: :{similarity_color}[{fact['similarity']:.3f}])
                        - **Content:** {fact['content']}
                        - **Category:** {fact['category']}
                        - **Sources:** {', '.join(fact['sources'])}
                        """)
                
                # Show fact analysis
                if fact_analysis["confirmations"]:
                    st.success("✅ **Supporting Facts Found:**")
                    for conf in fact_analysis["confirmations"]:
                        st.write(f"- {conf['fact']} (Similarity: {conf['similarity']:.3f})")
                
                if fact_analysis["contradictions"]:
                    st.error("❌ **Contradictory Facts Found:**")
                    for contra in fact_analysis["contradictions"]:
                        st.write(f"- {contra['fact']} (Similarity: {contra['similarity']:.3f})")
                
                if not fact_analysis["confirmations"] and not fact_analysis["contradictions"]:
                    st.info("ℹ️ No strong matches found in knowledge base")
        
        with tab3:
            st.subheader("Machine Learning Model Analysis")
            
            if models:
                with st.spinner("Analyzing with all ML models..."):
                    ml_results = analyze_with_all_models(input_text, models, vectorizer)
                
                # Create visualization
                if advanced_analysis and ml_results:
                    model_names = []
                    predictions = []
                    confidences = []
                    fake_probs = []
                    real_probs = []
                    
                    for model_name, result in ml_results.items():
                        if "error" not in result:
                            model_names.append(model_name)
                            predictions.append(result["prediction"])
                            confidences.append(result["confidence"])
                            fake_probs.append(result["fake_probability"])
                            real_probs.append(result["real_probability"])
                    
                    # Create subplots
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Model Predictions', 'Confidence Scores', 
                                       'Fake vs Real Probabilities', 'Model Consensus'),
                        specs=[[{"type": "bar"}, {"type": "bar"}],
                               [{"type": "bar"}, {"type": "pie"}]]
                    )
                    
                    # Predictions bar chart
                    colors = ['red' if p == 'FAKE' else 'green' for p in predictions]
                    fig.add_trace(go.Bar(x=model_names, y=[1]*len(model_names), 
                                        marker_color=colors, name="Predictions",
                                        text=predictions, textposition='inside'),
                                 row=1, col=1)
                    
                    # Confidence scores
                    fig.add_trace(go.Bar(x=model_names, y=confidences, 
                                        marker_color='blue', name="Confidence",
                                        text=[f"{c:.2%}" for c in confidences], 
                                        textposition='outside'),
                                 row=1, col=2)
                    
                    # Fake vs Real probabilities
                    fig.add_trace(go.Bar(x=model_names, y=fake_probs, 
                                        name="Fake Probability", marker_color='red'),
                                 row=2, col=1)
                    fig.add_trace(go.Bar(x=model_names, y=real_probs, 
                                        name="Real Probability", marker_color='green'),
                                 row=2, col=1)
                    
                    # Consensus pie chart
                    real_count = sum(1 for p in predictions if p == 'REAL')
                    fake_count = len(predictions) - real_count
                    fig.add_trace(go.Pie(labels=['REAL', 'FAKE'], values=[real_count, fake_count],
                                        marker_colors=['green', 'red']),
                                 row=2, col=2)
                    
                    fig.update_layout(height=600, showlegend=True, 
                                     title_text="Comprehensive ML Model Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed results
                for model_name, result in ml_results.items():
                    if "error" not in result:
                        if result["prediction"] == "REAL":
                            st.success(f"✅ **{model_name}**: {result['prediction']} "
                                     f"(Confidence: {result['confidence']:.2%})")
                        else:
                            st.error(f"❌ **{model_name}**: {result['prediction']} "
                                   f"(Confidence: {result['confidence']:.2%})")
                        
                        # Progress bars for probabilities
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Real Probability:")
                            st.progress(result["real_probability"])
                        with col2:
                            st.write("Fake Probability:")
                            st.progress(result["fake_probability"])
                    else:
                        st.warning(f"⚠️ **{model_name}**: Error - {result['error']}")
        
        with tab4:
            st.subheader("AI-Powered Assessment")
            
            if use_gemini:
                with st.spinner("Getting RAG-enhanced Gemini AI analysis..."):
                    try:
                        # Get relevant facts for enhanced analysis
                        if use_rag:
                            relevant_facts = rag_system.retrieve_relevant_facts(input_text, top_k=3)
                            gemini_analysis = rag_enhanced_gemini_analysis(input_text, relevant_facts)
                        else:
                            gemini_analysis = standard_gemini_analysis(input_text)
                        
                        st.markdown("### 🤖 Gemini AI Detailed Analysis")
                        st.markdown(gemini_analysis)
                        
                        # Extract key insights
                        if "REAL" in gemini_analysis.upper():
                            st.success("🎯 **Gemini Assessment: REAL NEWS**")
                        elif "FAKE" in gemini_analysis.upper():
                            st.error("🎯 **Gemini Assessment: FAKE NEWS**")
                        else:
                            st.warning("🎯 **Gemini Assessment: INCONCLUSIVE**")
                            
                    except Exception as e:
                        st.error(f"Gemini API error: {e}")
        
        with tab5:
            st.subheader("📊 Comprehensive Summary & Recommendations")
            
            # Show content validation summary first
            if input_text.strip() and content_analysis:
                if not content_analysis["is_news"]:
                    st.warning("⚠️ **Content Validation**: Input was identified as non-news content. Results may not be accurate.")
                    st.info(f"**Detected Content Type**: {content_analysis['content_type'].replace('_', ' ').title()}")
                    st.info(f"**Content Confidence**: {content_analysis['confidence']:.1%}")
                    
                    # Show key validation metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("News Score", f"{content_analysis['news_score']}/100")
                    with col2:
                        st.metric("News Patterns", content_analysis['news_pattern_count'])
                    with col3:
                        st.metric("Personal Patterns", content_analysis['personal_matches'])
                else:
                    st.success(f"✅ **Content Validation**: Input validated as news content (Confidence: {content_analysis['confidence']:.1%})")
            
            # Aggregate all results
            summary_data = []
            
            # News verification summary
            if check_existence and 'verification_results' in locals():
                if verification_results["sources_found"]:
                    summary_data.append(["News Verification", "VERIFIED", 
                                       f"{len(verification_results['sources_found'])} sources, "
                                       f"{verification_results['confidence_score']:.1f}% confidence"])
                else:
                    summary_data.append(["News Verification", "NOT FOUND", 
                                       "No matches across APIs"])
            
            # RAG analysis summary
            if use_rag and 'kb_confidence' in locals():
                consistency = fact_analysis["overall_consistency"]
                if consistency == "consistent":
                    summary_data.append(["RAG Knowledge Base", "CONSISTENT", 
                                       f"KB Confidence: {kb_confidence:.1f}%, {len(relevant_facts)} facts"])
                elif consistency == "contradictory":
                    summary_data.append(["RAG Knowledge Base", "CONTRADICTORY", 
                                       f"KB Confidence: {kb_confidence:.1f}%, conflicts found"])
                else:
                    summary_data.append(["RAG Knowledge Base", "NEUTRAL", 
                                       f"KB Confidence: {kb_confidence:.1f}%, no strong matches"])
            
            # ML models summary
            if models and 'ml_results' in locals():
                real_votes = sum(1 for r in ml_results.values() 
                               if "error" not in r and r["prediction"] == "REAL")
                fake_votes = len([r for r in ml_results.values() if "error" not in r]) - real_votes
                avg_confidence = np.mean([r["confidence"] for r in ml_results.values() 
                                        if "error" not in r])
                
                if real_votes > fake_votes:
                    summary_data.append(["ML Models Consensus", "REAL", 
                                       f"{real_votes}/{real_votes+fake_votes} models, "
                                       f"Avg confidence: {avg_confidence:.2%}"])
                elif fake_votes > real_votes:
                    summary_data.append(["ML Models Consensus", "FAKE", 
                                       f"{fake_votes}/{real_votes+fake_votes} models, "
                                       f"Avg confidence: {avg_confidence:.2%}"])
                else:
                    summary_data.append(["ML Models Consensus", "SPLIT", 
                                       f"Equal votes, Avg confidence: {avg_confidence:.2%}"])
            
            # Gemini AI summary
            if use_gemini and 'gemini_analysis' in locals():
                if "REAL" in gemini_analysis.upper():
                    summary_data.append(["Gemini AI", "REAL", "AI analysis indicates authentic news"])
                elif "FAKE" in gemini_analysis.upper():
                    summary_data.append(["Gemini AI", "FAKE", "AI analysis indicates fake news"])
                else:
                    summary_data.append(["Gemini AI", "UNCERTAIN", "AI analysis inconclusive"])
            
            # Display summary table
            if summary_data:
                df_summary = pd.DataFrame(summary_data, columns=["Method", "Result", "Details"])
                st.table(df_summary)
                
                # Final verdict with RAG consideration
                real_indicators = sum(1 for row in summary_data if "REAL" in row[1] or "CONSISTENT" in row[1] or "VERIFIED" in row[1])
                fake_indicators = sum(1 for row in summary_data if "FAKE" in row[1] or "CONTRADICTORY" in row[1])
                
                st.markdown("### 🎯 RAG-Enhanced Final Assessment")
                
                # Check if content was non-news and adjust verdict accordingly
                if input_text.strip() and content_analysis and not content_analysis["is_news"]:
                    st.error("⚠️ **ANALYSIS LIMITATION**: Input was not identified as news content. Results may not be meaningful for fake news detection.")
                
                if real_indicators > fake_indicators:
                    st.success(f"**LIKELY AUTHENTIC NEWS** ({real_indicators}/{len(summary_data)} positive indicators)")
                    st.markdown("✅ **Recommendation**: This appears to be legitimate news based on multiple verification methods including knowledge base analysis.")
                elif fake_indicators > real_indicators:
                    st.error(f"**LIKELY FAKE NEWS** ({fake_indicators}/{len(summary_data)} negative indicators)")
                    st.markdown("❌ **Recommendation**: This appears to be fake or misleading news. Cross-reference with trusted sources before sharing.")
                else:
                    st.warning("**INCONCLUSIVE** - Mixed or insufficient evidence")
                    st.markdown("⚠️ **Recommendation**: Insufficient evidence to make a definitive determination. Seek additional verification from trusted sources.")
                
                # RAG-specific insights
                if use_rag and 'relevant_facts' in locals() and relevant_facts:
                    st.markdown("### 🧠 Knowledge Base Insights")
                    if fact_analysis["confirmations"]:
                        st.success(f"✅ Found {len(fact_analysis['confirmations'])} supporting facts in knowledge base")
                    if fact_analysis["contradictions"]:
                        st.error(f"❌ Found {len(fact_analysis['contradictions'])} contradictory facts in knowledge base")
                    
                    st.markdown("**Key Knowledge Base Matches:**")
                    for fact in relevant_facts[:3]:
                        similarity_emoji = "🎯" if fact["similarity"] > 0.7 else "📍" if fact["similarity"] > 0.4 else "📌"
                        st.markdown(f"{similarity_emoji} {fact['content'][:100]}... (Similarity: {fact['similarity']:.3f})")
            else:
                st.info("No analysis results to display. Please run the analysis first.")

# Performance metrics and system info
st.markdown("---")
with st.expander("🔧 System Information"):
    # RAG Health Check
    rag_health = check_rag_health(rag_system)
    
    st.markdown("**RAG System Health Check:**")
    for component, status in rag_health.items():
        st.markdown(f"- {component}: {status}")
    
    # Content validation statistics
    st.markdown("**Content Validation Info:**")
    
    if input_text and input_text.strip() and content_analysis is not None:
        detailed_analysis = get_detailed_content_analysis(input_text)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", detailed_analysis["character_count"])
        with col2:
            st.metric("Words", detailed_analysis["word_count"])
        with col3:
            st.metric("Sentences", detailed_analysis["sentence_count"])
        with col4:
            st.metric("News Score", f"{detailed_analysis['news_score']}/100")
        
        # Validation cache info
        st.markdown("**Validation Cache:**")
        validation_keys = [key for key in st.session_state.keys() if key.startswith('content_validation_')]
        st.write(f"• Cached validations: {len(validation_keys)}")
        if content_analysis:
            st.write(f"• Current text hash: {content_analysis['text_hash']}")
            st.write(f"• Validation status: {content_analysis['validation_status']}")
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
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset RAG System"):
            try:
                # Clear ChromaDB
                if hasattr(rag_system, 'chroma_client') and rag_system.chroma_client:
                    try:
                        rag_system.chroma_client.delete_collection("news_facts")
                    except:
                        pass
                
                # Clear cache and reinitialize
                st.cache_resource.clear()
                st.toast("✅ RAG system reset successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")
    
    with col2:
        if st.button("🧹 Clear Validation Cache"):
            reset_content_validation_state()
            st.toast("✅ Validation cache cleared!")
            st.rerun()
    
    st.markdown("---")
    st.markdown(f"""
    **Available ML Models:** {', '.join(models.keys()) if models else 'None'}
    
    **API Status:**
    - NewsAPI: {'✅' if NEWSAPI_KEY else '❌'}
    - GNews: {'✅' if GNEWS_KEY else '❌'}
    - CurrentsAPI: {'✅' if CURRENTS_KEY else '❌'}
    - Gemini AI: {'✅' if GEMINI_API_KEY else '❌'}
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p><b>🔍 RAG-Enhanced Multi-Source Fake News Detection System</b></p>
        <p><small>Combining RAG Technology, ML Models, AI Analysis, and Multi-API Verification • Always verify important news independently</small></p>
        <p><small><em>⚠️ Content validation ensures meaningful analysis of news content only</em></small></p>
    </div>
    """, 
    unsafe_allow_html=True
)