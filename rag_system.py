import os
import json
import hashlib
import requests
from typing import List, Dict, Optional
import numpy as np

# RAG imports - with proper error handling
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available")

# Streamlit import with fallback
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create a fallback for st functions
    class DummyStreamlit:
        def success(self, msg): print(f"SUCCESS: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    st = DummyStreamlit()

# Load Google Fact Check API Key
GOOGLEFACT_KEY = os.getenv("GOOGLE_FACTCHECK")


def fetch_google_fact_checks(query: str = None, max_results: int = 10) -> List[Dict]:
    """
    Fetch fact check claims from Google Fact Check API
    """
    if not GOOGLEFACT_KEY:
        return []
        
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "key": GOOGLEFACT_KEY, 
        "query": query or "", 
        "languageCode": "en"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        claims = data.get("claims", [])
        
        result_claims = []
        for i, claim in enumerate(claims[:max_results]):
            text = claim.get("text", "").strip()
            if text:
                claimant = claim.get("claimant", "GoogleFactCheck")
                review = claim.get("claimReview", [])
                sources = [r.get("publisher", {}).get("name", "") for r in review]
                sources = [s for s in sources if s]
                
                result_claims.append({
                    "id": f"google_fact_{i}",
                    "text": text,
                    "claimant": claimant,
                    "sources": sources or [claimant]
                })
                
        return result_claims
        
    except Exception as e:
        st.warning(f"Could not fetch Google Fact Check claims: {e}")
        return []


class RAGKnowledgeBase:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.fact_database = []
        self.vectorizer = None
        self.fact_vectors = None
        self.chroma_persist_dir = None
        self.initialize_rag_system()

    def initialize_rag_system(self):
        """Initialize ChromaDB + embeddings, fallback to TF-IDF"""
        # Load fact database first
        self.load_fact_database()
        
        # Try to initialize ChromaDB and embeddings
        if CHROMADB_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.chroma_persist_dir = os.path.join(os.getcwd(), "chroma_db")
                os.makedirs(self.chroma_persist_dir, exist_ok=True)
                
                self.chroma_client = chromadb.PersistentClient(
                    path=self.chroma_persist_dir,
                    settings=Settings(anonymized_telemetry=False, allow_reset=True)
                )
                
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.toast("✅ ChromaDB and SentenceTransformer initialized")
                
                # Initialize collection
                self._initialize_collection()
                return
                
            except Exception as e:
                st.warning(f"ChromaDB or Embeddings init failed: {e}")
                self.embedding_model = None
                self.chroma_client = None
        
        # Fall back to TF-IDF if ChromaDB/embeddings fail
        self.use_fallback_rag()

    def _initialize_collection(self):
        """Initialize or get ChromaDB collection"""
        if not self.chroma_client:
            return
            
        try:
            self.collection = self.chroma_client.get_collection("news_facts")
            st.toast(f"✅ Connected to existing ChromaDB collection")
        except Exception:
            try:
                self.collection = self.chroma_client.create_collection(
                    "news_facts", 
                    metadata={"description": "news facts"}
                )
                st.toast("✅ Created new ChromaDB collection")
                self.populate_chroma_collection()
            except Exception as e:
                st.warning(f"ChromaDB collection creation failed: {e}")
                self.collection = None
                self.use_fallback_rag()
                return

        # Fetch Google Fact Checks if API key is available
        if GOOGLEFACT_KEY:
            self.fetch_google_fact_checks_and_add()

    def use_fallback_rag(self):
        """Fallback RAG implementation using TF-IDF"""
        if not SKLEARN_AVAILABLE:
            st.error("❌ Neither ChromaDB nor scikit-learn available for RAG")
            return
            
        try:
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            if self.fact_database:
                fact_texts = [fact['content'] for fact in self.fact_database]
                self.fact_vectors = self.vectorizer.fit_transform(fact_texts)
                st.info("ℹ️ Using TF-IDF fallback for RAG functionality")
            else:
                st.warning("⚠️ No fact database available for RAG")
        except Exception as e:
            st.error(f"❌ Fallback RAG initialization failed: {e}")
            self.fact_database = []

    def load_fact_database(self):
        """Load or create fact database"""
        try:
            with open('fact_database.json', 'r', encoding='utf-8') as f:
                self.fact_database = json.load(f)
            st.toast(f"📚 Loaded {len(self.fact_database)} facts from database")
        except FileNotFoundError:
            # Create sample fact database
            self.fact_database = [
                {
                    "id": "fact_001",
                    "content": "The Earth is round and orbits the Sun. This has been scientifically proven.",
                    "category": "science",
                    "verified": True,
                    "sources": ["NASA", "Scientific consensus"]
                },
                {
                    "id": "fact_002", 
                    "content": "Vaccines are safe and effective for preventing diseases when approved by health authorities.",
                    "category": "health",
                    "verified": True,
                    "sources": ["WHO", "CDC", "FDA"]
                },
                {
                    "id": "fact_003",
                    "content": "Climate change is real and primarily caused by human activities according to scientific consensus.",
                    "category": "environment",
                    "verified": True,
                    "sources": ["IPCC", "NASA", "NOAA"]
                },
                {
                    "id": "fact_004",
                    "content": "5G technology does not cause COVID-19 or other health problems according to scientific studies.",
                    "category": "technology",
                    "verified": True,
                    "sources": ["WHO", "FCC", "IEEE"]
                }
            ]
            self.save_fact_database()
            st.info("📚 Created sample fact database")
        except Exception as e:
            st.error(f"❌ Error loading fact database: {e}")
            self.fact_database = []

    def save_fact_database(self):
        """Save fact database to file"""
        try:
            with open('fact_database.json', 'w', encoding='utf-8') as f:
                json.dump(self.fact_database, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.warning(f"Could not save fact database: {e}")

    def populate_chroma_collection(self):
        """Populate ChromaDB collection with existing facts"""
        if not self.collection or not self.embedding_model or not self.fact_database:
            return
            
        try:
            contents = [f['content'] for f in self.fact_database]
            embeddings = self.embedding_model.encode(contents).tolist()
            ids = [f['id'] for f in self.fact_database]
            metadatas = [
                {
                    "category": f['category'], 
                    "verified": f['verified'],
                    "sources": json.dumps(f.get('sources', []))
                } 
                for f in self.fact_database
            ]
            
            # Clear existing data
            try:
                existing_ids = self.collection.get()['ids']
                if existing_ids:
                    self.collection.delete(ids=existing_ids)
            except:
                pass
                
            # Add facts to collection
            self.collection.add(
                documents=contents, 
                embeddings=embeddings, 
                metadatas=metadatas, 
                ids=ids
            )
            st.toast(f"✅ Populated ChromaDB with {len(contents)} facts")
            
        except Exception as e:
            st.warning(f"Could not populate ChromaDB: {e}")

    def add_fact(self, content: str, category: str = "news", verified: bool = True, sources: Optional[List[str]] = None):
        """Add a new fact to the database"""
        if not content or not content.strip():
            return
            
        fact_id = f"fact_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        
        # Check if fact already exists
        if any(f['id'] == fact_id for f in self.fact_database):
            return
            
        fact = {
            "id": fact_id, 
            "content": content.strip(), 
            "category": category, 
            "verified": verified, 
            "sources": sources or []
        }
        
        self.fact_database.append(fact)
        self.save_fact_database()
        
        # Add to ChromaDB if available
        if self.collection and self.embedding_model:
            try:
                emb = self.embedding_model.encode([content]).tolist()
                self.collection.add(
                    documents=[content], 
                    embeddings=emb,
                    metadatas=[{
                        "category": category, 
                        "verified": verified,
                        "sources": json.dumps(sources or [])
                    }], 
                    ids=[fact_id]
                )
            except Exception as e:
                st.warning(f"Could not add to ChromaDB: {e}")
        
        # Update TF-IDF vectors if using fallback
        elif self.vectorizer and SKLEARN_AVAILABLE:
            try:
                fact_texts = [fact['content'] for fact in self.fact_database]
                self.fact_vectors = self.vectorizer.fit_transform(fact_texts)
            except Exception as e:
                st.warning(f"Could not update TF-IDF vectors: {e}")

    def retrieve_relevant_facts(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant facts for a given query"""
        if not query or not query.strip():
            return []
            
        # Try ChromaDB first
        if self.collection and self.embedding_model:
            return self._retrieve_with_chromadb(query, top_k)
        # Fall back to TF-IDF
        elif self.vectorizer and self.fact_vectors is not None:
            return self._retrieve_with_fallback(query, top_k)
        else:
            # Simple keyword matching as last resort
            return self._retrieve_with_keywords(query, top_k)

    def _retrieve_with_chromadb(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using ChromaDB"""
        try:
            q_emb = self.embedding_model.encode([query]).tolist()
            res = self.collection.query(query_embeddings=q_emb, n_results=min(top_k, 50))
            
            docs = res.get('documents', [[]])[0]
            distances = res.get('distances', [[]])[0]
            
            results = []
            for i, doc in enumerate(docs):
                fact = next((f for f in self.fact_database if f['content'] == doc), None)
                if fact:
                    fact_copy = fact.copy()
                    # Convert distance to similarity score (assuming cosine distance)
                    similarity = 1.0 - (distances[i] if i < len(distances) else 0.5)
                    fact_copy["similarity"] = max(0, min(1, similarity))
                    results.append(fact_copy)
            
            return results
            
        except Exception as e:
            st.warning(f"ChromaDB retrieval error: {e}")
            return self._retrieve_with_fallback(query, top_k)

    def _retrieve_with_fallback(self, query: str, top_k: int) -> List[Dict]:
        """Fallback retrieval using TF-IDF"""
        if not SKLEARN_AVAILABLE or not hasattr(self, 'vectorizer') or not self.fact_database:
            return []
            
        try:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.fact_vectors)[0]
            top_idx = sims.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_idx:
                if sims[idx] > 0.1:  # Similarity threshold
                    fact = self.fact_database[idx].copy()
                    fact["similarity"] = float(sims[idx])
                    results.append(fact)
            
            return results
            
        except Exception as e:
            st.warning(f"TF-IDF retrieval failed: {e}")
            return self._retrieve_with_keywords(query, top_k)

    def _retrieve_with_keywords(self, query: str, top_k: int) -> List[Dict]:
        """Simple keyword-based retrieval as last resort"""
        query_words = set(query.lower().split())
        scored_facts = []
        
        for fact in self.fact_database:
            fact_words = set(fact['content'].lower().split())
            overlap = len(query_words.intersection(fact_words))
            if overlap > 0:
                similarity = overlap / len(query_words.union(fact_words))
                fact_copy = fact.copy()
                fact_copy["similarity"] = similarity
                scored_facts.append(fact_copy)
        
        # Sort by similarity and return top_k
        scored_facts.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_facts[:top_k]

    def fetch_google_fact_checks_and_add(self):
        """
        Fetches Google Fact Check claims and adds them into ChromaDB collection
        """
        claims = fetch_google_fact_checks(query="politics", max_results=5)
        if not claims:
            st.warning("⚠️ No Google Fact Check claims fetched")
            return

        for claim in claims:
            try:
                if self.collection and self.embedding_model:
                    self.collection.add(
                        ids=[claim["id"]],
                        documents=[claim["text"]],
                        embeddings=[self.embedding_model.encode(claim["text"]).tolist()]
                    )
            except Exception as e:
                st.warning(f"Failed to add claim: {e}")

        st.toast(f"✅ Added {len(claims)} Google Fact Check claims to RAG KB")

    def get_all_facts(self) -> List[Dict]:
        """Get all facts in the database"""
        return self.fact_database

    def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = {
            "total_facts": len(self.fact_database),
            "categories": {},
            "verified_facts": 0,
            "chromadb_active": bool(self.collection),
            "embeddings_active": bool(self.embedding_model),
            "fallback_active": bool(self.vectorizer)
        }
        
        for fact in self.fact_database:
            category = fact.get("category", "unknown")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            if fact.get("verified", True):
                stats["verified_facts"] += 1
        
        return stats


# Helper functions
def analyze_against_facts(news_text: str, relevant_facts: List[Dict]) -> Dict:
    """Analyze news text against retrieved facts"""
    analysis = {
        "contradictions": [],
        "confirmations": [],
        "similarity_scores": [],
        "overall_consistency": "unknown"
    }
    
    if not relevant_facts:
        return analysis
    
    for fact in relevant_facts:
        similarity = fact.get("similarity", 0)
        analysis["similarity_scores"].append(similarity)
        
        if similarity > 0.7:  # High similarity threshold
            if fact.get("verified", True):
                analysis["confirmations"].append({
                    "fact": fact["content"],
                    "similarity": similarity,
                    "sources": fact.get("sources", [])
                })
            else:
                analysis["contradictions"].append({
                    "fact": fact["content"],
                    "similarity": similarity,
                    "sources": fact.get("sources", [])
                })
    
    # Determine overall consistency
    if len(analysis["contradictions"]) > len(analysis["confirmations"]):
        analysis["overall_consistency"] = "contradictory"
    elif len(analysis["confirmations"]) > 0:
        analysis["overall_consistency"] = "consistent"
    else:
        analysis["overall_consistency"] = "neutral"
    
    return analysis


def calculate_kb_confidence(relevant_facts: List[Dict], fact_analysis: Dict) -> float:
    """Calculate confidence score based on knowledge base analysis"""
    if not relevant_facts:
        return 0.0
    
    confirmations = len(fact_analysis["confirmations"])
    contradictions = len(fact_analysis["contradictions"])
    avg_similarity = np.mean(fact_analysis["similarity_scores"]) if fact_analysis["similarity_scores"] else 0
    
    # Calculate confidence based on various factors
    consistency_score = (confirmations - contradictions) / max(len(relevant_facts), 1)
    similarity_score = avg_similarity
    
    confidence = (consistency_score * 0.6 + similarity_score * 0.4) * 100
    return max(0, min(100, confidence))


def comprehensive_news_check_with_rag(news_text: str, rag: RAGKnowledgeBase) -> Dict:
    """
    Extended comprehensive news check with Google Fact Check + RAG integration.
    """
    # Basic structure for results
    results = {
        "sources_found": [],
        "total_matches": 0,
        "matched_articles": [],
        "search_summary": {},
        "confidence_score": 0.0,
        "rag_analysis": None
    }
    
    # Try to import and use fetch_news functions
    try:
        from fetch_news import comprehensive_news_check
        results = comprehensive_news_check(news_text)
        
        # Fetch Google Fact Check claims
        google_claims = []
        if news_text.strip():
            try:
                google_claims = fetch_google_fact_checks(news_text)
            except Exception as e:
                results["search_summary"]["GoogleFactCheck"] = {
                    "found": False,
                    "error": str(e)
                }

        if google_claims:
            results["sources_found"].append("GoogleFactCheck")
            results["total_matches"] += len(google_claims)
            results["matched_articles"].extend(google_claims[:3])
            results["search_summary"]["GoogleFactCheck"] = {
                "found": True,
                "count": len(google_claims)
            }

            # Add claims into the RAG knowledge base
            for claim in google_claims:
                rag.add_fact(
                    content=claim.get("text", ""),
                    category="factcheck",
                    verified=True,
                    sources=[claim.get("claimant", "GoogleFactCheck")]
                )
        else:
            if "GoogleFactCheck" not in results["search_summary"]:
                results["search_summary"]["GoogleFactCheck"] = {"found": False, "count": 0}
                
    except ImportError:
        st.warning("fetch_news module not available, using basic RAG analysis")
    except Exception as e:
        st.error(f"Error in comprehensive news check: {e}")
    
    # RAG Analysis
    try:
        relevant_facts = rag.retrieve_relevant_facts(news_text, top_k=5)
        fact_analysis = analyze_against_facts(news_text, relevant_facts)
        
        results["rag_analysis"] = {
            "relevant_facts": relevant_facts,
            "fact_check_results": fact_analysis,
            "knowledge_base_confidence": calculate_kb_confidence(relevant_facts, fact_analysis)
        }
        
        # Update confidence score considering RAG analysis
        rag_confidence = results["rag_analysis"]["knowledge_base_confidence"]
        base_confidence = results.get("confidence_score", 0)
        results["confidence_score"] = (base_confidence * 0.7 + rag_confidence * 0.3)
        
    except Exception as e:
        st.warning(f"RAG analysis failed: {e}")
        results["rag_analysis"] = {
            "error": str(e),
            "relevant_facts": [],
            "fact_check_results": {},
            "knowledge_base_confidence": 0
        }
    
    return results


def check_rag_health(rag: RAGKnowledgeBase) -> Dict:
    """Check RAG system health and provide diagnostics"""
    try:
        stats = rag.get_stats()
        
        health_status = {
            "ChromaDB": "❌ Not Available",
            "Embeddings": "❌ Not Available", 
            "Knowledge Base": f"✅ {stats['total_facts']} facts" if stats['total_facts'] > 0 else "❌ No facts",
            "Overall": "❌ Limited Functionality"
        }
        
        if stats.get("chromadb_active"):
            try:
                count = rag.collection.count() if rag.collection else 0
                health_status["ChromaDB"] = f"✅ Active ({count} items)"
            except:
                health_status["ChromaDB"] = "⚠️ Connected but unstable"
        
        if stats.get("embeddings_active"):
            health_status["Embeddings"] = "✅ SentenceTransformer loaded"
        elif stats.get("fallback_active"):
            health_status["Embeddings"] = "⚠️ TF-IDF fallback"
        
        # Determine overall status
        if "✅" in health_status["ChromaDB"] and "✅" in health_status["Embeddings"]:
            health_status["Overall"] = "✅ Fully Operational"
        elif "⚠️" in str(health_status.values()) or stats['total_facts'] > 0:
            health_status["Overall"] = "⚠️ Partial Functionality"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "ChromaDB": "❌ Error",
            "Embeddings": "❌ Error", 
            "Knowledge Base": "❌ Error",
            "Overall": "❌ System Error"
        }


# AI Analysis Functions
def rag_enhanced_gemini_analysis(text: str, relevant_facts: List[Dict]) -> str:
    """RAG-enhanced Gemini analysis"""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Gemini API key not available"
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Create context from relevant facts
        context = "\n".join([f"- {fact['content']} (Sources: {', '.join(fact.get('sources', []))})" 
                           for fact in relevant_facts])
        
        prompt = f"""
        Analyze this news text for authenticity using the following knowledge base context:
        
        Knowledge Base Context:
        {context}
        
        News Text to Analyze:
        {text}
        
        Provide a detailed analysis considering:
        1. Consistency with known facts
        2. Credibility indicators
        3. Potential misinformation markers
        4. Overall assessment (REAL/FAKE/UNCERTAIN)
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Gemini analysis failed: {e}"


def standard_gemini_analysis(text: str) -> str:
    """Standard Gemini analysis without RAG context"""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Gemini API key not available"
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Analyze this news text for authenticity:
        
        {text}
        
        Consider:
        1. Language patterns that might indicate fake news
        2. Logical consistency
        3. Factual plausibility
        4. Overall credibility assessment (REAL/FAKE/UNCERTAIN)
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Gemini analysis failed: {e}"


# Example usage and testing
def main():
    """Main function for testing the RAG system"""
    print("🚀 Initializing RAG Knowledge Base...")
    rag = RAGKnowledgeBase()
    
    # Check system health
    health = check_rag_health(rag)
    print("\n📊 System Health:")
    for component, status in health.items():
        print(f"  {component}: {status}")
    
    # Test with sample news
    news_text = "5G technology does not cause COVID-19"
    print(f"\n🔍 Testing with news: '{news_text}'")
    
    # Test RAG retrieval
    relevant_facts = rag.retrieve_relevant_facts(news_text, top_k=3)
    print(f"\n📚 Found {len(relevant_facts)} relevant facts:")
    for i, fact in enumerate(relevant_facts, 1):
        print(f"  {i}. {fact['content']}")
        print(f"     Sources: {fact.get('sources', [])} | Similarity: {fact.get('similarity', 0):.3f}")
    
    # Test comprehensive analysis
    print(f"\n🔬 Running comprehensive analysis...")
    results = comprehensive_news_check_with_rag(news_text, rag)
    print(f"  Confidence Score: {results['confidence_score']:.2f}")
    print(f"  Sources Found: {results['sources_found']}")
    
    if results.get('rag_analysis'):
        rag_analysis = results['rag_analysis']
        print(f"  RAG Confidence: {rag_analysis.get('knowledge_base_confidence', 0):.2f}")
        print(f"  Consistency: {rag_analysis.get('fact_check_results', {}).get('overall_consistency', 'unknown')}")


if __name__ == "__main__":
    main()