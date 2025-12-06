"""
Test script for RAG system corrections
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_rag_system():
    """Test the RAG system functionality"""
    print("=" * 60)
    print("Testing RAG System")
    print("=" * 60)
    
    # Test 1: Import modules
    print("\n1. Testing imports...")
    try:
        from rag_system import RAGKnowledgeBase, check_rag_health
        from rag_pipeline import RAGPipeline, get_or_create_rag_pipeline
        from ai_analysis import standard_gemini_analysis, rag_enhanced_gemini_analysis, check_api_keys
        print("   ✅ All imports successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Initialize RAG Knowledge Base
    print("\n2. Testing RAG Knowledge Base initialization...")
    try:
        rag = RAGKnowledgeBase()
        print("   ✅ RAG Knowledge Base initialized")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False
    
    # Test 3: Check RAG health
    print("\n3. Checking RAG system health...")
    try:
        health = check_rag_health(rag)
        print("   System Health:")
        for component, status in health.items():
            print(f"     {component}: {status}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    # Test 4: Test fact addition
    print("\n4. Testing fact addition...")
    try:
        rag.add_fact(
            content="This is a test fact for verification.",
            category="test",
            verified=True,
            sources=["Test Source"]
        )
        print("   ✅ Fact added successfully")
    except Exception as e:
        print(f"   ❌ Fact addition failed: {e}")
    
    # Test 5: Test fact retrieval
    print("\n5. Testing fact retrieval...")
    try:
        query = "test fact verification"
        results = rag.retrieve_relevant_facts(query, top_k=3)
        print(f"   Retrieved {len(results)} facts for query: '{query}'")
        for i, fact in enumerate(results[:3], 1):
            print(f"     {i}. {fact.get('content', '')[:60]}...")
            print(f"        Similarity: {fact.get('similarity', 0):.3f}")
        print("   ✅ Fact retrieval successful")
    except Exception as e:
        print(f"   ❌ Fact retrieval failed: {e}")
    
    # Test 6: Test RAG Pipeline
    print("\n6. Testing RAG Pipeline...")
    try:
        pipeline = get_or_create_rag_pipeline()
        print(f"   Pipeline mode: {pipeline.embedding_mode}")
        
        # Test retrieval
        contexts = pipeline.retrieve("climate change", k=3)
        print(f"   Retrieved {len(contexts)} contexts")
        
        # Test answer generation
        answer = pipeline.generate_answer("Is climate change real?", k=3, use_gemini=False)
        print(f"   Answer: {answer.answer[:100]}...")
        print(f"   Confidence: {answer.confidence:.2f}")
        print("   ✅ RAG Pipeline successful")
    except Exception as e:
        print(f"   ❌ RAG Pipeline failed: {e}")
    
    # Test 7: Check API keys
    print("\n7. Checking API configuration...")
    try:
        api_status = check_api_keys()
        print(f"   Gemini Available: {'✅' if api_status['gemini_available'] else '❌'}")
        print(f"   NewsAPI Available: {'✅' if api_status['newsapi_available'] else '❌'}")
    except Exception as e:
        print(f"   ❌ API check failed: {e}")
    
    # Test 8: Test AI analysis functions
    print("\n8. Testing AI analysis functions...")
    try:
        test_text = "The Earth is round and orbits the Sun."
        
        # Test standard analysis
        print("   Testing standard_gemini_analysis...")
        result = standard_gemini_analysis(test_text)
        if "unavailable" in result.lower() or "error" in result.lower():
            print(f"   ⚠️  Gemini not configured: {result[:80]}...")
        else:
            print(f"   ✅ Standard analysis: {result[:80]}...")
        
        # Test RAG-enhanced analysis
        print("   Testing rag_enhanced_gemini_analysis...")
        relevant_facts = rag.retrieve_relevant_facts(test_text, top_k=3)
        result = rag_enhanced_gemini_analysis(test_text, relevant_facts)
        if "unavailable" in result.lower() or "error" in result.lower():
            print(f"   ⚠️  Gemini not configured: {result[:80]}...")
        else:
            print(f"   ✅ RAG-enhanced analysis: {result[:80]}...")
    except Exception as e:
        print(f"   ❌ AI analysis test failed: {e}")
    
    # Test 9: Get statistics
    print("\n9. Getting RAG statistics...")
    try:
        stats = rag.get_stats()
        print(f"   Total facts: {stats['total_facts']}")
        print(f"   Verified facts: {stats['verified_facts']}")
        print(f"   Categories: {list(stats['categories'].keys())}")
        print(f"   ChromaDB active: {stats['chromadb_active']}")
        print(f"   Embeddings active: {stats['embeddings_active']}")
        print(f"   Fallback active: {stats['fallback_active']}")
        print("   ✅ Statistics retrieved")
    except Exception as e:
        print(f"   ❌ Statistics retrieval failed: {e}")
    
    print("\n" + "=" * 60)
    print("RAG System Test Complete!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_rag_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test script failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
