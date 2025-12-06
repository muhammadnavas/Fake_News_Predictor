"""Advanced RAG Pipeline for Fake News Project.

This module builds on top of (but does NOT break) the existing `rag_system.RAGKnowledgeBase`.
It adds a higher-level pipeline that supports:

1. Dataset ingestion (streaming large CSVs) from `True.csv` and `Fake.csv`.
2. Chunking & cleaning of article text (title + body) into knowledge base facts.
3. Dual-mode vector storage:
   - Preferred: ChromaDB persistent collection with SentenceTransformer embeddings.
   - Fallback: In-memory TF-IDF vectors.
4. Retrieval interface returning structured contexts.
5. Simple answer generation that synthesizes an answer by:
   - Scoring contexts.
   - Producing a traceable JSON answer object.
6. Optional LLM (Gemini) answer augmentation re-using existing Gemini key if available.

The pipeline is intentionally decoupled so the Streamlit app can:
   - Initialize once and reuse.
   - Trigger ingestion only when requested (avoid long startup times).

Assumptions:
   - The CSV files follow the Kaggle FakeNews dataset format with columns:
     title,text,subject,date (case-insensitive). We handle graceful fallbacks.
"""
from __future__ import annotations

import os
import csv
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterable, Tuple


from rag_system import RAGKnowledgeBase


    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import chromadb  # type: ignore
    _HAS_CHROMA = True
except Exception:
    _HAS_CHROMA = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    import numpy as np  # type: ignore

    np = None  # type: ignore

try:
    import streamlit as st  # type: ignore

    class _Dummy:
        def write(self, *a, **k): print(*a)
        def warning(self, *a, **k): print("WARNING:", *a)
        def error(self, *a, **k): print("ERROR:", *a)
        def success(self, *a, **k): print("SUCCESS:", *a)
        def info(self, *a, **k): print("INFO:", *a)
        def toast(self, *a, **k): print("TOAST:", *a)
    st = _Dummy()  # type: ignore


@dataclass
class RetrievedContext:
    id: str
    content: str
    score: float
    sources: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

@dataclass
class RAGAnswer:
    query: str
    answer: str
    contexts: List[RetrievedContext]
    reasoning: str
    confidence: float
    mode: str  # "embedding" | "tfidf" | "keyword"



def _hash_text(txt: str) -> str:
    return hashlib.md5(txt.encode("utf-8")).hexdigest()[:12]


def _normalize_fieldnames(fields: List[str]) -> Dict[str, str]:
    lower_map = {f.lower(): f for f in fields}
    return lower_map



    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        field_map = _normalize_fieldnames(reader.fieldnames)
        count = 0
        for row in reader:

            yield {
                "title": row.get(field_map.get("title", "title"), "").strip(),
                "text": row.get(field_map.get("text", "text"), "").strip(),
                "subject": row.get(field_map.get("subject", "subject"), "").strip(),
                "date": row.get(field_map.get("date", "date"), "").strip(),
            }
            count += 1
            if limit and count >= limit:
                break



    if not text:
        return []
    parts: List[str] = []
    current = []
    length = 0
    for segment in text.replace("\n", " ").split("."):
        seg = segment.strip()
        if not seg:
            continue
        seg_full = seg + "."
        if length + len(seg_full) > max_chars and current:
            parts.append(" ".join(current).strip())
            current = [seg_full]
            length = len(seg_full)
        else:
            current.append(seg_full)
            length += len(seg_full)
    if current:
        parts.append(" ".join(current).strip())
    return parts


class RAGPipeline:
    """High-level ingestion + retrieval + answer synthesis pipeline.

    Wraps existing `RAGKnowledgeBase` while adding dataset ingestion & answer generation.
    """
    def __init__(self, kb: Optional[RAGKnowledgeBase] = None):
        self.kb = kb or RAGKnowledgeBase()
        self.embedding_mode = "chromadb" if (self.kb.collection and self.kb.embedding_model) else (
            "tfidf" if self.kb.vectorizer else "keyword"
        )


    def ingest_csv_dataset(self, path: str, label: str, limit: Optional[int] = 200):
        """Ingest a CSV dataset, converting each article into 1..N facts.

        label: "real" or "fake" (affects category & verified flag heuristics)
        limit: optional row cap for performance.
        """
        if not os.path.exists(path):
            st.warning(f"Dataset not found: {path}")
            return 0

        added = 0
        try:
            for row in _iter_csv_rows(path, limit=limit):
                title = row.get("title", "")
                body = row.get("text", "")
                if not (title or body):
                    continue
                base = f"{title}. {body}".strip()

                chunks = _chunk_text(base, max_chars=800)
                if not chunks:
                    continue
                for chunk in chunks:
                    fact_id = _hash_text(chunk)

                    if any(f.get("id") == f"fact_{fact_id}" for f in self.kb.fact_database):
                        continue
                    category = row.get("subject", label) or label
                    verified = label == "real"
                    sources = [category.title()]
                    self.kb.add_fact(chunk, category=category, verified=verified, sources=sources)
                    added += 1

            st.error(f"Ingestion error for {path}: {e}")
        if added:
            st.toast(f"✅ Ingested {added} fact chunks from {os.path.basename(path)}")
        else:
            st.info(f"No new facts ingested from {os.path.basename(path)}")
        return added

    def bulk_ingest_default_datasets(self, true_path: str = "True.csv", fake_path: str = "Fake.csv", per_file_limit: int = 150):
        total_added = 0
        total_added += self.ingest_csv_dataset(true_path, label="real", limit=per_file_limit)
        total_added += self.ingest_csv_dataset(fake_path, label="fake", limit=per_file_limit)
        return total_added


    def retrieve(self, query: str, k: int = 5) -> List[RetrievedContext]:
        raw = self.kb.retrieve_relevant_facts(query, top_k=k)
        contexts: List[RetrievedContext] = []
        for fact in raw:
            contexts.append(
                RetrievedContext(
                    id=fact.get("id", _hash_text(fact.get("content", ""))),
                    content=fact.get("content", ""),
                    score=float(fact.get("similarity", 0.0)),
                    sources=fact.get("sources", []),
                    metadata={
                        "category": fact.get("category"),
                        "verified": fact.get("verified"),
                    },
                )
            )
        return contexts

    def generate_answer(self, query: str, k: int = 5, use_gemini: bool = False) -> RAGAnswer:
        contexts = self.retrieve(query, k=k)
        if not contexts:
            return RAGAnswer(
                query=query,
                answer="No relevant facts found in knowledge base.",
                contexts=[],
                reasoning="Knowledge base returned no contexts.",
                confidence=0.0,
                mode=self.embedding_mode,
            )

        supporting = [c for c in contexts if str(c.metadata.get("verified", "False")).lower() in ["true", "1"]]
        contradicting = [c for c in contexts if str(c.metadata.get("verified", "False")).lower() in ["false", "0"]]
        avg_score = sum(c.score for c in contexts) / max(1, len(contexts))
        if supporting and not contradicting:
            base_answer = "Query aligns with verified knowledge base facts."
        elif contradicting and not supporting:
            base_answer = "Query appears to conflict with verified knowledge; may indicate misinformation."
        elif supporting and contradicting:
            base_answer = "Mixed evidence: some facts support while others contradict the claim."
        else:
            base_answer = "Insufficient evidence for a clear determination."

        reasoning_lines = [
            f"Total contexts: {len(contexts)} | Avg similarity: {avg_score:.3f}",
            f"Supporting (verified): {len(supporting)} | Contradicting (unverified): {len(contradicting)}",
        ]
        reasoning = "\n".join(reasoning_lines)
        confidence = min(0.95, max(0.05, avg_score * 0.9 + (len(supporting) - len(contradicting)) * 0.05))

        final_answer = base_answer

        if use_gemini:
            try:
                import google.generativeai as genai  # type: ignore
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-2.5-flash-latest")
                    context_block = "\n".join([f"- {c.content[:300]} (score={c.score:.2f})" for c in contexts])
                    prompt = f"""
You are assisting in fake news verification. Given the user query and retrieved knowledge base contexts, produce a concise factual assessment (<= 120 words).

Query: {query}

Retrieved Contexts:\n{context_block}\n
Guidelines:
1. Cite if evidence is supporting, contradicting, or mixed.
2. DO NOT fabricate facts.
3. Provide an overall classification: REAL / FAKE / MIXED / INSUFFICIENT.
"""
                    resp = model.generate_content(prompt)
                    if resp and getattr(resp, "text", None):
                        final_answer = resp.text.strip()
                        reasoning += "\nLLM augmentation applied."
                else:
                    reasoning += "\nGemini key missing; skipped LLM augmentation."

                reasoning += f"\nGemini augmentation failed: {e}"

        return RAGAnswer(
            query=query,
            answer=final_answer,
            contexts=contexts,
            reasoning=reasoning,
            confidence=confidence,
            mode=self.embedding_mode,
        )



_pipeline_singleton: Optional[RAGPipeline] = None

def get_or_create_rag_pipeline() -> RAGPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = RAGPipeline()
    return _pipeline_singleton



    pipe = get_or_create_rag_pipeline()
    print("Mode:", pipe.embedding_mode)
    added = pipe.bulk_ingest_default_datasets(per_file_limit=5)
    print("Added facts:", added)
    ans = pipe.generate_answer("The government approved a new health policy.", k=3)
    print("Answer:", ans.answer)
    print("Reasoning:\n", ans.reasoning)
