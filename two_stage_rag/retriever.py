"""
retriever.py — Stage 1: Hybrid Search (BM25 + Vector EnsembleRetriever)
========================================================================
This module implements HIGH RECALL retrieval using two complementary methods:

  ┌─────────────────────────────────────────────────────────────────┐
  │  BM25 Retriever       →  Keyword matching (lexical overlap)     │
  │  Vector Retriever     →  Semantic similarity (dense embeddings) │
  │         ↓                          ↓                            │
  │         └──────── EnsembleRetriever (50% + 50%) ───────────┘    │
  │                           ↓                                     │
  │              Top 100 Candidate Documents                         │
  └─────────────────────────────────────────────────────────────────┘

WHY HYBRID?
- BM25 excels at exact keyword matches (e.g., technical terms, names)
- Vector search excels at semantic similarity (paraphrases, synonyms)
- Combining both gives much higher recall than either alone

Usage:
    from retriever import build_hybrid_retriever, hybrid_search
    retriever = build_hybrid_retriever(chunks, vector_store)
    candidates = hybrid_search(retriever, "What is a transformer?")
"""

from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma  # modern standalone package

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
STAGE1_K = 100   # Number of candidates to retrieve in Stage 1
BM25_WEIGHT = 0.5    # Weight for BM25 retriever
VECTOR_WEIGHT = 0.5  # Weight for Vector retriever (must sum to 1.0)


def build_bm25_retriever(chunks: List[Document]) -> BM25Retriever:
    """
    Build a BM25 (Best Match 25) keyword retriever from document chunks.

    BM25 is a probabilistic ranking function based on TF-IDF that:
    - Rewards documents with high term frequency (TF)
    - Penalizes very long documents to avoid length bias
    - Uses inverse document frequency (IDF) to downweight common words
    
    Best for: Exact keyword matches, named entities, technical terms.

    Args:
        chunks: List of Document objects (from ingest.py).

    Returns:
        BM25Retriever configured to retrieve top-k documents.
    """
    print("  Building BM25 retriever (keyword-based)...")

    # BM25Retriever.from_documents() tokenizes all chunk texts
    # and builds an inverted index for fast keyword lookup
    bm25_retriever = BM25Retriever.from_documents(chunks)

    # Set k: how many documents BM25 should return per query
    bm25_retriever.k = STAGE1_K

    print(f"  ✓ BM25 retriever ready (k={STAGE1_K})")
    return bm25_retriever


def build_vector_retriever(vector_store: Chroma) -> object:
    """
    Build a dense vector retriever from the ChromaDB vector store.

    Vector retrieval uses the Bi-Encoder (all-MiniLM-L6-v2) to:
    - Encode the query into a dense embedding vector
    - Find the nearest neighbor chunk embeddings via cosine similarity
    - Returns semantically similar docs even without keyword overlap

    Best for: Paraphrases, synonyms, conceptual similarity.

    Args:
        vector_store: Initialized Chroma vector store (from ingest.py).

    Returns:
        VectorStoreRetriever configured for top-k similarity search.
    """
    print("  Building Vector retriever (semantic-based)...")

    # as_retriever() creates a retriever interface from the vector store
    # search_type="similarity" uses cosine similarity by default
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": STAGE1_K},
    )

    print(f"  ✓ Vector retriever ready (k={STAGE1_K})")
    return vector_retriever


def build_hybrid_retriever(
    chunks: List[Document],
    vector_store: Chroma,
) -> EnsembleRetriever:
    """
    Combine BM25 and Vector retrievers into a hybrid EnsembleRetriever.

    EnsembleRetriever uses Reciprocal Rank Fusion (RRF) to merge results:
    - Runs both retrievers independently in parallel
    - Assigns RRF scores based on rank position from each retriever
    - Combines scores using provided weights [0.5, 0.5]
    - Deduplicates merged results by document content

    Args:
        chunks: Raw document chunks (for BM25 indexing).
        vector_store: ChromaDB vector store (for vector retrieval).

    Returns:
        EnsembleRetriever combining both retrieval methods.
    """
    print("\n[Building Hybrid Retriever]")

    # Stage 1A: BM25 keyword retriever
    bm25_retriever = build_bm25_retriever(chunks)

    # Stage 1B: Dense vector retriever
    vector_retriever = build_vector_retriever(vector_store)

    # Combine using EnsembleRetriever with equal weights
    # weights=[0.5, 0.5] means both retrievers contribute equally
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[BM25_WEIGHT, VECTOR_WEIGHT],
    )

    print(f"  ✓ Hybrid EnsembleRetriever ready "
          f"(BM25={BM25_WEIGHT}, Vector={VECTOR_WEIGHT})")
    return hybrid_retriever


def hybrid_search(
    retriever: EnsembleRetriever,
    query: str,
) -> List[Document]:
    """
    Execute Stage 1 hybrid search: retrieve top candidates for a query.

    This is the HIGH RECALL stage — we cast a wide net by retrieving
    up to 100 candidates. Precision is handled in Stage 2 (reranker.py).

    Args:
        retriever: Built EnsembleRetriever from build_hybrid_retriever().
        query: User's natural language query string.

    Returns:
        List of up to 100 candidate Document objects (deduplicated).
    """
    print(f"\n[Stage 1] Running Hybrid Search...")
    print(f"  Query: '{query}'")

    # invoke() runs both BM25 and Vector search, merges and deduplicates
    candidates = retriever.invoke(query)

    print(f"  [Stage 1] Hybrid Search retrieved {len(candidates)} candidates "
          f"via Hybrid Search (BM25 + Vector)")

    return candidates
