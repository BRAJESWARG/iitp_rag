"""
reranker.py — Stage 2: Cross Encoder Reranker (High Precision)
===============================================================
This module implements HIGH PRECISION reranking using a Cross Encoder model.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 1 Output: 100 Candidate Documents                        │
  │         ↓                                                       │
  │  Cross Encoder: score([query, doc_i]) for each doc              │
  │         ↓    (Deep interaction between query & document)        │
  │  Sort by score (descending)                                     │
  │         ↓                                                       │
  │  Return Top 5 Highest-Scoring Documents                         │
  └─────────────────────────────────────────────────────────────────┘

WHY CROSS ENCODER vs BI-ENCODER?
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Property            │ Bi-Encoder           │ Cross Encoder        │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ How it works        │ Encode query & doc   │ Encode query+doc     │
│                     │ separately           │ TOGETHER in one pass │
│ Speed               │ Very fast (parallel) │ Slower (sequential)  │
│ Accuracy            │ Good (approximate)   │ Excellent (precise)  │
│ Best use            │ Stage 1 (100 docs)   │ Stage 2 (top-5 only) │
└─────────────────────┴──────────────────────┴──────────────────────┘

Usage:
    from reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    top5 = reranker.rerank(query, candidates_100)
"""

from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANKED = 5   # How many documents to keep after reranking


class CrossEncoderReranker:
    """
    Wraps the sentence-transformers CrossEncoder for document reranking.

    The Cross Encoder (ms-marco-MiniLM-L-6-v2) was fine-tuned on the
    MS MARCO dataset for passage ranking — specifically designed for
    measuring relevance between a question and a passage.

    It processes the [QUERY, DOCUMENT] pair through a single BERT-like
    transformer, allowing full attention between query and document tokens.
    This "deep interaction" produces much more accurate relevance scores
    than comparing independent embeddings.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        """
        Load the Cross Encoder model from HuggingFace Hub.

        Args:
            model_name: HuggingFace model identifier for the cross encoder.
        """
        print(f"  Loading Cross Encoder model: '{model_name}' ...")
        # CrossEncoder downloads and caches the model on first run
        self.model = CrossEncoder(model_name)
        self.top_k = TOP_K_RERANKED
        print(f"  ✓ Cross Encoder loaded successfully")

    def rerank(
        self,
        query: str,
        candidates: List[Document],
    ) -> List[Document]:
        """
        Rerank candidate documents using the Cross Encoder.

        For each candidate document:
          score = cross_encoder.predict([query, doc.page_content])

        This computes a relevance score by jointly encoding the query
        and document, enabling fine-grained matching.

        Args:
            query: The user's query string.
            candidates: List of 100 candidate Document objects from Stage 1.

        Returns:
            Top-5 Document objects sorted by relevance score (highest first).
        """
        if not candidates:
            print("  [WARNING] No candidates to rerank.")
            return []

        print(f"\n[Stage 2] Cross Encoder Reranking {len(candidates)} candidates...")

        # Build list of [query, document_text] pairs for batch prediction
        # Cross encoder needs BOTH query and document in a single input
        query_doc_pairs = [
            [query, doc.page_content]
            for doc in candidates
        ]

        # Batch predict relevance scores for all pairs simultaneously
        # Returns a numpy array of float scores (higher = more relevant)
        scores = self.model.predict(query_doc_pairs)

        # Pair each document with its relevance score
        scored_docs: List[Tuple[Document, float]] = list(
            zip(candidates, scores)
        )

        # Sort by score in DESCENDING order (highest relevance first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Extract top-k documents after reranking
        top_docs = [doc for doc, score in scored_docs[: self.top_k]]

        # Print scores for transparency
        print(f"  [Stage 2] Reranked to Top {self.top_k} using Cross Encoder")
        print(f"  Top {self.top_k} relevance scores:")
        for i, (doc, score) in enumerate(scored_docs[: self.top_k], 1):
            # Show a snippet of each top doc for visibility
            snippet = doc.page_content[:80].replace("\n", " ")
            print(f"    {i}. Score={score:.4f} | '{snippet}...'")

        return top_docs


# ------------------------------------------------------------------
# Module-level convenience: shared reranker instance (lazy loaded)
# ------------------------------------------------------------------
_reranker_instance: CrossEncoderReranker = None


def get_reranker() -> CrossEncoderReranker:
    """
    Return a singleton CrossEncoderReranker (loads model only once).
    Avoids re-loading the model on every query in the pipeline.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


def cross_encoder_rerank(
    query: str,
    candidates: List[Document],
) -> List[Document]:
    """
    Convenience function: rerank candidates using the singleton reranker.

    Args:
        query: User's query string.
        candidates: List of candidate Documents from Stage 1.

    Returns:
        Top-5 reranked Document objects.
    """
    reranker = get_reranker()
    return reranker.rerank(query, candidates)
