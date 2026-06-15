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

import os
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANKED = 5   # How many documents to keep after reranking

# Optional minimum cross-encoder relevance score, used to drop clearly-irrelevant
# chunks before they reach the LLM. NOTE: cross-encoder logits are NOT normalized
# across queries (a genuinely relevant chunk can still score negative), so this is
# DISABLED by default. Enable/tune it for your corpus via the env var, e.g.:
#   RERANK_MIN_SCORE=0
_MIN_SCORE_ENV = os.getenv("RERANK_MIN_SCORE")
RERANK_MIN_SCORE = float(_MIN_SCORE_ENV) if _MIN_SCORE_ENV not in (None, "") else None


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
        self.min_score = RERANK_MIN_SCORE
        print(f"  ✓ Cross Encoder loaded successfully")

    def rerank_with_scores(
        self,
        query: str,
        candidates: List[Document],
    ) -> List[Tuple[Document, float]]:
        """
        Rerank candidates and return (Document, score) pairs, highest first.

        For each candidate: score = cross_encoder.predict([query, doc]).
        Keeps the top-k, then applies the optional RERANK_MIN_SCORE threshold
        (always keeping at least the single best document).

        Args:
            query: The user's query string.
            candidates: Candidate Document objects from Stage 1.

        Returns:
            List of (Document, score) tuples sorted by relevance (highest first).
        """
        if not candidates:
            print("  [WARNING] No candidates to rerank.")
            return []

        print(f"\n[Stage 2] Cross Encoder Reranking {len(candidates)} candidates...")

        # Cross encoder needs BOTH query and document in a single input.
        # Batch-predict relevance scores for all pairs at once.
        query_doc_pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.model.predict(query_doc_pairs)

        # Pair docs with scores and sort descending (highest relevance first)
        scored_docs: List[Tuple[Document, float]] = sorted(
            zip(candidates, (float(s) for s in scores)),
            key=lambda x: x[1],
            reverse=True,
        )

        top = scored_docs[: self.top_k]

        # Optional relevance threshold — drop clearly-irrelevant docs, but always
        # keep at least the best one so the LLM still has context to ground on.
        if self.min_score is not None:
            filtered = [(d, s) for d, s in top if s >= self.min_score]
            if filtered:
                if len(filtered) < len(top):
                    print(f"  [Stage 2] Threshold {self.min_score}: "
                          f"kept {len(filtered)}/{len(top)} docs")
                top = filtered
            else:
                print(f"  [Stage 2] All docs below threshold {self.min_score}; "
                      f"keeping top 1")
                top = top[:1]

        # Print scores for transparency
        print(f"  [Stage 2] Reranked to Top {len(top)} using Cross Encoder")
        for i, (doc, score) in enumerate(top, 1):
            snippet = doc.page_content[:80].replace("\n", " ")
            print(f"    {i}. Score={score:.4f} | '{snippet}...'")

        return top

    def rerank(
        self,
        query: str,
        candidates: List[Document],
    ) -> List[Document]:
        """Back-compatible wrapper: return just the reranked Documents."""
        return [doc for doc, _ in self.rerank_with_scores(query, candidates)]


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
        Reranked Document objects (top-k, optionally score-filtered).
    """
    return get_reranker().rerank(query, candidates)


def cross_encoder_rerank_with_scores(
    query: str,
    candidates: List[Document],
) -> List[Tuple[Document, float]]:
    """
    Like cross_encoder_rerank but also returns each document's relevance score,
    so callers don't need to run the cross encoder a second time.

    Returns:
        List of (Document, score) tuples sorted by relevance (highest first).
    """
    return get_reranker().rerank_with_scores(query, candidates)
