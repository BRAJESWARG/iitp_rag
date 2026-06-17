"""
rrf_demo.py — Inspect Stage-1 hybrid retrieval (debug / learning tool)
======================================================================
Shows, for a single query, each chunk's BM25 rank, vector rank, and the fused
Reciprocal Rank Fusion (RRF) score that the EnsembleRetriever uses. Handy for
understanding *why* a chunk did or didn't make it into the Stage-1 candidate set.

No LLM calls — retrieval only. Runs against the current ChromaDB collection.

Usage (from the two_stage_rag/ directory):
    venv/bin/python rrf_demo.py
    venv/bin/python rrf_demo.py --query "What is multi-head attention?" --top 15

See RETRIEVAL_EXPLAINED.md for the BM25 + RRF math this demonstrates.
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Show BM25 vs vector ranks + fused RRF scores")
    parser.add_argument("--query", default="How does keyword-based document ranking work?",
                        help="Query to inspect")
    parser.add_argument("--top", type=int, default=10, help="Number of rows to show")
    args = parser.parse_args()

    from pipeline import create_pipeline

    pipeline = create_pipeline(file_paths=None)
    ens = pipeline.retriever                 # EnsembleRetriever (BM25 + Vector)
    bm25, vector = ens.retrievers
    w_bm25, w_vec = ens.weights              # e.g. [0.5, 0.5]
    c = getattr(ens, "c", 60)                # RRF smoothing constant (LangChain default 60)

    print(f"\nQUERY: {args.query}")
    print(f"weights = BM25 {w_bm25}, Vector {w_vec} | RRF c = {c}\n")

    bm25_docs = bm25.invoke(args.query)
    vec_docs = vector.invoke(args.query)

    def rank_map(docs):
        m = {}
        for i, d in enumerate(docs):
            m.setdefault(d.page_content, i + 1)   # 1-based rank, first occurrence
        return m

    rb, rv = rank_map(bm25_docs), rank_map(vec_docs)

    # Preserve first-seen order across both lists (union of retrieved chunks)
    seen, order = set(), []
    for d in bm25_docs + vec_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            order.append(d.page_content)

    rows = []
    for content in order:
        a, b = rb.get(content), rv.get(content)
        score = (w_bm25 / (c + a) if a else 0.0) + (w_vec / (c + b) if b else 0.0)
        rows.append((score, a, b, content[:55].replace("\n", " ")))

    rows.sort(key=lambda x: x[0], reverse=True)

    print(f"{'RRF score':>10} | {'BM25 rk':>7} | {'Vec rk':>6} | chunk")
    print("-" * 92)
    for score, a, b, snip in rows[: args.top]:
        print(f"{score:>10.5f} | {str(a):>7} | {str(b):>6} | {snip}")
    print()


if __name__ == "__main__":
    main()
