"""
eval.py — Tiny retrieval-quality evaluation harness
====================================================
Runs a fixed set of (question → expected keywords) cases against the CURRENT
ChromaDB collection and checks whether the expected text shows up in the
top reranked documents. This is a *retrieval* eval — deterministic and cheap
(no LLM calls), so it's safe to run in CI to catch regressions when you change
chunk size, k, weights, or the models.

Usage:
    venv/bin/python eval.py              # retrieval eval (no Gemini, fast)
    venv/bin/python eval.py --answers    # also call Gemini and grade the answer text

The default test set targets the bundled sample.txt (transformers / attention /
RAG / BM25). Edit TEST_CASES for your own corpus. Exit code is 0 if every case
passes, 1 otherwise.
"""

import argparse
import sys
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Each case: query + keywords that should appear in the retrieved context.
# `any_of` passes if ANY listed keyword is found (case-insensitive).
TEST_CASES = [
    {"query": "What is a transformer model?",
     "any_of": ["Attention Is All You Need", "attention mechanism"]},
    {"query": "What does BM25 stand for?",
     "any_of": ["Best Match 25"]},
    {"query": "What dataset is the cross encoder fine-tuned on?",
     "any_of": ["MS MARCO"]},
    {"query": "How many dimensions does all-MiniLM-L6-v2 produce?",
     "any_of": ["384"]},
    {"query": "What is multi-head attention?",
     "any_of": ["Multi-Head Attention", "multiple attention heads", "parallel"]},
    {"query": "How does positional encoding work in transformers?",
     "any_of": ["positional encoding", "sinusoidal", "position information"]},
]

TOP_K_FOR_EVAL = 5


def _hit(text_blob: str, keywords) -> str | None:
    low = text_blob.lower()
    for kw in keywords:
        if kw.lower() in low:
            return kw
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval-quality eval harness")
    parser.add_argument("--answers", action="store_true",
                        help="Also call Gemini and check the answer text (slower, needs API key)")
    args = parser.parse_args()

    # Build the pipeline from the existing ChromaDB (no ingest).
    from pipeline import create_pipeline
    from retriever import hybrid_search
    from reranker import cross_encoder_rerank_with_scores

    print("Loading pipeline from existing ChromaDB...\n")
    pipeline = create_pipeline(file_paths=None)

    results = []
    for case in TEST_CASES:
        query = case["query"]
        expected = case["any_of"]

        candidates = hybrid_search(pipeline.retriever, query)
        ranked = cross_encoder_rerank_with_scores(query, candidates)[:TOP_K_FOR_EVAL]
        context = "\n".join(doc.page_content for doc, _ in ranked)
        retrieval_hit = _hit(context, expected)

        row = {
            "query": query,
            "expected": expected,
            "retrieval_pass": retrieval_hit is not None,
            "matched": retrieval_hit,
            "top_score": round(ranked[0][1], 3) if ranked else None,
        }

        if args.answers:
            from llm import generate_answer
            answer = generate_answer(query, [doc for doc, _ in ranked])
            row["answer_pass"] = _hit(answer, expected) is not None
            row["answer"] = answer[:120]

        results.append(row)

    # ── Report ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RETRIEVAL EVAL RESULTS")
    print("=" * 72)
    retr_passed = 0
    ans_passed = 0
    for r in results:
        ok = "PASS" if r["retrieval_pass"] else "FAIL"
        retr_passed += r["retrieval_pass"]
        line = f"[{ok}] retrieval | score={r['top_score']} | {r['query']}"
        if not r["retrieval_pass"]:
            line += f"  (expected one of: {r['expected']})"
        print(line)
        if args.answers:
            aok = "PASS" if r["answer_pass"] else "FAIL"
            ans_passed += r["answer_pass"]
            print(f"      [{aok}] answer: {r['answer']}")

    total = len(results)
    print("-" * 72)
    print(f"Retrieval: {retr_passed}/{total} passed")
    if args.answers:
        print(f"Answers:   {ans_passed}/{total} passed")
    print("=" * 72)

    # Fail the run (non-zero exit) if any retrieval case missed.
    return 0 if retr_passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
