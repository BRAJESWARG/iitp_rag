# Improvements Log — `feature/rag-improvements`

A running record of changes made on the `feature/rag-improvements` branch, why, and where. Newest first within each section.

> Base: branched from `master` at `0398e63`. All changes here are in `two_stage_rag/` unless noted.

---

## Batch 1 — Robustness & efficiency quick wins

### 1. Retry on transient server errors (not just 429) — `llm.py`
**Problem:** The retry loop only handled HTTP `429` and only caught `ClientError`. A Gemini **503 "high demand"** is a `ServerError`, so it wasn't caught or retried — it crashed the request (we hit this live).
**Fix:** Catch both `ClientError` and `ServerError`; retry on `{429, 500, 502, 503, 504}` with the existing backoff. Applied to both `generate_answer()` and `generate_answer_stream()`.
**Files:** [llm.py](two_stage_rag/llm.py)

### 2. Relevance score threshold in the reranker — `reranker.py`
**Problem:** The reranker always returned exactly 5 docs, even when 4 scored ~ −11 (clearly irrelevant). Junk chunks were stuffed into Gemini's context.
**Fix:** Added a configurable minimum cross-encoder score (`RERANK_MIN_SCORE`, env-overridable). Docs below it are dropped, but at least the single top doc is always kept (so the grounded prompt can still answer or say "I don't know").
**Files:** [reranker.py](two_stage_rag/reranker.py)

### 3. Score the cross-encoder once, not twice — `reranker.py` + `api.py`
**Problem:** `api.py` reranked (which scores internally) and then called `reranker.model.predict()` again just to display scores — double cross-encoder inference per query.
**Fix:** Added `rerank_with_scores()` / `cross_encoder_rerank_with_scores()` returning `(doc, score)` pairs; `api.py` now reranks once and reuses the scores.
**Files:** [reranker.py](two_stage_rag/reranker.py), [api.py](two_stage_rag/api.py)

### 4. Fix CORS config — `api.py`
**Problem:** `allow_origins=["*"]` together with `allow_credentials=True` is rejected by browsers and is insecure.
**Fix:** Explicit dev origin allow-list (`5173`, `3000`, `3001`), keeping credentials valid.
**Files:** [api.py](two_stage_rag/api.py)

### 5. Complete & accurate `requirements.txt` — `two_stage_rag/requirements.txt`
**Problem:** It omitted packages the code imports (`google-genai`, `langchain-classic`, `fastapi`, `uvicorn`, `python-multipart`), causing the install failures we hit.
**Fix:** Added the missing packages, noted the CPU-Torch install step, and kept the existing ones.
**Files:** [requirements.txt](two_stage_rag/requirements.txt)

---

## Verification (Batch 1)
- `py_compile` clean on `llm.py`, `reranker.py`, `api.py`.
- Imports OK; `RETRYABLE_STATUS = {429,500,502,503,504}`; CORS origins = explicit list.
- `cross_encoder_rerank_with_scores` returns sorted `(Document, float)` pairs.
- `RERANK_MIN_SCORE=0` correctly dropped 2/3 irrelevant (−11) chunks, kept the relevant one.
- Full CLI query "What is a transformer model?" ran end-to-end in ~13s with a correct grounded answer.

## Notes
- After changing Python files, **restart uvicorn** (the server was started without `--reload`) for the running web app to pick them up.
- The relevance threshold is **opt-in**: set `RERANK_MIN_SCORE` in `.env` (e.g. `RERANK_MIN_SCORE=0`) to enable it.
- These are behavioural/robustness fixes; the deeper items (source-citation UI, low-relevance short-circuit, ingest replace-mode, Dockerization, fixing System 1's OpenAI SDK) are tracked separately and not in this batch.
