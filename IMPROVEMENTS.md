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

## Batch 2 — Tier 1 bug fixes (found during end-to-end testing)

### 6. Proxy no longer hangs on `POST /api/query` — `backend/src/server.ts`
**Problem:** `express.json()` / `express.urlencoded()` were mounted *before* the
`/api` proxy, so they consumed the request body; the proxied POST then reached
FastAPI with an empty body and hung (`http_code=000`, observed in testing).
WebSocket and multipart `/api/ingest` were unaffected, which is why the UI worked.
**Fix:** Mount the body parsers **after** `app.use('/api', apiProxy)` so proxied
requests keep their raw body stream.
**Files:** [server.ts](two_stage_rag/backend/src/server.ts)

### 7. WebSocket no longer starves the event loop — `api.py`
**Problem:** The WS handler ran the blocking pipeline (hybrid search, cross-encoder,
Gemini streaming) directly in the async coroutine, blocking the event loop. On a
cold/slow query this tripped strict WS keepalive timeouts (we saw a `1011` ping
timeout from the Python `websockets` client).
**Fix:** Wrap each blocking stage in `await asyncio.to_thread(...)`, and pull each
streamed chunk via `asyncio.to_thread(next, gen, sentinel)`, so the loop stays
responsive to pings between chunks. Applied to the HTTP `/api/query` path too for
consistency/concurrency.
**Files:** [api.py](two_stage_rag/api.py)

## Verification (Batch 2)
- `py_compile api.py` clean; Node proxy boots with the reordered middleware.
- **Proxy POST fix:** `POST /api/query` through :3001 now returns `200` in ~3s
  (was `http_code=000`, hung at 8s before the fix).
- **WS event-loop fix:** `test_ws.py` with its *default* 20s keepalive completed a
  **cold** query (12.8s, cross-encoder loading mid-request) with **no `1011`** —
  the exact scenario that failed before. All 3 stages + streamed answer received.

## Batch 3 — Tier 2 (RAG quality)

### 8. BM25 now indexes the FULL collection after ingest — `pipeline.py`
**Problem (found in testing):** after an ingest, `create_pipeline` set the BM25
chunks to the *newly ingested docs only*, so keyword search silently lost every
previously-ingested document until a server restart (masked by vector search).
**Fix:** `create_pipeline` always rebuilds chunks from the full persisted
collection, keeping BM25 and vector retrievers in sync.
**Files:** [pipeline.py](two_stage_rag/pipeline.py)

### 9. Ingest replace-mode — `pipeline.py`, `ingest.py`, `api.py`
**Why:** ingestion appends, so re-ingesting mixes old + new docs with no way to
reset. **Fix:** added `reset_vector_store()` and a `replace` flag
(`create_pipeline(..., replace=True)`, `POST /api/ingest` form field `replace`)
that clears the collection before ingesting. `/api/status` `documents_loaded`
now reflects the full store, not just the last upload.
**Files:** [ingest.py](two_stage_rag/ingest.py), [pipeline.py](two_stage_rag/pipeline.py), [api.py](two_stage_rag/api.py)

### 10. Low-relevance "I don't know" short-circuit — `llm.py`, `pipeline.py`, `api.py`
**Why:** when nothing relevant is retrieved, the LLM is still called and may
grasp at irrelevant chunks. **Fix:** opt-in `ANSWER_MIN_SCORE` (env) — if the top
reranked score is below it, return `NO_ANSWER_MESSAGE` and skip the LLM call
(saves cost, avoids hallucination). Applied to CLI, HTTP, and WS paths.
**Files:** [llm.py](two_stage_rag/llm.py), [pipeline.py](two_stage_rag/pipeline.py), [api.py](two_stage_rag/api.py)

### 11. Whitespace normalization on ingest — `ingest.py`
**Why:** PDF/DOCX extraction produced doubled spaces (`Acme  Technologies`) and
stray newlines, which hurts chunking. **Fix:** `_normalize_text()` collapses
non-breaking spaces, runs of spaces/tabs, and 3+ blank lines on load.
**Files:** [ingest.py](two_stage_rag/ingest.py)

### NOT done — #3 Source citations in the UI (BLOCKED)
Showing source filename/page in the chat requires editing the React UI and
rebuilding `frontend/dist`, but Vite 8 needs Node 20+ and this machine is on
Node 18. Deferred until Node is upgraded (or the dep is pinned to a Node-18
compatible Vite). The backend already returns scores/snippets to support it.

## Verification (Batch 3)
- `py_compile` clean on api/pipeline/reranker/llm/ingest; imports OK.
- **Normalization:** `'Acme  Technologies\xa0Inc.\n\n\n\nLeave   Policy'` → `'Acme Technologies Inc.\n\nLeave Policy'`.
- **BM25 full collection:** `create_pipeline(None)` builds BM25 over all 66 chunks.
- **Short-circuit:** with `ANSWER_MIN_SCORE=1000`, a query returned `NO_ANSWER_MESSAGE`
  and skipped the LLM (`[Relevance] top score 8.934 < 1000.0 — skipping LLM`).
- **Replace-mode:** `create_pipeline([zephyr], replace=True)` → collection cleared, 1 chunk.
- **BM25 fix:** appending `sample2` then → BM25 covers BOTH (`['sample2.txt','zephyr_test.txt']`),
  where previously it would have indexed only the new doc.
- ChromaDB restored to the committed sample-only baseline after testing.

## Batch 4 — Tier 3 (production-readiness)

### 12. Eval harness — `eval.py` (new)
A retrieval-quality test set: fixed `(question → expected keywords)` cases run
against the live ChromaDB, checking the expected text appears in the top reranked
docs. Deterministic, no LLM calls (CI-safe); `--answers` also grades Gemini output.
Exit code 0/1. **Verified:** 6/6 retrieval cases pass on the sample corpus.

### 13. Logging — `api.py`
Added a timestamped/leveled `logging` config and converted the server's `print()`
calls to `logging` (`logger.info` / `warning` / `exception`); dropped the now-unused
`traceback` import.

### 14. API hardening — `api.py`
- **Optional API-key auth** (`API_KEY` env): when set, `/api/query` + `/api/ingest`
  (and the WebSocket) require a matching `X-API-Key`; `/api/status` stays open.
- **Upload limits**: `MAX_UPLOAD_FILES` (default 10) and `MAX_UPLOAD_MB` (default 25)
  enforced on `/api/ingest` (400 / 413).
- **Bug fix:** the ingest handler's broad `except Exception` was turning validation
  errors (size/type) into 500s — now re-raises `HTTPException` unchanged.
**Verified** via FastAPI `TestClient`: 401 without/with wrong key, passes with key;
413 over size, 400 too-many/unsupported-type, 401 unauthenticated ingest.

### 15. Dockerize — `Dockerfile.api`, `backend/Dockerfile`, `docker-compose.yml`, `.dockerignore` (new)
One-command stack: `cd two_stage_rag && docker compose up --build` → open
http://localhost:3001. `api` (FastAPI, CPU-Torch) + `proxy` (multi-stage: builds the
React UI, serves it, proxies `/api` to `api:8000`). `compose config` validates;
a full build needs the Docker daemon running (wasn't available in this sandbox).

### 16. Fix System 1 (`pdf_rag`) for openai>=1.0 — `pdf_rag/embeddings.py`, `pdf_rag/retriever.py`
Migrated `openai.Embeddings.create` → `client.embeddings.create` and
`openai.ChatCompletion.create` → `client.chat.completions.create`, using a lazily
created `OpenAI()` client (import works without `OPENAI_API_KEY`; key needed only at
call time). **Verified:** compiles + imports under openai 2.41.0. End-to-end run still
needs a real OpenAI key.

## Notes
- New env knobs (both opt-in, set in `.env`): `RERANK_MIN_SCORE`, `ANSWER_MIN_SCORE`. Tier 3 adds `API_KEY`, `MAX_UPLOAD_FILES`, `MAX_UPLOAD_MB`.
- After changing Python files, **restart uvicorn** (the server was started without `--reload`) for the running web app to pick them up.
- The relevance threshold is **opt-in**: set `RERANK_MIN_SCORE` in `.env` (e.g. `RERANK_MIN_SCORE=0`) to enable it.
- These are behavioural/robustness fixes; the deeper items (source-citation UI, low-relevance short-circuit, ingest replace-mode, Dockerization, fixing System 1's OpenAI SDK) are tracked separately and not in this batch.
