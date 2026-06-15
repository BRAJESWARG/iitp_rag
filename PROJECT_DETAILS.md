# Project Details — IITP RAG

This repository contains **two independent Retrieval-Augmented Generation (RAG) systems**:

| # | System | Folder | Stack | Status |
|---|--------|--------|-------|--------|
| 1 | **PDF RAG (OpenAI + FAISS)** | `pdf_rag/` + `run_rag.py` | OpenAI embeddings + FAISS + GPT rerank | ⚠️ Uses pre-1.0 OpenAI API — needs a code fix to run (see [Known Issues](#known-issues)) |
| 2 | **Two-Stage RAG (Gemini)** | `two_stage_rag/` | LangChain + ChromaDB + Cross-Encoder + Gemini, with a FastAPI backend, Node proxy, and React UI | ✅ Verified working (CLI + Web UI) |

---

## Repository Layout

```
iitp_rag/
├── README.md                  # Documents System 1 (pdf_rag)
├── PROJECT_DETAILS.md         # ← this file (file-by-file reference)
├── CODE_EXPLAINED.md          # Concepts + Mermaid diagrams
├── RUN_GUIDE.md               # How to run everything
├── IMPROVEMENTS.md            # Change log (Tier 1 / Tier 2 fixes)
├── .gitignore
├── requirements.txt           # System 1 deps (openai, faiss, pdfplumber, numpy, dotenv)
├── run_rag.py                 # System 1 CLI entry point
│
├── pdf_rag/                   # ── SYSTEM 1: OpenAI + FAISS RAG ──
│   ├── __init__.py            # Exposes RAGPDFQA
│   ├── qa.py                  # RAGPDFQA orchestrator
│   ├── pdf_loader.py          # PDF text extraction + chunking
│   ├── embeddings.py          # OpenAI embeddings wrapper
│   ├── vector_store.py        # FAISS index build/search/save/load
│   └── retriever.py           # Semantic search + LLM rerank + answer gen
│
└── two_stage_rag/             # ── SYSTEM 2: Gemini Two-Stage RAG ──
    ├── .env                   # GOOGLE_API_KEY, PINECONE_API_KEY (never commit)
    ├── .env.example           # Template for .env
    ├── requirements.txt       # Python deps (complete; install CPU torch first)
    ├── setup.sh               # macOS-oriented setup script
    ├── README.md              # Architecture write-up
    ├── HOW_TO_RUN.md          # Original (macOS) run notes
    ├── ingest.py              # Stage 0: load → chunk → ChromaDB
    ├── retriever.py           # Stage 1: BM25 + Vector hybrid search
    ├── reranker.py            # Stage 2: Cross-Encoder reranking
    ├── llm.py                 # Final: Gemini answer generation
    ├── pipeline.py            # Orchestrates Stage 1 → 2 → LLM
    ├── main.py                # Interactive CLI
    ├── api.py                 # FastAPI REST + WebSocket backend
    ├── test_ws.py             # WebSocket smoke-test client
    ├── sample_docs/sample.txt # Sample document (transformers/attention/RAG)
    ├── chroma_db/             # Pre-ingested vector store (66 chunks of sample.txt)
    ├── backend/               # Node.js Express proxy (TypeScript)
    │   ├── src/server.ts      # Proxies /api → FastAPI, serves frontend/dist
    │   └── package.json
    └── frontend/              # React + Vite UI (TypeScript)
        ├── src/App.tsx
        ├── src/hooks/useRAG.ts
        ├── src/types/index.ts
        ├── src/components/     # Header, Sidebar, ChatWindow, MessageBubble, PipelineViz, DocumentUpload
        ├── dist/               # Pre-built production bundle (served by the Node proxy)
        └── package.json
```

---

## System 2 — Two-Stage RAG Architecture

```
User Query
   │
   ▼  STAGE 1 — High Recall (Hybrid Search)
   │   BM25 keyword retriever  (weight 0.5)
   │   Vector retriever        (weight 0.5, ChromaDB cosine)
   │   → EnsembleRetriever (Reciprocal Rank Fusion) → up to 100 candidates
   ▼  STAGE 2 — High Precision (Cross-Encoder)
   │   score([query, doc]) for every candidate, sort desc → Top 5
   ▼  FINAL — Grounded Generation
   │   Build context from Top 5 → gemini-2.5-flash (temp 0.2) → answer
   ▼
Grounded Answer
```

---

## File-by-File Details

### System 1 — `pdf_rag/` (OpenAI + FAISS)

| File | What it does | Key config |
|------|--------------|-----------|
| `run_rag.py` | CLI entry. Args: `pdf_path`, `--build`, `--load`, `--query`, `--index-path` (default `./store/pdf_rag_index`), `--top-k` (8), `--rerank-k` (5). Requires `OPENAI_API_KEY`. | — |
| `pdf_rag/qa.py` | `RAGPDFQA` dataclass; `build_knowledge_base()`, `load_knowledge_base()`, `query()`. | chunk_size 900, overlap 200, embed `text-embedding-3-large`, rerank/answer `gpt-4.1-mini` |
| `pdf_rag/pdf_loader.py` | `extract_text_from_pdf()` (pdfplumber, page-by-page); `split_text()` (word-based overlapping chunks). | — |
| `pdf_rag/embeddings.py` | `get_openai_embedding()` / `get_openai_embeddings()`. **⚠️ uses `openai.Embeddings.create` (pre-1.0 API).** | — |
| `pdf_rag/vector_store.py` | `build_faiss_index()` (`IndexFlatIP` + L2-normalize = cosine), `index_search()`, `save_index()`/`load_index()` (`.idx` + `.meta.pkl`). | — |
| `pdf_rag/retriever.py` | `semantic_search()`, `rerank_documents()` (LLM scores docs 0–100 as JSON), `generate_answer()`. **⚠️ uses `openai.ChatCompletion.create` (pre-1.0 API).** | temperature 0.0 |

### System 2 — `two_stage_rag/` (Gemini)

| File | What it does | Key config |
|------|--------------|-----------|
| `ingest.py` | Stage 0. Load PDF/TXT → `RecursiveCharacterTextSplitter` → ChromaDB. `ingest_documents()`, `load_vector_store()`. | chunk 500 / overlap 50, embed `all-MiniLM-L6-v2` (CPU), persist `./chroma_db`, collection `two_stage_rag` |
| `retriever.py` | Stage 1. `build_hybrid_retriever()` = BM25 + Vector via `EnsembleRetriever` (imported from `langchain_classic`); `hybrid_search()`. | `STAGE1_K=100`, weights 0.5 / 0.5 |
| `reranker.py` | Stage 2. `CrossEncoderReranker` (singleton). `cross_encoder_rerank()` (docs) + `cross_encoder_rerank_with_scores()` (docs+scores in ONE pass). Optional `RERANK_MIN_SCORE` drops low-scoring chunks (keeps ≥1). | model `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-k 5 |
| `llm.py` | Final stage. `GeminiLLM` using the **`google-genai`** SDK. `generate_answer()` + `generate_answer_stream()`. Grounded prompt. Retries transient `{429,500,502,503,504}` (catches `ClientError`+`ServerError`). Optional `ANSWER_MIN_SCORE` gate → `NO_ANSWER_MESSAGE`. | model `gemini-2.5-flash`, temp 0.2, max 1024 tokens, backoff [5,15,30]s |
| `pipeline.py` | `TwoStageRAGPipeline` (builds retriever on init; `run()` does Stage 1→2→LLM with the `ANSWER_MIN_SCORE` short-circuit). `create_pipeline(file_paths, replace=False)` — ingests if given (`replace` clears first), then **rebuilds BM25 over the full collection**. | — |
| `main.py` | Colored CLI. `--ingest FILE...`, `--query "..."`, or interactive loop (`exit`/`quit` to stop). | — |
| `api.py` | FastAPI. `GET /api/status`, `POST /api/ingest` (with `replace` flag), `POST /api/query`, `WS /api/ws/query` (streams stages + chunks). Blocking stages run via `asyncio.to_thread` (keeps the WS keepalive alive). Explicit CORS origins. Auto-loads ChromaDB on startup. | port 8000 |
| `test_ws.py` | Standalone WebSocket client that sends one query and prints streamed events. | — |
| `backend/src/server.ts` | Node/Express proxy. Forwards `/api/*` (incl. WebSocket upgrades) to FastAPI; serves `frontend/dist`; `GET /health`. Body parsers mounted **after** the proxy so proxied POST bodies aren't consumed. | `NODE_PORT=3001`, `PYTHON_API_URL=http://localhost:8000` |
| `frontend/src/hooks/useRAG.ts` | Core React hook. Polls `/api/status` every 10s; `sendQuery()` opens a WebSocket (HTTP `/api/query` fallback); `ingestFiles()` posts multipart. | `API_BASE = '/api'` |
| `frontend/src/components/*` | `Header` (status), `Sidebar` (`DocumentUpload` + `PipelineViz` + system info), `ChatWindow` (chat + input + example queries), `MessageBubble`, `PipelineViz` (animated stages). | — |
| `frontend/vite.config.ts` | Dev server on 5173; proxies `/api` → `http://localhost:3001`. | — |

---

## Models & Endpoints Summary

| Role | Model / Value |
|------|---------------|
| System 2 — Bi-Encoder (embeddings) | `all-MiniLM-L6-v2` (384-dim, CPU) |
| System 2 — Cross-Encoder (reranker) | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| System 2 — LLM | `gemini-2.5-flash` (temp 0.2) |
| System 1 — Embeddings | `text-embedding-3-large` |
| System 1 — Rerank / Answer LLM | `gpt-4.1-mini` |
| Vector DB (System 2) | ChromaDB, local persist at `two_stage_rag/chroma_db/` |
| Vector index (System 1) | FAISS `IndexFlatIP` (cosine), saved to `./store/` |

**Port map (System 2 web app):**

| URL | What |
|-----|------|
| http://localhost:3001 | The chat **UI** (open this) — Node proxy serving `frontend/dist` |
| http://localhost:8000 | Raw FastAPI (returns a status JSON at `/`) |
| http://localhost:8000/docs | Swagger API explorer |
| http://localhost:5173 | Vite dev server (hot reload) — needs Node 20+ (installed via nvm) |

---

## Tuning knobs (optional)

Both env vars are **off by default** — set them in `two_stage_rag/.env`:

- `RERANK_MIN_SCORE` — drop reranked chunks scoring below this (keeps ≥1). Cross-encoder scores aren't normalized across queries, so tune per corpus.
- `ANSWER_MIN_SCORE` — if the top reranked score is below this, return "I don't know" and skip the LLM call (saves cost, avoids hallucination).

## Status of Known Issues

**Resolved** (see [IMPROVEMENTS.md](IMPROVEMENTS.md) for details):
- ✅ `two_stage_rag/requirements.txt` completed (added `google-genai`, `langchain-classic`, `fastapi`, `uvicorn`, `python-multipart`).
- ✅ **Node upgraded to 20** (via nvm, set as default) — Vite 8 dev server works.
- ✅ macOS-origin `venv/`, `backend/node_modules/`, `frontend/node_modules/` reinstalled for Linux.
- ✅ Invalid CORS (`*` + credentials) → explicit dev origins.
- ✅ Proxy `POST /api/query` hang (body parsers ran before the proxy) → parsers moved after.
- ✅ WebSocket keepalive starvation (blocking the event loop) → blocking stages run in `asyncio.to_thread`.
- ✅ BM25 silently dropping prior docs after an ingest → rebuilt over the full collection.

**Still open:**
1. **System 1 (`pdf_rag`) won't run as-is** — `embeddings.py` / `retriever.py` use the pre-1.0 OpenAI API (`openai.Embeddings.create`, `openai.ChatCompletion.create`) but install `openai>=1.0`. Needs migrating to `client.embeddings.create` / `client.chat.completions.create`.
2. **Source citations in the UI (Tier 2 #3)** — the backend returns scores/snippets, but displaying source filenames needs a React change + `frontend/dist` rebuild (now possible on Node 20).
3. **Secrets** — `two_stage_rag/.env` was committed in early history. **Rotate** `GOOGLE_API_KEY` + `PINECONE_API_KEY`; never commit `.env` (GitHub push protection will block it).

**Reminders:**
- Install **CPU Torch first** — the default PyTorch wheel is the multi-GB CUDA build; this app is CPU-only.
- Don't bare-`npm install` in `frontend/` — npm bug [#4828](https://github.com/npm/cli/issues/4828) drops Vite 8's native binding. Remove `node_modules` + `package-lock.json` first.
