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
├── PROJECT_DETAILS.md         # ← this file
├── RUN_GUIDE.md               # How to run everything
├── .gitignore                 # Ignores .env, venv, node_modules, __pycache__, etc.
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
    ├── .env                   # GOOGLE_API_KEY, PINECONE_API_KEY (gitignored)
    ├── .env.example           # Template for .env
    ├── requirements.txt       # Python deps (INCOMPLETE — see Known Issues)
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
| `reranker.py` | Stage 2. `CrossEncoderReranker` (singleton via `get_reranker()`), `cross_encoder_rerank()`. | model `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-k 5 |
| `llm.py` | Final stage. `GeminiLLM` using the **`google-genai`** SDK (`from google import genai`). `generate_answer()` + `generate_answer_stream()`. Grounded prompt ("say I don't know"). Retries on HTTP 429. | model `gemini-2.5-flash`, temp 0.2, max 1024 tokens, backoff [5,15,30]s |
| `pipeline.py` | `TwoStageRAGPipeline` (builds retriever on init, runs Stage 1→2→LLM in `run()`). `create_pipeline(file_paths)` — ingests if given, else loads existing ChromaDB and rebuilds chunks for BM25. | — |
| `main.py` | Colored CLI. `--ingest FILE...`, `--query "..."`, or interactive loop (`exit`/`quit` to stop). | — |
| `api.py` | FastAPI. `GET /api/status`, `POST /api/ingest`, `POST /api/query`, `WS /api/ws/query` (streams stage events + answer chunks). Auto-loads existing ChromaDB on startup. CORS `*`. | port 8000 |
| `test_ws.py` | Standalone WebSocket client that sends one query and prints streamed events. | — |
| `backend/src/server.ts` | Node/Express proxy. Forwards `/api/*` (incl. WebSocket upgrades) to FastAPI; serves `frontend/dist` as a SPA; `GET /health`. | `NODE_PORT=3001`, `PYTHON_API_URL=http://localhost:8000` |
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
| http://localhost:5173 | Vite dev server — **only works on Node 20+** (see Known Issues) |

---

## Known Issues / Gotchas

1. **`two_stage_rag/requirements.txt` is incomplete.** The code imports packages it doesn't list. You must also install: `google-genai`, `langchain-classic`, and (for the web API) `fastapi`, `uvicorn`, `python-multipart`. See `RUN_GUIDE.md`.
2. **PyTorch defaults to the CUDA build (~multi-GB).** This app runs CPU-only (`device="cpu"`), so install the **CPU wheel** of Torch to avoid downloading gigabytes of unused GPU libraries.
3. **The bundled `venv/` and `backend/node_modules/` are macOS builds** (this repo was zipped on a Mac — note the `__MACOSX/` folder). They will **not run on Linux** — recreate the venv and reinstall node_modules on the target platform.
4. **Frontend pins Vite 8, which requires Node 20+.** On Node 18 the Vite dev server (5173) won't start; use the pre-built `frontend/dist` served by the Node proxy on **3001** instead.
5. **System 1 (`pdf_rag`) won't run as-is.** `embeddings.py` and `retriever.py` use the pre-1.0 OpenAI API (`openai.Embeddings.create`, `openai.ChatCompletion.create`) but `requirements.txt` pins `openai>=1.0`. They need migrating to `client.embeddings.create` / `client.chat.completions.create`.
6. **Secrets:** `two_stage_rag/.env` was previously committed to git with real API keys. It is now gitignored. **Rotate the `GOOGLE_API_KEY` and `PINECONE_API_KEY`** and keep the new values only in your local `.env`.
