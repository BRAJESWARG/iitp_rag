# Run Guide — IITP RAG (Linux)

**Verified-working** steps for this Linux machine. They account for the gotchas we
hit (macOS-only bundled venv/node_modules, CUDA-vs-CPU Torch, Node 18 → 20, the
npm rolldown-binding bug). See [PROJECT_DETAILS.md](PROJECT_DETAILS.md) for what
each file does and [CODE_EXPLAINED.md](CODE_EXPLAINED.md) for how it works.

> **Repo root used below:** `/home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag`

---

## Prerequisites

- **Python 3.10+** (tested on 3.12)
- **Node.js 20+** — installed here via nvm and set as the default. New terminals
  pick it up automatically. In an *older* terminal still on Node 18, run:
  `. ~/.nvm/nvm.sh && nvm use 20`  (check with `node -v` → `v20.x`)
- A **Google Gemini API key** → https://aistudio.google.com/app/apikey

---

## ⭐ System 2 — Two-Stage RAG (Gemini)

### One-time Python setup

```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag

# 1. Fresh Linux virtualenv (the bundled one is macOS-only)
rm -rf venv && python3 -m venv venv
venv/bin/python -m pip install --upgrade pip

# 2. Install CPU-only PyTorch FIRST (avoids the multi-GB CUDA download)
venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install everything else (requirements.txt is now complete)
venv/bin/python -m pip install -r requirements.txt

# 4. Configure your key
cp -n .env.example .env        # if .env doesn't exist yet
#    then edit .env:  GOOGLE_API_KEY=<your-rotated-key>
```

> Tip: call `venv/bin/python` directly so you don't have to activate the venv or
> worry about the current directory.

### Option A — CLI (simplest)

The sample doc is already indexed in `chroma_db/`, so you can query immediately.

```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag

venv/bin/python main.py --query "What is a transformer model?"   # single question
venv/bin/python main.py                                          # interactive ('exit' to quit)
venv/bin/python main.py --ingest /path/to/your.pdf --query "Summarize it"   # index your own
```

> First run downloads two small models (`all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2`,
> ~180 MB total) — ~1–2 min. Later runs are fast.

### Option B — Web UI (3 terminals)

Each terminal must be on **Node 20** (new terminals already are). FastAPI is Python,
so its Node version doesn't matter.

**First time only — reinstall the JS deps for Linux** (the bundled ones are macOS):
```bash
# backend
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag/backend
rm -rf node_modules package-lock.json && npm install
# frontend
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag/frontend
rm -rf node_modules package-lock.json && npm install
```
> ⚠️ **Do not re-run a bare `npm install` in `frontend/` afterward** — it re-triggers
> an npm bug ([#4828](https://github.com/npm/cli/issues/4828)) that drops Vite 8's
> native binding (`@rolldown/binding-linux-x64-gnu`) and you'll get
> *"Cannot find native binding"*. If you must reinstall, remove **both**
> `node_modules` and `package-lock.json` first (as above).

**Terminal 1 — FastAPI backend (start FIRST):**
```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag
venv/bin/python -m uvicorn api:app --port 8000      # add --reload to auto-apply code edits
# wait for "Application startup complete." + "Existing pipeline loaded from ChromaDB"
```

**Terminal 2 — Node proxy:**
```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag/backend
npm run dev      # serves the pre-built UI + proxies /api → :8000; wait for "port 3001"
```

**Terminal 3 — Vite dev server (optional, for live editing the React UI):**
```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag/frontend
npm run dev      # http://localhost:5173 (hot reload; proxies /api → :3001 → :8000)
```

**Open in your browser:**

> ## 👉 http://localhost:5173  (hot-reload dev)  —  or  http://localhost:3001  (pre-built)

| URL | What it is |
|-----|------------|
| **http://localhost:5173** | Vite dev UI (hot reload) — needs Terminal 3 |
| **http://localhost:3001** | Pre-built UI via the Node proxy — needs only Terminals 1 & 2 |
| http://localhost:8000 | Raw FastAPI (returns a small JSON, *not* the UI) |
| http://localhost:8000/docs | Swagger API explorer |

**Using the UI:** wait for "Pipeline Ready", type a question, press Enter, and watch
the stages animate (Hybrid Search → Cross Encoder → Gemini) with the answer streaming
in. Drag-drop a PDF/TXT into the sidebar to index your own docs. **Stop:** `Ctrl+C` in each terminal.

### Optional tuning (env vars in `.env`)

Both are **off by default** — add to `.env` to enable:

| Variable | Effect |
|----------|--------|
| `RERANK_MIN_SCORE` | Drop reranked chunks scoring below this (keeps ≥1). Cross-encoder scores aren't normalized across queries, so tune per corpus, e.g. `RERANK_MIN_SCORE=0`. |
| `ANSWER_MIN_SCORE` | If the top reranked score is below this, return "I don't know" and skip the LLM call entirely. |

**Replace vs append on ingest:** `POST /api/ingest` accepts a `replace` form field
(`true` clears the collection first). The CLI/`create_pipeline(..., replace=True)`
does the same. Default is append.

---

## System 1 — PDF RAG (OpenAI + FAISS)

⚠️ **Does not run as-is** — `pdf_rag/embeddings.py` and `pdf_rag/retriever.py` use
the pre-1.0 OpenAI API, which errors on the `openai>=1.0` it installs. Needs migrating
to the new SDK first (`client.embeddings.create` / `client.chat.completions.create`).

```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."           # a real key, not the placeholder
python run_rag.py /real/path/to/document.pdf --build --query "What are the main findings?"
```

---

## Troubleshooting (errors we actually hit)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError` (dotenv, langchain, chromadb…) | Deps not installed / wrong venv | Finish setup; use `venv/bin/python` |
| `uvicorn: command not found` | Activated the broken macOS venv | `rm -rf venv && python3 -m venv venv` then reinstall |
| pip stuck downloading for minutes | Pulling the CUDA build of Torch (multi-GB) | Install CPU Torch first (step 2 above) |
| `You are using Node.js 18… Vite requires 20+` / `CustomEvent is not defined` | Old terminal on Node 18 | `. ~/.nvm/nvm.sh && nvm use 20`, or open a new terminal |
| `Cannot find native binding` / `@rolldown/binding-linux-x64-gnu` | npm optional-dep bug after a bare `npm install` | In `frontend/`: `rm -rf node_modules package-lock.json && npm install` |
| `You installed esbuild for another platform` | `backend/node_modules` came from macOS | `cd backend && rm -rf node_modules package-lock.json && npm install` |
| `EADDRINUSE :::3001` (or 8000/5173) | Server already on that port | `ss -ltnp \| grep -E ':(3001\|8000\|5173)'` then `fuser -k 3001/tcp 8000/tcp 5173/tcp` |
| Browser shows `{"message":"Two-Stage RAG Pipeline API"…}` | You opened port 8000 (raw API) | Open **:5173** or **:3001** instead |
| UI loads but every request errors / hangs | FastAPI (8000) not running | Start Terminal 1 first |
| `python: can't open file '.../main.py'` | Wrong directory | `cd` into `two_stage_rag` |

---

## Security Reminder

`two_stage_rag/.env` holds your API keys — **never commit it** (GitHub push protection
will block it, and the key would leak). It was committed once in early history;
**rotate** the `GOOGLE_API_KEY` and `PINECONE_API_KEY` and keep the new values only
in your local `.env`.
