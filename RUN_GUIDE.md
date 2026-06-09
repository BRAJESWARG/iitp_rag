# Run Guide — IITP RAG (Linux)

These are the **actual, verified-working** steps for this Linux machine. They account for the gotchas described in `PROJECT_DETAILS.md` (macOS-only bundled venv/node_modules, missing deps, CUDA-vs-CPU Torch, Node 18 vs Vite 8).

> **Repo root used below:**
> `/home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag`

---

## Prerequisites

- Python 3.10+ (tested on 3.12)
- Node.js (v18 works for the proxy; **Node 20+** only needed for the Vite dev server)
- A **Google Gemini API key** → https://aistudio.google.com/app/apikey

---

## ⭐ System 2 — Two-Stage RAG (Gemini)

### One-time setup

```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag

# 1. Build a fresh Linux virtualenv (the bundled one is macOS-only)
rm -rf venv
python3 -m venv venv

# 2. Install CPU-only PyTorch FIRST (avoids the multi-GB CUDA download)
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install the rest + the packages requirements.txt forgets
venv/bin/python -m pip install -r requirements.txt google-genai langchain-classic
#    For the web app, also:
venv/bin/python -m pip install fastapi uvicorn python-multipart

# 4. Configure your key
cp -n .env.example .env    # if .env doesn't exist yet
#    then edit .env:  GOOGLE_API_KEY=<your-rotated-key>
```

> Tip: use `venv/bin/python` directly (as above) so you don't have to `source venv/bin/activate` or worry about the current directory.

### Option A — CLI (simplest)

The sample doc is already indexed in `chroma_db/`, so you can query immediately.

```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag

# single question:
venv/bin/python main.py --query "What is a transformer model?"

# interactive (ask many; type 'exit' to quit):
venv/bin/python main.py

# index your own documents:
venv/bin/python main.py --ingest /path/to/your.pdf --query "Summarize it"
```

> First run downloads two small models (`all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2`, ~180 MB total), so it takes ~1–2 minutes. Later runs are fast.

### Option B — Web UI

Because Node here is v18 and the frontend pins Vite 8 (needs Node 20+), use the **pre-built** frontend served by the Node proxy. This needs **2 servers** in **2 terminals**, then you open port **3001**.

**First time only — reinstall the Node proxy deps for Linux:**
```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag/backend
rm -rf node_modules
npm install
```

**Terminal 1 — FastAPI backend (start this FIRST):**
```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag
venv/bin/python -m uvicorn api:app --port 8000
# wait for: "Application startup complete." + "Existing pipeline loaded from ChromaDB"
```

**Terminal 2 — Node proxy:**
```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag/backend
npm run dev
# wait for: "Server is running on port 3001"
```

**Open in your browser:**

> ## 👉 http://localhost:3001

| URL | What it is |
|-----|------------|
| **http://localhost:3001** | ✅ The chat UI — open this |
| http://localhost:8000 | Raw API (returns a small JSON, *not* the UI) |
| http://localhost:8000/docs | Swagger API explorer |

**Using the UI:** wait for "Pipeline Ready" (top-right), type a question, press Enter, and watch the stages animate (Hybrid Search → Cross Encoder → Gemini) with the answer streaming in. Drag-drop a PDF/TXT into the sidebar to index your own docs.

**To stop:** press `Ctrl+C` in each terminal.

### (Optional) Live dev server with hot-reload
Only if you upgrade to **Node 20+** (e.g. via `nvm install 20`):
```bash
cd /home/mihup/Documents/GitHub/bgmaster/iitp_rag/iitp_rag/two_stage_rag/frontend
rm -rf node_modules
npm install
npm run dev        # http://localhost:5173 (proxies /api → :3001 → :8000)
```

---

## System 1 — PDF RAG (OpenAI + FAISS)

⚠️ **This does not run as-is** — `pdf_rag/embeddings.py` and `pdf_rag/retriever.py` use the pre-1.0 OpenAI API, which errors on the `openai>=1.0` it installs. The code needs migrating to the new SDK first.

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
| `ModuleNotFoundError: No module named 'dotenv'` (or langchain, chromadb…) | Deps not installed yet / wrong venv | Finish the install; use `venv/bin/python` |
| `uvicorn: command not found` | Activated the broken macOS venv | Recreate venv (`rm -rf venv && python3 -m venv venv`) and reinstall |
| pip stuck downloading for many minutes | Pulling the CUDA build of Torch (multi-GB) | Install CPU Torch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `You installed esbuild for another platform` | `backend/node_modules` came from macOS | `cd backend && rm -rf node_modules && npm install` |
| `EADDRINUSE: address already in use :::3001` | A server is already on that port | `ss -ltnp \| grep -E ':(3001\|8000)'` then `fuser -k 3001/tcp 8000/tcp` |
| Browser shows `{"message":"Two-Stage RAG Pipeline API"...}` | You opened port 8000 (raw API) | Open **http://localhost:3001** instead |
| UI loads but every request errors | FastAPI (8000) isn't running | Start Terminal 1 (uvicorn) before using the UI |
| `python: can't open file '.../main.py'` | Wrong directory | `cd` into `two_stage_rag`; `main.py` lives there |

---

## Security Reminder

`two_stage_rag/.env` previously held real API keys committed to git. It is now gitignored. **Rotate** the `GOOGLE_API_KEY` and `PINECONE_API_KEY`, and keep the new values only in your local `.env` (never commit it).
