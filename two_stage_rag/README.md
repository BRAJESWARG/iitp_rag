# Two-Stage RAG Pipeline

A production-ready **Two-Stage Retrieval-Augmented Generation (RAG)** system built with Python, LangChain, and Google Gemini.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────┐
│ STAGE 1: HIGH RECALL (Hybrid Search)     │
│                                          │
│   BM25 Retriever ──┐                     │
│   (keyword-based)  ├─ EnsembleRetriever  │
│   Vector Retriever ┘  [50% + 50%]        │
│   (semantic-based)                       │
│                                          │
│   Output: Top 100 Candidate Documents    │
└──────────────────────────────────────────┘
    │ 100 candidates
    ▼
┌──────────────────────────────────────────┐
│ STAGE 2: HIGH PRECISION (Cross Encoder)  │
│                                          │
│   score([query, doc_i]) for each doc     │
│   Sort descending by relevance score     │
│                                          │
│   Output: Top 5 Refined Documents        │
└──────────────────────────────────────────┘
    │ 5 docs
    ▼
┌──────────────────────────────────────────┐
│ FINAL: LLM CONTEXT WINDOW (Gemini)       │
│                                          │
│   Context: {top_5_docs}                  │
│   Question: {user_query}                 │
│   → gemini-1.5-flash API                 │
│                                          │
│   Output: Grounded Final Answer          │
└──────────────────────────────────────────┘
```

---

## File Structure

```
two_stage_rag/
├── .env                    # GOOGLE_API_KEY=your_key_here
├── requirements.txt        # All pip packages
├── setup.sh                # Automated environment setup script
├── ingest.py               # Stage 0: Document loading + chunking + ChromaDB
├── retriever.py            # Stage 1: BM25 + Vector EnsembleRetriever
├── reranker.py             # Stage 2: Cross Encoder reranking (top 5)
├── llm.py                  # Final: Gemini LLM response generation
├── pipeline.py             # Main orchestrator combining all stages
├── main.py                 # Interactive CLI interface
├── sample_docs/
│   └── sample.txt          # Sample document for testing
└── chroma_db/              # ChromaDB storage (created after ingestion)
```

---

## Quick Start

### 1. Setup Environment

```bash
cd two_stage_rag
chmod +x setup.sh
./setup.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `.env`:
```
GOOGLE_API_KEY=your_actual_key_here
```

Get your free key at: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 3. Ingest Documents

```bash
# Activate virtual environment first
source venv/bin/activate

# Ingest the sample document (or your own PDF/TXT)
python main.py --ingest sample_docs/sample.txt

# Ingest a PDF
python main.py --ingest your_paper.pdf

# Ingest multiple files
python main.py --ingest doc1.pdf doc2.txt doc3.pdf
```

### 4. Ask Questions

```bash
# Interactive mode (recommended)
python main.py

# Single query mode
python main.py --query "What is a transformer model?"

# Ingest AND query in one command
python main.py --ingest sample_docs/sample.txt --query "What is BM25?"
```

---

## Example Session

```
$ python main.py --ingest sample_docs/sample.txt

╔══════════════════════════════════════════════════════════════╗
║          TWO-STAGE RAG PIPELINE  •  Powered by Gemini        ║
╠══════════════════════════════════════════════════════════════╣
║  Stage 1: Hybrid Search  (BM25 + Vector)  →  100 candidates  ║
║  Stage 2: Cross Encoder  Reranker         →  Top 5 docs      ║
║  Final:   Google Gemini  LLM              →  Final Answer    ║
╚══════════════════════════════════════════════════════════════╝

STAGE 0: DOCUMENT INGESTION PIPELINE
  Loading TXT: sample_docs/sample.txt
  ✓ Loaded 1 page(s) from sample.txt
  ✓ Split into 42 chunks (size=500, overlap=50)
  ✓ ChromaDB vector store built and persisted

> Enter your query: What is a transformer model?

[Stage 1] Hybrid Search retrieved 42 candidates via Hybrid Search (BM25 + Vector)
[Stage 2] Reranked to Top 5 using Cross Encoder
[Gemini]  Generating answer...

Answer:
  A transformer model is a type of deep learning architecture introduced in the
  landmark 2017 paper "Attention Is All You Need" by Vaswani et al. Unlike RNNs
  and LSTMs, transformers rely entirely on attention mechanisms to draw global
  dependencies between input and output...
```

---

## Module Details

### `ingest.py` — Stage 0: Document Ingestion

| Parameter | Value |
|-----------|-------|
| Chunk size | 500 characters |
| Chunk overlap | 50 characters |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (local persistence) |
| Supported formats | `.pdf`, `.txt` |

### `retriever.py` — Stage 1: Hybrid Search

| Component | Detail |
|-----------|--------|
| BM25 | `rank-bm25` library, k=100 |
| Vector | ChromaDB cosine similarity, k=100 |
| Fusion | `EnsembleRetriever` [50%, 50%] via RRF |
| Output | Up to 100 deduplicated candidates |

### `reranker.py` — Stage 2: Cross Encoder Reranking

| Parameter | Value |
|-----------|-------|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Input | 100 candidates |
| Output | Top 5 by relevance score |
| Training data | MS MARCO (8.8M passages) |

### `llm.py` — Final Stage: Gemini Generation

| Parameter | Value |
|-----------|-------|
| Model | `gemini-1.5-flash` |
| Temperature | 0.2 (factual, low creativity) |
| Max tokens | 1024 |
| Grounding | Context-only, says "I don't know" if not found |

---

## Key Design Decisions

**Why Hybrid Search?**
- BM25 excels at exact keyword matches (technical terms, names)
- Vector search captures semantic similarity (synonyms, paraphrases)
- Ensemble of both achieves higher recall than either alone

**Why Cross Encoder for reranking?**
- Bi-encoders encode query and document *independently* (fast but approximate)
- Cross encoders encode them *together* for deep token-level interaction
- Much more accurate relevance scoring, applied only to top-100 candidates

**Why 100 → 5?**
- 100 candidates ensures high recall (don't miss relevant docs)
- Reranking to 5 ensures high precision in the LLM context window
- Fewer docs = less noise, more focused answer from Gemini

---

## Dependencies

```
langchain>=0.1.0          # Core RAG framework
langchain-community>=0.0.20  # BM25, ChromaDB, HuggingFace integrations
langchain-google-genai>=0.0.6  # Gemini LLM integration
chromadb>=0.4.22          # Vector database
sentence-transformers>=2.3.1  # Bi-Encoder + Cross Encoder models
rank-bm25>=0.2.2          # BM25 retrieval
google-generativeai>=0.4.0  # Google AI SDK
pypdf>=4.0.0              # PDF loading
python-dotenv>=1.0.0      # .env file loading
```
