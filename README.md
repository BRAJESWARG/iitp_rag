# PDF RAG Extraction System

> ℹ️ This README documents **System 1** (`pdf_rag/` + `run_rag.py`, OpenAI + FAISS).
> The repo also contains **System 2** (`two_stage_rag/`, Gemini + web UI — the actively
> developed one). For the full picture see **[PROJECT_DETAILS.md](PROJECT_DETAILS.md)**,
> **[CODE_EXPLAINED.md](CODE_EXPLAINED.md)**, and **[RUN_GUIDE.md](RUN_GUIDE.md)**.
> Note: System 1 currently needs an `openai>=1.0` SDK fix before it runs (see PROJECT_DETAILS).

This project implements a two-stage retrieval-augmented generation (RAG) pipeline for extracting answers from PDF documents.

## What it does

- extracts text from PDF pages
- chunks text into overlapping segments
- creates vector embeddings for each chunk
- builds a FAISS similarity index
- performs semantic retrieval on user queries
- reranks top documents with an LLM
- generates a final answer from the most relevant context

## Setup

1. Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Usage

Build the index from a PDF:

```bash
python run_rag.py /path/to/document.pdf --build
```

Load the index and ask a question:

```bash
python run_rag.py /path/to/document.pdf --load --query "What are the main findings?"
```

Build and query in one command:

```bash
python run_rag.py /path/to/document.pdf --build --query "Summarize the introduction."
```

## Notes

- `run_rag.py` uses `text-embedding-3-large` for embeddings and `gpt-4.1-mini` for reranking/answer generation.
- You can adjust `chunk_size`, `chunk_overlap`, and top-k retrieval values inside `pdf_rag/qa.py`.
- The system is designed as a general RAG pipeline that can be extended for multiple PDFs, database persistence, or local embedding models.
