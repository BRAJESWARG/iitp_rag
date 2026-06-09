"""
api.py — FastAPI Python Backend for Two-Stage RAG Pipeline
==========================================================
Wraps the complete RAG pipeline in a REST API with:
  - POST /api/query    → Run full pipeline and return answer
  - POST /api/ingest   → Ingest uploaded documents
  - GET  /api/status   → System health and loaded state
  - GET  /api/docs     → Auto-generated Swagger UI

Run with:
    source venv/bin/activate
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import shutil
import tempfile
import traceback
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ──────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Two-Stage RAG Pipeline API",
    description="Production RAG system: BM25 + Vector Hybrid Search → Cross Encoder Reranking → Gemini LLM",
    version="1.0.0",
)

# Allow all origins for local development (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Global pipeline state (loaded once, reused across requests)
# ──────────────────────────────────────────────────────────────
pipeline_state = {
    "pipeline": None,          # TwoStageRAGPipeline instance
    "chunks_count": 0,
    "documents_loaded": [],    # List of ingested file names
    "initialized": False,
    "initializing": False,
    "error": None,
    "started_at": None,
}


# ──────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class StageInfo(BaseModel):
    stage: str
    candidates: int
    top_docs: Optional[int] = None
    scores: Optional[List[float]] = None

class QueryResponse(BaseModel):
    answer: str
    query: str
    stage1_candidates: int
    stage2_top_docs: int
    top_doc_scores: List[float]
    top_doc_snippets: List[str]
    model_used: str
    duration_ms: float

class IngestResponse(BaseModel):
    success: bool
    message: str
    files_ingested: List[str]
    total_chunks: int

class StatusResponse(BaseModel):
    status: str
    initialized: bool
    documents_loaded: List[str]
    chunks_count: int
    model_stage1_embed: str
    model_stage2_reranker: str
    model_llm: str
    api_key_set: bool
    started_at: Optional[str]
    error: Optional[str]


# ──────────────────────────────────────────────────────────────
# Helper: Load pipeline from existing ChromaDB
# ──────────────────────────────────────────────────────────────

def _try_load_existing_pipeline():
    """Attempt to load pipeline from already-ingested ChromaDB (non-blocking)."""
    chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    if not os.path.exists(chroma_path):
        return  # Nothing to load yet

    try:
        pipeline_state["initializing"] = True
        from pipeline import create_pipeline
        p = create_pipeline(file_paths=None)
        pipeline_state["pipeline"] = p
        pipeline_state["initialized"] = True
        pipeline_state["chunks_count"] = len(p.chunks)
        
        # Populate documents_loaded from document chunks metadata
        docs = set()
        for doc in p.chunks:
            source = doc.metadata.get("source")
            if source:
                docs.add(os.path.basename(source))
        pipeline_state["documents_loaded"] = sorted(list(docs))

        pipeline_state["started_at"] = datetime.now().isoformat()
        pipeline_state["error"] = None
        print("✓ Existing pipeline loaded from ChromaDB on startup")
    except Exception as e:
        pipeline_state["error"] = str(e)
        print(f"⚠ Could not auto-load pipeline: {e}")
    finally:
        pipeline_state["initializing"] = False


# ──────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Try to load an existing pipeline when the server starts."""
    print("=" * 55)
    print(" Two-Stage RAG API starting up...")
    print("=" * 55)
    _try_load_existing_pipeline()


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/api/status", response_model=StatusResponse)
def get_status():
    """Return the current state of the RAG pipeline system."""
    from llm import GEMINI_MODEL
    return StatusResponse(
        status="ready" if pipeline_state["initialized"] else (
            "initializing" if pipeline_state["initializing"] else "not_initialized"
        ),
        initialized=pipeline_state["initialized"],
        documents_loaded=pipeline_state["documents_loaded"],
        chunks_count=pipeline_state["chunks_count"],
        model_stage1_embed="all-MiniLM-L6-v2",
        model_stage2_reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_llm=GEMINI_MODEL,
        api_key_set=bool(os.getenv("GOOGLE_API_KEY")),
        started_at=pipeline_state.get("started_at"),
        error=pipeline_state.get("error"),
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Ingest one or more PDF or TXT documents into the RAG system.
    Accepts multipart/form-data with one or more file fields.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved_paths = []
    tmp_dir = tempfile.mkdtemp()

    try:
        # Save uploaded files to a temp directory
        for upload in files:
            ext = os.path.splitext(upload.filename)[1].lower()
            if ext not in (".pdf", ".txt"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {upload.filename}. Only PDF and TXT allowed."
                )
            dest = os.path.join(tmp_dir, upload.filename)
            with open(dest, "wb") as f:
                content = await upload.read()
                f.write(content)
            saved_paths.append(dest)

        # Run ingestion pipeline
        pipeline_state["initializing"] = True
        from pipeline import create_pipeline

        p = create_pipeline(file_paths=saved_paths)
        pipeline_state["pipeline"] = p
        pipeline_state["initialized"] = True
        pipeline_state["chunks_count"] = len(p.chunks)
        pipeline_state["documents_loaded"] = [
            os.path.basename(path) for path in saved_paths
        ]
        pipeline_state["started_at"] = datetime.now().isoformat()
        pipeline_state["error"] = None

        return IngestResponse(
            success=True,
            message=f"Successfully ingested {len(saved_paths)} document(s).",
            files_ingested=[os.path.basename(p) for p in saved_paths],
            total_chunks=len(pipeline_state["pipeline"].chunks),
        )

    except Exception as e:
        pipeline_state["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        pipeline_state["initializing"] = False
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """
    Run a full two-stage RAG query and return the answer with metadata.
    Requires documents to be ingested first via /api/ingest.
    """
    import time

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not pipeline_state["initialized"] or pipeline_state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Please ingest documents first via POST /api/ingest."
        )

    start_time = time.time()

    try:
        p = pipeline_state["pipeline"]
        query = request.query.strip()

        # Stage 1: Hybrid Search
        from retriever import hybrid_search
        candidates = hybrid_search(p.retriever, query)
        stage1_count = len(candidates)

        # Stage 2: Cross Encoder Reranking
        from reranker import cross_encoder_rerank, get_reranker
        top_docs = cross_encoder_rerank(query, candidates)

        # Compute scores for the top docs
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in top_docs]
        scores = reranker.model.predict(pairs).tolist()

        # Final: Gemini LLM
        from llm import generate_answer, GEMINI_MODEL
        answer = generate_answer(query, top_docs)

        duration_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            answer=answer,
            query=query,
            stage1_candidates=stage1_count,
            stage2_top_docs=len(top_docs),
            top_doc_scores=[round(s, 4) for s in scores],
            top_doc_snippets=[doc.page_content[:200] for doc in top_docs],
            model_used=GEMINI_MODEL,
            duration_ms=round(duration_ms, 2),
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.websocket("/api/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for real-time stage transitions and response streaming.
    Receives {"query": "..."} from client.
    Sends stages, text chunks, and completion results.
    """
    await websocket.accept()
    try:
        while True:
            # Receive input from client
            data = await websocket.receive_json()
            query = data.get("query")
            if not query or not query.strip():
                await websocket.send_json({"type": "error", "message": "Query cannot be empty."})
                continue

            if not pipeline_state["initialized"] or pipeline_state["pipeline"] is None:
                await websocket.send_json({
                    "type": "error", 
                    "message": "Pipeline not initialized. Please ingest documents first."
                })
                continue

            import time
            start_time = time.time()
            p = pipeline_state["pipeline"]
            query = query.strip()

            # ──────────────────────────────────────────────────────
            # Stage 1: Searching
            # ──────────────────────────────────────────────────────
            await websocket.send_json({"type": "stage", "stage": "stage1_searching"})
            
            from retriever import hybrid_search
            # Non-blocking search execution
            candidates = hybrid_search(p.retriever, query)
            stage1_count = len(candidates)

            if not candidates:
                await websocket.send_json({
                    "type": "error",
                    "message": "No relevant documents found. Please ingest documents."
                })
                continue

            # ──────────────────────────────────────────────────────
            # Stage 2: Reranking
            # ──────────────────────────────────────────────────────
            await websocket.send_json({
                "type": "stage",
                "stage": "stage2_reranking",
                "stage1_candidates": stage1_count
            })
            
            from reranker import cross_encoder_rerank, get_reranker
            top_docs = cross_encoder_rerank(query, candidates)
            stage2_count = len(top_docs)

            if not top_docs:
                await websocket.send_json({
                    "type": "error",
                    "message": "Reranking yielded zero results."
                })
                continue

            # Compute scores for the top docs
            reranker = get_reranker()
            pairs = [[query, doc.page_content] for doc in top_docs]
            scores = reranker.model.predict(pairs).tolist()
            snippets = [doc.page_content[:200] for doc in top_docs]

            # ──────────────────────────────────────────────────────
            # Stage 3: LLM Generating (Send Rerank Results)
            # ──────────────────────────────────────────────────────
            await websocket.send_json({
                "type": "stage",
                "stage": "llm_generating",
                "stage2_top_docs": stage2_count,
                "top_doc_scores": [round(s, 4) for s in scores],
                "top_doc_snippets": snippets
            })

            # Stream Gemini answer chunk-by-chunk
            from llm import generate_answer_stream, GEMINI_MODEL
            
            for text_chunk in generate_answer_stream(query, top_docs):
                await websocket.send_json({
                    "type": "chunk",
                    "text": text_chunk
                })

            duration_ms = (time.time() - start_time) * 1000

            # Done
            await websocket.send_json({
                "type": "done",
                "duration_ms": round(duration_ms, 2),
                "model_used": GEMINI_MODEL
            })

    except WebSocketDisconnect:
        print("✓ WebSocket client disconnected")
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": f"Pipeline Query failed: {str(e)}"})
        except Exception:
            pass


@app.get("/")
def root():
    return {
        "message": "Two-Stage RAG Pipeline API",
        "docs": "/docs",
        "status": "/api/status",
    }
