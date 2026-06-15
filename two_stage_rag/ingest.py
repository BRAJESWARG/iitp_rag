"""
ingest.py — Stage 0: Document Ingestion Pipeline
=================================================
Responsibilities:
  1. Load documents (PDF, TXT) using LangChain DocumentLoaders
  2. Split into chunks via RecursiveCharacterTextSplitter
  3. Store chunks in ChromaDB vector store (for vector retrieval)
  4. Return raw text chunks for BM25 indexing

Usage:
    from ingest import ingest_documents
    docs = ingest_documents(["path/to/file.pdf", "path/to/file.txt"])
"""

import os
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# Use modern standalone packages (replaces deprecated langchain_community versions)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ------------------------------------------------------------------
# Configuration Constants
# ------------------------------------------------------------------
CHUNK_SIZE = 500          # Max characters per chunk
CHUNK_OVERLAP = 50        # Overlap between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Bi-Encoder for vector store
CHROMA_PERSIST_DIR = "./chroma_db"      # Where ChromaDB data is saved
COLLECTION_NAME = "two_stage_rag"       # ChromaDB collection name


def _normalize_text(text: str) -> str:
    """
    Collapse common extraction artifacts so chunking/retrieval behave better:
    non-breaking spaces, runs of spaces/tabs (e.g. "Acme  Technologies"), and
    3+ consecutive blank lines. Paragraph breaks are preserved.
    """
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Load documents from provided file paths.
    Supports: PDF (.pdf) and plain text (.txt) files.

    Args:
        file_paths: List of absolute or relative paths to document files.

    Returns:
        List of LangChain Document objects.
    """
    all_docs = []

    for path in file_paths:
        if not os.path.exists(path):
            print(f"  [WARNING] File not found, skipping: {path}")
            continue

        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            # PyPDFLoader extracts text page-by-page from PDF files
            print(f"  Loading PDF: {path}")
            loader = PyPDFLoader(path)
        elif ext == ".txt":
            # TextLoader reads plain text files with UTF-8 encoding
            print(f"  Loading TXT: {path}")
            loader = TextLoader(path, encoding="utf-8")
        else:
            print(f"  [WARNING] Unsupported file type '{ext}', skipping: {path}")
            continue

        docs = loader.load()
        # Normalize whitespace artifacts from PDF/DOCX extraction before chunking
        for d in docs:
            d.page_content = _normalize_text(d.page_content)
        all_docs.extend(docs)
        print(f"  ✓ Loaded {len(docs)} page(s) from {os.path.basename(path)}")

    return all_docs


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split loaded documents into smaller overlapping chunks.
    
    Uses RecursiveCharacterTextSplitter which tries to split on:
    paragraph breaks → sentence breaks → word breaks (in that order)
    ensuring semantically coherent chunks.

    Args:
        docs: List of full LangChain Document objects.

    Returns:
        List of chunked LangChain Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Preferred split points (tries these in order)
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    print(f"  ✓ Split into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Create or update a ChromaDB vector store with document chunks.

    Uses HuggingFace 'all-MiniLM-L6-v2' as the Bi-Encoder embedding model:
    - Lightweight yet powerful sentence embedding model
    - Encodes query and documents INDEPENDENTLY (fast at scale)
    - Used for approximate nearest-neighbor search in Stage 1

    Args:
        chunks: List of chunked Document objects.

    Returns:
        Chroma vector store instance.
    """
    print(f"  Loading embedding model: '{EMBEDDING_MODEL}' ...")

    # HuggingFaceEmbeddings wraps sentence-transformers models
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},    # Use 'cuda' if GPU available
        encode_kwargs={"normalize_embeddings": True},  # Cosine similarity
    )

    print(f"  Building ChromaDB vector store at: '{CHROMA_PERSIST_DIR}' ...")

    # Chroma.from_documents() encodes all chunks and stores them
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # NOTE: In ChromaDB 1.x+, data is auto-persisted when persist_directory
    # is set in Chroma.from_documents(). No explicit persist() call needed.
    print(f"  ✓ ChromaDB vector store built and persisted with "
          f"{len(chunks)} chunks")

    return vector_store


def ingest_documents(file_paths: List[str]) -> List[Document]:
    """
    Main ingestion pipeline: Load → Split → Store in ChromaDB.
    Also returns raw text chunks for BM25 indexing (done in retriever.py).

    Args:
        file_paths: List of paths to documents (PDF or TXT).

    Returns:
        List of chunked Document objects (for BM25 use in retriever.py).
    """
    print("\n" + "=" * 60)
    print("STAGE 0: DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    # Step 1: Load raw documents
    print("\n[Step 1] Loading documents...")
    docs = load_documents(file_paths)

    if not docs:
        raise ValueError(
            "No documents were loaded. Check file paths and formats."
        )
    print(f"  → Total pages/documents loaded: {len(docs)}")

    # Step 2: Split into chunks
    print("\n[Step 2] Splitting documents into chunks...")
    chunks = split_documents(docs)

    # Step 3: Store embeddings in ChromaDB (for vector retrieval in Stage 1)
    print("\n[Step 3] Building ChromaDB vector store...")
    build_vector_store(chunks)

    print("\n✓ Ingestion complete!")
    print(f"  → {len(chunks)} chunks ready for retrieval")
    print("=" * 60 + "\n")

    # Return raw chunks so BM25Retriever can index them in retriever.py
    return chunks


# ------------------------------------------------------------------
# Convenience: Load existing ChromaDB without re-ingesting
# ------------------------------------------------------------------
def load_vector_store() -> Chroma:
    """
    Load an already-persisted ChromaDB vector store from disk.
    Call this when documents have already been ingested.

    Returns:
        Chroma vector store instance.
    
    Raises:
        FileNotFoundError: If ChromaDB directory doesn't exist yet.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at '{CHROMA_PERSIST_DIR}'. "
            "Run ingest_documents() first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    return vector_store


def reset_vector_store() -> None:
    """
    Delete the persisted ChromaDB collection. Used by replace-mode ingestion so
    a new document set isn't mixed with previously-ingested documents.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return
    try:
        vs = load_vector_store()
        vs.delete_collection()
        print(f"  ✓ Cleared existing collection '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"  [WARNING] Could not reset collection: {e}")
