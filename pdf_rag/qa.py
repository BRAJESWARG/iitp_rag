import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .embeddings import get_openai_embeddings, get_openai_embedding
from .pdf_loader import extract_text_from_pdf, split_text
from .retriever import generate_answer, rerank_documents, semantic_search
from .vector_store import build_faiss_index, load_index, save_index


@dataclass
class RAGPDFQA:
    pdf_path: str
    index_path: Optional[str] = None
    chunk_size: int = 900
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-large"
    reranker_model: str = "gpt-4.1-mini"
    answer_model: str = "gpt-4.1-mini"

    def __post_init__(self):
        self.chunks: list[str] = []
        self.index = None

    def build_knowledge_base(self) -> None:
        text = extract_text_from_pdf(self.pdf_path)
        self.chunks = split_text(text, self.chunk_size, self.chunk_overlap)
        if not self.chunks:
            raise ValueError("No text chunks could be created from the PDF.")

        embeddings = get_openai_embeddings(self.chunks, model=self.embedding_model)
        matrix = np.vstack(embeddings)
        self.index = build_faiss_index(matrix)

        if self.index_path:
            save_index(self.index, {"chunks": self.chunks}, self.index_path)

    def load_knowledge_base(self) -> None:
        if not self.index_path:
            raise ValueError("index_path must be set to load a saved index.")
        self.index, meta = load_index(self.index_path)
        self.chunks = meta["chunks"]

    def query(self, question: str, search_top_k: int = 8, rerank_top_k: int = 5) -> str:
        if self.index is None or not self.chunks:
            raise RuntimeError("Knowledge base is not loaded or built.")

        candidates = semantic_search(
            question,
            self.chunks,
            self.index,
            top_k=search_top_k,
            embedding_model=self.embedding_model,
        )
        texts = [text for text, score in candidates]
        reranked = rerank_documents(question, texts, reranker_model=self.reranker_model)
        top_texts = [text for text, _ in reranked[:rerank_top_k]]
        return generate_answer(question, top_texts, answer_model=self.answer_model)
