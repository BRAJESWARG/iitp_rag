import os
import pickle

import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS index using inner product similarity."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def index_search(index: faiss.IndexFlatIP, query_embedding: np.ndarray, top_k: int = 5) -> tuple[list[int], list[float]]:
    """Search the FAISS index and return top-k indices and scores."""
    query = query_embedding.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)
    scores, indices = index.search(query, top_k)
    return indices[0].tolist(), scores[0].tolist()


def save_index(index: faiss.IndexFlatIP, meta: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, f"{path}.idx")
    with open(f"{path}.meta.pkl", "wb") as f:
        pickle.dump(meta, f)


def load_index(path: str) -> tuple[faiss.IndexFlatIP, dict]:
    index = faiss.read_index(f"{path}.idx")
    with open(f"{path}.meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return index, meta
