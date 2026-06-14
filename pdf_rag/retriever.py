from typing import Iterable

from openai import OpenAI

from .embeddings import get_openai_embedding
from .vector_store import index_search


# Lazy OpenAI client (reads OPENAI_API_KEY on first use)
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def semantic_search(query: str, chunks: list[str], index, top_k: int = 8, embedding_model: str = "text-embedding-3-large") -> list[tuple[str, float]]:
    query_embedding = get_openai_embedding(query, model=embedding_model)
    indices, scores = index_search(index, query_embedding, top_k)
    candidates = [(chunks[i], float(scores[idx])) for idx, i in enumerate(indices) if i != -1]
    return candidates


def rerank_documents(query: str, documents: Iterable[str], reranker_model: str = "gpt-4.1-mini") -> list[tuple[str, float]]:
    prompt = (
        "You are a helpful retrieval re-ranker. "
        "Given a user question and a set of document excerpts, score each excerpt from 0 to 100 based on relevance to the query. "
        "Return a JSON array of objects with keys 'score' and 'text'."
        "\n\nQuestion:\n" + query + "\n\nDocuments:\n"
    )
    for idx, doc in enumerate(documents, start=1):
        prompt += f"[{idx}] {doc}\n\n"
    prompt += "\nReturn only valid JSON."

    response = _get_client().chat.completions.create(
        model=reranker_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    try:
        import json

        results = json.loads(text)
        ranked = [(item["text"], float(item["score"])) for item in results]
    except Exception:
        ranked = [(doc, 0.0) for doc in documents]
    return sorted(ranked, key=lambda x: x[1], reverse=True)


def generate_answer(query: str, top_documents: Iterable[str], answer_model: str = "gpt-4.1-mini") -> str:
    prompt = (
        "Use the following document excerpts to answer the question precisely. "
        "If the answer is not contained in the text, say that you do not know.\n\n"
        "Question:\n" + query + "\n\n"
        "Context:\n"
    )
    for idx, doc in enumerate(top_documents, start=1):
        prompt += f"[{idx}] {doc}\n\n"
    prompt += "\nAnswer:" 

    response = _get_client().chat.completions.create(
        model=answer_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()
