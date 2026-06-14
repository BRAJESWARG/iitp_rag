from typing import Iterable

import numpy as np
from openai import OpenAI


# Lazily create the OpenAI client (reads OPENAI_API_KEY) so importing this
# module doesn't require the key — only calling these functions does.
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def get_openai_embedding(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    """Generate a single dense embedding from OpenAI."""
    response = _get_client().embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


def get_openai_embeddings(texts: Iterable[str], model: str = "text-embedding-3-large") -> list[np.ndarray]:
    """Generate embeddings for a batch of texts."""
    response = _get_client().embeddings.create(model=model, input=list(texts))
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]
