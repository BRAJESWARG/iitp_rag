import os
from typing import Iterable

import numpy as np
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")


def get_openai_embedding(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    """Generate a single dense embedding from OpenAI."""
    response = openai.Embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


def get_openai_embeddings(texts: Iterable[str], model: str = "text-embedding-3-large") -> list[np.ndarray]:
    """Generate embeddings for a batch of texts."""
    response = openai.Embeddings.create(model=model, input=list(texts))
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]
