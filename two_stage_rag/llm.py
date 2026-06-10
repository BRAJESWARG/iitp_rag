"""
llm.py — Final Stage: Google Gemini LLM Response Generation
============================================================
This module takes the Top 5 reranked documents and generates
a grounded answer using Google Gemini (gemini-2.0-flash-lite).

  ┌─────────────────────────────────────────────────────────────────┐
  │  Input: Top 5 Refined Docs + User Query                        │
  │         ↓                                                       │
  │  Build Prompt:                                                  │
  │    Context: [doc1_text + doc2_text + ... + doc5_text]          │
  │    Question: [user_query]                                       │
  │    Answer based only on the context above.                     │
  │         ↓                                                       │
  │  google.genai (gemini-2.0-flash)                               │
  │         ↓                                                       │
  │  Final Grounded Answer                                         │
  └─────────────────────────────────────────────────────────────────┘

WHY GROUNDED GENERATION?
- Prevents hallucinations by restricting Gemini to provided context
- The "If not in context, say I don't know" instruction forces honesty
- Ensures answers are traceable back to source documents

NOTE: Uses google.genai (the modern SDK) directly instead of the
      LangChain wrapper, which has compatibility issues with Gemini 2.0.

Usage:
    from llm import GeminiLLM
    llm = GeminiLLM()
    answer = llm.generate_answer(query, top_5_docs)
"""

import os
import time
from typing import List, Generator

from langchain_core.documents import Document

# Use the modern google-genai SDK directly
# (langchain_google_genai wrapper has compatibility issues with Gemini 2.0+)
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# gemini-2.5-flash: Latest flash model, separate quota bucket from 2.0
# Fallback chain: gemini-2.5-flash → gemini-2.0-flash → gemini-2.0-flash-lite
GEMINI_MODEL = "gemini-2.5-flash"       # Latest, best free-tier availability
TEMPERATURE = 0.2                        # Low = factual, deterministic
MAX_OUTPUT_TOKENS = 1024                 # Maximum response length
MAX_RETRIES = 3                          # Retry on transient errors
RETRY_BACKOFF = [5, 15, 30]             # Seconds to wait between retries
# Transient HTTP statuses worth retrying: 429 rate-limit + 5xx server errors
# (e.g. 503 "model is currently experiencing high demand")
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# ------------------------------------------------------------------
# Prompt Template
# ------------------------------------------------------------------
# This is a grounded QA prompt that instructs Gemini to:
# 1. Only use information from the provided context
# 2. Explicitly say "I don't know" if the answer isn't in context
# This prevents hallucinations — a key RAG best practice
RAG_PROMPT_TEMPLATE = """\
You are an expert assistant that answers questions based STRICTLY on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer the question using ONLY the information from the context above.
- If the answer is not explicitly contained in the context, say: "I don't know based on the provided documents."
- Be concise and precise. Cite relevant parts of the context where appropriate.
- Do NOT use any outside knowledge or make assumptions beyond what's in the context.

ANSWER:"""


class GeminiLLM:
    """
    Wrapper for Google Gemini LLM using the modern google-genai SDK.

    Uses gemini-2.0-flash with:
    - Low temperature (0.2) for deterministic, factual responses
    - Structured grounded prompt to prevent hallucinations
    - Direct google.genai client (avoids LangChain wrapper issues)
    """

    def __init__(self, api_key: str = None):
        """
        Initialize Gemini client with API key.

        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.

        Raises:
            ValueError: If GOOGLE_API_KEY is not set.
        """
        # Resolve API key: parameter > environment variable
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key or self.api_key == "your_google_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY is not set or is still the placeholder value.\n"
                "Please set it in the .env file:\n"
                "  GOOGLE_API_KEY=your_actual_key_here\n"
                "Get your key from: https://aistudio.google.com/app/apikey"
            )

        print(f"  Initializing Gemini LLM (model: '{GEMINI_MODEL}')...")

        # Initialize the modern google-genai client
        # This client is the official replacement for google.generativeai
        self.client = genai.Client(api_key=self.api_key)

        print(f"  ✓ Gemini LLM ready (model={GEMINI_MODEL}, temperature={TEMPERATURE})")

    def _format_context(self, top_docs: List[Document]) -> str:
        """
        Combine top-5 document texts into a single numbered context string.

        Documents are numbered and separated so Gemini can distinguish
        between different source passages.

        Args:
            top_docs: List of top-ranked Document objects from Stage 2.

        Returns:
            Formatted multi-document context string.
        """
        context_parts = []
        for i, doc in enumerate(top_docs, 1):
            # Include source metadata if available (e.g., filename, page number)
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page != "" else ""

            context_parts.append(
                f"[Document {i} — {source}{page_info}]\n{doc.page_content}"
            )

        return "\n\n" + "\n\n".join(context_parts) + "\n"

    def generate_answer(
        self,
        query: str,
        top_docs: List[Document],
    ) -> str:
        """
        Generate a grounded answer from Gemini using top-5 reranked docs.

        Builds the RAG prompt with context, sends it to Gemini 2.0 Flash,
        and returns the grounded answer text.

        Args:
            query: The user's question.
            top_docs: Top-5 reranked Document objects from Stage 2.

        Returns:
            Gemini's answer string (grounded in provided context).
        """
        if not top_docs:
            return (
                "I couldn't find any relevant documents to answer your question. "
                "Please ensure documents have been ingested into the system."
            )

        print(f"\n[Gemini] Generating answer...")
        print(f"  Using {len(top_docs)} context documents")
        print(f"  Model: {GEMINI_MODEL}")

        # Combine top documents into a single context block
        context = self._format_context(top_docs)

        # Fill the prompt template with context and question
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )

        # Build generation config: controls output quality and length
        generation_config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            candidate_count=1,
        )

        # Retry loop: handles 429 rate-limit errors with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=generation_config,
                )
                answer = response.text
                print(f"  ✓ Answer generated successfully")
                return answer

            except (ClientError, ServerError) as e:
                # e.code = HTTP status (429 = rate-limit, 5xx = server overload)
                # Retry transient errors with backoff; re-raise everything else
                code = getattr(e, "code", None)
                if code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF[attempt]
                    print(f"  [Transient {code}] Gemini unavailable. "
                          f"Retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
                # Non-retryable error or final attempt — re-raise
                raise

        return "Error: Could not generate answer after retries."

    def generate_answer_stream(
        self,
        query: str,
        top_docs: List[Document],
    ) -> Generator[str, None, None]:
        """
        Generate a grounded answer stream from Gemini using top-5 reranked docs.

        Builds the RAG prompt with context, sends it to Gemini 2.0/2.5 Flash,
        and yields text chunks as they arrive.
        """
        if not top_docs:
            yield "I couldn't find any relevant documents to answer your question."
            return

        print(f"\n[Gemini] Generating answer stream...")
        print(f"  Using {len(top_docs)} context documents")
        print(f"  Model: {GEMINI_MODEL}")

        # Combine top documents into a single context block
        context = self._format_context(top_docs)

        # Fill the prompt template with context and question
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )

        # Build generation config: controls output quality and length
        generation_config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            candidate_count=1,
        )

        # Retry loop: handles 429 rate-limit errors
        for attempt in range(MAX_RETRIES):
            try:
                response_stream = self.client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=generation_config,
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                print(f"  ✓ Answer stream completed successfully")
                return

            except (ClientError, ServerError) as e:
                code = getattr(e, "code", None)
                if code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF[attempt]
                    print(f"  [Transient {code}] Gemini unavailable. "
                          f"Retrying stream in {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
                raise

        yield "Error: Could not generate answer stream after retries."


# ------------------------------------------------------------------
# Module-level convenience: shared LLM instance (lazy loaded)
# ------------------------------------------------------------------
_llm_instance: GeminiLLM = None


def get_llm() -> GeminiLLM:
    """
    Return a singleton GeminiLLM instance (initializes only once).
    Avoids re-initializing the client on every query.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = GeminiLLM()
    return _llm_instance


def generate_answer(query: str, top_docs: List[Document]) -> str:
    """
    Convenience function: generate answer using the singleton LLM.

    Args:
        query: User's question string.
        top_docs: Top-5 Documents from the Cross Encoder reranker.

    Returns:
        Final grounded answer string from Gemini.
    """
    llm = get_llm()
    return llm.generate_answer(query, top_docs)


def generate_answer_stream(query: str, top_docs: List[Document]) -> Generator[str, None, None]:
    """
    Convenience function: generate answer stream using the singleton LLM.
    """
    llm = get_llm()
    return llm.generate_answer_stream(query, top_docs)

