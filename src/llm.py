"""
LLM module — initializes Google Gemini via LangChain.
Uses singleton instances to share connections and a rate limiter
to stay within Gemini's API quotas and avoid 429 errors.
"""

import threading
import time

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.config import Config


class _RateLimiter:
    """
    Token-bucket rate limiter that enforces a minimum interval between
    Gemini API calls so we don't burst past the quota.
    """

    def __init__(self, max_calls_per_minute: int = 15):
        self._interval = 60.0 / max_calls_per_minute  # seconds between calls
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def wait(self):
        """Block until it is safe to make the next API call."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_call = time.monotonic()


# Module-level singletons (created once, reused everywhere)
_llm: ChatGoogleGenerativeAI | None = None
_embeddings: GoogleGenerativeAIEmbeddings | None = None
_rate_limiter = _RateLimiter(max_calls_per_minute=Config.LLM_RATE_LIMIT)


class _RateLimitedLLM(ChatGoogleGenerativeAI):
    """Thin wrapper that applies rate-limiting before every invoke."""

    _rl: _RateLimiter | None = None  # set after construction

    def invoke(self, *args, **kwargs):
        if self._rl:
            self._rl.wait()
        return super().invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        if self._rl:
            self._rl.wait()
        return await super().ainvoke(*args, **kwargs)


def get_llm() -> ChatGoogleGenerativeAI:
    """Return the shared, rate-limited Gemini chat model singleton."""
    global _llm
    if _llm is None:
        _llm = _RateLimitedLLM(
            model=Config.LLM_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=Config.LLM_TEMPERATURE,
            max_retries=Config.LLM_MAX_RETRIES,
            timeout=Config.LLM_TIMEOUT,
        )
        _llm._rl = _rate_limiter
    return _llm


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the shared Gemini embedding model singleton."""
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
        )
    return _embeddings
