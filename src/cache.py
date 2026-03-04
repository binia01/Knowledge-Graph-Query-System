"""
Query result cache — avoids re-hitting Neo4j for repeated identical questions.
Uses a simple in-memory LRU cache.
"""

from collections import OrderedDict
from hashlib import sha256


class QueryCache:
    """Simple LRU cache for query results."""

    def __init__(self, max_size: int = 128):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(question: str) -> str:
        """Generate a cache key from the question text."""
        return sha256(question.strip().lower().encode()).hexdigest()

    def get(self, question: str) -> dict | None:
        """Retrieve a cached result, or None if not found."""
        key = self._key(question)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, question: str, result: dict):
        """Store a result in the cache."""
        key = self._key(question)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self):
        """Clear the entire cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / total * 100:.1f}%" if total > 0 else "N/A",
        }
