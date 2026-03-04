"""
Configuration module — loads environment variables and provides
centralized settings for the entire application.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # Google Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

    # LLM settings
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    LLM_TEMPERATURE: float = 0.0

    # LLM rate-limiting & resilience
    LLM_RATE_LIMIT: int = int(os.getenv("LLM_RATE_LIMIT", "15"))  # max calls/min
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "5"))  # auto-retry on 429
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))  # seconds per request

    # Validation thresholds
    CONFIDENCE_HIGH: float = 0.7   # Execute directly
    CONFIDENCE_MED: float = 0.4    # Auto-correct and retry
    # Below 0.4 → reject and ask user to clarify

    # Vector search
    VECTOR_DIMENSIONS: int = 3072
    SIMILARITY_THRESHOLD: float = 0.7
    VECTOR_TOP_K: int = 5

    # Flask
    FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of missing required configuration items."""
        errors: list[str] = []
        if not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is not set")
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD is not set")
        return errors
