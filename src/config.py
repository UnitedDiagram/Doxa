"""Configuration constants, environment variables, and logging setup."""

import logging
import os

from dotenv import load_dotenv

# Load .env from the project root before reading any environment variables.
load_dotenv()

# ---------------------------------------------------------------------------
# Environment variables (with defaults)
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("DOXA_LOG_LEVEL", "INFO")
HISTORY_PERIOD: str = os.environ.get("DOXA_HISTORY_PERIOD", "1y")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
SENTIMENT_MAX_HEADLINES: int = int(os.environ.get("DOXA_MAX_HEADLINES", "10"))

CACHE_BACKEND: str = os.environ.get("DOXA_CACHE_BACKEND", "memory")
CACHE_MAX_ENTRIES: int = int(
    os.environ.get("DOXA_CACHE_MAX_ENTRIES", "1000")
)


def configure_logging() -> None:
    """Set up root logging based on DOXA_LOG_LEVEL env var.

    Call this once from entry points, not on module import.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
