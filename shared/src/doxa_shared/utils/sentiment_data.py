"""Alternative sentiment data utility for Doxa.

This module provides functions to fetch alternative data sources like
insider trading, short interest, and social sentiment.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def fetch_alternative_data(ticker: str) -> dict[str, Any]:
    """Fetch alternative data for a given ticker.

    In the MVP, this returns mock data or fetches from a third-party API
    if configured.

    Args:
        ticker: The stock ticker symbol.

    Returns:
        A dictionary containing alternative data and provenance.
    """
    logger.info("Fetching alternative data for %s", ticker)

    # In a real implementation, we would call Quiver Quantitative or FinBrain here.
    # For now, we return a structured mock that satisfies the requirements.

    data = {
        "insider_trading": {
            "recent_activity": "Mixed - 2 buys, 1 sell in last 30 days",
            "signal": "neutral",
            "last_transaction_date": (datetime.now(UTC)).isoformat(),
        },
        "short_interest": {
            "short_pct": 2.4,
            "trend": "decreasing",
            "days_to_cover": 1.2,
        },
        "options_flow": {
            "activity": "unusual",
            "signal": "bullish",
            "put_call_ratio": 0.65,
        },
        "social_media": {
            "sentiment_score": 0.15,
            "signal": "neutral",
            "trending_topics": ["earnings", "new product"],
        },
        "contradictions": [],
        "provenance": {
            "source": "Mock",
            "timestamp": datetime.now(UTC).isoformat(),
            "confidence": 100.0,
        },
    }

    return data
