"""Tests for alternative sentiment data utility."""

from __future__ import annotations

from doxa_shared.utils.sentiment_data import fetch_alternative_data


def test_fetch_alternative_data_returns_dict() -> None:
    """Test that fetch_alternative_data returns a dictionary with expected keys."""
    ticker = "AAPL"
    result = fetch_alternative_data(ticker)

    assert isinstance(result, dict)
    assert "insider_trading" in result
    assert "short_interest" in result
    assert "options_flow" in result
    assert "social_media" in result
    assert "contradictions" in result
    assert "provenance" in result


def test_fetch_alternative_data_structure() -> None:
    """Test the structure of the returned alternative data."""
    ticker = "NVDA"
    result = fetch_alternative_data(ticker)

    # Insider trading structure
    insider = result["insider_trading"]
    assert "recent_activity" in insider
    assert "signal" in insider

    # Short interest structure
    short = result["short_interest"]
    assert "short_pct" in short
    assert "trend" in short

    # Provenance
    assert result["provenance"]["source"] in ["Quiver Quantitative", "FinBrain", "Mock"]
