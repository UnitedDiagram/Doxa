"""Smoke test: render a minimal state to PDF and check for valid output.

Skips when WeasyPrint native libraries (pango) are not installed.
"""

from __future__ import annotations

import pytest

from src.export.pdf_export import (
    WeasyPrintUnavailableError,
    render_report_pdf,
)
from src.state import create_initial_state


def _sample_state() -> dict:
    state = create_initial_state("AAPL")
    state["market_data"] = {
        "current_price": 150.0,
        "market_cap": 2_500_000_000_000,
        "fifty_two_week_high": 200.0,
        "fifty_two_week_low": 120.0,
        "beta": 1.2,
        "shares_outstanding": 15_000_000_000,
        "dividend_yield": 0.005,
        "company_name": "Apple Inc.",
        "peer_comparison": {
            "sector": "Technology",
            "stock_metrics": {"pe_trailing": 28.5},
        },
    }
    state["valuation_analysis"] = {
        "confidence": 82.0,
        "altman_z_score": {"z_score": 5.1},
    }
    state["final_report"] = (
        "# AAPL — Equity Research Note\n\n"
        "Rating: Buy | 12-Mo Price Target: $180.00 | Date: 2026-05-12\n\n"
        "## Snapshot\n\n"
        "| Metric | Value |\n|--------|-------|\n| Price | $150 |\n\n"
        "## I. Investment Summary\n\n"
        "Apple remains a high-quality compounder with services tailwinds.\n\n"
        "## II. Company Overview\n\n"
        "Apple designs consumer electronics and software.\n\n"
        "## VI. Valuation\n\n"
        "DCF supports $180 fair value.\n"
    )
    return state


def test_render_returns_pdf_bytes() -> None:
    """End-to-end render produces a non-empty bytes object with PDF magic header."""
    try:
        result = render_report_pdf(_sample_state())
    except WeasyPrintUnavailableError as e:
        pytest.skip(f"WeasyPrint native libs not installed: {e}")

    assert isinstance(result, bytes)
    assert len(result) > 1000
    assert result.startswith(b"%PDF")


def test_render_raises_on_empty_report() -> None:
    """An empty final_report raises ValueError, not produces a blank PDF."""
    state = create_initial_state("AAPL")
    state["final_report"] = ""
    with pytest.raises(ValueError, match="empty"):
        render_report_pdf(state)
