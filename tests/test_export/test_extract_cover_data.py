"""Verify cover data extraction handles missing/None fields gracefully."""

from __future__ import annotations

from src.export.pdf_export import _extract_cover_data
from src.state import create_initial_state


def test_extract_cover_data_with_empty_state() -> None:
    """Empty ResearchState yields N/A defaults without raising."""
    state = create_initial_state("NVDA")
    state["final_report"] = "# NVDA — Equity Research Note\n"
    data = _extract_cover_data(state)

    assert data["ticker"] == "NVDA"
    assert data["rating"] == "N/A"
    assert data["price_target"] == "N/A"
    assert data["current_price"] == "N/A"
    assert data["market_cap"] == "N/A"
    assert data["report_type"] == "Initiating Coverage"


def test_extract_cover_data_parses_header_line() -> None:
    """Rating, price target, and upside are parsed from the WriterAgent header."""
    state = create_initial_state("AAPL")
    state["final_report"] = (
        "# AAPL — Equity Research Note\n\n"
        "Rating: Buy | 12-Mo Price Target: $180.00 | Date: 2026-05-12\n\n"
        "## Snapshot\n"
    )
    state["market_data"] = {
        "current_price": 150.0,
        "market_cap": 2_500_000_000_000,
        "fifty_two_week_high": 200.0,
        "beta": 1.2,
    }
    data = _extract_cover_data(state)

    assert data["ticker"] == "AAPL"
    assert data["rating"] == "Buy"
    assert "$180" in data["price_target"]
    assert "$150" in data["current_price"]
    assert "T" in data["market_cap"]  # trillion suffix
    assert data["upside_sign"] == "pos"  # 180 > 150


def test_extract_cover_data_reads_pipeline_state_locations() -> None:
    """Sector, P/E, and Altman-Z come from where the pipeline really stores them."""
    state = create_initial_state("MSFT")
    state["final_report"] = "# MSFT — Equity Research Note\n"
    state["market_data"] = {
        "company_name": "Microsoft Corporation",
        "peer_comparison": {
            "sector": "Technology",
            "industry": "Software",
            "stock_metrics": {"pe_trailing": 35.5},
        },
    }
    state["valuation_analysis"] = {"altman_z_score": {"z_score": 4.72}}
    data = _extract_cover_data(state)

    assert data["company"] == "Microsoft Corporation"
    assert data["sector"] == "Technology"
    assert data["pe_ratio"] == "35.50x"
    assert data["altman_z"] == "4.72"


def test_extract_cover_data_handles_negative_upside() -> None:
    """Negative upside is flagged so the cover renders it in red."""
    state = create_initial_state("XYZ")
    state["final_report"] = (
        "# XYZ\n\n"
        "Rating: Sell | 12-Mo Price Target: $50.00 | Date: 2026-05-12\n"
    )
    state["market_data"] = {"current_price": 100.0}
    data = _extract_cover_data(state)
    assert data["rating"] == "Sell"
    assert data["upside_sign"] == "neg"
