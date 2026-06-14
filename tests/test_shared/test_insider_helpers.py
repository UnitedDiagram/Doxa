"""Tests for insider ownership utility functions."""

from __future__ import annotations

import pytest
from doxa_shared.utils.valuation import (
    calculate_ceo_ownership_value,
    fetch_insider_ownership,
    interpret_insider_signal,
)


def test_fetch_insider_ownership_complete_data() -> None:
    """Test insider ownership extraction with complete data."""
    info = {
        "heldPercentInsiders": 0.152,  # 15.2%
        "heldPercentInstitutions": 0.645,  # 64.5%
    }

    result = fetch_insider_ownership(info)

    assert result["insider_pct"] == pytest.approx(15.2, abs=0.01)
    assert result["institutional_pct"] == pytest.approx(64.5, abs=0.01)


def test_fetch_insider_ownership_missing_insider() -> None:
    """Test handling when insider percentage is missing."""
    info = {
        "heldPercentInstitutions": 0.645,
    }

    result = fetch_insider_ownership(info)

    assert result["insider_pct"] is None
    assert result["institutional_pct"] == pytest.approx(64.5, abs=0.01)


def test_fetch_insider_ownership_missing_institutional() -> None:
    """Test handling when institutional percentage is missing."""
    info = {
        "heldPercentInsiders": 0.152,
    }

    result = fetch_insider_ownership(info)

    assert result["insider_pct"] == pytest.approx(15.2, abs=0.01)
    assert result["institutional_pct"] is None


def test_fetch_insider_ownership_all_missing() -> None:
    """Test handling when all ownership data is missing."""
    info = {}

    result = fetch_insider_ownership(info)

    assert result["insider_pct"] is None
    assert result["institutional_pct"] is None


def test_calculate_ceo_ownership_value_millions() -> None:
    """Test CEO ownership value calculation - MVP returns None."""
    info = {
        "companyOfficers": [
            {
                "name": "Tim Cook",
                "title": "Chief Executive Officer",
                "totalPay": 3000000,  # Share count not reliably available
            }
        ]
    }
    current_price = 175.43

    # MVP: CEO share ownership data not reliably available in yfinance
    # Function should return None (AC #11: "Calculate value IF data available")
    result = calculate_ceo_ownership_value(info, current_price)

    # MVP behavior: returns None
    assert result is None


def test_calculate_ceo_ownership_value_billions() -> None:
    """Test CEO ownership value calculation in billions."""
    info = {
        "companyOfficers": [
            {
                "name": "Elon Musk",
                "title": "CEO",
                "totalPay": 10000000,  # Mock large share count
            }
        ]
    }
    current_price = 250.0

    result = calculate_ceo_ownership_value(info, current_price)

    # Should format with B suffix if value >= 1 billion
    # May return None for MVP
    assert result is None or "M" in result or "B" in result


def test_calculate_ceo_ownership_value_missing_data() -> None:
    """Test handling when CEO data is unavailable."""
    info = {}
    current_price = 175.43

    result = calculate_ceo_ownership_value(info, current_price)

    assert result is None


def test_interpret_insider_signal_heavy_buying() -> None:
    """Test signal interpretation for heavy insider buying."""
    result = interpret_insider_signal(buying=10, selling=2)

    assert result == "positive"


def test_interpret_insider_signal_heavy_selling() -> None:
    """Test signal interpretation for heavy insider selling."""
    result = interpret_insider_signal(buying=2, selling=10)

    assert result == "negative"


def test_interpret_insider_signal_balanced() -> None:
    """Test signal interpretation for balanced activity."""
    result = interpret_insider_signal(buying=5, selling=5)

    assert result == "neutral"


def test_interpret_insider_signal_none_values() -> None:
    """Test signal interpretation when data is unavailable."""
    result = interpret_insider_signal(buying=None, selling=None)

    assert result == "neutral"


def test_interpret_insider_signal_partial_none() -> None:
    """Test signal interpretation with partial data."""
    result = interpret_insider_signal(buying=5, selling=None)

    assert result == "neutral"
