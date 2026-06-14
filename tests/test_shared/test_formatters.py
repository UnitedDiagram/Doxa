"""Tests for shared formatter utility functions."""

from __future__ import annotations

from doxa_shared.utils.formatters import fmt_number, fmt_pct, fmt_ratio


class TestFmtNumber:
    """Tests for fmt_number."""

    def test_basic_formatting(self) -> None:
        assert fmt_number(1500000) == "1,500,000.00"

    def test_with_prefix(self) -> None:
        assert fmt_number(1500000, "$") == "$1,500,000.00"

    def test_with_suffix(self) -> None:
        assert fmt_number(2.5, "", "B") == "2.50B"

    def test_with_prefix_and_suffix(self) -> None:
        assert fmt_number(2.5, "$", "B") == "$2.50B"

    def test_returns_na_for_none(self) -> None:
        assert fmt_number(None) == "N/A"

    def test_returns_na_for_invalid(self) -> None:
        assert fmt_number("not a number") == "N/A"


class TestFmtPct:
    """Tests for fmt_pct."""

    def test_basic_percentage(self) -> None:
        assert fmt_pct(0.1523) == "15.2%"

    def test_small_percentage(self) -> None:
        assert fmt_pct(0.05) == "5.0%"

    def test_returns_na_for_none(self) -> None:
        assert fmt_pct(None) == "N/A"

    def test_returns_na_for_invalid(self) -> None:
        assert fmt_pct("bad") == "N/A"


class TestFmtRatio:
    """Tests for fmt_ratio."""

    def test_basic_ratio(self) -> None:
        assert fmt_ratio(15.67) == "15.67x"

    def test_small_ratio(self) -> None:
        assert fmt_ratio(2.3) == "2.30x"

    def test_returns_na_for_none(self) -> None:
        assert fmt_ratio(None) == "N/A"

    def test_returns_na_for_invalid(self) -> None:
        assert fmt_ratio("bad") == "N/A"
