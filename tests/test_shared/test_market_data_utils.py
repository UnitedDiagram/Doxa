"""Tests for shared market data utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock

from doxa_shared.utils.market_data import df_get, safe_get


class TestDfGet:
    """Tests for df_get - yfinance label variation handler."""

    def test_finds_first_matching_label(self) -> None:
        df = MagicMock()
        df.index = ["Total Revenue"]
        df.iloc.__getitem__ = MagicMock(
            return_value=MagicMock(name="2024-01-01")
        )
        df.loc.__getitem__ = MagicMock(return_value=1000.0)
        result = df_get(df, ["Total Revenue", "totalRevenue"], 0)
        assert result == 1000.0

    def test_returns_none_when_no_labels_match(self) -> None:
        df = MagicMock()
        df.index = ["Something Else"]
        result = df_get(df, ["Total Revenue", "totalRevenue"], 0)
        assert result is None

    def test_returns_none_for_nan_value(self) -> None:
        df = MagicMock()
        df.index = ["Total Revenue"]
        col_mock = MagicMock(name="2024-01-01")
        df.iloc.__getitem__ = MagicMock(return_value=col_mock)
        df.loc.__getitem__ = MagicMock(return_value="nan")
        result = df_get(df, ["Total Revenue"], 0)
        assert result is None

    def test_returns_none_for_none_value(self) -> None:
        df = MagicMock()
        df.index = ["Total Revenue"]
        col_mock = MagicMock(name="2024-01-01")
        df.iloc.__getitem__ = MagicMock(return_value=col_mock)
        df.loc.__getitem__ = MagicMock(return_value=None)
        result = df_get(df, ["Total Revenue"], 0)
        assert result is None

    def test_handles_exception_gracefully(self) -> None:
        df = MagicMock()
        df.index = ["Total Revenue"]
        df.iloc.__getitem__ = MagicMock(side_effect=KeyError("bad"))
        result = df_get(df, ["Total Revenue"], 0)
        assert result is None


class TestSafeGet:
    """Tests for safe_get - safe attribute accessor."""

    def test_returns_attribute_value(self) -> None:
        obj = MagicMock()
        obj.last_price = 150.0
        assert safe_get(obj, "last_price") == 150.0

    def test_returns_none_for_missing_attribute(self) -> None:
        obj = MagicMock(spec=[])
        assert safe_get(obj, "nonexistent") is None

    def test_returns_none_on_exception(self) -> None:
        class BadObj:
            """Object that raises on any attribute access."""

            def __getattr__(self, name: str) -> None:
                raise RuntimeError("fail")

        result = safe_get(BadObj(), "bad_attr")
        assert result is None
