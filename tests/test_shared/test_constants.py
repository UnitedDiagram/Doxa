"""Tests for shared yfinance constants."""

from __future__ import annotations

from doxa_shared.constants.yfinance import (
    FAST_INFO_LAST_PRICE,
    FAST_INFO_MARKET_CAP,
    FAST_INFO_YEAR_HIGH,
    FAST_INFO_YEAR_LOW,
    INFO_NET_INCOME,
    INFO_TOTAL_CASH,
    INFO_TOTAL_DEBT,
    INFO_TOTAL_REVENUE,
)


class TestYfinanceConstants:
    """Verify shared constants match expected yfinance API field names."""

    def test_fast_info_keys(self) -> None:
        assert FAST_INFO_LAST_PRICE == "last_price"
        assert FAST_INFO_MARKET_CAP == "market_cap"
        assert FAST_INFO_YEAR_HIGH == "year_high"
        assert FAST_INFO_YEAR_LOW == "year_low"

    def test_info_keys(self) -> None:
        assert INFO_TOTAL_REVENUE == "totalRevenue"
        assert INFO_NET_INCOME == "netIncomeToCommon"
        assert INFO_TOTAL_CASH == "totalCash"
        assert INFO_TOTAL_DEBT == "totalDebt"
