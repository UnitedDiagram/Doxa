"""Tests for market_data helper functions (institutional depth)."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
from doxa_shared.utils.market_data import (
    analyze_volume_patterns,
    calculate_52week_range,
    calculate_beta,
    fetch_5y_price_history,
    fetch_dividend_history,
    fetch_split_history,
)


class TestFetch5yPriceHistory:
    """Test 5-year price history fetching."""

    def test_returns_dataframe_with_expected_columns(self) -> None:
        """Test function returns DataFrame with OHLCV columns."""
        # Arrange
        mock_ticker = Mock()
        mock_df = pd.DataFrame({
            "Open": [100, 101],
            "High": [105, 106],
            "Low": [99, 100],
            "Close": [102, 103],
            "Volume": [1000, 1100],
        })
        mock_ticker.history.return_value = mock_df
        mock_ticker.ticker = "AAPL"

        # Act
        result = fetch_5y_price_history(mock_ticker, cache=None)

        # Assert
        assert result is not None
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        mock_ticker.history.assert_called_once_with(period="5y")

    def test_returns_none_on_empty_dataframe(self) -> None:
        """Test function returns None when history is empty."""
        # Arrange
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker.ticker = "INVALID"

        # Act
        result = fetch_5y_price_history(mock_ticker, cache=None)

        # Assert
        assert result is None

    def test_returns_none_on_exception(self) -> None:
        """Test function returns None when yfinance raises exception."""
        # Arrange
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("API error")
        mock_ticker.ticker = "ERROR"

        # Act
        result = fetch_5y_price_history(mock_ticker, cache=None)

        # Assert
        assert result is None

    @patch("doxa_shared.utils.cache.cached_fetch")
    def test_uses_cached_fetch_when_cache_provided(
        self, mock_cached_fetch: Mock,
    ) -> None:
        """Test function uses cached_fetch() when cache is provided."""
        # Arrange
        mock_ticker = Mock()
        mock_ticker.ticker = "AAPL"
        mock_cache = Mock()
        mock_df = pd.DataFrame({
            "Open": [100], "High": [105], "Low": [99],
            "Close": [102], "Volume": [1000],
        })
        mock_cached_fetch.return_value = mock_df

        # Act
        result = fetch_5y_price_history(mock_ticker, cache=mock_cache)

        # Assert
        assert result is not None
        mock_cached_fetch.assert_called_once()
        mock_ticker.history.assert_not_called()


class TestFetchDividendHistory:
    """Test dividend history fetching."""

    def test_returns_dividend_data_for_dividend_stock(self) -> None:
        """Test function returns dividend data for stocks that pay dividends."""
        # Arrange
        mock_ticker = Mock()
        cutoff = datetime.now() - timedelta(days=5 * 365)
        dates = pd.date_range(start=cutoff, periods=5, freq="3ME")
        dividends = pd.Series([0.5, 0.5, 0.6, 0.6, 0.7], index=dates)
        mock_ticker.dividends = dividends
        mock_ticker.fast_info = Mock()
        mock_ticker.fast_info.last_price = 100.0
        mock_ticker.ticker = "AAPL"

        # Act
        result = fetch_dividend_history(mock_ticker, years=5)

        # Assert
        assert result is not None
        assert "total_dividends" in result
        assert "yield" in result
        assert isinstance(result["yield"], float)
        assert "payment_count" in result
        assert result["payment_count"] > 0

    def test_returns_none_for_non_dividend_stock(self) -> None:
        """Test function returns None for stocks with no dividends."""
        # Arrange
        mock_ticker = Mock()
        mock_ticker.dividends = pd.Series()
        mock_ticker.ticker = "BRK.B"

        # Act
        result = fetch_dividend_history(mock_ticker, years=5)

        # Assert
        assert result is None


class TestFetchSplitHistory:
    """Test stock split history fetching."""

    def test_returns_split_data_with_proper_formatting(self) -> None:
        """Test function returns split data with proper ratio formatting."""
        # Arrange
        mock_ticker = Mock()
        cutoff = datetime.now() - timedelta(days=5 * 365)
        dates = [cutoff + timedelta(days=365)]
        splits = pd.Series([2.0], index=pd.DatetimeIndex(dates))
        mock_ticker.splits = splits
        mock_ticker.ticker = "TSLA"

        # Act
        result = fetch_split_history(mock_ticker, years=5)

        # Assert
        assert result is not None
        assert len(result) == 1
        assert "2-for-1 split" in result[0]["ratio"]
        assert "date" in result[0]

    def test_returns_none_for_no_splits(self) -> None:
        """Test function returns None when no splits in period."""
        # Arrange
        mock_ticker = Mock()
        mock_ticker.splits = pd.Series()
        mock_ticker.ticker = "AAPL"

        # Act
        result = fetch_split_history(mock_ticker, years=5)

        # Assert
        assert result is None


class TestAnalyzeVolumePatterns:
    """Test volume pattern analysis."""

    def test_detects_normal_volume(self) -> None:
        """Test function detects normal trading volume."""
        # Arrange
        df = pd.DataFrame({
            "Volume": [1000, 1000, 1000, 1000, 1200],
        })

        # Act
        result = analyze_volume_patterns(df)

        # Assert
        assert result is not None
        assert result["unusual_activity_detected"] is False
        assert result["avg_volume"] == 1040.0

    def test_detects_unusual_volume(self) -> None:
        """Test function detects unusual volume (>2x average)."""
        # Arrange
        df = pd.DataFrame({
            "Volume": [1000, 1000, 1000, 1000, 5000],
        })

        # Act
        result = analyze_volume_patterns(df)

        # Assert
        assert result is not None
        assert result["unusual_activity_detected"] is True
        assert result["volume_ratio"] > 2.0

    def test_returns_none_on_missing_volume_column(self) -> None:
        """Test function returns None when Volume column missing."""
        # Arrange
        df = pd.DataFrame({
            "Close": [100, 101, 102],
        })

        # Act
        result = analyze_volume_patterns(df)

        # Assert
        assert result is None


class TestCalculate52WeekRange:
    """Test 52-week range calculation."""

    def test_calculates_position_correctly(self) -> None:
        """Test function calculates current position in 52-week range."""
        # Arrange
        mock_ticker = Mock()
        mock_df = pd.DataFrame({
            "Close": [100, 90, 110, 105],
        })
        mock_ticker.history.return_value = mock_df

        # Act
        result = calculate_52week_range(mock_ticker, current_price=105)

        # Assert
        assert result is not None
        assert result["week_52_high"] == 110
        assert result["week_52_low"] == 90
        assert result["current_position_pct"] == 75.0  # (105-90)/(110-90)*100

    def test_handles_flat_price_edge_case(self) -> None:
        """Test function handles edge case where high == low."""
        # Arrange
        mock_ticker = Mock()
        mock_df = pd.DataFrame({
            "Close": [100, 100, 100],
        })
        mock_ticker.history.return_value = mock_df

        # Act
        result = calculate_52week_range(mock_ticker, current_price=100)

        # Assert
        assert result is not None
        assert result["current_position_pct"] == 50.0


class TestCalculateBeta:
    """Test beta calculation."""

    def test_calculates_beta_with_sufficient_data(self) -> None:
        """Test function calculates beta with sufficient data (252+ days)."""
        # Arrange
        mock_ticker = Mock()

        # Create 300 days of data
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
        stock_prices = pd.DataFrame({
            "Close": range(100, 400),
        }, index=dates)
        market_prices = pd.DataFrame({
            "Close": range(1000, 1300),
        }, index=dates)

        mock_ticker.history.return_value = stock_prices

        # Patch yfinance Ticker to return mock market
        with patch("yfinance.Ticker") as mock_yf:
            mock_market = Mock()
            mock_market.history.return_value = market_prices
            mock_yf.return_value = mock_market

            # Act
            result = calculate_beta(mock_ticker, market_ticker="^GSPC")

            # Assert
            assert result is not None
            assert isinstance(result, float)

    def test_returns_none_with_insufficient_data(self) -> None:
        """Test function returns None with insufficient data (<252 days)."""
        # Arrange
        mock_ticker = Mock()
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        stock_prices = pd.DataFrame({
            "Close": range(100, 200),
        }, index=dates)
        mock_ticker.history.return_value = stock_prices

        # Act
        result = calculate_beta(mock_ticker)

        # Assert
        assert result is None
