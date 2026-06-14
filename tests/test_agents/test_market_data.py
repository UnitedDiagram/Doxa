"""Tests for MarketDataAgent execution tracing and institutional depth."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import pandas as pd
from doxa_shared.types.state import create_initial_state

from src.agents.market_data import MarketDataAgent


def _mock_yf_ticker() -> Mock:
    """Build a mock yfinance Ticker with realistic data."""
    ticker = Mock()

    # fast_info
    fi = Mock()
    fi.last_price = 150.0
    fi.market_cap = 2_500_000_000_000
    fi.year_high = 180.0
    fi.year_low = 120.0
    type(ticker).fast_info = PropertyMock(return_value=fi)

    # info
    type(ticker).info = PropertyMock(return_value={
        "totalRevenue": 380_000_000_000,
        "netIncomeToCommon": 95_000_000_000,
        "totalCash": 60_000_000_000,
        "totalDebt": 110_000_000_000,
    })

    # history
    hist = pd.DataFrame({"Close": [148.0, 149.0, 150.0]})
    ticker.history.return_value = hist

    # financials, balance_sheet, cashflow (empty for simplicity)
    type(ticker).financials = PropertyMock(return_value=pd.DataFrame())
    type(ticker).balance_sheet = PropertyMock(return_value=pd.DataFrame())
    type(ticker).cashflow = PropertyMock(return_value=pd.DataFrame())

    # news
    type(ticker).news = PropertyMock(return_value=[
        {
            "content": {
                "title": "Test headline",
                "provider": {"displayName": "Reuters"},
            },
            "canonicalUrl": {"url": "https://example.com"},
        },
    ])

    return ticker


class TestMarketDataAgentTracing:
    """Tests that MarketDataAgent emits correct trace events."""

    @patch("src.agents.market_data.yf.Ticker")
    def test_logs_agent_started(
        self, mock_ticker_cls: Mock, caplog: Any,
    ) -> None:
        """Agent should log agent_started event."""
        mock_ticker_cls.return_value = _mock_yf_ticker()
        state = create_initial_state("AAPL")
        with caplog.at_level(logging.DEBUG):
            MarketDataAgent().fetch_data(state)

        started_records = [
            r for r in caplog.records
            if "agent_started" in r.message
        ]
        assert len(started_records) >= 1
        parsed = json.loads(started_records[0].message)
        assert parsed["agent"] == "MarketDataAgent"
        assert parsed["ticker"] == "AAPL"

    @patch("src.agents.market_data.yf.Ticker")
    def test_logs_agent_completed(
        self, mock_ticker_cls: Mock, caplog: Any,
    ) -> None:
        """Agent should log agent_completed with execution_time_ms."""
        mock_ticker_cls.return_value = _mock_yf_ticker()
        state = create_initial_state("AAPL")
        with caplog.at_level(logging.DEBUG):
            MarketDataAgent().fetch_data(state)

        completed = [
            r for r in caplog.records
            if "agent_completed" in r.message
        ]
        assert len(completed) >= 1
        parsed = json.loads(completed[0].message)
        assert parsed["agent"] == "MarketDataAgent"
        assert "execution_time_ms" in parsed
        assert parsed["execution_time_ms"] >= 0
        assert "errors_count" in parsed

    @patch("src.agents.market_data.yf.Ticker")
    def test_logs_api_call_for_fast_info(
        self, mock_ticker_cls: Mock, caplog: Any,
    ) -> None:
        """Agent should log api_call for yfinance.fast_info."""
        mock_ticker_cls.return_value = _mock_yf_ticker()
        state = create_initial_state("AAPL")
        with caplog.at_level(logging.DEBUG):
            MarketDataAgent().fetch_data(state)

        fast_info_calls = [
            r for r in caplog.records
            if "yfinance.fast_info" in r.message
            and "api_call" in r.message
        ]
        assert len(fast_info_calls) >= 1
        parsed = json.loads(fast_info_calls[0].message)
        assert "fields_requested" in parsed
        assert isinstance(parsed["fields_requested"], list)

    @patch("src.agents.market_data.yf.Ticker")
    def test_logs_api_call_for_history(
        self, mock_ticker_cls: Mock, caplog: Any,
    ) -> None:
        """Agent should log api_call for yfinance.history."""
        mock_ticker_cls.return_value = _mock_yf_ticker()
        state = create_initial_state("AAPL")
        with caplog.at_level(logging.DEBUG):
            MarketDataAgent().fetch_data(state)

        history_calls = [
            r for r in caplog.records
            if "yfinance.history" in r.message
            and "api_call" in r.message
        ]
        assert len(history_calls) >= 1
        # Post-fetch trace should include data_points
        post_fetch = [
            r for r in history_calls
            if "data_points" in r.message
        ]
        assert len(post_fetch) >= 1
        parsed = json.loads(post_fetch[0].message)
        assert parsed["data_points"] == 3
        assert "period" in parsed

    @patch("src.agents.market_data.yf.Ticker")
    def test_logs_api_call_for_news(
        self, mock_ticker_cls: Mock, caplog: Any,
    ) -> None:
        """Agent should log api_call for yfinance.news."""
        mock_ticker_cls.return_value = _mock_yf_ticker()
        state = create_initial_state("AAPL")
        with caplog.at_level(logging.DEBUG):
            MarketDataAgent().fetch_data(state)

        news_calls = [
            r for r in caplog.records
            if "yfinance.news" in r.message
            and "api_call" in r.message
        ]
        assert len(news_calls) >= 1
        # Post-fetch trace should include headlines_count
        post_fetch = [
            r for r in news_calls
            if "headlines_count" in r.message
        ]
        assert len(post_fetch) >= 1
        parsed = json.loads(post_fetch[0].message)
        assert parsed["headlines_count"] == 1

    @patch("src.agents.market_data.yf.Ticker")
    def test_logs_api_call_for_statements(
        self, mock_ticker_cls: Mock, caplog: Any,
    ) -> None:
        """Agent should log api_call for financial statements."""
        mock_ticker_cls.return_value = _mock_yf_ticker()
        state = create_initial_state("AAPL")
        with caplog.at_level(logging.DEBUG):
            MarketDataAgent().fetch_data(state)

        sources = [
            "yfinance.financials",
            "yfinance.balance_sheet",
            "yfinance.cashflow",
        ]
        for source in sources:
            calls = [
                r for r in caplog.records
                if source in r.message
            ]
            assert len(calls) >= 1, (
                f"Missing api_call trace for {source}"
            )

    @patch("src.agents.market_data.yf.Ticker")
    def test_logs_agent_error_on_failure(
        self, mock_ticker_cls: Mock, caplog: Any,
    ) -> None:
        """Agent should log agent_error when data fetch fails."""
        mock = _mock_yf_ticker()
        type(mock).fast_info = PropertyMock(
            side_effect=RuntimeError("API down"),
        )
        mock_ticker_cls.return_value = mock

        state = create_initial_state("AAPL")
        with caplog.at_level(logging.DEBUG):
            MarketDataAgent().fetch_data(state)

        error_traces = [
            r for r in caplog.records
            if "agent_error" in r.message
        ]
        assert len(error_traces) >= 1
        parsed = json.loads(error_traces[0].message)
        assert parsed["error_type"] == "RuntimeError"
        assert "API down" in parsed["error_message"]
        assert parsed["data_source"] == "yfinance.fast_info"

    @patch("src.agents.market_data.yf.Ticker")
    def test_logs_agent_error_on_statement_failure(
        self, mock_ticker_cls: Mock, caplog: Any,
    ) -> None:
        """Agent should log agent_error when statements fail."""
        mock = _mock_yf_ticker()
        type(mock).financials = PropertyMock(
            side_effect=RuntimeError("financials down"),
        )
        type(mock).balance_sheet = PropertyMock(
            side_effect=RuntimeError("bs down"),
        )
        type(mock).cashflow = PropertyMock(
            side_effect=RuntimeError("cf down"),
        )
        mock_ticker_cls.return_value = mock

        state = create_initial_state("AAPL")
        with caplog.at_level(logging.DEBUG):
            MarketDataAgent().fetch_data(state)

        error_traces = [
            r for r in caplog.records
            if "agent_error" in r.message
        ]
        sources_traced = {
            json.loads(r.message)["data_source"]
            for r in error_traces
        }
        assert "yfinance.financials" in sources_traced
        assert "yfinance.balance_sheet" in sources_traced
        assert "yfinance.cashflow" in sources_traced

    @patch("src.agents.market_data.yf.Ticker")
    def test_all_traces_are_valid_json(
        self, mock_ticker_cls: Mock, caplog: Any,
    ) -> None:
        """Every trace log message should be valid JSON."""
        mock_ticker_cls.return_value = _mock_yf_ticker()
        state = create_initial_state("AAPL")
        with caplog.at_level(logging.DEBUG):
            MarketDataAgent().fetch_data(state)

        trace_records = [
            r for r in caplog.records
            if r.message.startswith("{")
        ]
        assert len(trace_records) >= 3  # at least started + calls + completed
        for record in trace_records:
            parsed = json.loads(record.message)
            assert "event" in parsed
            assert "timestamp" in parsed

    @patch("src.agents.market_data.yf.Ticker")
    def test_existing_behavior_unchanged(
        self, mock_ticker_cls: Mock,
    ) -> None:
        """Tracing should not change agent's data output."""
        mock_ticker_cls.return_value = _mock_yf_ticker()
        state = create_initial_state("AAPL")
        result = MarketDataAgent().fetch_data(state)

        assert result is state
        assert result["market_data"]["current_price"] == 150.0
        assert len(result["news"]) == 1
        assert len(result["errors"]) == 0


class TestMarketDataAgentInstitutionalDepth:
    """Tests for institutional depth data fetching."""

    def _mock_institutional_ticker(self) -> Mock:
        """Create mock ticker with institutional depth data."""
        ticker = _mock_yf_ticker()

        # Add 5-year price history
        dates = pd.date_range(start="2021-01-01", periods=1260, freq="D")
        hist_5y = pd.DataFrame({
            "Open": range(100, 1360),
            "High": range(105, 1365),
            "Low": range(95, 1355),
            "Close": range(100, 1360),
            "Volume": [1000000] * 1260,
        }, index=dates)
        ticker.history.side_effect = lambda period: (
            hist_5y if period == "5y"
            else pd.DataFrame({"Close": [148.0, 149.0, 150.0]})
        )

        # Add dividends
        div_dates = pd.date_range(start="2021-01-01", periods=20, freq="3ME")
        ticker.dividends = pd.Series([0.5] * 20, index=div_dates)

        # Add splits
        ticker.splits = pd.Series([2.0], index=[div_dates[0]])

        # Enhanced info with institutional data
        type(ticker).info = PropertyMock(return_value={
            "totalRevenue": 380_000_000_000,
            "netIncomeToCommon": 95_000_000_000,
            "totalCash": 60_000_000_000,
            "totalDebt": 110_000_000_000,
            "marketCap": 2_500_000_000_000,
            "sharesOutstanding": 16_000_000_000,
            "floatShares": 15_000_000_000,
            "heldPercentInstitutions": 0.65,
            "sector": "Technology",
            "industry": "Semiconductors",
            "trailingPE": 25.0,
        })

        return ticker

    @patch("src.agents.market_data.yf.Ticker")
    @patch("src.agents.market_data.get_cache")
    def test_fetches_all_institutional_depth_fields(
        self, mock_cache: Mock, mock_ticker_cls: Mock,
    ) -> None:
        """Agent should fetch all institutional depth fields."""
        mock_cache.return_value = Mock()
        mock_ticker_cls.return_value = self._mock_institutional_ticker()
        state = create_initial_state("AAPL")

        result = MarketDataAgent().fetch_data(state)

        # Verify all new fields present
        assert "historical_prices_5y" in result["market_data"]
        assert "dividend_history" in result["market_data"]
        assert "split_history" in result["market_data"]
        assert "volume_analysis" in result["market_data"]
        assert "week_52_range" in result["market_data"]
        assert "beta" in result["market_data"]
        assert "institutional_ownership_pct" in result["market_data"]
        assert "float_pct" in result["market_data"]
        assert "peer_comparison" in result["market_data"]
        assert result["market_data"]["peer_comparison"]["sector"] == "Technology"
        # Verify institutional_ownership_pct is percentage (0-100), not fraction
        assert result["market_data"]["institutional_ownership_pct"] == 65.0

    @patch("src.agents.market_data.yf.Ticker")
    @patch("src.agents.market_data.get_cache")
    def test_handles_partial_data_gracefully(
        self, mock_cache: Mock, mock_ticker_cls: Mock,
    ) -> None:
        """Agent should handle missing institutional data without crashing."""
        mock_cache.return_value = Mock()
        mock = _mock_yf_ticker()  # Basic mock without institutional data
        mock.dividends = pd.Series()  # No dividends
        mock.splits = pd.Series()  # No splits
        type(mock).info = PropertyMock(return_value={})  # Empty info
        mock_ticker_cls.return_value = mock

        state = create_initial_state("AAPL")
        result = MarketDataAgent().fetch_data(state)

        # Verify no exceptions raised, fields set to None
        assert result["market_data"]["dividend_history"] is None
        assert result["market_data"]["split_history"] is None
        assert len(result["errors"]) == 0  # Missing data is not an error

    @patch("src.agents.market_data.yf.Ticker")
    def test_adds_provenance_with_confidence_score(
        self, mock_ticker_cls: Mock,
    ) -> None:
        """Agent should add provenance metadata with confidence score."""
        mock_ticker_cls.return_value = self._mock_institutional_ticker()
        state = create_initial_state("AAPL")

        result = MarketDataAgent().fetch_data(state)

        # Verify provenance metadata
        assert "provenance_metadata" in result
        assert "market_data" in result["provenance_metadata"]
        prov = result["provenance_metadata"]["market_data"]
        assert prov["agent"] == "MarketDataAgent"
        assert prov["source"] == "yfinance"
        assert "timestamp" in prov
        assert "confidence" in prov
        assert isinstance(prov["confidence"], float)
        assert "data_fields" in prov

    @patch("src.agents.market_data.yf.Ticker")
    def test_returns_same_state_object(
        self, mock_ticker_cls: Mock,
    ) -> None:
        """Agent should return same state object (not create new dict)."""
        mock_ticker_cls.return_value = self._mock_institutional_ticker()
        state = create_initial_state("AAPL")

        result = MarketDataAgent().fetch_data(state)

        # Verify state identity (same object)
        assert result is state
