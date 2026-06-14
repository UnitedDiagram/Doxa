"""Tests that MarketDataAgent posts insights to insights_board.

These tests test both the full agent pipeline (state identity, no-spike case)
and the _post_market_insights helper directly for signal-specific assertions.
"""

from __future__ import annotations

from unittest.mock import Mock, PropertyMock, patch

import pandas as pd
from doxa_shared.types.state import create_initial_state

from src.agents.market_data import MarketDataAgent, _post_market_insights


def _mock_ticker_base() -> Mock:
    """Create a baseline mock yfinance Ticker."""
    ticker = Mock()

    fi = Mock()
    fi.last_price = 50.0
    fi.market_cap = 10_000_000_000
    fi.year_high = 80.0
    fi.year_low = 40.0
    type(ticker).fast_info = PropertyMock(return_value=fi)

    type(ticker).info = PropertyMock(return_value={
        "sector": "Technology",
        "industry": "Software",
        "sharesOutstanding": 1_000_000_000,
        "floatShares": 900_000_000,
        "heldPercentInstitutions": 0.50,
    })

    hist = pd.DataFrame({"Close": [50.0] * 30})
    ticker.history.return_value = hist
    type(ticker).financials = PropertyMock(return_value=pd.DataFrame())
    type(ticker).balance_sheet = PropertyMock(return_value=pd.DataFrame())
    type(ticker).cashflow = PropertyMock(return_value=pd.DataFrame())
    type(ticker).news = PropertyMock(return_value=[])
    ticker.dividends = pd.Series(dtype=float)
    ticker.splits = pd.Series(dtype=float)

    return ticker


def test_volume_spike_posts_insight() -> None:
    """Volume spike in market_data posts a volume insight."""
    state = create_initial_state("NVDA")
    state["market_data"]["volume_analysis"] = {
        "spike_detected": True, "spike_ratio": 3.5,
    }
    state["market_data"]["week_52_range"] = {"high": 80.0, "low": 40.0}
    state["market_data"]["current_price"] = 50.0
    state["market_data"]["price_history"] = [50.0] * 30

    _post_market_insights(state)

    volume_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "volume"
    ]
    assert len(volume_insights) >= 1
    assert volume_insights[0]["agent"] == "MarketDataAgent"
    assert "3.5" in volume_insights[0]["signal"]
    assert 0.0 <= volume_insights[0]["confidence"] <= 1.0
    assert "timestamp" in volume_insights[0]


def test_no_spike_posts_no_volume_insight() -> None:
    """Without spike, no volume insight is posted."""
    state = create_initial_state("NVDA")
    state["market_data"]["volume_analysis"] = {
        "spike_detected": False, "spike_ratio": 1.1,
    }
    state["market_data"]["week_52_range"] = {"high": 80.0, "low": 40.0}
    state["market_data"]["current_price"] = 50.0
    state["market_data"]["price_history"] = [50.0] * 30

    _post_market_insights(state)

    volume_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "volume"
    ]
    assert len(volume_insights) == 0


def test_52week_low_breach_posts_price_action_insight() -> None:
    """Price within 2% of 52-week low triggers a price_action insight."""
    state = create_initial_state("NVDA")
    # current=50.0, low=49.5 → (50.0-49.5)/49.5 = 1.01% <= 2%
    state["market_data"]["volume_analysis"] = {"spike_detected": False}
    state["market_data"]["week_52_range"] = {"high": 80.0, "low": 49.5}
    state["market_data"]["current_price"] = 50.0
    state["market_data"]["price_history"] = [50.0] * 30

    _post_market_insights(state)

    price_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "price_action"
    ]
    assert len(price_insights) >= 1
    assert "52-week low" in price_insights[0]["signal"]
    assert len(state["errors"]) == 0


def test_52week_high_breach_posts_price_action_insight() -> None:
    """Price within 2% of 52-week high triggers a price_action insight."""
    state = create_initial_state("NVDA")
    # current=79.5, high=80.0 → (80.0-79.5)/80.0 = 0.625% <= 2%
    state["market_data"]["volume_analysis"] = {"spike_detected": False}
    state["market_data"]["week_52_range"] = {"high": 80.0, "low": 40.0}
    state["market_data"]["current_price"] = 79.5
    state["market_data"]["price_history"] = [79.5] * 30

    _post_market_insights(state)

    price_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "price_action"
    ]
    assert len(price_insights) >= 1
    assert "52-week high" in price_insights[0]["signal"]
    assert len(state["errors"]) == 0


def test_insights_errors_dont_crash_with_malformed_data() -> None:
    """_post_market_insights does not raise when given malformed state data."""
    state = create_initial_state("NVDA")
    state["market_data"]["volume_analysis"] = None
    state["market_data"]["week_52_range"] = None
    state["market_data"]["current_price"] = None
    state["market_data"]["price_history"] = None

    # Should not raise — errors are appended to state["errors"] if any occur
    _post_market_insights(state)
    assert isinstance(state["insights_board"], list)


@patch("src.agents.market_data.yf.Ticker")
@patch("src.agents.market_data.get_cache")
def test_insights_errors_dont_crash_agent(
    mock_cache: Mock, mock_ticker_cls: Mock,
) -> None:
    """Even if _post_market_insights is patched to throw, agent still returns state."""
    mock_cache.return_value = Mock()
    mock_ticker_cls.return_value = _mock_ticker_base()

    with patch(
        "src.agents.market_data._post_market_insights",
        side_effect=RuntimeError("test-forced-error"),
    ):
        state = create_initial_state("NVDA")
        result = MarketDataAgent().fetch_data(state)

    # Agent still returns the same state object
    assert result is state


@patch("src.agents.market_data.yf.Ticker")
@patch("src.agents.market_data.get_cache")
def test_state_identity_preserved(mock_cache: Mock, mock_ticker_cls: Mock) -> None:
    """fetch_data returns the same state dict object."""
    mock_cache.return_value = Mock()
    mock_ticker_cls.return_value = _mock_ticker_base()
    state = create_initial_state("AAPL")

    result = MarketDataAgent().fetch_data(state)
    assert result is state
