"""Tests that ValuationAgent posts insights to insights_board."""

from __future__ import annotations

from unittest.mock import Mock, PropertyMock, patch

import pandas as pd
from doxa_shared.types.state import create_initial_state

from src.agents.valuation import ValuationAgent, _post_valuation_insights


def _make_cashflow_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Operating Cash Flow": [5_000_000_000, 4_000_000_000]},
        index=pd.RangeIndex(2),
    ).T


def _make_income_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Net Income": [2_000_000_000, 1_800_000_000]},
        index=pd.RangeIndex(2),
    ).T


def _make_balance_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Stockholders Equity": [10_000_000_000, 9_000_000_000],
         "Total Debt": [5_000_000_000, 5_000_000_000]},
        index=pd.RangeIndex(2),
    ).T


def _mock_yf_ticker_valuation() -> Mock:
    """Create a mock yfinance Ticker for ValuationAgent."""
    ticker = Mock()
    type(ticker).info = PropertyMock(return_value={
        "sector": "Technology",
        "currentPrice": 100.0,
        "sharesOutstanding": 1_000_000_000,
        "marketCap": 100_000_000_000,
        "totalRevenue": 20_000_000_000,
        "netIncome": 2_000_000_000,
        "totalDebt": 5_000_000_000,
        "totalCash": 10_000_000_000,
        "ebitda": 3_000_000_000,
        "bookValue": 10.0,
        "beta": 1.2,
        "returnOnEquity": 0.20,
        "profitMargins": 0.10,
    })
    cf = _make_cashflow_df()
    type(ticker).cashflow = PropertyMock(return_value=cf)
    inc = _make_income_df()
    type(ticker).income_stmt = PropertyMock(return_value=inc)
    bs = _make_balance_df()
    type(ticker).balance_sheet = PropertyMock(return_value=bs)
    type(ticker).financials = PropertyMock(return_value=pd.DataFrame())
    return ticker


def test_altman_distress_posts_leverage_insight() -> None:
    """When Altman Z is in distress zone, a leverage insight is posted."""
    state = create_initial_state("NVDA")
    state["valuation_analysis"] = {
        "altman_z_score": {
            "z_score": 1.5,
            "interpretation": "Distress Zone - High bankruptcy risk",
            "components": {},
        },
        "dupont_analysis": None,
        "trend_analysis": None,
        "dcf": {},
    }

    _post_valuation_insights(state)

    leverage_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "leverage"
    ]
    assert len(leverage_insights) >= 1
    assert leverage_insights[0]["agent"] == "ValuationAgent"
    assert "distress zone" in leverage_insights[0]["signal"].lower()
    assert leverage_insights[0]["confidence"] >= 0.8


def test_low_roe_posts_profitability_insight() -> None:
    """When ROE < 5%, a profitability insight is posted."""
    state = create_initial_state("NVDA")
    state["valuation_analysis"] = {
        "altman_z_score": None,
        "dupont_analysis": {
            "roe": 0.03,  # 3% ROE — below 5% threshold
            "profit_margin": 0.05,
            "asset_turnover": 0.3,
            "equity_multiplier": 2.0,
        },
        "trend_analysis": None,
        "dcf": {},
    }

    _post_valuation_insights(state)

    prof_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "profitability"
    ]
    assert len(prof_insights) >= 1
    assert "3.0%" in prof_insights[0]["signal"]


def test_no_insights_when_metrics_healthy() -> None:
    """No distress or low-ROE insights when metrics are healthy."""
    state = create_initial_state("NVDA")
    state["valuation_analysis"] = {
        "altman_z_score": {
            "z_score": 3.5,
            "interpretation": "Safe Zone - Low bankruptcy risk",
            "components": {},
        },
        "dupont_analysis": {
            "roe": 0.20,  # 20% ROE — healthy
        },
        "trend_analysis": None,
        "dcf": {},
    }

    _post_valuation_insights(state)

    leverage_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "leverage"
    ]
    prof_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "profitability"
    ]
    assert len(leverage_insights) == 0
    assert len(prof_insights) == 0


@patch("src.agents.valuation.yf.Ticker")
def test_state_identity_preserved(mock_ticker_cls: Mock) -> None:
    """execute() returns the same state object."""
    mock_ticker_cls.return_value = _mock_yf_ticker_valuation()
    state = create_initial_state("NVDA")
    result = ValuationAgent().execute(state)
    assert result is state
