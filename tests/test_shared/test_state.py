"""Tests for shared ResearchState type and factory."""

from __future__ import annotations

import pytest
from doxa_shared.types.state import create_initial_state


class TestCreateInitialState:
    """Tests for create_initial_state factory function."""

    def test_creates_state_with_uppercase_ticker(self) -> None:
        state = create_initial_state("aapl")
        assert state["ticker"] == "AAPL"

    def test_creates_state_with_empty_defaults(self) -> None:
        state = create_initial_state("NVDA")
        assert state["market_data"] == {}
        assert state["financials"] == {}
        assert state["news"] == []
        assert state["sentiment_score"] == 0.0
        assert state["sentiment_rationale"] == ""
        assert state["quant_analysis"] == {}
        assert state["valuation_analysis"] == {}
        assert state["alternative_data"] == {}
        assert state["human_notes"] == ""
        assert state["final_report"] == ""
        assert state["errors"] == []

    def test_raises_on_empty_ticker(self) -> None:
        with pytest.raises(ValueError, match="Ticker must not be empty"):
            create_initial_state("")

    def test_raises_on_whitespace_ticker(self) -> None:
        with pytest.raises(ValueError, match="Ticker must not be empty"):
            create_initial_state("   ")

    def test_strips_whitespace_from_ticker(self) -> None:
        state = create_initial_state("  msft  ")
        assert state["ticker"] == "MSFT"


class TestResearchStateMutation:
    """Tests for state mutation pattern - modify in place, not recreate."""

    def test_state_is_mutable_dict(self) -> None:
        state = create_initial_state("TEST")
        state["errors"].append("test error")
        assert len(state["errors"]) == 1

    def test_state_returns_same_object(self) -> None:
        state = create_initial_state("TEST")
        original_id = id(state)
        state["market_data"]["price"] = 100.0
        assert id(state) == original_id
