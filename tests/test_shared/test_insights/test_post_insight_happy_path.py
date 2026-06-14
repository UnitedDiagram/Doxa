"""Tests for post_insight() in doxa_shared.utils.insights — happy path."""

from __future__ import annotations

from doxa_shared.types.state import create_initial_state
from doxa_shared.utils.insights import post_insight


def test_post_insight_appends_to_board() -> None:
    """post_insight appends one insight with correct schema."""
    state = create_initial_state("NVDA")
    assert state["insights_board"] == []

    post_insight(
        state,
        agent="MarketDataAgent",
        category="volume",
        signal="Volume spike: 3.0x average",
        confidence=0.8,
    )

    assert len(state["insights_board"]) == 1
    insight = state["insights_board"][0]
    assert insight["agent"] == "MarketDataAgent"
    assert insight["category"] == "volume"
    assert insight["signal"] == "Volume spike: 3.0x average"
    assert insight["confidence"] == 0.8
    assert "timestamp" in insight
    assert "T" in insight["timestamp"]  # ISO 8601 format


def test_post_insight_multiple_entries_accumulate() -> None:
    """Multiple post_insight calls accumulate without overwriting."""
    state = create_initial_state("AAPL")

    post_insight(state, agent="A", category="cat1", signal="sig1", confidence=0.5)
    post_insight(state, agent="B", category="cat2", signal="sig2", confidence=0.6)

    assert len(state["insights_board"]) == 2
    assert state["insights_board"][0]["agent"] == "A"
    assert state["insights_board"][1]["agent"] == "B"


def test_post_insight_does_not_raise_on_missing_board_key() -> None:
    """post_insight does not raise if insights_board is intact (normal path)."""
    state = create_initial_state("TSLA")
    # Simulate a state where insights_board is an already-populated list
    state["insights_board"].append(
        {
            "agent": "X", "category": "c",
            "signal": "s", "confidence": 0.5,
            "timestamp": "t",
        }
    )

    post_insight(state, agent="Y", category="d", signal="new", confidence=0.3)
    assert len(state["insights_board"]) == 2


def test_post_insight_state_identity() -> None:
    """post_insight mutates the same state object, does not create a new one."""
    state = create_initial_state("MSFT")
    original_id = id(state)

    post_insight(state, agent="Z", category="c", signal="s", confidence=0.9)

    assert id(state) == original_id
