"""Tests for post_insight() confidence clamping behavior."""

from __future__ import annotations

import logging

from doxa_shared.types.state import create_initial_state
from doxa_shared.utils.insights import post_insight


def test_post_insight_clamps_confidence_above_one(caplog: logging.LogRecord) -> None:
    """Confidence > 1.0 is clamped to 1.0 with a warning."""
    state = create_initial_state("NVDA")

    with caplog.at_level(logging.WARNING, logger="doxa_shared.utils.insights"):
        post_insight(
            state,
            agent="MarketDataAgent",
            category="volume",
            signal="Test signal",
            confidence=1.5,
        )

    assert len(state["insights_board"]) == 1
    assert state["insights_board"][0]["confidence"] == 1.0
    assert "clamping" in caplog.text.lower() or "out of" in caplog.text.lower()


def test_post_insight_clamps_confidence_below_zero(caplog: logging.LogRecord) -> None:
    """Confidence < 0.0 is clamped to 0.0 with a warning."""
    state = create_initial_state("AAPL")

    with caplog.at_level(logging.WARNING, logger="doxa_shared.utils.insights"):
        post_insight(
            state,
            agent="ValuationAgent",
            category="leverage",
            signal="Test signal",
            confidence=-0.3,
        )

    assert len(state["insights_board"]) == 1
    assert state["insights_board"][0]["confidence"] == 0.0
    assert len(state["errors"]) == 0  # should not append to errors


def test_post_insight_boundary_values_are_valid() -> None:
    """Confidence exactly 0.0 and 1.0 are accepted without warnings."""
    state = create_initial_state("GOOG")

    post_insight(state, agent="A", category="c", signal="s", confidence=0.0)
    post_insight(state, agent="B", category="c", signal="s", confidence=1.0)

    assert len(state["insights_board"]) == 2
    assert state["insights_board"][0]["confidence"] == 0.0
    assert state["insights_board"][1]["confidence"] == 1.0
    assert len(state["errors"]) == 0
