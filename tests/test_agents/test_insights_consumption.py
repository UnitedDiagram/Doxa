"""Tests for Writer/Editor insight board consumption formatting.

Verifies that _format_insights_for_prompt (WriterAgent) and
_format_insights_for_editor (EditorAgent) correctly format the
insights_board entries and handle the empty-board fallback.
"""

from __future__ import annotations

from src.agents.editor import _format_insights_for_editor
from src.agents.writer import _format_insights_for_prompt

_SAMPLE_INSIGHTS = [
    {
        "agent": "MarketDataAgent",
        "category": "volume",
        "signal": "NVDA volume spike: 3.5x avg",
        "confidence": 0.85,
        "timestamp": "2024-01-15T10:00:00Z",
    },
    {
        "agent": "ValuationAgent",
        "category": "leverage",
        "signal": "NVDA Altman Z distress zone",
        "confidence": 0.9,
        "timestamp": "2024-01-15T10:01:00Z",
    },
]


# -- WriterAgent: _format_insights_for_prompt --


def test_writer_empty_board_returns_placeholder() -> None:
    """Empty list returns the placeholder string."""
    result = _format_insights_for_prompt([])
    assert result == "No cross-domain insights available."


def test_writer_formats_insights_grouped_by_category() -> None:
    """Populated board groups entries by category."""
    result = _format_insights_for_prompt(_SAMPLE_INSIGHTS)

    assert "VOLUME:" in result
    assert "LEVERAGE:" in result
    assert "MarketDataAgent" in result
    assert "ValuationAgent" in result
    assert "3.5x" in result
    assert "distress" in result


def test_writer_includes_confidence_as_percent() -> None:
    """Confidence is formatted as a percent (e.g. 85%)."""
    result = _format_insights_for_prompt(_SAMPLE_INSIGHTS)
    assert "85%" in result
    assert "90%" in result


# -- EditorAgent: _format_insights_for_editor --


def test_editor_empty_board_returns_placeholder() -> None:
    """Empty list returns the placeholder string."""
    result = _format_insights_for_editor([])
    assert result == "No cross-domain insights available."


def test_editor_formats_insights_as_flat_list() -> None:
    """Populated board produces a flat bullet list."""
    result = _format_insights_for_editor(_SAMPLE_INSIGHTS)

    lines = result.strip().split("\n")
    assert len(lines) == 2
    assert all(line.startswith("- [") for line in lines)
    assert "MarketDataAgent" in result
    assert "ValuationAgent" in result


def test_editor_includes_confidence_as_percent() -> None:
    """Confidence is formatted as a percent (e.g. 85%)."""
    result = _format_insights_for_editor(_SAMPLE_INSIGHTS)
    assert "85%" in result
    assert "90%" in result
