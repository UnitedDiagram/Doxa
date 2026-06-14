"""Tests that SentimentAgent posts insights to insights_board."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from doxa_shared.types.state import create_initial_state

from src.agents.sentiment import SentimentAgent


def _make_mock_claude_response(
    score: float = 0.3,
    rationale: str = "Positive outlook",
    contradictions: list[str] | None = None,
) -> Mock:
    """Create a mock Anthropic streaming response."""
    import json

    contradictions = contradictions or []
    response_data = {
        "score": score,
        "rationale": rationale,
        "key_catalyst": "Strong earnings",
        "contradictions": contradictions,
    }

    mock_block = Mock()
    mock_block.type = "text"
    mock_block.text = json.dumps(response_data)

    mock_message = Mock()
    mock_message.content = [mock_block]

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    mock_stream.get_final_message = AsyncMock(return_value=mock_message)

    return mock_stream


def _run_sentiment(state: dict, patch_stream: Mock) -> dict:
    """Run SentimentAgent.analyze() synchronously for testing."""
    with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_cls, \
         patch(
             "src.agents.sentiment.fetch_alternative_data",
             return_value={
                 "insider_trading": None,
                 "short_interest": None,
                 "options_flow": None,
                 "social_media": None,
                 "contradictions": [],
             },
         ), \
         patch("src.agents.sentiment.ANTHROPIC_API_KEY", "fake-key"):
        mock_client = Mock()
        mock_client.messages.stream.return_value = patch_stream
        mock_client_cls.return_value = mock_client

        return asyncio.run(
            SentimentAgent().analyze(state)
        )


def test_insider_selling_posts_insight() -> None:
    """Net insider selling triggers an insider_activity insight."""
    stream = _make_mock_claude_response(score=0.1)

    state = create_initial_state("NVDA")
    state["news"] = [{"headline": "NVDA Q3 beat estimates"}]

    with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_cls, \
         patch(
             "src.agents.sentiment.fetch_alternative_data",
             return_value={
                 "insider_trading": {
                     "net_shares_sold": 50_000,
                     "net_shares_bought": 5_000,
                 },
                 "short_interest": None,
                 "options_flow": None,
                 "social_media": None,
                 "contradictions": [],
             },
         ), \
         patch("src.agents.sentiment.ANTHROPIC_API_KEY", "fake-key"):
        mock_client = Mock()
        mock_client.messages.stream.return_value = stream
        mock_client_cls.return_value = mock_client
        result = asyncio.run(
            SentimentAgent().analyze(state)
        )

    insider_insights = [
        ins for ins in result["insights_board"]
        if ins.get("category") == "insider_activity"
    ]
    assert len(insider_insights) >= 1
    assert insider_insights[0]["agent"] == "SentimentAgent"
    assert "50,000" in insider_insights[0]["signal"]


def test_high_short_interest_posts_insight() -> None:
    """Short interest >= 10% triggers a short_interest insight."""
    stream = _make_mock_claude_response(score=-0.2)

    state = create_initial_state("NVDA")
    state["news"] = [{"headline": "NVDA under pressure"}]

    with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_cls, \
         patch(
             "src.agents.sentiment.fetch_alternative_data",
             return_value={
                 "insider_trading": None,
                 "short_interest": {"short_interest_pct": 15.0},
                 "options_flow": None,
                 "social_media": None,
                 "contradictions": [],
             },
         ), \
         patch("src.agents.sentiment.ANTHROPIC_API_KEY", "fake-key"):
        mock_client = Mock()
        mock_client.messages.stream.return_value = stream
        mock_client_cls.return_value = mock_client
        result = asyncio.run(
            SentimentAgent().analyze(state)
        )

    short_insights = [
        ins for ins in result["insights_board"]
        if ins.get("category") == "short_interest"
    ]
    assert len(short_insights) >= 1
    assert "15.0%" in short_insights[0]["signal"]


def test_strong_negative_sentiment_posts_insight() -> None:
    """Sentiment score <= -0.5 triggers a sentiment_contradiction insight."""
    stream = _make_mock_claude_response(score=-0.7)

    state = create_initial_state("NVDA")
    state["news"] = [{"headline": "NVDA faces major headwinds"}]

    with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_cls, \
         patch(
             "src.agents.sentiment.fetch_alternative_data",
             return_value={
                 "insider_trading": None,
                 "short_interest": None,
                 "options_flow": None,
                 "social_media": None,
                 "contradictions": [],
             },
         ), \
         patch("src.agents.sentiment.ANTHROPIC_API_KEY", "fake-key"):
        mock_client = Mock()
        mock_client.messages.stream.return_value = stream
        mock_client_cls.return_value = mock_client
        result = asyncio.run(
            SentimentAgent().analyze(state)
        )

    neg_insights = [
        ins for ins in result["insights_board"]
        if ins.get("category") == "sentiment_contradiction"
    ]
    assert len(neg_insights) >= 1
    # The score value should appear in the signal
    assert "-0.70" in neg_insights[0]["signal"]


def test_state_identity_preserved() -> None:
    """analyze() returns the same state dict object."""
    stream = _make_mock_claude_response(score=0.3)

    state = create_initial_state("NVDA")
    state["news"] = [{"headline": "Test"}]

    with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_cls, \
         patch(
             "src.agents.sentiment.fetch_alternative_data",
             return_value={
                 "insider_trading": None,
                 "short_interest": None,
                 "options_flow": None,
                 "social_media": None,
                 "contradictions": [],
             },
         ), \
         patch("src.agents.sentiment.ANTHROPIC_API_KEY", "fake-key"):
        mock_client = Mock()
        mock_client.messages.stream.return_value = stream
        mock_client_cls.return_value = mock_client
        result = asyncio.run(
            SentimentAgent().analyze(state)
        )

    assert result is state


def test_no_insights_when_normal_market_conditions() -> None:
    """When all metrics are healthy, no insights are posted."""
    stream = _make_mock_claude_response(score=0.3)

    state = create_initial_state("NVDA")
    state["news"] = [{"headline": "NVDA is doing great"}]

    with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_cls, \
         patch(
             "src.agents.sentiment.fetch_alternative_data",
             return_value={
                 "insider_trading": {
                     "net_shares_sold": 100,
                     "net_shares_bought": 500,  # net buying - no insight
                 },
                 "short_interest": {"short_interest_pct": 3.5},  # below 10%
                 "options_flow": None,
                 "social_media": None,
                 "contradictions": [],
             },
         ), \
         patch("src.agents.sentiment.ANTHROPIC_API_KEY", "fake-key"):
        mock_client = Mock()
        mock_client.messages.stream.return_value = stream
        mock_client_cls.return_value = mock_client
        result = asyncio.run(
            SentimentAgent().analyze(state)
        )

    # score is 0.3, not <= -0.5; short interest below 10%; net buying
    assert len(result["insights_board"]) == 0
    assert len(result["errors"]) == 0
