"""Tests for SentimentAgent async implementation and confidence scoring."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.sentiment import SentimentAgent
from src.state import create_initial_state


@pytest.fixture
def sample_state():
    """Create sample research state with news headlines."""
    state = create_initial_state("AAPL")
    state["news"] = [
        {"headline": "AAPL announces record-breaking quarterly earnings"},
        {"headline": "AAPL stock hits all-time high on strong iPhone sales"},
    ]
    return state


def create_mock_anthropic_response(
    score: float,
    rationale: str | None = None,
    key_catalyst: str | None = None,
) -> tuple[AsyncMock, MagicMock]:
    """Create mocked AsyncAnthropic client and message.

    Args:
        score: Sentiment score to return.
        rationale: Optional rationale text.
        key_catalyst: Optional key catalyst text.

    Returns:
        A tuple of (mock_client, mock_message).
    """
    # Build JSON response
    response_data: dict[str, Any] = {"score": score}
    if rationale is not None:
        response_data["rationale"] = rationale
    if key_catalyst is not None:
        response_data["key_catalyst"] = key_catalyst

    json_text = json.dumps(response_data)

    # Create mock message with text block
    mock_message = MagicMock()
    mock_message.content = [MagicMock(type="text", text=json_text)]

    # Create mock stream that properly implements async context manager
    mock_stream_enter = AsyncMock()
    mock_stream_enter.get_final_message = AsyncMock(return_value=mock_message)

    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream_enter)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    # Create mock client
    mock_client = AsyncMock()
    mock_client.messages.stream = MagicMock(return_value=mock_stream)

    return mock_client, mock_message


@pytest.mark.asyncio
async def test_sentiment_async_execution(sample_state):
    """Test that analyze() is properly async and returns state."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(
                score=0.8,
                rationale="Positive earnings report",
                key_catalyst="Record revenue",
            )
            mock_client_class.return_value = mock_client

            agent = SentimentAgent()
            result = await agent.analyze(sample_state)

            # Verify async execution worked
            assert result is sample_state  # Same object (mutation pattern)
            assert mock_client.messages.stream.called
            assert "alternative_data" in result
            assert result["alternative_data"] != {}
            assert "insider_trading" in result["alternative_data"]
            assert "contradictions" in result["alternative_data"]


@pytest.mark.asyncio
async def test_sentiment_enhanced_contradiction_detection(sample_state):
    """Test that enhanced sentiment analysis detects contradictions."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(
                score=0.2,
                rationale="Record earnings but insiders selling heavily",
                key_catalyst="Mixed signals",
            )
            # Add contradictions to mock response data
            mock_message = MagicMock()
            json_text = json.dumps({
                "score": 0.2,
                "rationale": "Record earnings but insiders selling heavily",
                "contradictions": ["Insiders selling despite record earnings"],
                "key_catalyst": "Mixed signals"
            })
            mock_message.content = [MagicMock(type="text", text=json_text)]
            mock_stream = mock_client.messages.stream.return_value
            mock_stream.__aenter__.return_value.get_final_message.return_value = (
                mock_message
            )
            mock_client_class.return_value = mock_client

            agent = SentimentAgent()
            result = await agent.analyze(sample_state)

            assert (
                "Insiders selling despite record earnings"
                in result["alternative_data"]["contradictions"]
            )
            assert result["sentiment_score"] == 0.2


@pytest.mark.asyncio
async def test_enhanced_confidence_calculation(sample_state):
    """Test the 50+10+10+10+10+10 confidence scoring logic."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(
                score=0.5, rationale="Test", key_catalyst="Test"
            )
            mock_client_class.return_value = mock_client

            # Mock alternative data with only insider and short interest
            with patch("src.agents.sentiment.fetch_alternative_data") as mock_fetch:
                mock_fetch.return_value = {
                    "insider_trading": {"signal": "positive"},
                    "short_interest": {"short_pct": 1.0},
                    "options_flow": None,
                    "social_media": None,
                    "provenance": {"source": "Mock"}
                }

                agent = SentimentAgent()
                result = await agent.analyze(sample_state)

                # Base 50 + 10 (insider) + 10 (short) + 10 (Claude success) = 80
                # Wait, Task 6 says:
                # - Base score: 50
                # - +10 for insider data
                # - +10 for short interest
                # - +10 for options flow
                # - +10 for social sentiment
                # - +10 for Claude's successfully parsed analysis
                assert result["sentiment_confidence"] == 80.0


@pytest.mark.asyncio
async def test_confidence_100_percent(sample_state):
    """Test 100% confidence when all fields present."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(
                score=0.8,
                rationale="Strong positive sentiment",
                key_catalyst="Record earnings",
            )
            mock_client_class.return_value = mock_client

            agent = SentimentAgent()
            result = await agent.analyze(sample_state)

            assert result["sentiment_score"] == 0.8
            assert "Strong positive sentiment" in result["sentiment_rationale"]
            # 50 base + 40 alt data + 10 Claude success = 100
            assert result["sentiment_confidence"] == 100.0
            assert result["sentiment_confidence_details"]["missing_fields"] == []
            assert "Claude analysis success" in result["sentiment_confidence_details"][
                "reason"
            ]


@pytest.mark.asyncio
async def test_confidence_90_percent_missing_catalyst(sample_state):
    """Test 90% confidence when key_catalyst missing."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(
                score=0.6, rationale="Moderate positive sentiment"
            )
            mock_client_class.return_value = mock_client

            agent = SentimentAgent()
            result = await agent.analyze(sample_state)

            assert result["sentiment_score"] == 0.6
            # 50 base + 40 alt data + 0 Claude (missing field) = 90
            assert result["sentiment_confidence"] == 90.0
            assert "key_catalyst" in result["sentiment_confidence_details"][
                "missing_fields"
            ]


@pytest.mark.asyncio
async def test_confidence_90_percent_partial_claude(sample_state):
    """Test 90% confidence when only score present."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(score=0.3)
            mock_client_class.return_value = mock_client

            agent = SentimentAgent()
            result = await agent.analyze(sample_state)

            assert result["sentiment_score"] == 0.3
            # 50 base + 40 alt data + 0 Claude = 90
            assert result["sentiment_confidence"] == 90.0
            assert "rationale" in result["sentiment_confidence_details"][
                "missing_fields"
            ]
            assert "key_catalyst" in result["sentiment_confidence_details"][
                "missing_fields"
            ]


@pytest.mark.asyncio
async def test_confidence_90_percent_api_failure(sample_state):
    """Test 90% confidence when API call fails but alt data present."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            # Create mock client
            mock_client = MagicMock()

            # Make stream() raise synchronously (not return an awaitable)
            # This simulates API client creation/connection failures
            mock_client.messages.stream.side_effect = Exception("API error")
            mock_client_class.return_value = mock_client

            agent = SentimentAgent()
            result = await agent.analyze(sample_state)

            assert result["sentiment_score"] == 0.0
            # 50 base + 40 alt data + 0 Claude = 90
            assert result["sentiment_confidence"] == 90.0
            assert (
                "claude_analysis"
                in result["sentiment_confidence_details"]["missing_fields"]
            )



@pytest.mark.asyncio
async def test_backward_compatibility_no_headlines(sample_state):
    """Test backward compatibility when no headlines available."""
    sample_state["news"] = []

    agent = SentimentAgent()
    result = await agent.analyze(sample_state)

    # Should return state unchanged with defaults
    assert result is sample_state
    assert result["sentiment_score"] == 0.0
    assert result["sentiment_rationale"] == ""


@pytest.mark.asyncio
async def test_score_clamping(sample_state):
    """Test that scores are clamped to [-1.0, 1.0]."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            # Test score > 1.0 gets clamped
            mock_client, _ = create_mock_anthropic_response(
                score=1.5, rationale="Test", key_catalyst="Test"
            )
            mock_client_class.return_value = mock_client

            agent = SentimentAgent()
            result = await agent.analyze(sample_state)

            assert result["sentiment_score"] == 1.0  # Clamped

            # Test score < -1.0 gets clamped
            mock_client, _ = create_mock_anthropic_response(
                score=-1.8, rationale="Test", key_catalyst="Test"
            )
            mock_client_class.return_value = mock_client

            result = await agent.analyze(sample_state)
            assert result["sentiment_score"] == -1.0  # Clamped


@pytest.mark.asyncio
async def test_trace_logging(sample_state, caplog):
    """Test that confidence_calculated trace event is logged."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(
                score=0.7, rationale="Positive", key_catalyst="Growth"
            )
            mock_client_class.return_value = mock_client

            with caplog.at_level(logging.DEBUG):
                agent = SentimentAgent()
                await agent.analyze(sample_state)

            # Check for trace log
            trace_logs = [
                r for r in caplog.records if "confidence_calculated" in r.message
            ]
            assert len(trace_logs) > 0

            trace_log = trace_logs[0]
            assert "SentimentAgent" in trace_log.message
            assert "confidence" in trace_log.message


@pytest.mark.asyncio
async def test_state_mutation_pattern(sample_state):
    """Test that same state object is returned (mutation pattern)."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(
                score=0.5, rationale="Neutral", key_catalyst="Mixed signals"
            )
            mock_client_class.return_value = mock_client

            agent = SentimentAgent()
            original_id = id(sample_state)
            result = await agent.analyze(sample_state)

            # Verify same object returned
            assert id(result) == original_id
            assert result is sample_state


@pytest.mark.asyncio
async def test_key_catalyst_appended_to_rationale(sample_state):
    """Test that key_catalyst is appended to rationale when present."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(
                score=0.8,
                rationale="Strong growth indicators",
                key_catalyst="iPhone 16 launch",
            )
            mock_client_class.return_value = mock_client

            agent = SentimentAgent()
            result = await agent.analyze(sample_state)

            assert "Strong growth indicators" in result["sentiment_rationale"]
            assert "Key catalyst: iPhone 16 launch" in result["sentiment_rationale"]


@pytest.mark.asyncio
async def test_no_api_key_error_handling(sample_state):
    """Test error handling when ANTHROPIC_API_KEY is not set."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", None):
        agent = SentimentAgent()
        result = await agent.analyze(sample_state)

        assert result["sentiment_score"] == 0.0
        assert result["sentiment_rationale"] == ""
        assert len(result["errors"]) > 0
        assert "ANTHROPIC_API_KEY" in result["errors"][0]
