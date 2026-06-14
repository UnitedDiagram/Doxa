"""Enhanced tests for SentimentAgent alternative data and contradiction detection."""

from __future__ import annotations

import json
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
    ]
    return state


def create_mock_anthropic_response(
    score: float,
    rationale: str,
    contradictions: list[str] | None = None,
    key_catalyst: str = "Test catalyst",
) -> tuple[AsyncMock, MagicMock]:
    """Create mocked AsyncAnthropic client and message."""
    response_data = {
        "score": score,
        "rationale": rationale,
        "contradictions": contradictions or [],
        "key_catalyst": key_catalyst
    }
    json_text = json.dumps(response_data)

    mock_message = MagicMock()
    mock_message.content = [MagicMock(type="text", text=json_text)]

    mock_stream_enter = AsyncMock()
    mock_stream_enter.get_final_message = AsyncMock(return_value=mock_message)

    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream_enter)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.messages.stream = MagicMock(return_value=mock_stream)

    return mock_client, mock_message


@pytest.mark.asyncio
async def test_sentiment_enhanced_full_flow(sample_state):
    """Test full enhanced sentiment flow with alt data and contradictions."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(
                score=0.1,
                rationale="Mixed signals between earnings and insider activity",
                contradictions=["Insiders selling despite record earnings"],
                key_catalyst="Insider selling"
            )
            mock_client_class.return_value = mock_client

            # Mock alternative data
            with patch("src.agents.sentiment.fetch_alternative_data") as mock_fetch:
                mock_fetch.return_value = {
                    "insider_trading": {
                        "signal": "negative",
                        "recent_activity": "Heavy selling",
                    },
                    "short_interest": {"short_pct": 5.0, "trend": "increasing"},
                    "options_flow": {"signal": "bearish"},
                    "social_media": {"sentiment_score": -0.1},
                    "provenance": {"source": "Quiver"}
                }

                agent = SentimentAgent()
                result = await agent.analyze(sample_state)

                assert result["sentiment_score"] == 0.1
                assert (
                    "Insiders selling despite record earnings"
                    in result["alternative_data"]["contradictions"]
                )
                # 50 base + 40 alt data + 10 Claude = 100
                assert result["sentiment_confidence"] == 100.0


@pytest.mark.asyncio
async def test_sentiment_enhanced_partial_data_confidence(sample_state):
    """Test confidence scoring with partial alternative data."""
    with patch("src.agents.sentiment.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.sentiment.AsyncAnthropic") as mock_client_class:
            mock_client, _ = create_mock_anthropic_response(
                score=0.5, rationale="Test", key_catalyst="Test"
            )
            mock_client_class.return_value = mock_client

            # Mock alternative data with only short interest
            with patch("src.agents.sentiment.fetch_alternative_data") as mock_fetch:
                mock_fetch.return_value = {
                    "insider_trading": None,
                    "short_interest": {"short_pct": 1.0},
                    "options_flow": None,
                    "social_media": None,
                    "provenance": {"source": "Mock"}
                }

                agent = SentimentAgent()
                result = await agent.analyze(sample_state)

                # 50 base + 10 (short) + 10 (Claude) = 70
                assert result["sentiment_confidence"] == 70.0
                assert (
                    "insider_trading"
                    in result["sentiment_confidence_details"]["missing_fields"]
                )
