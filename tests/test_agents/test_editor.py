"""Tests for EditorAgent implementation and distillation logic."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.editor import EditorAgent
from src.state import create_initial_state


@pytest.fixture
def sample_state():
    """Create sample research state with a long report."""
    state = create_initial_state("AAPL")
    state["final_report"] = "Initial comprehensive report content... " * 100
    return state


def create_mock_anthropic_response(edited_content: str, rationale: str):
    """Create mocked AsyncAnthropic client and message."""
    response_data = {
        "edited_report": edited_content,
        "rationale": rationale
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

    return mock_client


@pytest.mark.asyncio
async def test_editor_agent_initialization():
    """Test that EditorAgent can be initialized."""
    agent = EditorAgent()
    assert agent is not None


@pytest.mark.asyncio
async def test_editor_distillation_flow(sample_state):
    """Test full distillation flow with Claude mock."""
    with patch("src.agents.editor.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.editor.AsyncAnthropic") as mock_client_class:
            mock_client = create_mock_anthropic_response(
                edited_content="High-signal distilled report.",
                rationale="Removed 90% boilerplate."
            )
            mock_client_class.return_value = mock_client

            agent = EditorAgent()
            result = await agent.analyze(sample_state)

            assert result["final_report"] == "High-signal distilled report."
            assert (
                result["provenance_metadata"]["editor"]["rationale"]
                == "Removed 90% boilerplate."
            )


@pytest.mark.asyncio
async def test_editor_preserves_provenance(sample_state):
    """Test that distillation preserves source tags."""
    sample_state["final_report"] = "Important data [Source: SEC 10-K]. Boilerplate."

    with patch("src.agents.editor.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.editor.AsyncAnthropic") as mock_client_class:
            # Mock Claude to actually include the source tag in its response
            mock_client = create_mock_anthropic_response(
                edited_content="High-signal: Important data [Source: SEC 10-K].",
                rationale="Kept sources."
            )
            mock_client_class.return_value = mock_client

            agent = EditorAgent()
            result = await agent.analyze(sample_state)

            assert "[Source: SEC 10-K]" in result["final_report"]


@pytest.mark.asyncio
async def test_editor_handles_empty_report():
    """Test that EditorAgent handles empty input gracefully."""
    state = create_initial_state("TEST")
    state["final_report"] = ""

    agent = EditorAgent()
    result = await agent.analyze(state)

    assert result["final_report"] == ""
    assert "editor" not in result.get("provenance_metadata", {})


@pytest.mark.asyncio
async def test_editor_handles_api_failure(sample_state):
    """Test that EditorAgent handles Claude API failure gracefully."""
    original_report = sample_state["final_report"]

    with patch("src.agents.editor.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.editor.AsyncAnthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client.messages.stream.side_effect = Exception("API error")
            mock_client_class.return_value = mock_client

            agent = EditorAgent()
            result = await agent.analyze(sample_state)

            # Report should remain unchanged
            assert result["final_report"] == original_report
            assert (
                result["provenance_metadata"]["editor"]["rationale"]
                == "API call failed"
            )


