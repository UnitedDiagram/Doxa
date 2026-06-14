"""Tests for shared prompt templates."""

from __future__ import annotations

from doxa_shared.prompts.sentiment import SENTIMENT_PROMPT
from doxa_shared.prompts.writer import NARRATIVE_PROMPT


class TestSentimentPrompt:
    """Tests for SENTIMENT_PROMPT template."""

    def test_contains_required_placeholders(self) -> None:
        assert "{ticker}" in SENTIMENT_PROMPT
        assert "{headlines}" in SENTIMENT_PROMPT

    def test_requests_json_response(self) -> None:
        assert "JSON" in SENTIMENT_PROMPT

    def test_is_non_empty_string(self) -> None:
        assert isinstance(SENTIMENT_PROMPT, str)
        assert len(SENTIMENT_PROMPT) > 100


class TestNarrativePrompt:
    """Tests for NARRATIVE_PROMPT template."""

    def test_contains_required_placeholders(self) -> None:
        assert "{ticker}" in NARRATIVE_PROMPT
        assert "{current_price}" in NARRATIVE_PROMPT
        assert "{roe}" in NARRATIVE_PROMPT
        assert "{rating}" in NARRATIVE_PROMPT

    def test_contains_section_headers(self) -> None:
        assert "I. Investment Summary" in NARRATIVE_PROMPT
        assert "II. Company Overview" in NARRATIVE_PROMPT
        assert "V. Financial Analysis" in NARRATIVE_PROMPT
        assert "VII. Investment Risks" in NARRATIVE_PROMPT

    def test_is_non_empty_string(self) -> None:
        assert isinstance(NARRATIVE_PROMPT, str)
        assert len(NARRATIVE_PROMPT) > 100
