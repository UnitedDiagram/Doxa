"""Tests for RegulatoryAgent and regulatory analysis pipeline."""

from __future__ import annotations

from unittest.mock import Mock, patch

from doxa_shared.types.state import create_initial_state

from src.agents.regulatory import (
    RegulatoryAgent,
    _calculate_confidence,
    _empty_analysis,
    _fallback_risk_extraction,
)

# ---------------------------------------------------------------------------
# Sample data for mocks
# ---------------------------------------------------------------------------

SAMPLE_CIK = "0000320193"

SAMPLE_FILINGS = [
    {
        "accession_number": "0000320193-24-000123",
        "filing_date": "2024-11-01",
        "primary_document": "aapl-20240928.htm",
        "form": "10-K",
    },
]

SAMPLE_SECTIONS = {
    "risk_factors": (
        "The Company faces significant regulatory risks related to "
        "data privacy laws in multiple jurisdictions.\n\n"
        "Competition in the consumer electronics market continues "
        "to intensify, potentially impacting margins.\n\n"
        "Supply chain disruptions could materially affect the "
        "Company's ability to meet product demand."
    ),
    "legal_proceedings": (
        "The Company is involved in patent infringement litigation "
        "with several competitors in the mobile device space."
    ),
    "md_and_a": (
        "Revenue increased 8% year-over-year driven by strong "
        "services segment performance."
    ),
}

SAMPLE_CLAUDE_RESPONSE = {
    "risk_factors": [
        "Data privacy regulations (Item 1A, para 1): GDPR and similar "
        "laws could restrict data monetization, impacting services revenue.",
        "Competitive pressure (Item 1A, para 2): Intensifying competition "
        "may compress margins in consumer electronics.",
        "Supply chain risk (Item 1A, para 3): Disruptions could reduce "
        "product availability and increase costs.",
    ],
    "legal_proceedings": (
        "Active patent infringement litigation with several mobile "
        "device competitors. Outcome remains uncertain."
    ),
    "risk_score": "Medium",
}


# ---------------------------------------------------------------------------
# Helper to create a mock Claude streaming response
# ---------------------------------------------------------------------------


def _make_claude_stream_mock(response_dict: dict) -> Mock:  # type: ignore[type-arg]
    """Build a mock that simulates anthropic streaming."""
    text_block = Mock()
    text_block.type = "text"
    text_block.text = __import__("json").dumps(response_dict)

    message = Mock()
    message.content = [text_block]

    stream_ctx = Mock()
    stream_ctx.__enter__ = Mock(return_value=stream_ctx)
    stream_ctx.__exit__ = Mock(return_value=False)
    stream_ctx.get_final_message.return_value = message

    return stream_ctx


# ---------------------------------------------------------------------------
# _empty_analysis tests
# ---------------------------------------------------------------------------


class TestEmptyAnalysis:
    """Tests for _empty_analysis helper."""

    def test_returns_default_structure(self) -> None:
        result = _empty_analysis()
        assert result["risk_factors"] == []
        assert result["risk_score"] == "Low"
        assert result["confidence"] == 0.0
        assert result["cik"] == ""
        assert result["filing_url"] == ""


# ---------------------------------------------------------------------------
# _calculate_confidence tests
# ---------------------------------------------------------------------------


class TestCalculateConfidence:
    """Tests for confidence scoring (updated for Story 2.6 institutional depth)."""

    def test_full_data_gives_high_score(self) -> None:
        score, details = _calculate_confidence(
            cik_found=True,
            filing_found=True,
            sections=SAMPLE_SECTIONS,
            claude_succeeded=True,
            multi_year_count=3,
            claude_evolution_succeeded=True,
            insider_data_complete=True,
        )
        # 15+15+15+5+5+5 = 60 (base) + 20 (3yr) + 10 (insider) + 10 (evolution) = 100
        assert score == 100.0
        assert details["filings_analyzed"] == 3

    def test_no_data_gives_zero(self) -> None:
        score, details = _calculate_confidence(
            cik_found=False,
            filing_found=False,
            sections={"risk_factors": "", "legal_proceedings": "", "md_and_a": ""},
            claude_succeeded=False,
        )
        assert score == 0.0

    def test_cik_only_gives_15(self) -> None:
        score, _ = _calculate_confidence(
            cik_found=True,
            filing_found=False,
            sections={"risk_factors": "", "legal_proceedings": "", "md_and_a": ""},
            claude_succeeded=False,
        )
        # 15 (CIK) only
        assert score == 15.0

    def test_partial_sections_gives_45(self) -> None:
        score, _ = _calculate_confidence(
            cik_found=True,
            filing_found=True,
            sections={
                "risk_factors": "Some risks",
                "legal_proceedings": "",
                "md_and_a": "",
            },
            claude_succeeded=False,
        )
        # 15 (CIK) + 15 (filing) + 15 (risk factors) = 45
        assert score == 45.0

    def test_all_sections_no_claude_gives_55(self) -> None:
        score, _ = _calculate_confidence(
            cik_found=True,
            filing_found=True,
            sections=SAMPLE_SECTIONS,
            claude_succeeded=False,
        )
        # 15 + 15 + 15 + 5 + 5 = 55
        assert score == 55.0


# ---------------------------------------------------------------------------
# _fallback_risk_extraction tests
# ---------------------------------------------------------------------------


class TestFallbackRiskExtraction:
    """Tests for fallback risk extraction without Claude."""

    def test_extracts_first_three_paragraphs(self) -> None:
        text = (
            "First risk paragraph that is long enough to pass the filter. "
            "It contains details about regulatory compliance issues.\n\n"
            "Second risk paragraph about competitive dynamics. "
            "The market is becoming more competitive each year.\n\n"
            "Third risk paragraph about supply chain concerns. "
            "Disruptions may impact the company significantly.\n\n"
            "Fourth paragraph should not be included in results."
        )
        result = _fallback_risk_extraction(text)
        assert len(result) == 3

    def test_returns_empty_for_empty_text(self) -> None:
        assert _fallback_risk_extraction("") == []

    def test_skips_short_paragraphs(self) -> None:
        text = "Short.\n\nAlso short.\n\n" + "A" * 60
        result = _fallback_risk_extraction(text)
        assert len(result) == 1

    def test_truncates_long_paragraphs(self) -> None:
        text = "A" * 1000
        result = _fallback_risk_extraction(text)
        assert len(result) == 1
        assert len(result[0]) == 500


# ---------------------------------------------------------------------------
# RegulatoryAgent full pipeline tests
# ---------------------------------------------------------------------------


class TestRegulatoryAgentHappyPath:
    """Tests for the full happy path: CIK → filing → sections → Claude."""

    @patch("src.agents.regulatory.ANTHROPIC_API_KEY", "test-key")
    @patch("src.agents.regulatory.fetch_multi_year_10k")
    @patch("src.agents.regulatory.anthropic.Anthropic")
    @patch("src.agents.regulatory.extract_10k_sections")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_full_pipeline(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_sections: Mock,
        mock_anthropic: Mock,
        mock_multi_year: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "filing text"
        mock_sections.return_value = SAMPLE_SECTIONS
        mock_multi_year.return_value = SAMPLE_MULTI_YEAR_FILINGS

        # Mock Claude client for both calls
        stream_mock = _make_claude_stream_mock(SAMPLE_CLAUDE_RESPONSE)
        evolution_mock = _make_claude_stream_mock(SAMPLE_RISK_EVOLUTION)
        mock_client = Mock()
        mock_client.messages.stream.side_effect = [stream_mock, evolution_mock]
        mock_anthropic.return_value = mock_client

        state = create_initial_state("AAPL")
        # Add insider data for full score
        state["financials"] = {
            "heldPercentInsiders": 0.15,
            "heldPercentInstitutions": 0.65,
        }
        agent = RegulatoryAgent()
        result = agent.analyze(state)

        reg = result["regulatory_analysis"]
        assert reg["cik"] == SAMPLE_CIK
        assert reg["filing_date"] == "2024-11-01"
        assert reg["risk_score"] == "Medium"
        assert len(reg["risk_factors"]) == 3
        # 60 (base) + 20 (3yr) + 10 (insider) + 10 (evolution) = 100
        assert reg["confidence"] == 100.0
        assert "filing_url" in reg
        assert len(result["errors"]) == 0

    @patch("src.agents.regulatory.ANTHROPIC_API_KEY", "test-key")
    @patch("src.agents.regulatory.anthropic.Anthropic")
    @patch("src.agents.regulatory.extract_10k_sections")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_returns_same_state_object(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_sections: Mock,
        mock_anthropic: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "text"
        mock_sections.return_value = SAMPLE_SECTIONS

        stream_mock = _make_claude_stream_mock(SAMPLE_CLAUDE_RESPONSE)
        mock_client = Mock()
        mock_client.messages.stream.return_value = stream_mock
        mock_anthropic.return_value = mock_client

        state = create_initial_state("AAPL")
        agent = RegulatoryAgent()
        result = agent.analyze(state)
        assert result is state


class TestRegulatoryAgentMissingCik:
    """Tests when CIK cannot be resolved."""

    @patch("src.agents.regulatory.resolve_cik")
    def test_missing_cik_sets_zero_confidence(
        self, mock_cik: Mock,
    ) -> None:
        mock_cik.return_value = None

        state = create_initial_state("ZZZZZ")
        result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]
        assert reg["confidence"] == 0.0
        assert len(result["errors"]) > 0

    @patch("src.agents.regulatory.resolve_cik")
    def test_cik_network_error_handled(
        self, mock_cik: Mock,
    ) -> None:
        mock_cik.side_effect = Exception("Connection timeout")

        state = create_initial_state("AAPL")
        result = RegulatoryAgent().analyze(state)

        assert result["regulatory_analysis"]["confidence"] == 0.0
        assert any("CIK" in e for e in result["errors"])


class TestRegulatoryAgentMissingFiling:
    """Tests when no 10-K filing is found."""

    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_no_filing_sets_low_confidence(
        self, mock_cik: Mock, mock_filings: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = []

        state = create_initial_state("AAPL")
        result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]
        assert reg["cik"] == SAMPLE_CIK
        assert reg["confidence"] == 0.0
        assert len(result["errors"]) > 0


class TestRegulatoryAgentNoApiKey:
    """Tests when ANTHROPIC_API_KEY is not set."""

    @patch("src.agents.regulatory.ANTHROPIC_API_KEY", "")
    @patch("src.agents.regulatory.fetch_multi_year_10k")
    @patch("src.agents.regulatory.extract_10k_sections")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_uses_fallback_without_api_key(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_sections: Mock,
        mock_multi_year: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "filing text"
        mock_sections.return_value = SAMPLE_SECTIONS
        mock_multi_year.return_value = SAMPLE_MULTI_YEAR_FILINGS

        state = create_initial_state("AAPL")
        result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]
        # Without Claude: 15+15+15+5+5 = 55 (base) + 20 (3yr) = 75 (no Claude bonus)
        assert reg["confidence"] == 75.0
        # Should still have risk factors from fallback
        assert isinstance(reg["risk_factors"], list)
        assert any("API_KEY" in e for e in result["errors"])


class TestRegulatoryAgentClaudeFailure:
    """Tests when Claude API call fails."""

    @patch("src.agents.regulatory.ANTHROPIC_API_KEY", "test-key")
    @patch("src.agents.regulatory.fetch_multi_year_10k")
    @patch("src.agents.regulatory.anthropic.Anthropic")
    @patch("src.agents.regulatory.extract_10k_sections")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_falls_back_on_claude_error(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_sections: Mock,
        mock_anthropic: Mock,
        mock_multi_year: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "filing text"
        mock_sections.return_value = SAMPLE_SECTIONS
        mock_multi_year.return_value = SAMPLE_MULTI_YEAR_FILINGS

        mock_client = Mock()
        mock_client.messages.stream.side_effect = Exception("API error")
        mock_anthropic.return_value = mock_client

        state = create_initial_state("AAPL")
        result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]
        # Without Claude success: 55 (base) + 20 (3yr) = 75
        assert reg["confidence"] == 75.0
        assert isinstance(reg["risk_factors"], list)


class TestRegulatoryAgentPartialSections:
    """Tests when only some filing sections are extracted."""

    @patch("src.agents.regulatory.ANTHROPIC_API_KEY", "")
    @patch("src.agents.regulatory.fetch_multi_year_10k")
    @patch("src.agents.regulatory.extract_10k_sections")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_partial_sections_reduce_confidence(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_sections: Mock,
        mock_multi_year: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "filing text"
        mock_sections.return_value = {
            "risk_factors": "Some risk content here.",
            "legal_proceedings": "",
            "md_and_a": "",
        }
        mock_multi_year.return_value = SAMPLE_MULTI_YEAR_FILINGS

        state = create_initial_state("AAPL")
        result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]
        # 15+15+15 = 45 (base) + 20 (3yr) = 65
        assert reg["confidence"] == 65.0


class TestRegulatoryAgentErrorAccumulation:
    """Tests that errors are appended, never raised."""

    @patch("src.agents.regulatory.resolve_cik")
    def test_never_raises_exception(
        self, mock_cik: Mock,
    ) -> None:
        mock_cik.side_effect = RuntimeError("Unexpected error")

        state = create_initial_state("AAPL")
        # Should not raise
        result = RegulatoryAgent().analyze(state)
        assert len(result["errors"]) > 0

    def test_handles_empty_ticker(self) -> None:
        state = create_initial_state("TEST")
        state["ticker"] = ""  # type: ignore[typeddict-item]
        result = RegulatoryAgent().analyze(state)
        assert len(result["errors"]) > 0
        assert result["regulatory_analysis"]["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Enhanced RegulatoryAgent tests (Story 2.6)
# ---------------------------------------------------------------------------


SAMPLE_MULTI_YEAR_FILINGS = [
    {
        "year": 2024,
        "filing_date": "2024-11-01",
        "risk_factors_text": "New cybersecurity risks emerged in 2024.",
    },
    {
        "year": 2023,
        "filing_date": "2023-11-03",
        "risk_factors_text": "Supply chain risks continue to impact operations.",
    },
    {
        "year": 2022,
        "filing_date": "2022-10-28",
        "risk_factors_text": "Regulatory risks in European markets.",
    },
]

SAMPLE_RISK_EVOLUTION = {
    "new_risks": ["Cybersecurity threats from state actors"],
    "removed_risks": ["European regulatory uncertainty"],
    "escalated_risks": ["Supply chain disruptions"],
    "trend": "increasing",
    "interpretation": "Risk profile increasing with new cyber threats.",
}


class TestEnhancedConfidenceCalculation:
    """Tests for enhanced confidence scoring with institutional depth."""

    def test_confidence_with_3_year_multi_year_data(self) -> None:
        confidence, details = _calculate_confidence(
            cik_found=True,
            filing_found=True,
            sections={"risk_factors": "text", "legal_proceedings": "", "md_and_a": ""},
            claude_succeeded=True,
            multi_year_count=3,
            claude_evolution_succeeded=True,
            insider_data_complete=True,
        )

        # 15+15+15+5 = 50 (base) + 20 (3 years) + 10 (insider) + 10 (evolution) = 90
        assert confidence == 90.0
        assert details["filings_analyzed"] == 3
        assert details["insider_data_available"] is True
        assert details["claude_evolution_success"] is True

    def test_confidence_with_2_year_multi_year_data(self) -> None:
        confidence, details = _calculate_confidence(
            cik_found=True,
            filing_found=True,
            sections={"risk_factors": "text", "legal_proceedings": "", "md_and_a": ""},
            claude_succeeded=True,
            multi_year_count=2,
            claude_evolution_succeeded=False,
            insider_data_complete=False,
        )

        # 50 (base) + 10 (2 years) = 60
        assert confidence == 60.0
        assert details["filings_analyzed"] == 2
        assert details["insider_data_available"] is False

    def test_confidence_with_1_year_only(self) -> None:
        confidence, details = _calculate_confidence(
            cik_found=True,
            filing_found=True,
            sections={"risk_factors": "text", "legal_proceedings": "", "md_and_a": ""},
            claude_succeeded=True,
            multi_year_count=1,
            claude_evolution_succeeded=False,
            insider_data_complete=False,
        )

        # 50 (base) + 0 (1 year) = 50
        assert confidence == 50.0
        assert details["filings_analyzed"] == 1


class TestMultiYearAnalysis:
    """Tests for multi-year 10-K analysis."""

    @patch("src.agents.regulatory.ANTHROPIC_API_KEY", "test-key")
    @patch("src.agents.regulatory.fetch_multi_year_10k")
    @patch("src.agents.regulatory.anthropic.Anthropic")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_multi_year_analysis_3_years(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_anthropic: Mock,
        mock_multi_year: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "Sample 10-K text"
        mock_multi_year.return_value = SAMPLE_MULTI_YEAR_FILINGS

        # Mock Claude responses
        claude_stream = _make_claude_stream_mock(SAMPLE_CLAUDE_RESPONSE)
        evolution_stream = _make_claude_stream_mock(SAMPLE_RISK_EVOLUTION)

        mock_client = Mock()
        mock_client.messages.stream.side_effect = [claude_stream, evolution_stream]
        mock_anthropic.return_value = mock_client

        with patch(
            "src.agents.regulatory.extract_10k_sections",
            return_value=SAMPLE_SECTIONS,
        ):
            state = create_initial_state("AAPL")
            result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]

        # Verify multi-year filings stored
        assert len(reg["multi_year_filings"]) == 3
        assert reg["multi_year_filings"][0]["year"] == 2024

        # Verify risk evolution analysis stored
        assert reg["risk_evolution"] is not None
        assert "new_risks" in reg["risk_evolution"]
        assert reg["risk_evolution"]["trend"] == "increasing"

        # Verify confidence includes multi-year bonus
        assert reg["confidence"] >= 70.0  # Base + multi-year + evolution

    @patch("src.agents.regulatory.fetch_multi_year_10k")
    @patch("src.agents.regulatory.anthropic.Anthropic")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_multi_year_fallback_to_1_year(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_anthropic: Mock,
        mock_multi_year: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "Sample 10-K text"
        mock_multi_year.return_value = []  # No multi-year data available

        claude_stream = _make_claude_stream_mock(SAMPLE_CLAUDE_RESPONSE)
        mock_client = Mock()
        mock_client.messages.stream.return_value = claude_stream
        mock_anthropic.return_value = mock_client

        with patch(
            "src.agents.regulatory.extract_10k_sections",
            return_value=SAMPLE_SECTIONS,
        ):
            state = create_initial_state("AAPL")
            result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]

        # Should work with single-year analysis
        assert len(reg["multi_year_filings"]) == 0
        assert reg["risk_evolution"] is None
        assert len(reg["risk_factors"]) > 0  # Still has single-year risks


class TestInsiderOwnershipIntegration:
    """Tests for insider ownership integration."""

    @patch("src.agents.regulatory.anthropic.Anthropic")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_insider_ownership_complete_data(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_anthropic: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "Sample 10-K text"

        claude_stream = _make_claude_stream_mock(SAMPLE_CLAUDE_RESPONSE)
        mock_client = Mock()
        mock_client.messages.stream.return_value = claude_stream
        mock_anthropic.return_value = mock_client

        with patch(
            "src.agents.regulatory.extract_10k_sections",
            return_value=SAMPLE_SECTIONS,
        ):
            state = create_initial_state("AAPL")
            # Add financials with insider data
            state["financials"] = {
                "heldPercentInsiders": 0.152,
                "heldPercentInstitutions": 0.645,
            }
            state["current_price"] = 175.43

            result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]

        # Verify management signals populated
        assert "management_signals" in reg
        assert reg["management_signals"]["insider_ownership_pct"] == 15.2
        assert reg["management_signals"]["institutional_ownership_pct"] == 64.5
        assert reg["management_signals"]["ceo_ownership_value"] is None  # MVP
        assert reg["management_signals"]["signal"] == "neutral"  # MVP

    @patch("src.agents.regulatory.anthropic.Anthropic")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_insider_ownership_missing_data(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_anthropic: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "Sample 10-K text"

        claude_stream = _make_claude_stream_mock(SAMPLE_CLAUDE_RESPONSE)
        mock_client = Mock()
        mock_client.messages.stream.return_value = claude_stream
        mock_anthropic.return_value = mock_client

        with patch(
            "src.agents.regulatory.extract_10k_sections",
            return_value=SAMPLE_SECTIONS,
        ):
            state = create_initial_state("AAPL")
            # No financials - insider data missing
            state["financials"] = {}

            result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]

        # Should handle missing data gracefully
        assert reg["management_signals"]["insider_ownership_pct"] is None
        assert reg["management_signals"]["institutional_ownership_pct"] is None
        assert reg["management_signals"]["ceo_ownership_value"] is None
        assert reg["management_signals"]["signal"] == "neutral"

    @patch("src.agents.regulatory.anthropic.Anthropic")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_insider_ownership_empty_market_data(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_anthropic: Mock,
    ) -> None:
        """Test insider data extraction when market_data is empty dict."""
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "Sample 10-K text"

        claude_stream = _make_claude_stream_mock(SAMPLE_CLAUDE_RESPONSE)
        mock_client = Mock()
        mock_client.messages.stream.return_value = claude_stream
        mock_anthropic.return_value = mock_client

        with patch(
            "src.agents.regulatory.extract_10k_sections",
            return_value=SAMPLE_SECTIONS,
        ):
            state = create_initial_state("AAPL")
            # Empty market_data dict (MarketDataAgent failed or not run yet)
            state["market_data"] = {}
            state["financials"] = {}

            result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]

        # Verify current_price defaults to 0.0 when missing - should handle gracefully
        assert reg["management_signals"]["signal"] == "neutral"
        assert reg["management_signals"]["ceo_ownership_value"] is None


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing fields."""

    @patch("src.agents.regulatory.anthropic.Anthropic")
    @patch("src.agents.regulatory.fetch_filing_text")
    @patch("src.agents.regulatory.fetch_recent_filings")
    @patch("src.agents.regulatory.resolve_cik")
    def test_existing_fields_preserved(
        self,
        mock_cik: Mock,
        mock_filings: Mock,
        mock_text: Mock,
        mock_anthropic: Mock,
    ) -> None:
        mock_cik.return_value = SAMPLE_CIK
        mock_filings.return_value = SAMPLE_FILINGS
        mock_text.return_value = "Sample 10-K text"

        claude_stream = _make_claude_stream_mock(SAMPLE_CLAUDE_RESPONSE)
        mock_client = Mock()
        mock_client.messages.stream.return_value = claude_stream
        mock_anthropic.return_value = mock_client

        with patch(
            "src.agents.regulatory.extract_10k_sections",
            return_value=SAMPLE_SECTIONS,
        ):
            state = create_initial_state("AAPL")
            result = RegulatoryAgent().analyze(state)

        reg = result["regulatory_analysis"]

        # Verify all 7 existing fields still present
        assert "risk_factors" in reg
        assert "legal_proceedings" in reg
        assert "risk_score" in reg
        assert "filing_date" in reg
        assert "confidence" in reg
        assert "cik" in reg
        assert "filing_url" in reg

        # Verify new fields added
        assert "multi_year_filings" in reg
        assert "risk_evolution" in reg
        assert "management_signals" in reg
        assert "confidence_details" in reg
