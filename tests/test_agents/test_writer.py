"""Tests for WriterAgent provenance tracking functionality."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from doxa_shared.types.state import ResearchState, create_initial_state

from src.agents.writer import WriterAgent


@pytest.fixture
def sample_state_with_provenance() -> ResearchState:
    """Create a sample state with provenance metadata for testing."""
    state = create_initial_state("NVDA")
    state["market_data"] = {
        "current_price": 184.32,
        "market_cap": 4_487_641_447_357,
        "fifty_two_week_high": 212.19,
        "fifty_two_week_low": 86.62,
    }
    state["financials"] = {
        "total_revenue": 187_141_996_544,
        "net_income": 99_198_001_152,
        "total_cash": 60_608_000_000,
        "total_debt": 10_821_999_616,
        "operating_cash_flow": 71_000_000_000,
    }
    state["quant_analysis"] = {
        "roe": 0.31,
        "profit_margin": 0.53,
        "asset_turnover": 0.62,
        "equity_multiplier": 2.64,
        "dupont_driver": "High Profitability",
        "altman_z": 4.72,
        "altman_zone": "Safe",
        "pe_ratio": 45.24,
        "signal": "BULLISH",
    }
    state["sentiment_score"] = 0.72
    state["sentiment_rationale"] = "Positive market sentiment."
    state["news"] = [
        {"headline": "NVDA announces new GPU"},
        {"headline": "NVDA beats earnings"},
    ]
    state["valuation_analysis"] = {
        "dcf": {
            "fair_value_per_share": 225.50,
            "current_price": 184.32,
            "upside_downside_pct": 22.3,
            "wacc": 0.10,
            "terminal_value": 5_000_000_000_000,
            "fcf_projections": [70e9, 80e9, 90e9, 100e9, 110e9],
            "sensitivity_table": {
                "wacc_range": [0.08, 0.10, 0.12],
                "growth_range": [0.02, 0.025, 0.03],
                "values": [
                    [280.0, 225.5, 190.0],
                    [300.0, 240.0, 200.0],
                    [325.0, 260.0, 215.0],
                ],
            },
        },
        "comps": {},
        "confidence": 85.0,
    }
    state["regulatory_analysis"] = {
        "risk_factors": [
            "Supply chain concentration risk",
            "Export control regulatory risk",
        ],
        "legal_proceedings": "No material proceedings.",
        "risk_score": "Medium",
        "filing_date": "2025-12-31",
        "confidence": 80.0,
        "cik": "0001045810",
        "filing_url": "",
    }

    # Add provenance metadata
    ts = datetime.now(UTC).isoformat()
    state["provenance_metadata"] = {
        "market_data": {
            "agent": "MarketDataAgent",
            "source": "yfinance.fast_info + yfinance.info",
            "timestamp": ts,
            "confidence": None,
        },
        "quant_analysis": {
            "agent": "QuantAgent",
            "source": "calculated from financials",
            "timestamp": ts,
            "confidence": 95.0,
        },
        "sentiment": {
            "agent": "SentimentAgent",
            "source": "Claude analysis",
            "timestamp": ts,
            "confidence": 100.0,
            "headline_count": 2,
        },
        "valuation": {
            "agent": "ValuationAgent",
            "source": "DCF model",
            "timestamp": ts,
            "confidence": 85.0,
        },
        "regulatory": {
            "agent": "RegulatoryAgent",
            "source": "SEC 10-K",
            "timestamp": ts,
            "filing_date": "2025-12-31",
            "citation": "Risk Factors section",
        },
    }

    return state


@pytest.fixture
def sample_state_without_provenance() -> ResearchState:
    """Create a sample state without provenance metadata (backward compat)."""
    state = create_initial_state("AAPL")
    state["market_data"] = {
        "current_price": 150.0,
        "market_cap": 2_500_000_000_000,
        "fifty_two_week_high": 180.0,
        "fifty_two_week_low": 120.0,
    }
    state["financials"] = {
        "total_revenue": 400_000_000_000,
        "net_income": 100_000_000_000,
        "total_cash": 50_000_000_000,
        "total_debt": 20_000_000_000,
        "operating_cash_flow": 110_000_000_000,
    }
    state["quant_analysis"] = {
        "roe": 0.25,
        "profit_margin": 0.25,
        "asset_turnover": 1.0,
        "equity_multiplier": 1.0,
        "dupont_driver": "Balanced",
        "altman_z": 5.0,
        "altman_zone": "Safe",
        "pe_ratio": 25.0,
        "signal": "NEUTRAL",
    }
    state["sentiment_score"] = 0.0
    state["sentiment_rationale"] = "Neutral sentiment."
    state["news"] = []

    # Delete provenance_metadata to test true backward compat
    del state["provenance_metadata"]  # type: ignore[misc]

    return state


def test_provenance_comments_in_market_overview(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify market data appears in Snapshot block; provenance in Appendix D."""
    agent = WriterAgent()

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test narrative")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_with_provenance)

    report = result["final_report"]

    # Current Price is still in the Snapshot header block
    assert "Current Price" in report
    # Provenance is consolidated in Appendix D (markdown table, not HTML comments)
    assert "### D. Data Provenance" in report
    assert "MarketDataAgent" in report
    # No inline HTML provenance comments in the main body
    assert "<!-- Source: MarketDataAgent, yfinance.fast_info" not in report


def test_provenance_comments_in_financial_summary(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify market data provenance is captured in Appendix D."""
    agent = WriterAgent()

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test narrative")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_with_provenance)

    report = result["final_report"]

    # Provenance is consolidated in Appendix D (no inline HTML comments)
    assert "### D. Data Provenance" in report
    assert "MarketDataAgent" in report
    assert "<!-- Source: MarketDataAgent, yfinance.info" not in report


def test_provenance_comments_in_dupont_analysis(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify DuPont data is in Appendix A; provenance in Appendix D."""
    agent = WriterAgent()

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test narrative")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_with_provenance)

    report = result["final_report"]

    # DuPont table is in Appendix A
    assert "### A. DuPont Analysis" in report
    assert "Return on Equity (ROE)" in report
    # No inline HTML provenance comments on DuPont rows
    assert "<!-- Source: QuantAgent" not in report
    # Provenance flows through Appendix D table
    assert "QuantAgent" in report


def test_provenance_comments_in_sentiment_analysis(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify sentiment data is in Appendix B; provenance in Appendix D."""
    agent = WriterAgent()

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test narrative")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_with_provenance)

    report = result["final_report"]

    # Sentiment section is in Appendix B
    assert "### B. Sentiment Analysis" in report
    assert "+0.72" in report  # sentiment score
    # No inline HTML provenance comments
    assert "<!-- Source: SentimentAgent" not in report
    # Provenance flows through Appendix D table
    assert "SentimentAgent" in report


def test_data_provenance_summary_exists(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify Appendix C (Data Quality) and Appendix D (Provenance) exist."""
    agent = WriterAgent()

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test narrative")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_with_provenance)

    report = result["final_report"]

    # Verify Appendix C: Data Quality & Confidence
    assert "### C. Data Quality & Confidence" in report
    assert "| Agent | Confidence | Status |" in report
    # Verify Appendix D: Data Provenance
    assert "### D. Data Provenance" in report


def test_data_provenance_summary_table_formatting(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify Data Quality & Confidence table has correct formatting."""
    agent = WriterAgent()

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test narrative")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_with_provenance)

    report = result["final_report"]

    # Verify all agents appear in confidence table
    assert "Market Data" in report
    assert "Quant Analysis" in report
    assert "Sentiment" in report

    # Verify confidence scores formatted correctly
    assert "100.0%" in report  # Market Data (has current_price and market_cap)
    assert "85.0%" in report   # Valuation confidence

    # Verify status indicators present
    assert "✅" in report or "Complete" in report


def test_backward_compatibility_without_provenance(
    sample_state_without_provenance: ResearchState,
) -> None:
    """Verify report generates successfully without provenance_metadata."""
    agent = WriterAgent()

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test narrative")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_without_provenance)

    report = result["final_report"]

    # Report should generate with new IC title
    assert "# AAPL — Equity Research | Initiating Coverage" in report

    # Appendix C: Data Quality & Confidence (nested heading)
    assert "### C. Data Quality & Confidence" in report

    # Valuation shows fallback (no valuation data in fixture)
    assert "## Valuation Analysis" in report
    assert "*No valuation data available.*" in report

    # Regulatory section is NOT a standalone section in IC format
    assert "## Regulatory & Risk Assessment" not in report


def test_state_mutation_pattern(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify WriterAgent returns same state object (not a new dict)."""
    agent = WriterAgent()
    original_state = sample_state_with_provenance

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test narrative")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_with_provenance)

    # Should return the same state object, not a new one
    assert result is original_state


def test_provenance_preserved_after_writer(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify provenance_metadata preserved after WriterAgent runs."""
    agent = WriterAgent()

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test narrative")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_with_provenance)

    # Provenance metadata should still be in state
    assert "provenance_metadata" in result
    assert "market_data" in result["provenance_metadata"]
    assert "quant_analysis" in result["provenance_metadata"]
    assert "sentiment" in result["provenance_metadata"]
    assert "valuation" in result["provenance_metadata"]
    assert "regulatory" in result["provenance_metadata"]


def test_narrative_sections_unchanged(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify narrative sections remain unchanged with provenance."""
    agent = WriterAgent()
    narrative_text = "This is the Claude-generated narrative content."

    with patch("src.agents.writer.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(type="text", text=narrative_text)]
            mock_client.return_value.messages.create.return_value = mock_msg

            result = agent.generate_report(sample_state_with_provenance)

        report = result["final_report"]

        # Narrative content should be present and unchanged
        assert narrative_text in report


def test_provenance_comments_in_valuation_section(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify comprehensive Valuation Analysis section."""
    agent = WriterAgent()

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_with_provenance)

    report = result["final_report"]

    # Verify Valuation section exists with new structure
    assert "## Valuation Analysis" in report
    assert "### Price Target Methodology" in report
    assert "### DCF Valuation" in report
    assert "Bull Case" in report
    assert "Base Case" in report
    assert "Bear Case" in report
    assert "**Valuation Confidence:** 85.0%" in report


def test_provenance_comments_in_regulatory_section(
    sample_state_with_provenance: ResearchState,
) -> None:
    """Verify regulatory data flows into Snapshot; no standalone section."""
    agent = WriterAgent()

    with patch("src.agents.writer.anthropic.Anthropic") as mock_client:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="Test")]
        mock_client.return_value.messages.create.return_value = mock_msg

        result = agent.generate_report(sample_state_with_provenance)

    report = result["final_report"]

    # IC format: no standalone Regulatory & Risk Assessment section
    assert "## Regulatory & Risk Assessment" not in report
    # Regulatory risk score appears in Snapshot table
    assert "Regulatory Risk" in report
    assert "Medium" in report
    # RegulatoryAgent provenance is in Appendix D
    assert "RegulatoryAgent" in report


# -------------------------------------------------------------------------
# TASK 9: Comprehensive Edge Case Tests for Institutional-Grade Features
# -------------------------------------------------------------------------


def test_price_target_with_full_data() -> None:
    """Test price target calculation with DCF + comps."""
    state = create_initial_state("AAPL")
    state["market_data"] = {"current_price": 150.0}
    state["quant_analysis"] = {"signal": "BULLISH"}
    state["sentiment_score"] = 0.5
    state["valuation_analysis"] = {
        "dcf": {
            "fair_value_per_share": 200.0,
            "current_price": 150.0,
        },
        "comps": {
            "implied_valuations": {"P/E": 220.0, "EV/EBITDA": 210.0},
        },
    }

    agent = WriterAgent()
    result = agent._calculate_price_target(state)

    assert result["price_target"] > 0
    assert result["bull_target"] > result["base_target"] > result["bear_target"]
    assert sum([result["bull_prob"], result["base_prob"], result["bear_prob"]]) == 1.0
    assert result["upside_pct"] > 0


def test_price_target_without_dcf() -> None:
    """Test graceful fallback when DCF unavailable."""
    state = create_initial_state("AAPL")
    state["market_data"] = {"current_price": 150.0}
    state["quant_analysis"] = {"signal": "NEUTRAL"}
    state["sentiment_score"] = 0.0
    state["valuation_analysis"] = {"dcf": {}, "comps": {}}

    agent = WriterAgent()
    result = agent._calculate_price_target(state)

    assert result["price_target"] == 150.0  # Falls back to current price
    assert result["upside_pct"] == 0.0
    assert "Insufficient data" in result["methodology"]


def test_price_target_dcf_only() -> None:
    """Test price target using 100% DCF when no comps."""
    state = create_initial_state("AAPL")
    state["market_data"] = {"current_price": 150.0}
    state["quant_analysis"] = {"signal": "NEUTRAL"}
    state["sentiment_score"] = 0.0
    state["valuation_analysis"] = {
        "dcf": {
            "fair_value_per_share": 180.0,
            "current_price": 150.0,
        },
        "comps": {},
    }

    agent = WriterAgent()
    result = agent._calculate_price_target(state)

    assert result["base_target"] == 180.0  # 100% DCF
    assert "100% DCF" in result["methodology"]


def test_rating_valuation_based() -> None:
    """Test rating driven by valuation upside."""
    state = create_initial_state("AAPL")
    state["quant_analysis"] = {"signal": "NEUTRAL", "altman_z": 5.0}
    state["sentiment_score"] = 0.0

    agent = WriterAgent()

    # Test Strong Buy (>20% upside)
    rating, explanation = agent._calculate_rating(state, 25.0)
    assert rating == "Strong Buy"
    assert "25.0%" in explanation

    # Test Buy (>10% upside)
    rating, explanation = agent._calculate_rating(state, 15.0)
    assert rating == "Buy"

    # Test Hold (-10% to +10%)
    rating, explanation = agent._calculate_rating(state, 5.0)
    assert rating == "Hold"

    # Test Sell (<-10%)
    rating, explanation = agent._calculate_rating(state, -15.0)
    assert rating == "Sell"


def test_rating_tiebreaker_bullish() -> None:
    """Test tiebreaker upgrades rating when near threshold."""
    state = create_initial_state("AAPL")
    state["quant_analysis"] = {"signal": "BULLISH", "altman_z": 5.0}
    state["sentiment_score"] = 0.5

    agent = WriterAgent()

    # Near Buy/Strong Buy threshold (10%) - should upgrade
    rating, explanation = agent._calculate_rating(state, 10.5)
    assert rating == "Strong Buy"
    assert "upgraded" in explanation or "bullish" in explanation.lower()


def test_rating_tiebreaker_altman_distress() -> None:
    """Test Altman Z-Score downgrade at distress threshold."""
    state = create_initial_state("AAPL")
    state["quant_analysis"] = {"signal": "NEUTRAL", "altman_z": 1.5}
    state["sentiment_score"] = 0.0

    agent = WriterAgent()

    # Near Hold/Sell threshold with distress Z-Score
    rating, explanation = agent._calculate_rating(state, -10.0)
    assert rating == "Sell"
    assert "Altman" in explanation or "distress" in explanation.lower()


def test_valuation_section_with_sensitivity_table() -> None:
    """Test valuation section renders dynamic sensitivity table."""
    state = create_initial_state("AAPL")
    state["valuation_analysis"] = {
        "dcf": {
            "fair_value_per_share": 200.0,
            "current_price": 150.0,
            "upside_downside_pct": 33.3,
            "wacc": 0.08,
            "terminal_value": 1_000_000_000,
            "fcf_projections": [100, 110, 120, 130, 140],
            "sensitivity_table": {
                "wacc_range": [0.07, 0.08, 0.09],
                "growth_range": [0.02, 0.03, 0.04],
                "values": [
                    [220, 200, 180],
                    [240, 220, 200],
                    [260, 240, 220],
                ],
            },
        },
        "comps": {},
        "confidence": 90.0,
    }
    price_target_dict = {
        "price_target": 210.0,
        "bull_target": 230.0,
        "base_target": 200.0,
        "bear_target": 170.0,
        "bull_prob": 0.2,
        "base_prob": 0.6,
        "bear_prob": 0.2,
        "upside_pct": 40.0,
        "methodology": "Test",
    }

    agent = WriterAgent()
    section = agent._build_valuation_section(state, price_target_dict)

    assert "### Sensitivity Analysis" in section
    assert "WACC 7.0%" in section
    assert "WACC 8.0%" in section
    assert "$220" in section
    assert "2.0%" in section  # Growth rate
    assert "### Scenario Analysis" in section
    assert "Bull" in section
    assert "Base" in section
    assert "Bear" in section
    assert "**Valuation Confidence:** 90.0%" in section


def test_valuation_section_with_peer_comps() -> None:
    """Test valuation section renders peer comparison table with target row."""
    state = create_initial_state("AAPL")
    state["quant_analysis"] = {"pe_ratio": 34.0}
    state["valuation_analysis"] = {
        "dcf": {
            "fair_value_per_share": 200.0,
            "current_price": 150.0,
        },
        "comps": {
            "peer_companies": ["MSFT", "GOOGL"],
            "peer_multiples": {
                "MSFT": {"P/E": 30.0, "EV/EBITDA": 20.0, "P/B": 10.0, "P/S": 8.0},
                "GOOGL": {"P/E": 25.0, "EV/EBITDA": 15.0, "P/B": 5.0, "P/S": 6.0},
            },
            "median_multiples": {
                "P/E": 27.5, "EV/EBITDA": 17.5,
                "P/B": 7.5, "P/S": 7.0,
            },
        },
        "confidence": 85.0,
    }
    price_target_dict = {
        "price_target": 210.0,
        "bull_target": 230.0,
        "base_target": 200.0,
        "bear_target": 170.0,
        "bull_prob": 0.2,
        "base_prob": 0.6,
        "bear_prob": 0.2,
        "upside_pct": 40.0,
        "methodology": "Test",
    }

    agent = WriterAgent()
    section = agent._build_valuation_section(state, price_target_dict)

    assert "### Comparable Company Analysis" in section
    assert "MSFT" in section
    assert "GOOGL" in section
    assert "30.0x" in section
    assert "**Median**" in section
    # Target company row and premium/discount
    assert "**AAPL**" in section
    assert "34.0x" in section
    assert "**Premium/(Discount)**" in section


def test_regulatory_section_with_risk_truncation() -> None:
    """Test regulatory section truncates long risk factors."""
    long_risk = " ".join(["word"] * 400)  # 400 words
    state = create_initial_state("AAPL")
    state["regulatory_analysis"] = {
        "risk_factors": [long_risk],
        "legal_proceedings": "None",
        "risk_score": "High",
        "filing_date": "2025-01-01",
        "confidence": 75.0,
        "cik": "0000320193",
        "filing_url": "https://sec.gov/...",
    }

    agent = WriterAgent()
    section = agent._build_regulatory_section(state)

    assert "## Regulatory & Risk Assessment" in section
    assert "[truncated]" in section
    assert "**Regulatory Analysis Confidence:** 75%" in section


def test_confidence_section_low_confidence_warning() -> None:
    """Test low confidence warning appears when overall < 50%."""
    state = create_initial_state("AAPL")
    state["market_data"] = {}  # Empty
    state["quant_analysis"] = {"confidence": 0.0}
    state["valuation_analysis"] = {"confidence": 0.0}
    state["regulatory_analysis"] = {"confidence": 0.0}
    state["sentiment_rationale"] = ""

    agent = WriterAgent()
    section = agent._build_confidence_section(state, narrative_was_generated=False)

    assert "⚠️ **LOW CONFIDENCE WARNING**" in section
    assert "Use with caution" in section


def test_investment_summary_table() -> None:
    """Test IC Snapshot header block rendering."""
    state = create_initial_state("AAPL")
    state["market_data"] = {
        "current_price": 150.0,
        "market_cap": 2_500_000_000_000,
        "fifty_two_week_high": 180.0,
        "fifty_two_week_low": 120.0,
    }
    state["financials"] = {
        "total_cash": 50_000_000_000,
        "total_debt": 20_000_000_000,
    }
    state["quant_analysis"] = {
        "pe_ratio": 25.0,
        "altman_z": 5.0,
        "altman_zone": "Safe",
        "signal": "BULLISH",
    }
    state["regulatory_analysis"] = {"risk_score": "Low"}
    price_target_dict = {
        "price_target": 180.0,
        "upside_pct": 20.0,
    }

    agent = WriterAgent()
    table = agent._build_header_block(state, price_target_dict, "Strong Buy")

    assert "## Snapshot" in table
    assert "$150.00" in table
    assert "$180.00" in table
    assert "+20.0%" in table
    assert "Strong Buy" in table
    assert "BULLISH" in table
    assert "Implied Upside" in table
    assert "Enterprise Value" in table


def test_narrative_prompt_token_limit_safeguard() -> None:
    """Test NARRATIVE_PROMPT token limit triggers trimming."""
    state = create_initial_state("AAPL")
    state["market_data"] = {"current_price": 150.0, "market_cap": 1e12}
    state["financials"] = {
        "total_revenue": 1e12,
        "net_income": 1e11,
        "total_cash": 1e11,
        "total_debt": 1e10,
        "operating_cash_flow": 1e11,
    }
    state["quant_analysis"] = {
        "roe": 0.3,
        "profit_margin": 0.25,
        "asset_turnover": 1.0,
        "equity_multiplier": 2.0,
        "dupont_driver": "Test",
        "altman_z": 5.0,
        "altman_zone": "Safe",
        "pe_ratio": 25.0,
        "signal": "BULLISH",
    }
    state["sentiment_score"] = 0.5
    state["sentiment_rationale"] = "Test " * 500
    state["human_notes"] = "Test " * 500
    state["valuation_analysis"] = {
        "dcf": {"fair_value_per_share": 200.0, "current_price": 150.0}
    }
    state["regulatory_analysis"] = {
        "risk_factors": ["Test " * 100] * 5
    }

    price_target_dict = {
        "price_target": 180.0,
        "upside_pct": 20.0,
        "bull_target": 200.0,
        "base_target": 180.0,
        "bear_target": 160.0,
    }

    from src.agents.writer import _generate_narrative

    with patch("src.agents.writer.ANTHROPIC_API_KEY", "test-key"):
        with patch("src.agents.writer.anthropic.Anthropic") as mock_cls:
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(type="text", text="Narrative")]
            mock_cls.return_value.messages.create.return_value = mock_msg

            result = _generate_narrative(state, "Buy", price_target_dict)

    assert isinstance(result, str)
    assert result == "Narrative"

    # Verify Claude was called (trimming didn't crash)
    call_args = mock_cls.return_value.messages.create.call_args
    prompt_text = call_args[1]["messages"][0]["content"]
    # If token limit was hit, summaries get replaced with short refs
    if len(prompt_text) <= 14000:
        assert "See Valuation" in prompt_text or "DCF Fair Value" in prompt_text


def test_valuation_section_with_5x5_sensitivity_table() -> None:
    """Test valuation section renders 5x5 sensitivity table correctly."""
    state = create_initial_state("AAPL")
    state["valuation_analysis"] = {
        "dcf": {
            "fair_value_per_share": 200.0,
            "current_price": 150.0,
            "upside_downside_pct": 33.3,
            "wacc": 0.08,
            "terminal_value": 1_000_000_000,
            "sensitivity_table": {
                "wacc_range": [0.06, 0.07, 0.08, 0.09, 0.10],
                "growth_range": [0.01, 0.02, 0.03, 0.04, 0.05],
                "values": [
                    [180, 170, 160, 150, 140],
                    [200, 190, 180, 170, 160],
                    [220, 210, 200, 190, 180],
                    [250, 240, 220, 210, 200],
                    [290, 270, 250, 230, 220],
                ],
            },
        },
        "comps": {},
        "confidence": 90.0,
    }
    price_target_dict = {
        "price_target": 210.0,
        "bull_target": 230.0,
        "base_target": 200.0,
        "bear_target": 170.0,
        "bull_prob": 0.2,
        "base_prob": 0.6,
        "bear_prob": 0.2,
        "upside_pct": 40.0,
        "methodology": "Test",
    }

    section = WriterAgent._build_valuation_section(state, price_target_dict)

    assert "WACC 6.0%" in section
    assert "WACC 10.0%" in section
    assert "1.0%" in section
    assert "5.0%" in section
    # Should have 5 data columns + 1 row header
    header_line = [line for line in section.split("\n") if "WACC 6.0%" in line][0]
    assert header_line.count("WACC") == 5


def test_valuation_section_no_peers() -> None:
    """Test valuation section omits comps table when no peers found."""
    state = create_initial_state("AAPL")
    state["quant_analysis"] = {"pe_ratio": 25.0}
    state["valuation_analysis"] = {
        "dcf": {
            "fair_value_per_share": 200.0,
            "current_price": 150.0,
        },
        "comps": {"peer_companies": []},
        "confidence": 70.0,
    }
    price_target_dict = {
        "price_target": 200.0,
        "bull_target": 230.0,
        "base_target": 200.0,
        "bear_target": 170.0,
        "bull_prob": 0.2,
        "base_prob": 0.6,
        "bear_prob": 0.2,
        "upside_pct": 33.3,
        "methodology": "Test",
    }

    section = WriterAgent._build_valuation_section(state, price_target_dict)

    assert "## Valuation Analysis" in section
    assert "### Comparable Company Analysis" not in section


def test_regulatory_section_legal_truncation() -> None:
    """Test regulatory section truncates long legal proceedings (>500 words)."""
    long_legal = " ".join(["proceeding"] * 600)
    state = create_initial_state("AAPL")
    state["regulatory_analysis"] = {
        "risk_factors": ["Short risk"],
        "legal_proceedings": long_legal,
        "risk_score": "High",
        "filing_date": "2025-01-01",
        "confidence": 75.0,
        "cik": "0000320193",
    }

    section = WriterAgent._build_regulatory_section(state)

    assert "[full details in SEC filing]" in section


def test_price_target_negative_dcf_uses_comps() -> None:
    """Test price target falls back to comps when DCF is negative."""
    state = create_initial_state("AAPL")
    state["market_data"] = {
        "current_price": 50.0,
        "market_cap": 10_000_000_000,
    }
    state["quant_analysis"] = {"signal": "BEARISH"}
    state["sentiment_score"] = -0.5
    state["valuation_analysis"] = {
        "dcf": {
            "fair_value_per_share": -10.0,
            "current_price": 50.0,
        },
        "comps": {
            "implied_valuations": {"P/E": 80_000_000_000},
        },
    }

    agent = WriterAgent()
    result = agent._calculate_price_target(state)

    assert result["price_target"] > 0
    assert "comps" in result["methodology"].lower()


def test_confidence_section_weighted_average() -> None:
    """Test confidence section calculates correct weighted average."""
    state = create_initial_state("AAPL")
    state["market_data"] = {"current_price": 150.0, "market_cap": 1e12}
    state["quant_analysis"] = {"confidence": 100.0}
    state["valuation_analysis"] = {"confidence": 80.0}
    state["regulatory_analysis"] = {"confidence": 60.0}
    state["sentiment_rationale"] = "Some rationale"

    section = WriterAgent._build_confidence_section(state, narrative_was_generated=True)

    # Weights: MD 15%, Q 20%, V 25%, R 20%, S 10%, W 10%
    # = 100*0.15 + 100*0.20 + 80*0.25 + 60*0.20 + 100*0.10 + 100*0.10
    # = 15 + 20 + 20 + 12 + 10 + 10 = 87.0
    assert "87.0%" in section
    assert "**Overall**" in section
