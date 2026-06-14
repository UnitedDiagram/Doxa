"""Tests that RegulatoryAgent posts insights to insights_board."""

from __future__ import annotations

from unittest.mock import patch

from doxa_shared.types.state import create_initial_state

from src.agents.regulatory import RegulatoryAgent


def _make_regulatory_state(ticker: str = "NVDA", **reg_overrides: object) -> dict:
    """Create a state pre-populated with regulatory_analysis for testing insights."""
    state = create_initial_state(ticker)
    base_reg = {
        "risk_factors": ["Competition risk", "Supply chain risk"],
        "legal_proceedings": (
            "The company is involved in material "
            "litigation regarding patents."
        ),
        "risk_score": "Medium",
        "filing_date": "2024-01-15",
        "confidence": 80.0,
        "cik": "0001045810",
        "filing_url": "https://sec.gov/test",
        "multi_year_filings": [],
        "risk_evolution": {
            "new_risks": ["New AI regulation risk"],
            "removed_risks": [],
            "escalated_risks": [],
            "trend": "stable",
            "interpretation": "New regulatory pressures emerging",
        },
        "management_signals": {},
        "confidence_details": {},
    }
    base_reg.update(reg_overrides)
    state["regulatory_analysis"] = base_reg
    return state


def _patch_regulatory_io() -> tuple:
    """Return a stack of patches to bypass all IO in RegulatoryAgent.analyze()."""
    return (
        patch("src.agents.regulatory._resolve_cik_safe", return_value="0001045810"),
        patch("src.agents.regulatory._fetch_filings_safe", return_value=[{
            "accession_number": "0001045810-24-000001",
            "filing_date": "2024-01-15",
            "primary_document": "nvda-20240127.htm",
        }]),
        patch(
            "src.agents.regulatory._fetch_filing_text_safe",
            return_value="<risk>...</risk>",
        ),
        patch("src.agents.regulatory.extract_10k_sections", return_value={
            "risk_factors": "Competition risk...",
            "legal_proceedings": "Material litigation...",
            "md_and_a": "Revenue grew...",
        }),
        patch("src.agents.regulatory._call_claude_analysis", return_value={
            "risk_factors": ["Competition risk"],
            "legal_proceedings": (
                "Material litigation regarding patents"
                " that could affect business."
            ),
            "risk_score": "Medium",
        }),
        patch("src.agents.regulatory._fetch_multi_year_filings", return_value=[]),
        patch("src.agents.regulatory._fetch_insider_data", return_value={
            "insider_ownership_pct": 5.0,
            "institutional_ownership_pct": 65.0,
            "ceo_ownership_value": None,
            "signal": "neutral",
        }),
    )


def test_new_material_risks_post_regulatory_insights() -> None:
    """_post_regulatory_insights posts insights for new risks in risk_evolution."""
    from src.agents.regulatory import _post_regulatory_insights

    state = create_initial_state("NVDA")
    state["regulatory_analysis"] = {
        "risk_factors": ["Competition risk"],
        "legal_proceedings": "No material legal proceedings disclosed.",
        "risk_score": "Medium",
        "filing_date": "2024-01-15",
        "confidence": 80.0,
        "cik": "0001045810",
        "filing_url": "https://sec.gov/test",
        "multi_year_filings": [],
        "risk_evolution": {
            "new_risks": ["AI regulation risk", "Chip export controls"],
            "removed_risks": [],
            "escalated_risks": [],
            "trend": "escalating",
            "interpretation": "New regulatory pressures",
        },
        "management_signals": {},
        "confidence_details": {},
    }

    _post_regulatory_insights(state)

    reg_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "regulatory"
    ]
    assert len(reg_insights) >= 1
    assert all(ins["agent"] == "RegulatoryAgent" for ins in reg_insights)
    signals = " ".join(ins["signal"] for ins in reg_insights)
    assert "new material risk" in signals.lower()


def test_litigation_posts_insight_when_substantial() -> None:
    """Substantial litigation posts a litigation insight."""
    from src.agents.regulatory import _post_regulatory_insights

    state = create_initial_state("NVDA")
    state["regulatory_analysis"] = {
        "risk_factors": ["Competition risk"],
        "legal_proceedings": (
            "The Company is a defendant in multiple patent infringement suits "
            "which, if adversely determined, could have a material adverse effect "
            "on business operations and financial condition."
        ),
        "risk_score": "Medium",
        "filing_date": "2024-01-15",
        "confidence": 80.0,
        "cik": "0001045810",
        "filing_url": "https://sec.gov/test",
        "multi_year_filings": [],
        "risk_evolution": None,
        "management_signals": {},
        "confidence_details": {},
    }

    _post_regulatory_insights(state)

    lit_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "litigation"
    ]
    assert len(lit_insights) >= 1
    assert lit_insights[0]["agent"] == "RegulatoryAgent"


def test_low_confidence_posts_disclosure_quality_insight() -> None:
    """Low regulatory confidence (<60) posts a disclosure_quality insight."""
    patches = list(_patch_regulatory_io())
    p_claude = patch(
        "src.agents.regulatory._call_claude_analysis",
        return_value=None,
    )
    p_evolution = patch(
        "src.agents.regulatory._call_claude_risk_evolution",
        return_value=None,
    )
    p_conf = patch(
        "src.agents.regulatory._calculate_confidence",
        return_value=(35.0, {}),
    )
    with (
        patches[0], patches[1], patches[2],
        patches[3], p_claude,
        patches[5], patches[6],
        p_evolution, p_conf,
    ):
        state = create_initial_state("NVDA")
        RegulatoryAgent().analyze(state)

    dq_insights = [
        ins for ins in state["insights_board"]
        if ins.get("category") == "disclosure_quality"
    ]
    assert len(dq_insights) >= 1
    assert "35" in dq_insights[0]["signal"]


def test_state_identity_preserved() -> None:
    """analyze() returns the same state object."""
    patches = _patch_regulatory_io()
    p_evolution = patch(
        "src.agents.regulatory._call_claude_risk_evolution",
        return_value=None,
    )
    with (
        patches[0], patches[1], patches[2],
        patches[3], patches[4], patches[5],
        patches[6], p_evolution,
    ):
        state = create_initial_state("NVDA")
        result = RegulatoryAgent().analyze(state)

    assert result is state
