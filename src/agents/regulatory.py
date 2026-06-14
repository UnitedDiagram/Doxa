"""RegulatoryAgent — SEC EDGAR filing analysis for regulatory risks.

This agent resolves a company's SEC CIK, fetches the most recent 10-K
filing, extracts key disclosure sections, and uses Claude AI to identify
material regulatory risks with citations.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import anthropic
from doxa_shared.prompts.regulatory import (
    REGULATORY_RISK_PROMPT,
    RISK_EVOLUTION_PROMPT,
)
from doxa_shared.types.state import ResearchState
from doxa_shared.utils.edgar import (
    extract_10k_sections,
    fetch_filing_text,
    fetch_multi_year_10k,
    fetch_recent_filings,
    resolve_cik,
)
from doxa_shared.utils.insights import post_insight
from doxa_shared.utils.valuation import (
    calculate_ceo_ownership_value,
    fetch_insider_ownership,
    interpret_insider_signal,
)

from src.config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

# Maximum characters per section to send to Claude (10-K filings are huge).
_MAX_SECTION_CHARS = 8000


class RegulatoryAgent:
    """Agent for analyzing SEC EDGAR filings for regulatory risks.

    Resolves a company's CIK, fetches 10-K filings, extracts risk
    sections, and uses Claude to produce a regulatory risk assessment.
    Falls back to basic extraction when Claude is unavailable.
    """

    def analyze(self, state: ResearchState) -> ResearchState:
        """Execute regulatory analysis and update state.

        Fetches SEC EDGAR filings, extracts sections, and runs Claude
        analysis. Never raises exceptions — errors are appended to
        state['errors'].

        Args:
            state: ResearchState dict containing 'ticker' key.

        Returns:
            Updated ResearchState with 'regulatory_analysis' populated.
        """
        ticker = state.get("ticker", "")
        if not ticker:
            error_msg = "RegulatoryAgent: No ticker provided in state"
            logger.warning(error_msg)
            state["errors"].append(error_msg)
            state["regulatory_analysis"] = _empty_analysis()
            return state

        logger.info("RegulatoryAgent starting for %s", ticker)

        # Step 1: Resolve ticker to CIK
        cik = _resolve_cik_safe(ticker, state)
        if not cik:
            state["regulatory_analysis"] = _empty_analysis()
            return state

        # Step 2: Fetch recent 10-K filings
        filings = _fetch_filings_safe(cik, ticker, state)
        if not filings:
            analysis = _empty_analysis()
            analysis["cik"] = cik
            state["regulatory_analysis"] = analysis
            return state

        latest = filings[0]
        accession = latest["accession_number"]
        filing_date = latest["filing_date"]
        primary_doc = latest["primary_document"]

        # Build filing URL for reference
        cik_int = str(int(cik))
        accession_nodashes = accession.replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{accession_nodashes}/{primary_doc}"
        )

        # Step 3: Fetch filing text
        filing_text = _fetch_filing_text_safe(
            cik, accession, primary_doc, ticker, state,
        )

        # Step 4: Extract sections
        sections: dict[str, str] = {
            "risk_factors": "",
            "legal_proceedings": "",
            "md_and_a": "",
        }
        if filing_text:
            try:
                sections = extract_10k_sections(filing_text)
            except Exception as exc:
                error_msg = (
                    f"RegulatoryAgent: Section extraction failed "
                    f"for {ticker}: {exc}"
                )
                logger.warning(error_msg)
                state["errors"].append(error_msg)

        # Step 5: Claude analysis or fallback
        claude_succeeded = False
        risk_factors: list[str] = []
        legal_proceedings = "No material legal proceedings disclosed."
        risk_score = "Low"

        if ANTHROPIC_API_KEY and any(sections.values()):
            result = _call_claude_analysis(ticker, sections)
            if result:
                claude_succeeded = True
                risk_factors = result.get("risk_factors", [])
                legal_proceedings = result.get(
                    "legal_proceedings",
                    "No material legal proceedings disclosed.",
                )
                risk_score = result.get("risk_score", "Low")
        elif not ANTHROPIC_API_KEY:
            msg = (
                "RegulatoryAgent: ANTHROPIC_API_KEY not set; "
                "using fallback extraction"
            )
            logger.warning(msg)
            state["errors"].append(msg)

        # Fallback: extract first few paragraphs from risk factors
        if not claude_succeeded and not risk_factors:
            risk_factors = _fallback_risk_extraction(
                sections.get("risk_factors", ""),
            )

        # Step 6: Multi-year analysis
        multi_year_filings: list[dict[str, Any]] = []
        risk_evolution: dict[str, Any] | None = None
        claude_evolution_succeeded = False

        if cik:
            multi_year_filings = _fetch_multi_year_filings(cik, ticker, state)

            # If we have 2+ years, run Claude risk evolution analysis
            if len(multi_year_filings) >= 2 and ANTHROPIC_API_KEY:
                risk_evolution = _call_claude_risk_evolution(
                    ticker,
                    multi_year_filings,
                )
                if risk_evolution:
                    claude_evolution_succeeded = True

        # Step 7: Insider ownership analysis
        management_signals: dict[str, Any] = {}
        try:
            management_signals = _fetch_insider_data(state)
        except Exception as exc:
            error_msg = (
                f"RegulatoryAgent: Insider data fetch failed "
                f"for {ticker}: {exc}"
            )
            logger.warning(error_msg)
            state["errors"].append(error_msg)

        # Step 8: Calculate confidence
        # Check if insider data is complete (both fields available)
        insider_data_complete = (
            management_signals.get("insider_ownership_pct") is not None
            and management_signals.get("institutional_ownership_pct") is not None
        )

        confidence, confidence_details = _calculate_confidence(
            cik_found=True,
            filing_found=True,
            sections=sections,
            claude_succeeded=claude_succeeded,
            multi_year_count=len(multi_year_filings),
            claude_evolution_succeeded=claude_evolution_succeeded,
            insider_data_complete=insider_data_complete,
        )

        # Write results
        state["regulatory_analysis"] = {
            "risk_factors": risk_factors,
            "legal_proceedings": legal_proceedings,
            "risk_score": risk_score,
            "filing_date": filing_date,
            "confidence": confidence,
            "cik": cik,
            "filing_url": filing_url,
            "multi_year_filings": multi_year_filings,
            "risk_evolution": risk_evolution,
            "management_signals": management_signals,
            "confidence_details": confidence_details,
        }

        logger.info(
            "RegulatoryAgent completed for %s "
            "(risk_score=%s, confidence=%.0f)",
            ticker,
            risk_score,
            confidence,
        )

        # Add provenance metadata
        if "provenance_metadata" not in state:
            state["provenance_metadata"] = {}
        state["provenance_metadata"]["regulatory"] = {
            "agent": "RegulatoryAgent",
            "source": "SEC 10-K",
            "timestamp": datetime.now(UTC).isoformat(),
            "filing_date": filing_date,
            "citation": "Risk Factors section",
        }

        _post_regulatory_insights(state)

        return state


def _empty_analysis() -> dict[str, Any]:
    """Return an empty regulatory analysis dict with default values.

    Returns:
        A dict matching the regulatory_analysis structure with defaults.
    """
    return {
        "risk_factors": [],
        "legal_proceedings": "No data available.",
        "risk_score": "Low",
        "filing_date": "",
        "confidence": 0.0,
        "cik": "",
        "filing_url": "",
        "multi_year_filings": [],
        "risk_evolution": None,
        "management_signals": {},
        "confidence_details": {},
    }


def _resolve_cik_safe(
    ticker: str,
    state: ResearchState,
) -> str | None:
    """Resolve CIK with error handling.

    Args:
        ticker: Stock ticker symbol.
        state: ResearchState for error accumulation.

    Returns:
        CIK string or None if resolution failed.
    """
    try:
        cik = resolve_cik(ticker)
        if not cik:
            error_msg = (
                f"RegulatoryAgent: Could not resolve CIK "
                f"for {ticker}"
            )
            logger.warning(error_msg)
            state["errors"].append(error_msg)
            return None
        return cik
    except Exception as exc:
        error_msg = (
            f"RegulatoryAgent: CIK lookup failed for {ticker}: {exc}"
        )
        logger.warning(error_msg)
        state["errors"].append(error_msg)
        return None


def _fetch_filings_safe(
    cik: str,
    ticker: str,
    state: ResearchState,
) -> list[dict[str, Any]]:
    """Fetch filings with error handling.

    Args:
        cik: 10-digit zero-padded CIK string.
        ticker: Stock ticker for logging.
        state: ResearchState for error accumulation.

    Returns:
        List of filing metadata dicts, or empty list on failure.
    """
    try:
        filings = fetch_recent_filings(cik)
        if not filings:
            error_msg = (
                f"RegulatoryAgent: No 10-K filings found "
                f"for {ticker} (CIK: {cik})"
            )
            logger.warning(error_msg)
            state["errors"].append(error_msg)
            return []
        return filings
    except Exception as exc:
        error_msg = (
            f"RegulatoryAgent: Filing fetch failed "
            f"for {ticker}: {exc}"
        )
        logger.warning(error_msg)
        state["errors"].append(error_msg)
        return []


def _fetch_filing_text_safe(
    cik: str,
    accession_number: str,
    primary_document: str,
    ticker: str,
    state: ResearchState,
) -> str:
    """Fetch filing document text with error handling.

    Args:
        cik: 10-digit zero-padded CIK string.
        accession_number: Filing accession number.
        primary_document: Filename of the primary document.
        ticker: Stock ticker for logging.
        state: ResearchState for error accumulation.

    Returns:
        Plain text content of the filing, or empty string on failure.
    """
    try:
        return fetch_filing_text(cik, accession_number, primary_document)
    except Exception as exc:
        error_msg = (
            f"RegulatoryAgent: Filing text fetch failed "
            f"for {ticker}: {exc}"
        )
        logger.warning(error_msg)
        state["errors"].append(error_msg)
        return ""


def _call_claude_analysis(
    ticker: str,
    sections: dict[str, str],
) -> dict[str, Any] | None:
    """Call Claude to analyze 10-K sections for regulatory risks.

    Truncates each section to _MAX_SECTION_CHARS before sending to
    stay within token limits. Parses the JSON response.

    Args:
        ticker: Stock ticker symbol.
        sections: Dict with risk_factors, legal_proceedings, md_and_a.

    Returns:
        Parsed dict with risk_factors, legal_proceedings, risk_score,
        or None on failure.
    """
    risk_text = sections.get("risk_factors", "")[:_MAX_SECTION_CHARS]
    legal_text = sections.get("legal_proceedings", "")[:_MAX_SECTION_CHARS]
    mda_text = sections.get("md_and_a", "")[:_MAX_SECTION_CHARS]

    if not risk_text and not legal_text and not mda_text:
        return None

    prompt = REGULATORY_RISK_PROMPT.format(
        ticker=ticker,
        risk_factors=risk_text or "Not available.",
        legal_proceedings=legal_text or "Not available.",
        md_and_a=mda_text or "Not available.",
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()

        raw_text = ""
        for block in message.content:
            if block.type == "text":
                raw_text = block.text.strip()
                break

        if not raw_text:
            logger.warning(
                "Claude returned no text block for %s", ticker,
            )
            return None

        # Strip markdown code fences if Claude wraps JSON in ```json ... ```
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        parsed: dict[str, Any] = json.loads(raw_text)

        # Validate expected keys
        risk_factors = parsed.get("risk_factors", [])
        if not isinstance(risk_factors, list):
            risk_factors = []

        legal = parsed.get(
            "legal_proceedings",
            "No material legal proceedings disclosed.",
        )
        if not isinstance(legal, str):
            legal = str(legal)

        score = parsed.get("risk_score", "Low")
        if score not in ("Low", "Medium", "High"):
            score = "Medium"

        return {
            "risk_factors": [str(r) for r in risk_factors[:5]],
            "legal_proceedings": legal,
            "risk_score": score,
        }

    except Exception as exc:
        logger.warning(
            "Claude regulatory call failed for %s: %s",
            ticker,
            exc,
        )
        return None


def _fallback_risk_extraction(risk_text: str) -> list[str]:
    """Extract basic risk factors without Claude.

    Splits the risk factors section into paragraphs and returns the
    first 3 non-trivial ones as a simple fallback.

    Args:
        risk_text: Plain text of the risk factors section.

    Returns:
        List of up to 3 risk factor paragraph strings.
    """
    if not risk_text:
        return []

    paragraphs = [
        p.strip() for p in risk_text.split("\n\n") if len(p.strip()) > 50
    ]
    # Take first 3 paragraphs, truncated to 500 chars each
    return [p[:500] for p in paragraphs[:3]]


def _fetch_multi_year_filings(
    cik: str,
    ticker: str,
    state: ResearchState,
) -> list[dict[str, Any]]:
    """Fetch multiple years of 10-K filings with error handling.

    Args:
        cik: 10-digit zero-padded CIK string.
        ticker: Stock ticker for logging.
        state: ResearchState for error accumulation.

    Returns:
        List of filing dicts with year, filing_date, risk_factors_text.
        Returns empty list on failure.
    """
    try:
        filings = fetch_multi_year_10k(cik, years=3)
        logger.info("Fetched %d years of 10-K filings for %s", len(filings), ticker)
        return filings
    except Exception as exc:
        error_msg = (
            f"RegulatoryAgent: Multi-year fetch failed for {ticker}: {exc}"
        )
        logger.warning(error_msg)
        state["errors"].append(error_msg)
        return []


def _call_claude_risk_evolution(
    ticker: str,
    filings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Call Claude to analyze year-over-year risk evolution.

    Args:
        ticker: Stock ticker symbol.
        filings: List of 3 filing dicts with year and risk_factors_text.

    Returns:
        Dict with new_risks, removed_risks, escalated_risks, trend,
        interpretation, or None on failure.
    """
    if len(filings) < 2:
        # Need at least 2 years for comparison
        return None

    # Get 3 years (or as many as available)
    latest = filings[0]
    prior_1 = filings[1] if len(filings) >= 2 else None
    prior_2 = filings[2] if len(filings) >= 3 else None

    # Truncate risk factors to stay within token limits
    risk_latest = latest["risk_factors_text"][:_MAX_SECTION_CHARS]
    risk_prior_1 = (
        prior_1["risk_factors_text"][:_MAX_SECTION_CHARS]
        if prior_1
        else "Not available."
    )
    risk_prior_2 = (
        prior_2["risk_factors_text"][:_MAX_SECTION_CHARS]
        if prior_2
        else "Not available."
    )

    if not risk_latest:
        return None

    prompt = RISK_EVOLUTION_PROMPT.format(
        ticker=ticker,
        year_latest=latest["year"],
        risk_factors_latest=risk_latest,
        year_prior_1=prior_1["year"] if prior_1 else "N/A",
        risk_factors_prior_1=risk_prior_1,
        year_prior_2=prior_2["year"] if prior_2 else "N/A",
        risk_factors_prior_2=risk_prior_2,
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()

        raw_text = ""
        for block in message.content:
            if block.type == "text":
                raw_text = block.text.strip()
                break

        if not raw_text:
            logger.warning(
                "Claude risk evolution returned no text for %s", ticker,
            )
            return None

        # Strip markdown code fences if Claude wraps JSON in ```json ... ```
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        parsed: dict[str, Any] = json.loads(raw_text)

        # Validate and sanitize
        return {
            "new_risks": parsed.get("new_risks", []),
            "removed_risks": parsed.get("removed_risks", []),
            "escalated_risks": parsed.get("escalated_risks", []),
            "trend": parsed.get("trend", "stable"),
            "interpretation": parsed.get("interpretation", ""),
        }

    except Exception as exc:
        logger.warning(
            "Claude risk evolution call failed for %s: %s",
            ticker,
            exc,
        )
        return None


def _fetch_insider_data(
    state: ResearchState,
) -> dict[str, Any]:
    """Fetch insider ownership and trading signals from state.

    Extracts insider/institutional ownership percentages and CEO ownership
    value from yfinance info dict (populated by MarketDataAgent).

    Args:
        state: ResearchState containing financials and current_price.

    Returns:
        Dict with insider_ownership_pct, institutional_ownership_pct,
        ceo_ownership_value, signal. Fields may be None if data unavailable.
    """
    # Extract yfinance info dict from state (set by MarketDataAgent)
    info = state.get("financials", {})
    current_price = float(state["market_data"].get("current_price", 0.0))

    # Fetch insider/institutional ownership percentages
    ownership = fetch_insider_ownership(info)

    # Calculate CEO ownership value (MVP returns None)
    ceo_value = calculate_ceo_ownership_value(info, current_price)

    # Interpret insider trading signal (MVP: buying and selling are None)
    signal = interpret_insider_signal(buying=None, selling=None)

    return {
        "insider_ownership_pct": ownership["insider_pct"],
        "institutional_ownership_pct": ownership["institutional_pct"],
        "ceo_ownership_value": ceo_value,
        "signal": signal,
    }


def _calculate_confidence(
    cik_found: bool,
    filing_found: bool,
    sections: dict[str, str],
    claude_succeeded: bool,
    multi_year_count: int = 1,
    claude_evolution_succeeded: bool = False,
    insider_data_complete: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Calculate confidence score based on data completeness.

    Scoring (reweighted for institutional depth):
      Base components (60 points total):
      - CIK resolved: +15
      - 10-K filing found: +15
      - Risk Factors extracted: +15
      - Legal Proceedings extracted: +5
      - MD&A extracted: +5
      - Claude analysis succeeded: +5

      Multi-year depth (20 points):
      - 3 years: +20
      - 2 years: +10
      - 1 year: +0

      Insider data (10 points):
      - Complete (both insider & institutional %): +10
      - Partial (only one field): +0
      - Missing (both None): -5

      Evolution analysis (10 points):
      - Claude risk evolution succeeded: +10

    Args:
        cik_found: Whether the CIK was resolved successfully.
        filing_found: Whether a 10-K filing was found.
        sections: Extracted section texts (may be empty strings).
        claude_succeeded: Whether Claude analysis completed.
        multi_year_count: Number of years of filings fetched (1-3).
        claude_evolution_succeeded: Whether Claude risk evolution completed.
        insider_data_complete: Whether both insider & institutional % available.

    Returns:
        Tuple of (confidence score 0-100, confidence_details dict).
    """
    score = 0.0

    # Base components (60 points)
    if cik_found:
        score += 15.0
    if filing_found:
        score += 15.0
    if sections.get("risk_factors"):
        score += 15.0
    if sections.get("legal_proceedings"):
        score += 5.0
    if sections.get("md_and_a"):
        score += 5.0
    if claude_succeeded:
        score += 5.0

    # Multi-year depth (20 points)
    if multi_year_count >= 3:
        score += 20.0
    elif multi_year_count >= 2:
        score += 10.0

    # Insider data (10 points / -5 penalty)
    if insider_data_complete:
        score += 10.0
    # Note: No penalty for missing data in MVP (too harsh)

    # Evolution analysis (10 points)
    if claude_evolution_succeeded:
        score += 10.0

    confidence_score = min(score, 100.0)

    # Confidence details breakdown
    confidence_details = {
        "filings_analyzed": multi_year_count,
        "insider_data_available": insider_data_complete,
        "claude_success": claude_succeeded,
        "claude_evolution_success": claude_evolution_succeeded,
    }

    return confidence_score, confidence_details


def _post_regulatory_insights(state: ResearchState) -> None:
    """Post cross-domain regulatory signals to the insights board.

    Reads regulatory_analysis from state and posts insights for new material
    risks, litigation concerns, and low disclosure quality. Appends to
    state['errors'] on failure; never raises.

    Args:
        state: ResearchState with regulatory_analysis populated.
    """
    try:
        ticker = state.get("ticker", "")
        reg = state.get("regulatory_analysis") or {}

        # New material risks from risk evolution
        evolution = reg.get("risk_evolution") or {}
        new_risks: list[str] = evolution.get("new_risks") or []
        for risk in new_risks[:3]:  # Cap at 3 to avoid flooding the board
            post_insight(
                state,
                agent="RegulatoryAgent",
                category="regulatory",
                signal=f"{ticker} new material risk in latest 10-K: {risk[:120]}",
                confidence=0.85,
            )

        # Litigation concern
        legal = reg.get("legal_proceedings", "")
        if legal and legal not in (
            "No material legal proceedings disclosed.",
            "No data available.",
            "",
        ) and len(legal) > 100:
            post_insight(
                state,
                agent="RegulatoryAgent",
                category="litigation",
                signal=f"{ticker} has material legal proceedings disclosed in 10-K",
                confidence=0.8,
            )

        # Low disclosure quality / confidence
        confidence_val = reg.get("confidence", 100.0)
        if confidence_val < 60.0:
            post_insight(
                state,
                agent="RegulatoryAgent",
                category="disclosure_quality",
                signal=(
                    f"{ticker} regulatory analysis confidence {confidence_val:.0f}% "
                    f"— limited SEC filing data available"
                ),
                confidence=0.7,
            )

        # High risk score from Claude
        risk_score = reg.get("risk_score", "")
        if risk_score == "High":
            post_insight(
                state,
                agent="RegulatoryAgent",
                category="regulatory",
                signal=f"{ticker} regulatory risk rated HIGH by Claude (10-K analysis)",
                confidence=0.85,
            )

    except Exception as exc:
        msg = f"_post_regulatory_insights failed: {exc}"
        logger.warning(msg)
        state["errors"].append(msg)
