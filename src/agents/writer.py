"""WriterAgent — generates a professional Markdown equity research report."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import anthropic
from doxa_shared.prompts.writer import NARRATIVE_PROMPT
from doxa_shared.utils.formatters import (
    fmt_large_number,
    fmt_number,
    fmt_pct,
    fmt_ratio,
)

from src.config import ANTHROPIC_API_KEY, configure_logging
from src.state import ResearchState, create_initial_state

logger = logging.getLogger(__name__)

_CLAUDE_MODEL = "claude-opus-4-6"


def _count_section_words(text: str) -> int:
    """Count non-table words in a markdown section.

    Excludes lines starting with ``|`` (table rows).
    """
    return len([
        w
        for line in text.split("\n")
        for w in line.split()
        if not line.strip().startswith("|")
    ])


class WriterAgent:
    """Synthesizes all gathered data into a formatted IC research report.

    Uses Claude to write narrative prose sections (I. Investment Summary
    through VII. Investment Risks) and embeds them alongside structured
    data tables in a 30-50 page institutional Markdown IC report.
    """

    def generate_report(self, state: ResearchState) -> ResearchState:
        """Build the final Markdown report and write it to state.

        Args:
            state: A fully-populated ResearchState.

        Returns:
            The same state dict with final_report populated.
        """
        # Calculate price target and rating
        price_target_dict = self._calculate_price_target(state)
        rating, rating_explanation = self._calculate_rating(
            state, price_target_dict["upside_pct"]
        )

        # Generate narrative with valuation context
        narrative = _generate_narrative(state, rating, price_target_dict)

        # Build comprehensive report
        report = _build_report(state, rating, narrative, price_target_dict)
        state["final_report"] = report

        logger.info(
            "Report generated for %s — Rating: %s (%s)",
            state["ticker"],
            rating,
            rating_explanation,
        )
        return state

    def _calculate_price_target(
        self, state: ResearchState
    ) -> dict[str, Any]:
        """Calculate bull/base/bear price targets with weighted average.

        Uses DCF fair value as base, applies scenario multipliers, and
        weights scenarios based on quant signal and sentiment.

        Args:
            state: Fully-populated ResearchState.

        Returns:
            Dict with price_target, upside_pct, bull/base/bear targets,
            probabilities, and methodology explanation.
        """
        val = state.get("valuation_analysis", {})
        dcf = val.get("dcf", {})
        comps = val.get("comps", {})

        current_price = dcf.get("current_price", 0) or state.get(
            "market_data", {}
        ).get("current_price", 0)

        if not current_price or current_price <= 0:
            return {
                "price_target": 0.0,
                "upside_pct": 0.0,
                "bull_target": 0.0,
                "base_target": 0.0,
                "bear_target": 0.0,
                "bull_prob": 0.2,
                "base_prob": 0.6,
                "bear_prob": 0.2,
                "methodology": "Insufficient data for price target",
            }

        # Get DCF fair value (primary input)
        dcf_fair_value = dcf.get("fair_value_per_share", 0)

        # Get comps-implied value (secondary input)
        # Note: implied_valuations are total market cap, need to convert to per-share
        comps_implied_value = 0.0
        if comps and comps.get("implied_valuations"):
            market_cap = state.get("market_data", {}).get("market_cap", 0) or 0
            shares_outstanding = (
                market_cap / current_price
                if current_price > 0 and market_cap > 0
                else 0
            )

            if shares_outstanding > 0:
                # Convert total market cap values to per-share values
                impl_vals = [
                    v / shares_outstanding for v in comps["implied_valuations"].values()
                    if v is not None and v > 0
                ]
                if impl_vals:
                    # Use median to avoid outliers from different multiples
                    sorted_vals = sorted(impl_vals)
                    mid = len(sorted_vals) // 2
                    if len(sorted_vals) % 2 == 0:
                        comps_implied_value = (
                            (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
                        )
                    else:
                        comps_implied_value = sorted_vals[mid]

        if comps and comps.get("implied_valuations") and comps_implied_value <= 0:
            logger.warning(
                "Comps data exists for %s but per-share conversion failed "
                "(market_cap=%s, current_price=%s)",
                state["ticker"],
                state.get("market_data", {}).get("market_cap"),
                current_price,
            )

        # Fallback if no DCF data
        if dcf_fair_value <= 0:
            if comps_implied_value > 0:
                dcf_fair_value = comps_implied_value
                methodology = "Price target based on peer comps only (DCF unavailable)"
            else:
                return {
                    "price_target": current_price,
                    "upside_pct": 0.0,
                    "bull_target": current_price,
                    "base_target": current_price,
                    "bear_target": current_price,
                    "bull_prob": 0.2,
                    "base_prob": 0.6,
                    "bear_prob": 0.2,
                    "methodology": "Insufficient data for price target",
                }

        # Calculate base case (weighted average of DCF and comps)
        if comps_implied_value > 0:
            base_target = (dcf_fair_value * 0.6) + (comps_implied_value * 0.4)
            methodology = (
                f"Weighted average: 60% DCF (${dcf_fair_value:.2f}) + "
                f"40% comps (${comps_implied_value:.2f})"
            )
        else:
            base_target = dcf_fair_value
            methodology = "100% DCF fair value (no peer comps available)"

        # Dynamic bull/bear multipliers based on quant signal and sentiment
        signal = state["quant_analysis"].get("signal", "")
        sentiment = state["sentiment_score"]

        if signal == "BULLISH" and sentiment > 0.3:
            bull_mult, bear_mult = 1.20, 0.90
        elif signal == "BEARISH" or sentiment < -0.3:
            bull_mult, bear_mult = 1.10, 0.80
        else:
            bull_mult, bear_mult = 1.15, 0.85

        bull_target = dcf_fair_value * bull_mult
        bear_target = dcf_fair_value * bear_mult

        # Assign scenario probabilities (reuse signal/sentiment from above)
        if signal == "BULLISH" and sentiment > 0.3:
            bull_prob, base_prob, bear_prob = 0.3, 0.6, 0.1
        elif signal == "BEARISH" or sentiment < -0.3:
            bull_prob, base_prob, bear_prob = 0.1, 0.6, 0.3
        else:
            bull_prob, base_prob, bear_prob = 0.2, 0.6, 0.2

        # Weighted average price target
        price_target = (
            (bull_target * bull_prob) +
            (base_target * base_prob) +
            (bear_target * bear_prob)
        )

        # Calculate upside %
        upside_pct = ((price_target - current_price) / current_price) * 100

        return {
            "price_target": round(price_target, 2),
            "upside_pct": round(upside_pct, 1),
            "bull_target": round(bull_target, 2),
            "base_target": round(base_target, 2),
            "bear_target": round(bear_target, 2),
            "bull_prob": bull_prob,
            "base_prob": base_prob,
            "bear_prob": bear_prob,
            "methodology": methodology,
        }

    def _calculate_rating(
        self, state: ResearchState, upside_pct: float
    ) -> tuple[str, str]:
        """Derive investment rating from valuation upside with tiebreakers.

        Valuation-based rating takes precedence, with quant signal and
        Altman Z-Score as tiebreakers when upside is near thresholds.

        Args:
            state: Fully-populated ResearchState.
            upside_pct: Upside percentage to price target.

        Returns:
            Tuple of (rating, precedence_explanation).
        """
        signal = state["quant_analysis"].get("signal", "")
        sentiment = state["sentiment_score"]
        altman_z = state["quant_analysis"].get("altman_z", 0)

        # Primary rating logic (valuation-based)
        if upside_pct > 20:
            base_rating = "Strong Buy"
        elif upside_pct > 10:
            base_rating = "Buy"
        elif upside_pct > -10:
            base_rating = "Hold"
        else:
            base_rating = "Sell"

        # Tiebreaker: bump one tier when near threshold (±2%)
        if 9 <= upside_pct <= 11:  # Near Hold/Buy threshold (10%)
            if signal == "BULLISH" and sentiment > 0.4:
                upgraded = "Buy" if base_rating == "Hold" else "Strong Buy"
                return (
                    upgraded,
                    f"Based on {upside_pct:.1f}% upside, "
                    f"upgraded due to bullish quant signal",
                )
        elif 19 <= upside_pct <= 21:  # Near Buy/Strong Buy threshold (20%)
            if signal == "BEARISH" or sentiment < -0.4:
                downgraded = "Buy" if base_rating == "Strong Buy" else "Hold"
                return (
                    downgraded,
                    f"Based on {upside_pct:.1f}% upside, "
                    f"downgraded due to bearish quant signal",
                )
        elif -11 <= upside_pct <= -9:  # Near Hold/Sell threshold (-10%)
            if altman_z < 1.81:  # Distress zone
                return (
                    "Sell",
                    f"Based on {upside_pct:.1f}% upside, "
                    f"downgraded due to Altman Z-Score distress zone",
                )

        # Default rating without tiebreaker
        return (base_rating, f"Based on {upside_pct:.1f}% upside to price target")

    @staticmethod
    def _build_valuation_section(
        state: ResearchState,
        price_target_dict: dict[str, Any],
        *,
        _compact: bool = False,
    ) -> str:
        """Generate comprehensive Valuation Analysis markdown section.

        Args:
            state: Fully-populated ResearchState.
            price_target_dict: Price target calculation results.
            _compact: Internal flag to reduce length when over 2,000 words.

        Returns:
            Formatted markdown string with valuation details (max 2,000 words).
        """
        val = state.get("valuation_analysis", {})
        if not val or not val.get("dcf"):
            return "## Valuation Analysis\n\n*No valuation data available.*\n"

        dcf = val["dcf"]
        comps = val.get("comps", {})
        confidence = val.get("confidence", 0)
        max_fcf = 3 if _compact else 5
        max_peers = 3 if _compact else 6

        lines: list[str] = ["## Valuation Analysis\n"]

        # Price Target Methodology
        lines.append("### Price Target Methodology\n")
        lines.append("| Scenario | Target | Probability |")
        lines.append("|----------|--------|-------------|")
        lines.append(
            f"| Bull Case | ${price_target_dict['bull_target']:.2f} | "
            f"{price_target_dict['bull_prob']*100:.0f}% |"
        )
        lines.append(
            f"| Base Case | ${price_target_dict['base_target']:.2f} | "
            f"{price_target_dict['base_prob']*100:.0f}% |"
        )
        lines.append(
            f"| Bear Case | ${price_target_dict['bear_target']:.2f} | "
            f"{price_target_dict['bear_prob']*100:.0f}% |"
        )
        lines.append(
            f"| **12-Month Target** | "
            f"**${price_target_dict['price_target']:.2f}** |"
            f" **Weighted Avg** |\n"
        )
        lines.append(
            f"**Methodology:** {price_target_dict['methodology']}\n"
        )

        # DCF Valuation
        lines.append("### DCF Valuation\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(
            f"| Fair Value per Share "
            f"| ${dcf.get('fair_value_per_share', 0):.2f} |"
        )
        lines.append(
            f"| Current Price | ${dcf.get('current_price', 0):.2f} |"
        )
        lines.append(
            f"| DCF Upside | {dcf.get('upside_downside_pct', 0):+.1f}% |"
        )
        lines.append(f"| WACC | {dcf.get('wacc', 0) * 100:.2f}% |")
        lines.append(
            f"| Terminal Value | ${dcf.get('terminal_value', 0):,.0f} |\n"
        )

        # FCF Projections
        fcf_proj = dcf.get("fcf_projections", [])
        if fcf_proj:
            label = "3-Year" if _compact else "5-Year"
            lines.append(f"**{label} FCF Projections:**")
            fcf_display = ", ".join(
                f"${v/1e9:.2f}B" for v in fcf_proj[:max_fcf]
            )
            lines.append(f"{fcf_display}\n")

        # Sensitivity Analysis
        sens_table = dcf.get("sensitivity_table", {})
        if sens_table and sens_table.get("values"):
            lines.append("### Sensitivity Analysis\n")
            wacc_range = sens_table.get("wacc_range", [])
            growth_range = sens_table.get("growth_range", [])
            values = sens_table.get("values", [])

            if wacc_range and growth_range and values:
                header = "| Terminal Growth |"
                for wacc in wacc_range:
                    header += f" WACC {wacc*100:.1f}% |"
                lines.append(header)
                lines.append(
                    "|" + "---|" * (len(wacc_range) + 1)
                )

                for i, growth_rate in enumerate(growth_range):
                    row = f"| {growth_rate*100:.1f}% |"
                    if i < len(values):
                        for cell_val in values[i]:
                            row += f" ${cell_val:.2f} |"
                    lines.append(row)
                lines.append("")

        # Comparable Company Analysis
        if comps and comps.get("peer_companies"):
            lines.append("### Comparable Company Analysis\n")
            lines.append(
                "| Ticker | P/E | EV/EBITDA | P/B | P/S |"
            )
            lines.append(
                "|--------|-----|-----------|-----|-----|"
            )

            peer_multiples = comps.get("peer_multiples", {})
            for ticker in comps["peer_companies"][:max_peers]:
                multiples = peer_multiples.get(ticker, {})
                pe = multiples.get("P/E")
                ev = multiples.get("EV/EBITDA")
                pb = multiples.get("P/B")
                ps = multiples.get("P/S")

                lines.append(
                    f"| {ticker} | "
                    f"{f'{pe:.1f}x' if pe else 'N/A'} | "
                    f"{f'{ev:.1f}x' if ev else 'N/A'} | "
                    f"{f'{pb:.1f}x' if pb else 'N/A'} | "
                    f"{f'{ps:.1f}x' if ps else 'N/A'} |"
                )

            med = comps.get("median_multiples", {})
            pe_med = med.get("P/E")
            ev_med = med.get("EV/EBITDA")
            pb_med = med.get("P/B")
            ps_med = med.get("P/S")

            lines.append(
                f"| **Median** | "
                f"**{f'{pe_med:.1f}x' if pe_med else 'N/A'}** | "
                f"**{f'{ev_med:.1f}x' if ev_med else 'N/A'}** | "
                f"**{f'{pb_med:.1f}x' if pb_med else 'N/A'}** | "
                f"**{f'{ps_med:.1f}x' if ps_med else 'N/A'}** |"
            )

            # Target company row
            target_pe = state["quant_analysis"].get("pe_ratio")
            lines.append(
                f"| **{state['ticker']}** | "
                f"**{f'{target_pe:.1f}x' if target_pe else 'N/A'}** | "
                f"N/A | N/A | N/A |"
            )

            # Premium/discount row
            if target_pe and pe_med and pe_med > 0:
                pe_prem = (target_pe - pe_med) / pe_med * 100
                lines.append(
                    f"| **Premium/(Discount)** | "
                    f"**{pe_prem:+.0f}%** | — | — | — |"
                )
            lines.append("")

        # Scenario Analysis
        lines.append("### Scenario Analysis\n")
        bull = price_target_dict["bull_target"]
        base = price_target_dict["base_target"]
        bear = price_target_dict["bear_target"]
        bull_p = price_target_dict["bull_prob"] * 100
        base_p = price_target_dict["base_prob"] * 100
        bear_p = price_target_dict["bear_prob"] * 100

        bull_assumptions = _derive_scenario_assumptions(state, "bull")
        base_assumptions = _derive_scenario_assumptions(state, "base")
        bear_assumptions = _derive_scenario_assumptions(state, "bear")

        lines.append(
            "| Scenario | Target | Prob | Key Assumptions |"
        )
        lines.append(
            "|----------|--------|------|-----------------|"
        )
        lines.append(
            f"| Bull | ${bull:.2f} | {bull_p:.0f}% | "
            f"{bull_assumptions} |"
        )
        lines.append(
            f"| Base | ${base:.2f} | {base_p:.0f}% | "
            f"{base_assumptions} |"
        )
        lines.append(
            f"| Bear | ${bear:.2f} | {bear_p:.0f}% | "
            f"{bear_assumptions} |"
        )
        lines.append("")

        lines.append(f"**Valuation Confidence:** {confidence:.1f}%\n")

        section = "\n".join(lines)
        if not _compact and _count_section_words(section) > 2000:
            logger.warning(
                "Valuation section exceeds 2,000 words, using compact mode"
            )
            return WriterAgent._build_valuation_section(
                state, price_target_dict, _compact=True
            )
        return section

    @staticmethod
    def _build_regulatory_section(
        state: ResearchState, *, _compact: bool = False
    ) -> str:
        """Generate comprehensive Regulatory & Risk Assessment section.

        Args:
            state: Fully-populated ResearchState.
            _compact: Internal flag to reduce length when over 2,500 words.

        Returns:
            Formatted markdown string (max 2,500 words).
        """
        reg = state.get("regulatory_analysis", {})
        if not reg or not reg.get("filing_date"):
            return (
                "## Regulatory & Risk Assessment\n\n"
                "*No regulatory data available.*\n"
            )

        risk_score = reg.get("risk_score", "Unknown")
        risk_factors = reg.get("risk_factors", [])
        legal = reg.get("legal_proceedings", "No data available.")
        filing_date = reg.get("filing_date", "N/A")
        cik = reg.get("cik", "N/A")
        filing_url = reg.get("filing_url", "")
        confidence = reg.get("confidence", 0)
        max_risks = 3 if _compact else 5
        max_legal_words = 250 if _compact else 500

        lines: list[str] = ["## Regulatory & Risk Assessment\n"]

        risk_emoji = {
            "Low": "🟢", "Medium": "🟡", "High": "🔴",
        }.get(risk_score, "⚪")
        lines.append(f"**Risk Score:** {risk_emoji} **{risk_score}**\n")
        lines.append(
            f"**Most Recent 10-K:** {filing_date} | **CIK:** {cik}"
        )
        if filing_url:
            lines.append(f" | [View on SEC EDGAR]({filing_url})")
        lines.append("\n")

        # Key Risk Factors
        if risk_factors:
            lines.append("### Key Risk Factors\n")
            for i, risk_text in enumerate(risk_factors[:max_risks], 1):
                words = risk_text.split()
                if len(words) > 300:
                    risk_text = (
                        " ".join(words[:300]) + "... [truncated]"
                    )
                lines.append(f"{i}. {risk_text}\n")
        else:
            lines.append("### Key Risk Factors\n")
            lines.append("*No risk factors identified.*\n")

        # Legal Proceedings
        lines.append("### Legal Proceedings\n")
        if legal and legal != "No data available.":
            words = legal.split()
            if len(words) > max_legal_words:
                legal = (
                    " ".join(words[:max_legal_words])
                    + "... [full details in SEC filing]"
                )
            lines.append(f"{legal}\n")
        else:
            lines.append("*No material legal proceedings disclosed.*\n")

        lines.append(
            f"**Regulatory Analysis Confidence:** {confidence:.0f}%\n"
        )

        section = "\n".join(lines)
        if not _compact and _count_section_words(section) > 2500:
            logger.warning(
                "Regulatory section exceeds 2,500 words, using compact mode"
            )
            return WriterAgent._build_regulatory_section(
                state, _compact=True
            )
        return section

    @staticmethod
    def _build_header_block(
        state: ResearchState,
        price_target_dict: dict[str, Any],
        rating: str,
    ) -> str:
        """Generate IC Snapshot header block with key metrics.

        Args:
            state: Fully-populated ResearchState.
            price_target_dict: Price target calculation results.
            rating: Investment rating string.

        Returns:
            Formatted Snapshot markdown table for IC page-1 header.
        """
        md = state["market_data"]
        fin = state["financials"]
        qa = state["quant_analysis"]
        reg = state.get("regulatory_analysis", {})

        # Enterprise value
        market_cap = md.get("market_cap") or 0
        total_debt = fin.get("total_debt") or 0
        total_cash = fin.get("total_cash") or 0
        current_price = md.get("current_price") or 0
        ev = market_cap + total_debt - total_cash
        ev_str = fmt_large_number(ev) if ev > 0 else "N/A"

        # Shares outstanding
        shares = (
            market_cap / current_price
            if current_price > 0 and market_cap > 0
            else None
        )
        shares_str = fmt_large_number(shares, prefix="") if shares else "N/A"

        # Beta
        beta = md.get("beta")
        beta_str = f"{beta:.2f}" if beta is not None else "N/A"

        # Avg daily volume
        vol_analysis = md.get("volume_analysis") or {}
        avg_vol = vol_analysis.get("avg_volume")
        avg_vol_str = fmt_large_number(avg_vol, prefix="") if avg_vol else "N/A"

        # Dividend yield (stored as percentage, e.g. 0.5 means 0.5%)
        div_hist = md.get("dividend_history") or {}
        div_yield = div_hist.get("yield")
        div_yield_str = f"{div_yield:.1f}%" if div_yield is not None else "N/A"

        # Altman Z-Score
        altman_z = qa.get("altman_z")
        altman_zone = qa.get("altman_zone", "Unknown")
        altman_emoji = {
            "Safe": "🟢", "Grey": "🟡", "Distress": "🔴",
        }.get(altman_zone, "⚪")
        altman_str = (
            f"{altman_emoji} {altman_z:.2f}" if altman_z is not None else "N/A"
        )

        # Regulatory risk
        risk_score = reg.get("risk_score", "N/A")
        reg_emoji = {
            "Low": "🟢", "Medium": "🟡", "High": "🔴",
        }.get(risk_score, "⚪")
        reg_str = f"{reg_emoji} {risk_score}"

        pt = price_target_dict["price_target"]
        upside = price_target_dict["upside_pct"]

        lines = [
            "## Snapshot\n",
            "| Metric | Value | Metric | Value |",
            "|--------|-------|--------|-------|",
            f"| Current Price | {fmt_number(current_price, '$')} "
            f"| 52-Wk High | {fmt_number(md.get('fifty_two_week_high'), '$')} |",
            f"| Price Target | {fmt_number(pt, '$')} "
            f"| 52-Wk Low | {fmt_number(md.get('fifty_two_week_low'), '$')} |",
            f"| Rating | {rating} | Beta | {beta_str} |",
            f"| Implied Upside | {upside:+.1f}% | Avg Daily Vol | {avg_vol_str} |",
            f"| Market Cap | {fmt_large_number(market_cap)} "
            f"| Shares Out | {shares_str} |",
            f"| Enterprise Value | {ev_str} | Dividend Yield | {div_yield_str} |",
            f"| P/E (TTM) | {fmt_ratio(qa.get('pe_ratio'))} "
            f"| Regulatory Risk | {reg_str} |",
            f"| Altman Z-Score | {altman_str} "
            f"| Quant Signal | {qa.get('signal', 'N/A')} |",
            "",
        ]

        return "\n".join(lines)

    @staticmethod
    def _build_confidence_section(
        state: ResearchState, narrative_was_generated: bool
    ) -> str:
        """Generate Data Quality & Confidence section.

        Args:
            state: Fully-populated ResearchState.
            narrative_was_generated: Whether Claude narrative succeeded.

        Returns:
            Formatted markdown string with confidence table and warnings.
        """
        # Calculate agent-specific confidence scores
        md = state["market_data"]
        market_data_conf = 100.0 if all(
            md.get(k) for k in ["current_price", "market_cap"]
        ) else (50.0 if md.get("current_price") else 0.0)

        quant_conf = state["quant_analysis"].get("confidence", 0.0)

        val = state.get("valuation_analysis", {})
        val_conf = val.get("confidence", 0.0)

        reg = state.get("regulatory_analysis", {})
        reg_conf = reg.get("confidence", 0.0)

        sentiment_rationale = state.get("sentiment_rationale", "")
        sentiment_conf = 100.0 if (
            sentiment_rationale and sentiment_rationale != "..."
        ) else 0.0

        writer_conf = 100.0 if narrative_was_generated else 0.0

        # Weighted average (MarketData 15%, Quant 20%, Valuation 25%,
        # Regulatory 20%, Sentiment 10%, Writer 10%)
        overall_conf = (
            (market_data_conf * 0.15) +
            (quant_conf * 0.20) +
            (val_conf * 0.25) +
            (reg_conf * 0.20) +
            (sentiment_conf * 0.10) +
            (writer_conf * 0.10)
        )

        lines = ["## Data Quality & Confidence\n"]

        # Low confidence warning
        if overall_conf < 50:
            lines.append("⚠️ **LOW CONFIDENCE WARNING**")
            lines.append(
                "This report is based on incomplete or limited data. "
                "Use with caution."
            )
            lines.append(f"Overall confidence: {overall_conf:.1f}%\n")

        # Confidence table
        lines.append("| Agent | Confidence | Status |")
        lines.append("|-------|------------|--------|")

        def status_emoji(conf: float) -> tuple[str, str]:
            if conf >= 90:
                return ("✅", "Complete")
            if conf >= 50:
                return ("⚠️", "Partial")
            return ("❌", "Unavailable")

        agents = [
            ("Market Data", market_data_conf),
            ("Quant Analysis", quant_conf),
            ("Valuation", val_conf),
            ("Regulatory", reg_conf),
            ("Sentiment", sentiment_conf),
            ("Writer", writer_conf),
        ]

        for agent_name, conf in agents:
            emoji, status = status_emoji(conf)
            lines.append(f"| {agent_name} | {conf:.1f}% | {emoji} {status} |")

        lines.append(f"| **Overall** | **{overall_conf:.1f}%** | — |")
        lines.append("")

        # Methodology disclaimers
        lines.append("### Methodology\n")
        lines.append("- **Data Sources:** yfinance, SEC EDGAR, Anthropic Claude")
        lines.append(
            "- **Valuation Methods:** DCF (5-year FCF projections, "
            "WACC-based), Comparable Company Analysis"
        )
        lines.append(
            "- **Quant Metrics:** DuPont ROE Analysis, Altman Z-Score"
        )
        lines.append(
            "- **Limitations:** No forward EPS estimates, no charts, "
            "simplified peer selection"
        )
        lines.append("")

        return "\n".join(lines)


def _derive_scenario_assumptions(
    state: ResearchState, scenario: str
) -> str:
    """Generate ticker-specific scenario assumptions from state data.

    Args:
        state: Fully-populated ResearchState.
        scenario: One of "bull", "base", or "bear".

    Returns:
        A short string describing key assumptions for the scenario.
    """
    qa = state.get("quant_analysis", {})
    reg = state.get("regulatory_analysis", {})
    driver = qa.get("dupont_driver", "")
    margin = qa.get("profit_margin")
    zone = qa.get("altman_zone", "")
    risk_score = reg.get("risk_score", "")

    if scenario == "bull":
        parts = []
        if margin is not None and margin > 0.15:
            parts.append(f"Margin sustains above {margin*100:.0f}%")
        else:
            parts.append("Margin expansion via operating leverage")
        if driver:
            parts.append(f"{driver} trend accelerates")
        parts.append("Revenue growth above consensus")
        return "; ".join(parts[:3])

    if scenario == "base":
        parts = ["Current growth trajectory continues"]
        if margin is not None:
            parts.append(f"Margins stable near {margin*100:.0f}%")
        if zone:
            parts.append(f"Credit profile remains {zone}")
        return "; ".join(parts[:3])

    # bear
    parts = []
    if risk_score in ("Medium", "High"):
        top_risk = (reg.get("risk_factors") or ["regulatory headwinds"])[0]
        parts.append(top_risk.split(".")[0][:60])
    else:
        parts.append("Competitive pressure intensifies")
    if margin is not None and margin > 0.10:
        parts.append(f"Margin compresses from {margin*100:.0f}%")
    if zone == "Grey":
        parts.append("Credit deterioration into distress zone")
    elif zone != "Distress":
        parts.append("Multiple contraction on slowing growth")
    return "; ".join(parts[:3])


# ---------------------------------------------------------------------------
# Claude narrative generation
# ---------------------------------------------------------------------------


def _summarize_for_prompt(
    state: ResearchState, price_target_dict: dict[str, Any]
) -> dict[str, str]:
    """Prepare valuation and regulatory summaries for NARRATIVE_PROMPT.

    Returns dict with 'valuation_summary' and 'regulatory_summary' keys,
    each max 200 tokens (~800 chars).

    Args:
        state: Fully-populated ResearchState.
        price_target_dict: Price target calculation results.

    Returns:
        Dict with valuation_summary and regulatory_summary strings.
    """
    val = state.get("valuation_analysis", {})
    dcf = val.get("dcf", {})
    comps = val.get("comps", {})

    # Valuation summary
    if dcf:
        fv = fmt_number(dcf.get("fair_value_per_share"), "$")
        upside = price_target_dict.get("upside_pct", 0)
        val_summary = f"- DCF Fair Value: {fv} ({upside:+.1f}% upside to target)"

        if comps and comps.get("median_multiples"):
            med_pe = comps["median_multiples"].get("P/E")
            if med_pe:
                target_pe = state["quant_analysis"].get("pe_ratio", 0)
                premium = ((target_pe - med_pe) / med_pe * 100) if med_pe > 0 else 0
                val_summary += f"\n- Trading at {premium:+.1f}% vs peer median P/E"
    else:
        val_summary = "- Valuation data unavailable"

    # Regulatory summary
    reg = state.get("regulatory_analysis", {})
    if reg and reg.get("risk_factors"):
        risk_score = reg.get("risk_score", "Unknown")
        top_risk = reg["risk_factors"][0]
        # Extract first sentence only
        risk_first_sent = top_risk.split(".")[0][:150] + "..."
        reg_summary = (
            f"- Risk Score: {risk_score}\n"
            f"- Top Risk: {risk_first_sent}"
        )
    else:
        reg_summary = "- Regulatory data unavailable"

    return {
        "valuation_summary": val_summary,
        "regulatory_summary": reg_summary,
    }


def _generate_narrative(
    state: ResearchState, rating: str, price_target_dict: dict[str, Any]
) -> str:
    """Call Claude to write the narrative sections of the report.

    Falls back to an empty string if the API call fails, allowing the
    template-only report to still be assembled.

    Args:
        state: Fully-populated ResearchState.
        rating: Investment rating string.
        price_target_dict: Price target calculation results.

    Returns:
        Claude-generated narrative markdown or empty string on failure.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set; narrative sections will be skipped")
        return ""

    md = state["market_data"]
    fin = state["financials"]
    qa = state["quant_analysis"]

    # Prepare summaries with token limit safeguards
    summaries = _summarize_for_prompt(state, price_target_dict)

    # New IC fields: beta, avg_volume, enterprise_value, dividend_yield,
    # shares_outstanding, fcf_projections
    beta = md.get("beta")
    beta_str = f"{beta:.2f}" if beta is not None else "N/A"

    vol_analysis = md.get("volume_analysis") or {}
    avg_vol = vol_analysis.get("avg_volume")
    avg_vol_str = fmt_large_number(avg_vol, prefix="") if avg_vol else "N/A"

    market_cap_val = md.get("market_cap") or 0
    total_debt_val = fin.get("total_debt") or 0
    total_cash_val = fin.get("total_cash") or 0
    ev = market_cap_val + total_debt_val - total_cash_val
    ev_str = fmt_large_number(ev) if ev > 0 else "N/A"

    div_hist = md.get("dividend_history") or {}
    div_yield = div_hist.get("yield")
    div_yield_str = f"{div_yield:.1f}%" if div_yield is not None else "N/A"

    current_price_val = md.get("current_price") or 0
    shares = (
        market_cap_val / current_price_val
        if current_price_val > 0 and market_cap_val > 0
        else None
    )
    shares_str = fmt_large_number(shares, prefix="") if shares else "N/A"

    val = state.get("valuation_analysis") or {}
    fcf_proj = val.get("dcf", {}).get("fcf_projections", [])
    if fcf_proj:
        fcf_str = ", ".join(f"${v / 1e9:.2f}B" for v in fcf_proj[:5])
    else:
        fcf_str = "N/A (use revenue and earnings trends for estimates)"

    fmt_kwargs: dict[str, Any] = {
        "ticker": state["ticker"],
        "current_price": fmt_number(md.get("current_price"), "$"),
        "market_cap": fmt_large_number(md.get("market_cap")),
        "enterprise_value": ev_str,
        "wk_low": fmt_number(md.get("fifty_two_week_low"), "$"),
        "wk_high": fmt_number(md.get("fifty_two_week_high"), "$"),
        "beta": beta_str,
        "avg_volume": avg_vol_str,
        "shares_outstanding": shares_str,
        "dividend_yield": div_yield_str,
        "total_revenue": fmt_large_number(fin.get("total_revenue")),
        "net_income": fmt_large_number(fin.get("net_income")),
        "total_cash": fmt_large_number(fin.get("total_cash")),
        "total_debt": fmt_large_number(fin.get("total_debt")),
        "operating_cash_flow": fmt_large_number(
            fin.get("operating_cash_flow")
        ),
        "fcf_projections": fcf_str,
        "roe": fmt_pct(qa.get("roe")),
        "profit_margin": fmt_pct(qa.get("profit_margin")),
        "asset_turnover": fmt_ratio(qa.get("asset_turnover")),
        "equity_multiplier": fmt_ratio(qa.get("equity_multiplier")),
        "dupont_driver": qa.get("dupont_driver", "N/A"),
        "altman_z": fmt_ratio(qa.get("altman_z")),
        "altman_zone": qa.get("altman_zone", "Unknown"),
        "pe_ratio": fmt_ratio(qa.get("pe_ratio")),
        "price_target": fmt_number(
            price_target_dict.get("price_target"), "$"
        ),
        "bull_target": fmt_number(
            price_target_dict.get("bull_target"), "$"
        ),
        "base_target": fmt_number(
            price_target_dict.get("base_target"), "$"
        ),
        "bear_target": fmt_number(
            price_target_dict.get("bear_target"), "$"
        ),
        "upside_pct": f"{price_target_dict.get('upside_pct', 0):+.1f}%",
        "valuation_summary": summaries["valuation_summary"],
        "regulatory_summary": summaries["regulatory_summary"],
        "signal": qa.get("signal", "N/A"),
        "sentiment_score": f"{state['sentiment_score']:+.2f}",
        "sentiment_rationale": (
            state.get("sentiment_rationale") or "No sentiment data."
        ),
        "human_notes": (
            state["human_notes"] or "No analyst notes provided."
        ),
        "rating": rating,
        "cross_domain_insights": _format_insights_for_prompt(
            state.get("insights_board", [])
        ),
    }

    prompt = NARRATIVE_PROMPT.format(**fmt_kwargs)

    # Token limit safeguard (rough estimate: ~4 chars per token)
    if len(prompt) > 20000:  # ~5,000 tokens
        logger.warning(
            "NARRATIVE_PROMPT exceeded 5,000 tokens (%d chars), using trimmed data",
            len(prompt),
        )
        fmt_kwargs["valuation_summary"] = "- See Valuation Analysis section"
        fmt_kwargs["regulatory_summary"] = "- See Regulatory Assessment section"
        fmt_kwargs["bull_target"] = "(see Valuation)"
        fmt_kwargs["base_target"] = "(see Valuation)"
        fmt_kwargs["bear_target"] = "(see Valuation)"
        fmt_kwargs["fcf_projections"] = "N/A"
        prompt = NARRATIVE_PROMPT.format(**fmt_kwargs)

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in message.content:
            if block.type == "text":
                return block.text.strip()
        return ""
    except Exception as exc:
        logger.warning(
            "WriterAgent Claude call failed for %s: %s",
            state["ticker"], exc,
        )
        return ""


def _format_insights_for_prompt(insights: list[dict[str, Any]]) -> str:
    """Format insights board entries as a concise bullet list for prompts.

    Groups insights by category and formats them into a plain-text bullet
    list suitable for inclusion in a Claude prompt. Returns a placeholder
    string when the board is empty.

    Args:
        insights: List of insight dicts from state['insights_board'].

    Returns:
        Formatted multi-line string, or 'No cross-domain insights available.'
    """
    if not insights:
        return "No cross-domain insights available."

    by_category: dict[str, list[str]] = {}
    for insight in insights:
        cat = insight.get("category", "general")
        signal = insight.get("signal", "")
        agent = insight.get("agent", "")
        conf = insight.get("confidence", 0.0)
        entry = f"[{agent} | conf={conf:.0%}] {signal}"
        by_category.setdefault(cat, []).append(entry)

    lines: list[str] = []
    for cat, entries in by_category.items():
        lines.append(f"  {cat.upper()}:")
        for entry in entries:
            lines.append(f"    - {entry}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def _format_provenance_summary(prov_metadata: dict[str, Any]) -> str:
    """Generate Data Provenance Summary markdown section.

    Args:
        prov_metadata: Provenance metadata dict from ResearchState.

    Returns:
        Formatted markdown table showing data sources and confidence.
    """
    if not prov_metadata:
        return "## Data Provenance Summary\n\nNo provenance data available.\n"

    lines = ["## Data Provenance Summary\n"]
    lines.append("| Agent | Data Source | Timestamp | Confidence | Notes |")
    lines.append("|-------|-------------|-----------|------------|-------|")

    for key, metadata in prov_metadata.items():
        agent = metadata.get("agent", "Unknown")
        source = metadata.get("source", "Unknown")
        timestamp = metadata.get("timestamp", "N/A")
        confidence = metadata.get("confidence")
        conf_str = f"{confidence:.1f}%" if confidence is not None else "N/A"

        notes_parts = []
        if "headline_count" in metadata:
            notes_parts.append(f"{metadata['headline_count']} headlines")
        if "error_count" in metadata:
            notes_parts.append(f"{metadata['error_count']} errors")
        notes = ", ".join(notes_parts) if notes_parts else "—"

        lines.append(
            f"| {agent} | {source} | {timestamp} | {conf_str} | {notes} |"
        )

    return "\n".join(lines)


def _build_report(
    state: ResearchState,
    rating: str,
    narrative: str,
    price_target_dict: dict[str, Any],
) -> str:
    """Assemble the full institutional-grade IC Markdown report.

    Args:
        state: The fully-populated research state.
        rating: The calculated investment rating.
        narrative: Claude-generated prose sections (may be empty).
        price_target_dict: Price target calculation results.

    Returns:
        A formatted Markdown string (target: 8,000-15,000 words).
    """
    ticker = state["ticker"]
    qa = state["quant_analysis"]
    news = state["news"]
    sentiment = state["sentiment_score"]

    header_block = WriterAgent._build_header_block(state, price_target_dict, rating)
    valuation_section = WriterAgent._build_valuation_section(
        state, price_target_dict
    )
    narrative_was_generated = bool(narrative)
    confidence_section = WriterAgent._build_confidence_section(
        state, narrative_was_generated
    )

    # Narrative block — either Claude prose or fallback
    if narrative:
        narrative_block = narrative
    else:
        notes = state["human_notes"] or "No analyst notes provided."
        narrative_block = (
            f"## I. Investment Summary\n"
            f"{ticker} quant signal: **{qa.get('signal', 'N/A')}**. "
            f"Sentiment score: **{sentiment:+.2f}**. Rating: **{rating}**.\n\n"
            f"## II. Company Overview\n*(Claude narrative unavailable)*\n\n"
            f"## III. Industry & Competitive Positioning\n"
            f"*(Claude narrative unavailable)*\n\n"
            f"## IV. Investment Thesis & Catalysts\n"
            f"*(Claude narrative unavailable)*\n\n"
            f"## V. Financial Analysis\n*(Claude narrative unavailable)*\n\n"
            f"## VI. Valuation\n*(Claude narrative unavailable)*\n\n"
            f"## VII. Investment Risks\n{notes}"
        )

    altman_z = qa.get("altman_z")
    altman_zone = qa.get("altman_zone", "Unknown")
    altman_display = (
        f"{altman_z:.2f} — **{altman_zone}**" if altman_z is not None else "N/A"
    )

    headlines_md = (
        "\n".join(f"- {item.get('headline', 'N/A')}" for item in news)
        or "- No headlines available."
    )

    # Appendix sections
    dupont_appendix = (
        "### A. DuPont Analysis\n\n"
        "| Metric | Value |\n"
        "|--------|---------|\n"
        f"| Return on Equity (ROE) | {fmt_pct(qa.get('roe'))} |\n"
        f"| Net Profit Margin | {fmt_pct(qa.get('profit_margin'))} |\n"
        f"| Asset Turnover | {fmt_ratio(qa.get('asset_turnover'))} |\n"
        f"| Equity Multiplier | {fmt_ratio(qa.get('equity_multiplier'))} |\n"
        f"| ROE Driver | {qa.get('dupont_driver', 'N/A')} |\n"
        f"| P/E Ratio | {fmt_ratio(qa.get('pe_ratio'))} |\n"
        f"| Altman Z-Score | {altman_display} |\n"
    )

    sentiment_appendix = (
        f"### B. Sentiment Analysis\n\n"
        f"**Score: {sentiment:+.2f}**\n\n"
        f"{headlines_md}\n"
    )

    # Remap confidence and provenance headings for appendix nesting
    conf_text = confidence_section.replace(
        "## Data Quality & Confidence\n",
        "### C. Data Quality & Confidence\n",
        1,
    )

    prov = state.get("provenance_metadata", {})
    prov_text = _format_provenance_summary(prov).replace(
        "## Data Provenance Summary\n",
        "### D. Data Provenance\n",
        1,
    )

    pt = price_target_dict["price_target"]
    today = date.today().isoformat()

    report = f"""# {ticker} — Equity Research | Initiating Coverage

Rating: {rating} | 12-Mo Price Target: ${pt:.2f} | Date: {today}
Analyst: Doxa AI Research Platform

---

{header_block}

---

{narrative_block}

---

{valuation_section}

---

## Appendix

{dupont_appendix}

{sentiment_appendix}

{conf_text}

{prov_text}

### E. Disclosures & Methodology

This report is generated by Doxa, an AI-powered equity research
platform. All analysis is automated and should be validated by
qualified financial professionals before making investment decisions.

**Data Sources:** yfinance (market data), SEC EDGAR (regulatory
filings), Anthropic Claude (sentiment analysis, narrative generation)

**Methodology:** DCF valuation using 5-year FCF projections with
WACC-based discounting, comparable company analysis, DuPont ROE
decomposition, Altman Z-Score bankruptcy prediction, AI-powered
sentiment scoring.

**Disclaimer:** This report is for informational purposes only and
does not constitute investment advice. Past performance does not
guarantee future results.

---

*Report generated by Doxa — AI-Powered Equity Research*
"""

    # Quality check: count words (excluding tables)
    words = [
        w for line in report.split("\n")
        for w in line.split()
        if not line.strip().startswith("|")
    ]
    word_count = len(words)

    if word_count < 8000:
        logger.warning(
            "Report for %s below target length: %d non-table words "
            "(target: 8,000-15,000)",
            ticker,
            word_count,
        )
    elif word_count > 15000:
        logger.warning(
            "Report for %s exceeds target length: %d non-table words "
            "(target: 8,000-15,000)",
            ticker,
            word_count,
        )
    else:
        logger.info(
            "Report for %s meets quality benchmark: %d non-table words",
            ticker,
            word_count,
        )

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configure_logging()

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
        "total_assets": 111_000_000_000,
        "total_assets_prev": 65_000_000_000,
        "stockholders_equity": 42_000_000_000,
        "stockholders_equity_prev": 22_000_000_000,
        "total_liabilities": 69_000_000_000,
        "retained_earnings": 29_000_000_000,
        "ebit": 111_000_000_000,
        "working_capital": 35_000_000_000,
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
    state["sentiment_rationale"] = (
        "Market is reacting favorably to strong AI infrastructure demand. "
        "Key catalyst: Record quarterly earnings beat by 15%."
    )
    state["news"] = [
        {"headline": "NVDA announces strategic partnership with major cloud provider"},
        {"headline": "NVDA beats Q4 earnings estimates by 15%"},
        {"headline": "NVDA faces investigation over accounting practices"},
    ]
    state["human_notes"] = (
        "NVDA remains the dominant player in AI accelerators. "
        "Strong demand visibility through 2026. Risk is valuation."
    )

    agent = WriterAgent()
    result = agent.generate_report(state)
    print(state["final_report"])
