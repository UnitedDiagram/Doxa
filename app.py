"""Doxa Streamlit UI — web front-end for the equity research pipeline."""

from __future__ import annotations

import asyncio

import streamlit as st

from src.agents.market_data import MarketDataAgent
from src.agents.regulatory import RegulatoryAgent
from src.agents.sentiment import SentimentAgent
from src.agents.valuation import ValuationAgent
from src.agents.writer import WriterAgent
from src.state import create_initial_state

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Doxa",
    page_icon="📈",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "research_state" not in st.session_state:
    st.session_state.research_state = None

# ---------------------------------------------------------------------------
# Helper: signal badge colour
# ---------------------------------------------------------------------------

_SIGNAL_COLOUR = {
    "BULLISH": "green",
    "BEARISH": "red",
    "NEUTRAL": "orange",
}


def _signal_badge(signal: str) -> str:
    colour = _SIGNAL_COLOUR.get(signal, "grey")
    return (
        f'<span style="background:{colour};color:white;'
        f'padding:2px 10px;border-radius:4px;font-weight:bold">'
        f"{signal}</span>"
    )


def _fmt_number(value: object, prefix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{prefix}{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_ratio(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# Stage 1 — ticker input + agent pipeline
# ---------------------------------------------------------------------------

def render_input_stage() -> None:
    """Render the initial ticker input stage with agent pipeline execution."""
    st.title("📈 Doxa")
    st.caption("AI-powered equity research in seconds.")

    ticker = st.text_input(
        "Ticker symbol",
        placeholder="e.g. AAPL, NVDA, MSFT",
        key="ticker_input",
    ).strip().upper()

    if st.button("Run Analysis", type="primary", disabled=not ticker):
        state = create_initial_state(ticker)

        with st.status(f"Running analysis for **{ticker}**…", expanded=True) as status:
            # Agent 1 — Market Data
            st.write("🔍 Fetching market data…")
            state = MarketDataAgent().fetch_data(state)
            price = state["market_data"].get("current_price")
            st.write(
                f"✅ Market data fetched — current price: "
                f"{_fmt_number(price, '$')}"
            )

            # Agent 2 — Valuation (includes quantitative analysis)
            st.write("💰 Calculating valuation and quantitative analysis…")
            state = ValuationAgent().execute(state)
            valuation = state.get("valuation_analysis", {})
            dcf = valuation.get("dcf", {})
            fair_value = dcf.get("fair_value_per_share", 0)
            confidence = valuation.get("confidence", 0)
            st.write(
                f"✅ Valuation complete — DCF fair value: **${fair_value:.2f}** "
                f"| Confidence: **{confidence:.1f}%**"
            )

            # Agent 4 — Regulatory
            st.write("📜 Analyzing SEC filings…")
            state = RegulatoryAgent().analyze(state)
            reg = state.get("regulatory_analysis", {})
            reg_score = reg.get("risk_score", "N/A")
            reg_conf = reg.get("confidence", 0)
            st.write(
                f"✅ Regulatory analysis complete — "
                f"risk: **{reg_score}** | "
                f"Confidence: **{reg_conf:.0f}%**"
            )

            # Agent 5 — Sentiment
            st.write("🧠 Analysing news sentiment…")
            state = asyncio.run(SentimentAgent().analyze(state))
            score = state["sentiment_score"]
            st.write(f"✅ Sentiment scored: **{score:+.2f}**")

            status.update(label="Data collection complete!", state="complete")

        st.session_state.research_state = state
        st.session_state.stage = "collecting_notes"
        st.rerun()


# ---------------------------------------------------------------------------
# Stage 2 — summary tabs + analyst notes
# ---------------------------------------------------------------------------

def render_collecting_notes_stage() -> None:
    """Render the analyst notes collection stage with research summary tabs."""
    state = st.session_state.research_state
    ticker = state["ticker"]

    st.title(f"📋 {ticker} — Research Summary")

    md = state["market_data"]
    qa = state["quant_analysis"]
    news = state["news"]
    signal = qa.get("signal", "N/A")
    score = state["sentiment_score"]

    tab_market, tab_quant, tab_valuation, tab_regulatory, tab_sentiment = st.tabs(
        ["Market Data", "Quant Analysis", "Valuation", "Regulatory", "Sentiment"]
    )

    with tab_market:
        col1, col2 = st.columns(2)
        col1.metric("Current Price", _fmt_number(md.get("current_price"), "$"))
        col2.metric("Market Cap", _fmt_number(md.get("market_cap"), "$"))
        col1.metric("52-Week High", _fmt_number(md.get("fifty_two_week_high"), "$"))
        col2.metric("52-Week Low", _fmt_number(md.get("fifty_two_week_low"), "$"))

    with tab_quant:
        st.markdown(
            f"**Signal:** {_signal_badge(signal)}",
            unsafe_allow_html=True,
        )
        # DuPont Analysis
        st.subheader("DuPont Analysis")
        col1, col2 = st.columns(2)
        col1.metric("Return on Equity (ROE)", _fmt_pct(qa.get("roe")))
        col2.metric("Net Profit Margin", _fmt_pct(qa.get("profit_margin")))
        col1.metric("Asset Turnover", _fmt_ratio(qa.get("asset_turnover")))
        col2.metric("Equity Multiplier", _fmt_ratio(qa.get("equity_multiplier")))
        if qa.get("dupont_driver"):
            st.caption(f"ROE Driver: **{qa['dupont_driver']}**")
        # Altman Z-Score
        st.subheader("Bankruptcy Risk")
        altman_z = qa.get("altman_z")
        altman_zone = qa.get("altman_zone", "Unknown")
        zone_colours = {
            "Safe": "green",
            "Grey": "orange",
            "Distress": "red",
        }
        zone_colour = zone_colours.get(altman_zone, "grey")
        z_display = f"{altman_z:.2f}" if altman_z is not None else "N/A"
        st.markdown(
            f"**Altman Z-Score:** {z_display} — "
            f'<span style="color:{zone_colour};font-weight:bold">{altman_zone}</span>',
            unsafe_allow_html=True,
        )
        st.caption("Safe > 2.99 | Grey 1.81–2.99 | Distress < 1.81")
        # Valuation
        st.subheader("Valuation")
        col1, col2 = st.columns(2)
        col1.metric("P/E Ratio", _fmt_ratio(qa.get("pe_ratio")))

    with tab_valuation:
        valuation = state.get("valuation_analysis", {})
        confidence = valuation.get("confidence", 0)
        st.markdown(f"**Confidence:** {confidence:.1f}%")

        # DCF Analysis
        dcf = valuation.get("dcf", {})
        if dcf:
            st.subheader("DCF Valuation")
            col1, col2 = st.columns(2)
            col1.metric("Fair Value", _fmt_number(dcf.get("fair_value_per_share"), "$"))
            col2.metric("Current Price", _fmt_number(dcf.get("current_price"), "$"))
            upside = dcf.get("upside_downside_pct", 0)
            upside_color = "green" if upside > 0 else "red"
            st.markdown(
                f"**Upside/Downside:** "
                f'<span style="color:{upside_color};font-weight:bold">'
                f"{upside:+.1f}%</span>",
                unsafe_allow_html=True,
            )
            col1.metric("WACC", _fmt_pct(dcf.get("wacc")))
            col2.metric("Terminal Value", _fmt_number(dcf.get("terminal_value"), "$"))

        # Comparable Company Analysis
        comps = valuation.get("comps", {})
        if comps:
            st.subheader("Comparable Companies")
            peers = comps.get("peer_companies", [])
            if peers:
                st.caption(f"Peers: {', '.join(peers)}")
            median_multiples = comps.get("median_multiples", {})
            if median_multiples:
                col1, col2 = st.columns(2)
                col1.metric("Median P/E", _fmt_ratio(median_multiples.get("P/E")))
                col2.metric(
                    "Median EV/EBITDA", _fmt_ratio(median_multiples.get("EV/EBITDA"))
                )

    with tab_regulatory:
        reg = state.get("regulatory_analysis", {})
        reg_confidence = reg.get("confidence", 0)
        st.markdown(f"**Confidence:** {reg_confidence:.0f}%")

        risk_score = reg.get("risk_score", "N/A")
        risk_colour = {
            "Low": "green",
            "Medium": "orange",
            "High": "red",
        }.get(risk_score, "grey")
        st.markdown(
            f"**Risk Score:** "
            f'<span style="color:{risk_colour};'
            f'font-size:1.2em;font-weight:bold">'
            f"{risk_score}</span>",
            unsafe_allow_html=True,
        )

        filing_date = reg.get("filing_date", "")
        if filing_date:
            st.caption(f"Most recent 10-K filing: {filing_date}")

        cik = reg.get("cik", "")
        if cik:
            st.caption(f"CIK: {cik}")

        risk_factors = reg.get("risk_factors", [])
        if risk_factors:
            st.subheader("Top Regulatory Risks")
            for i, risk in enumerate(risk_factors, 1):
                st.markdown(f"{i}. {risk}")

        legal = reg.get("legal_proceedings", "")
        if legal and legal != "No data available.":
            st.subheader("Legal Proceedings")
            st.markdown(legal)

        filing_url = reg.get("filing_url", "")
        if filing_url:
            st.caption(f"[View filing on SEC EDGAR]({filing_url})")

    with tab_sentiment:
        # Simple colour-coded score
        colour = "green" if score > 0 else ("red" if score < 0 else "grey")
        st.markdown(
            f"**Sentiment Score:** "
            f'<span style="color:{colour};font-size:1.4em;font-weight:bold">'
            f"{score:+.2f}</span>",
            unsafe_allow_html=True,
        )
        if state.get("sentiment_rationale"):
            st.caption(state["sentiment_rationale"])
        st.divider()
        if news:
            st.subheader("Recent Headlines")
            for item in news:
                headline = item.get("headline", "")
                url = item.get("url", "")
                publisher = item.get("publisher", "")
                pub_suffix = f" *({publisher})*" if publisher else ""
                if url:
                    st.markdown(f"- [{headline}]({url}){pub_suffix}")
                else:
                    st.markdown(f"- {headline}{pub_suffix}")
        else:
            st.info("No headlines available.")

    st.divider()

    notes = st.text_area(
        "Your analyst notes",
        placeholder="Add your subjective take on this stock…",
        height=120,
        key="analyst_notes",
    )

    if st.button("Generate Report", type="primary"):
        state["human_notes"] = notes

        with st.status("Generating equity research report…", expanded=True) as status:
            st.write("✍️ WriterAgent composing report…")
            state = WriterAgent().generate_report(state)
            st.write("✅ Report ready.")
            status.update(label="Report generated!", state="complete")

        st.session_state.research_state = state
        st.session_state.stage = "report_ready"
        st.rerun()


# ---------------------------------------------------------------------------
# Stage 3 — report display
# ---------------------------------------------------------------------------

def render_report_stage() -> None:
    """Render the final report stage with generated Markdown report."""
    state = st.session_state.research_state
    ticker = state["ticker"]
    report = state["final_report"]

    st.title(f"📄 {ticker} — Equity Research Report")

    st.markdown(report)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=report,
            file_name=f"{ticker}_equity_research.txt",
            mime="text/plain",
        )

    with col2:
        try:
            from src.export.pdf_export import (
                WeasyPrintUnavailableError,
                render_report_pdf,
            )

            pdf_bytes = render_report_pdf(state)
            st.download_button(
                label="📄 Download Report (.pdf)",
                data=pdf_bytes,
                file_name=f"{ticker}_equity_research.pdf",
                mime="application/pdf",
            )
        except WeasyPrintUnavailableError as e:
            st.info(
                "PDF export requires system libraries. On macOS run "
                "`brew install pango`, then restart Streamlit."
            )
            st.caption(str(e).split("Original error:")[0].strip())
        except Exception as e:  # noqa: BLE001 - never crash the report stage
            st.warning(f"PDF export failed: {e}")

    with col3:
        if st.button("🔄 New Analysis"):
            st.session_state.stage = "input"
            st.session_state.research_state = None
            st.rerun()

    if state.get("errors"):
        with st.expander(f"⚠️ Warnings / Errors ({len(state['errors'])})"):
            for err in state["errors"]:
                st.warning(err)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

stage = st.session_state.stage

if stage == "input":
    render_input_stage()
elif stage == "collecting_notes":
    render_collecting_notes_stage()
elif stage == "report_ready":
    render_report_stage()
