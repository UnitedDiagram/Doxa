"""Render Doxa research reports to sell-side styled PDFs.

Converts the WriterAgent's markdown report (`state["final_report"]`) into a
professionally formatted PDF using WeasyPrint. The cover page is built from
structured `ResearchState` fields rather than parsed markdown, so the rating
box and key-stats panel use authoritative data.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import sys
from html import escape
from pathlib import Path
from typing import Any

import markdown as md_lib
from doxa_shared.utils.formatters import (
    fmt_large_number,
    fmt_number,
    fmt_pct,
    fmt_ratio,
)

from src.state import ResearchState

logger = logging.getLogger(__name__)

_STYLES_PATH = Path(__file__).parent / "styles" / "sell_side.css"

_MIFID_DISCLOSURE = (
    "This document is investment research produced by Doxa Research. "
    "It is provided for informational purposes and is not investment advice. "
    "Charges for this research are disclosed separately in accordance with "
    "MiFID II Article 13. Doxa Research is an AI-assisted research "
    "platform; outputs are reviewed by senior human analysts. Recipients "
    "should consult their own advisers before acting on any content herein."
)


class WeasyPrintUnavailableError(RuntimeError):
    """Raised when WeasyPrint cannot be imported (missing system libs)."""


def render_report_pdf(state: ResearchState) -> bytes:
    """Render a ResearchState's final report to a sell-side-styled PDF.

    Args:
        state: Fully-populated ResearchState with `final_report` markdown.

    Returns:
        PDF file contents as bytes.

    Raises:
        WeasyPrintUnavailableError: If WeasyPrint system libraries are missing
            (typically requires ``brew install pango`` on macOS).
        ValueError: If `state["final_report"]` is empty.
    """
    report_md = state.get("final_report", "")
    if not report_md.strip():
        raise ValueError("state['final_report'] is empty; nothing to render")

    _ensure_macos_homebrew_dyld_path()

    try:
        from weasyprint import CSS, HTML  # noqa: PLC0415
    except OSError as e:
        raise WeasyPrintUnavailableError(
            "WeasyPrint requires native libraries (pango, cairo). "
            "On macOS run: `brew install pango`. "
            f"Original error: {e}"
        ) from e

    cover_data = _extract_cover_data(state)
    body_md = _strip_title_and_snapshot(report_md)
    body_html = md_lib.markdown(
        body_md,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )

    full_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>"
        + _build_cover_html(cover_data)
        + "<main>"
        + body_html
        + _build_disclosures_html(state)
        + "</main></body></html>"
    )

    css = CSS(filename=str(_STYLES_PATH))
    return bytes(HTML(string=full_html).write_pdf(stylesheets=[css]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_macos_homebrew_dyld_path() -> None:
    """Add Homebrew lib dirs to DYLD_FALLBACK_LIBRARY_PATH on macOS.

    The python.org Python build doesn't search ``/opt/homebrew/lib`` by
    default, so WeasyPrint's ``dlopen`` of libgobject/pango fails even when
    the user has run ``brew install pango``. Setting the env var before the
    cffi dlopen call resolves this transparently.
    """
    if sys.platform != "darwin":
        return
    extra = [p for p in ("/opt/homebrew/lib", "/usr/local/lib") if Path(p).exists()]
    if not extra:
        return
    existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if any(p in existing.split(":") for p in extra):
        return
    parts = [*extra, existing] if existing else extra
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(parts)


def _extract_cover_data(state: ResearchState) -> dict[str, str]:
    """Pull structured cover-page fields from state with safe defaults.

    Rating and price target are parsed from the report header line emitted
    by WriterAgent (`Rating: X | 12-Mo Price Target: $Y | Date: Z`) to avoid
    duplicating the rating logic.

    Args:
        state: ResearchState (may have missing/None fields).

    Returns:
        Dict of strings safe for HTML embedding (already formatted).
    """
    md = state.get("market_data") or {}
    val = state.get("valuation_analysis") or {}
    peer = md.get("peer_comparison") or {}
    stock_metrics = peer.get("stock_metrics") or {}
    altman = val.get("altman_z_score") or {}
    report = state.get("final_report", "")

    rating = _parse_header_field(report, "Rating") or "N/A"
    price_target_raw = _parse_header_field(report, "12-Mo Price Target")
    report_date = _parse_header_field(report, "Date") or _dt.date.today().isoformat()

    current_price = md.get("current_price")
    try:
        pt_value = float((price_target_raw or "").replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        pt_value = None

    upside_pct: float | None = None
    if pt_value is not None and current_price:
        try:
            upside_pct = (pt_value - float(current_price)) / float(current_price)
        except (TypeError, ValueError, ZeroDivisionError):
            upside_pct = None

    return {
        "ticker": state.get("ticker", "N/A"),
        "company": md.get("company_name") or "",
        "sector": peer.get("sector") or peer.get("industry") or "",
        "report_type": "Initiating Coverage",
        "report_date": report_date,
        "rating": rating,
        "rating_class": _rating_css_class(rating),
        "price_target": fmt_number(pt_value, "$") if pt_value is not None else "N/A",
        "current_price": fmt_number(current_price, "$"),
        "upside": fmt_pct(upside_pct) if upside_pct is not None else "N/A",
        "upside_sign": (
            "pos" if (upside_pct is not None and upside_pct >= 0) else "neg"
        ),
        "market_cap": fmt_large_number(md.get("market_cap")),
        "enterprise_value": fmt_large_number(md.get("enterprise_value")),
        "fifty_two_high": fmt_number(md.get("fifty_two_week_high"), "$"),
        "fifty_two_low": fmt_number(md.get("fifty_two_week_low"), "$"),
        "beta": fmt_ratio(md.get("beta")) if md.get("beta") is not None else "N/A",
        "shares_out": fmt_large_number(md.get("shares_outstanding"), prefix=""),
        "dividend_yield": fmt_pct(md.get("dividend_yield")),
        "pe_ratio": fmt_ratio(stock_metrics.get("pe_trailing")),
        "altman_z": (
            f"{altman['z_score']:.2f}"
            if altman.get("z_score") is not None
            else "N/A"
        ),
        "summary": _extract_investment_summary(report),
        "analyst_name": "Doxa Research Team",
        "analyst_contact": "research@example.com",
        "confidence": _format_confidence(val.get("confidence")),
    }


def _strip_title_and_snapshot(markdown_text: str) -> str:
    """Remove the leading ``# title`` block and the first ``## Snapshot`` table.

    The PDF cover page replaces both, so they are stripped to avoid duplication.

    Args:
        markdown_text: Raw markdown from WriterAgent.

    Returns:
        Markdown with the title (and its meta line) and Snapshot section removed.
    """
    # Drop the leading "# TICKER ..." line and any immediately following
    # non-section lines (date, rating, separator) until we hit a "## " header.
    lines = markdown_text.splitlines()
    out: list[str] = []
    i = 0
    if i < len(lines) and lines[i].startswith("# "):
        i += 1
        while i < len(lines) and not lines[i].startswith("## "):
            i += 1
    # Skip the "## Snapshot" section if present at this position.
    if i < len(lines) and lines[i].strip().lower().startswith("## snapshot"):
        i += 1
        while i < len(lines) and not lines[i].startswith("## "):
            i += 1
    out.extend(lines[i:])
    return "\n".join(out).lstrip()


def _build_cover_html(c: dict[str, str]) -> str:
    """Render the cover page HTML from extracted cover data."""
    sector_html = (
        f'<div class="cover-sector">{escape(c["sector"])}</div>'
        if c["sector"]
        else ""
    )
    company_html = (
        f'<div class="cover-company">{escape(c["company"])}</div>'
        if c["company"]
        else ""
    )
    summary_html = (
        f'<div class="cover-summary"><h3>Investment Summary</h3>'
        f"<p>{escape(c['summary'])}</p></div>"
        if c["summary"]
        else ""
    )

    return f"""
    <section class="cover"
        data-ticker="{escape(c['ticker'])}"
        data-company="{escape(c['company'] or c['ticker'])}"
        data-type="{escape(c['report_type'])}"
        data-date="{escape(c['report_date'])}">
      <div class="cover-header">
        <div class="cover-firm">DOXA RESEARCH</div>
        <div class="cover-type">
          {escape(c['report_type'])} · {escape(c['report_date'])}
        </div>
      </div>

      <div class="cover-title-block">
        <h1 class="cover-ticker">{escape(c['ticker'])}</h1>
        {company_html}
        {sector_html}
      </div>

      <div class="cover-grid">
        <div class="rating-box {escape(c['rating_class'])}">
          <div class="rating-label">Recommendation</div>
          <div class="rating-value">{escape(c['rating'])}</div>
          <div class="rating-row">
            <span class="label">12-Mo Price Target</span>
            <span class="value">{escape(c['price_target'])}</span>
          </div>
          <div class="rating-row">
            <span class="label">Current Price</span>
            <span class="value">{escape(c['current_price'])}</span>
          </div>
          <div class="rating-row">
            <span class="label">Implied Upside</span>
            <span class="value upside {escape(c['upside_sign'])}">
              {escape(c['upside'])}
            </span>
          </div>
          <div class="rating-row">
            <span class="label">Report Date</span>
            <span class="value">{escape(c['report_date'])}</span>
          </div>
        </div>

        <div class="key-stats">
          <h3>Key Statistics</h3>
          {_stat_row("Market Cap", c["market_cap"])}
          {_stat_row("Enterprise Value", c["enterprise_value"])}
          {_stat_row("52-Wk High", c["fifty_two_high"])}
          {_stat_row("52-Wk Low", c["fifty_two_low"])}
          {_stat_row("Beta", c["beta"])}
          {_stat_row("Shares Outstanding", c["shares_out"])}
          {_stat_row("Dividend Yield", c["dividend_yield"])}
          {_stat_row("P/E (TTM)", c["pe_ratio"])}
          {_stat_row("Altman Z-Score", c["altman_z"])}
        </div>
      </div>

      <div class="cover-analyst">
        <div class="name">{escape(c['analyst_name'])}</div>
        <div>
          {escape(c['analyst_contact'])} · Confidence: {escape(c['confidence'])}
        </div>
      </div>

      {summary_html}

      <div class="cover-compliance">
        {escape(_MIFID_DISCLOSURE)}
      </div>
    </section>
    """


def _build_disclosures_html(state: ResearchState) -> str:
    """Build the final disclosures + methodology page."""
    errors = state.get("errors") or []
    error_html = ""
    if errors:
        items = "".join(f"<li>{escape(str(e))}</li>" for e in errors[:20])
        error_html = (
            "<h3>Data Quality Notices</h3>"
            f"<ul>{items}</ul>"
        )

    return f"""
    <section class="disclosures">
      <h2>Disclosures &amp; Methodology</h2>

      <h3>MiFID II Disclosure</h3>
      <p>{escape(_MIFID_DISCLOSURE)}</p>

      <h3>Analyst Certification</h3>
      <p>The views expressed in this report were generated by Doxa's
      multi-agent research pipeline and reviewed by senior human analysts
      prior to publication. No part of analyst compensation is directly
      tied to any specific recommendation in this report.</p>

      <h3>Methodology</h3>
      <p>This report was produced by a six-agent pipeline covering market
      data, fundamental valuation (DCF, comparable companies, Bull/Base/Bear
      scenarios), SEC EDGAR regulatory review, alternative-data sentiment
      analysis, and editorial distillation. Ratings reflect 12-month price
      target upside (Strong Buy &gt; 20%, Buy 10–20%, Hold ±10%, Sell &lt;
      -10%) with quantitative tiebreakers near thresholds.</p>

      <h3>Data Sources</h3>
      <p>Yahoo Finance (market data, fundamentals), SEC EDGAR (10-K filings,
      insider transactions), curated alternative-data feeds (insider trading,
      short interest, news sentiment). All data is as of the report date
      printed on the cover page unless otherwise noted.</p>

      <h3>Limitations</h3>
      <p>Forward-looking statements involve risks and uncertainties. Past
      performance is not indicative of future results. Doxa Research
      does not undertake to update this report after publication.</p>

      {error_html}
    </section>
    """


def _stat_row(label: str, value: str) -> str:
    return (
        f'<div class="stat-row"><span class="label">{escape(label)}</span>'
        f'<span class="value">{escape(value)}</span></div>'
    )


def _parse_header_field(report: str, field: str) -> str | None:
    """Extract a value from the WriterAgent header line.

    The header line format is:
        ``Rating: {rating} | 12-Mo Price Target: ${pt} | Date: {date}``
    """
    pattern = rf"{re.escape(field)}\s*:\s*([^|\n]+)"
    m = re.search(pattern, report)
    if not m:
        return None
    return m.group(1).strip().rstrip("*").strip()


def _rating_css_class(rating: str) -> str:
    r = rating.lower()
    if "strong buy" in r or "buy" in r:
        return "buy"
    if "sell" in r:
        return "sell"
    return "hold"


def _extract_investment_summary(report: str, max_chars: int = 700) -> str:
    """Pull the first paragraph of the Investment Summary section."""
    m = re.search(
        r"##\s*(?:I\.\s*)?Investment\s+Summary\s*\n+(.+?)(?=\n##\s)",
        report,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return ""
    body = m.group(1).strip()
    # First non-empty paragraph
    paragraph = next((p for p in body.split("\n\n") if p.strip()), "")
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    # The cover renders plain text, so drop markdown emphasis markers.
    paragraph = paragraph.replace("**", "")
    if len(paragraph) > max_chars:
        paragraph = paragraph[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return paragraph


def _format_confidence(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.0f}%"
    except (TypeError, ValueError):
        return "N/A"
