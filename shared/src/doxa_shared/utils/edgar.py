"""SEC EDGAR utility functions for Doxa.

This module provides helper functions for interacting with the SEC EDGAR
REST API to resolve company CIK numbers, fetch filing metadata, download
filing documents, and extract key sections from 10-K annual reports.

All functions include proper User-Agent headers as required by SEC EDGAR.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# SEC EDGAR requires a valid User-Agent header on all requests.
_SEC_HEADERS: dict[str, str] = {
    "User-Agent": "Doxa/1.0 (doxa-research@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# Module-level cache for the company tickers mapping (ticker -> CIK).
# Populated on first call to resolve_cik() and reused for subsequent calls.
# NOTE: Not thread-safe. For async/multi-threaded use (Epic 4), replace
# with a lock-protected cache or use functools.lru_cache.
_cik_cache: dict[str, str] | None = None

# SEC EDGAR rate limit: max 10 requests/second. The POC pipeline makes
# ~3 sequential requests per ticker which stays well under the limit.
# For batch processing (multiple tickers), callers must add throttling.


def resolve_cik(ticker: str) -> str | None:
    """Resolve a stock ticker to its SEC CIK (Central Index Key).

    Fetches the SEC company_tickers.json endpoint and caches the mapping
    in a module-level dict for subsequent calls.

    Args:
        ticker: Stock ticker symbol (case-insensitive).

    Returns:
        A 10-digit zero-padded CIK string, or None if not found.

    Raises:
        httpx.HTTPStatusError: If the SEC endpoint returns an error status.
        httpx.HTTPError: If a network-level error occurs.
    """
    global _cik_cache  # noqa: PLW0603

    if _cik_cache is None:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = httpx.get(url, headers=_SEC_HEADERS, timeout=15.0)
        resp.raise_for_status()
        data: dict[str, dict[str, Any]] = resp.json()

        _cik_cache = {}
        for entry in data.values():
            entry_ticker = str(entry.get("ticker", "")).upper()
            cik_str = str(entry.get("cik_str", ""))
            if entry_ticker and cik_str:
                _cik_cache[entry_ticker] = cik_str.zfill(10)

    return _cik_cache.get(ticker.upper().strip())


def fetch_recent_filings(
    cik: str,
    form_type: str = "10-K",
) -> list[dict[str, Any]]:
    """Fetch recent filing metadata for a company from SEC EDGAR.

    Args:
        cik: 10-digit zero-padded CIK string.
        form_type: SEC form type to filter by (default "10-K").

    Returns:
        A list of filing metadata dicts with keys: accession_number,
        filing_date, primary_document, form. Returns empty list on error.

    Raises:
        httpx.HTTPStatusError: If the SEC endpoint returns an error status.
        httpx.HTTPError: If a network-level error occurs.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = httpx.get(url, headers=_SEC_HEADERS, timeout=15.0)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()

    recent: dict[str, Any] = data.get("filings", {}).get("recent", {})
    forms: list[str] = recent.get("form", [])
    dates: list[str] = recent.get("filingDate", [])
    accessions: list[str] = recent.get("accessionNumber", [])
    documents: list[str] = recent.get("primaryDocument", [])

    filings: list[dict[str, Any]] = []
    for i, form in enumerate(forms):
        if form == form_type and i < len(dates):
            filings.append({
                "accession_number": accessions[i] if i < len(accessions) else "",
                "filing_date": dates[i] if i < len(dates) else "",
                "primary_document": documents[i] if i < len(documents) else "",
                "form": form,
            })

    return filings


def fetch_filing_text(
    cik: str,
    accession_number: str,
    primary_document: str,
) -> str:
    """Download the text content of a specific SEC filing document.

    Constructs the EDGAR archive URL and fetches the document. HTML tags
    are stripped to produce plain text suitable for analysis.

    Args:
        cik: 10-digit zero-padded CIK string.
        accession_number: Filing accession number (e.g. "0000320193-23-000106").
        primary_document: Filename of the primary document.

    Returns:
        Plain text content of the filing with HTML tags removed.

    Raises:
        httpx.HTTPStatusError: If the SEC endpoint returns an error status.
        httpx.HTTPError: If a network-level error occurs.
    """
    # EDGAR archive URLs use the integer CIK (no leading zeros)
    cik_int = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{accession_no_dashes}/{primary_document}"
    )
    resp = httpx.get(url, headers=_SEC_HEADERS, timeout=30.0)
    resp.raise_for_status()

    raw = resp.text
    return _strip_html(raw)


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities from text.

    Args:
        text: Raw HTML or plain text content.

    Returns:
        Plain text with HTML tags removed and entities decoded.
    """
    # Remove style and script blocks entirely
    cleaned = re.sub(
        r"<(style|script)[^>]*>.*?</\1>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Replace <br>, <p>, <div>, <tr> tags with newlines for readability
    cleaned = re.sub(
        r"<(?:br|/p|/div|/tr|/li)[^>]*>",
        "\n",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove all remaining HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Decode HTML entities
    cleaned = html.unescape(cleaned)
    # Collapse excessive whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


def extract_10k_sections(filing_text: str) -> dict[str, str]:
    """Extract key sections from a 10-K annual report filing.

    Uses regex patterns to locate Item 1A (Risk Factors), Item 3
    (Legal Proceedings), and Item 7 (MD&A). 10-K HTML formatting is
    wildly inconsistent across filers, so extraction is best-effort.

    Args:
        filing_text: Plain text content of the 10-K filing.

    Returns:
        A dict with keys ``risk_factors``, ``legal_proceedings``, and
        ``md_and_a``. Values are the extracted section text, or empty
        strings if a section could not be found.
    """
    sections: dict[str, str] = {
        "risk_factors": "",
        "legal_proceedings": "",
        "md_and_a": "",
    }

    sections["risk_factors"] = _extract_section(
        filing_text,
        start_patterns=[
            r"Item\s*1A[\.\s\u00a0]*[\-\u2014]*\s*Risk\s+Factors",
        ],
        end_patterns=[
            r"Item\s*1B[\.\s\u00a0]",
            r"Item\s*2[\.\s\u00a0]*[\-\u2014]*\s*Properties",
        ],
    )

    sections["legal_proceedings"] = _extract_section(
        filing_text,
        start_patterns=[
            r"Item\s*3[\.\s\u00a0]*[\-\u2014]*\s*Legal\s+Proceedings",
        ],
        end_patterns=[
            r"Item\s*4[\.\s\u00a0]",
            r"Item\s*5[\.\s\u00a0]",
        ],
    )

    sections["md_and_a"] = _extract_section(
        filing_text,
        start_patterns=[
            r"Item\s*7[\.\s\u00a0]*[\-\u2014]*\s*Management",
        ],
        end_patterns=[
            r"Item\s*7A[\.\s\u00a0]",
            r"Item\s*8[\.\s\u00a0]",
        ],
    )

    return sections


def fetch_multi_year_10k(cik: str, years: int = 3) -> list[dict[str, Any]]:
    """Fetch multiple years of 10-K filings with extracted Risk Factors.

    Fetches the last N years of 10-K filings for a company and extracts
    the Risk Factors (Item 1A) section from each filing. Used for year-over-year
    risk evolution analysis.

    Args:
        cik: 10-digit zero-padded CIK string.
        years: Number of years of 10-K filings to fetch (default 3).

    Returns:
        A list of filing dicts with keys:
        - year: int - Filing year extracted from filing_date
        - filing_date: str - Filing date in YYYY-MM-DD format
        - risk_factors_text: str - Extracted Risk Factors section text

        Returns empty list on API failures. If fewer than N years available
        (e.g., recent IPO), returns all available filings.
        List is sorted newest to oldest by filing_date.

    Example:
        >>> filings = fetch_multi_year_10k("0000320193", years=3)
        >>> len(filings)
        3
        >>> filings[0]["year"]
        2024
        >>> "competition" in filings[0]["risk_factors_text"]
        True
    """
    try:
        # Fetch filing metadata for all 10-K filings
        filings = fetch_recent_filings(cik, form_type="10-K")

        # Limit to requested number of years
        filings = filings[:years]

        results: list[dict[str, Any]] = []
        for filing in filings:
            accession = filing.get("accession_number", "")
            filing_date = filing.get("filing_date", "")
            primary_doc = filing.get("primary_document", "")

            if not all([accession, filing_date, primary_doc]):
                continue

            # Extract year from filing_date (format: YYYY-MM-DD)
            year = int(filing_date.split("-")[0])

            # Download filing text
            filing_text = fetch_filing_text(cik, accession, primary_doc)

            # Extract sections (we only need Risk Factors)
            sections = extract_10k_sections(filing_text)
            risk_factors = sections.get("risk_factors", "")

            results.append({
                "year": year,
                "filing_date": filing_date,
                "risk_factors_text": risk_factors,
            })

        return results

    except Exception as exc:
        # On any failure, log warning and return empty list (graceful degradation)
        logger.warning(
            "Multi-year 10-K fetch failed for CIK %s: %s",
            cik,
            exc,
        )
        return []


def _extract_section(
    text: str,
    start_patterns: list[str],
    end_patterns: list[str],
) -> str:
    """Extract a section of text between start and end header patterns.

    Finds ALL occurrences of the start pattern and picks the one that
    yields the longest content. This skips Table of Contents entries
    (which match the header but contain only page numbers).

    Args:
        text: The full filing plain text.
        start_patterns: Regex patterns for the section header.
        end_patterns: Regex patterns for the next section header.

    Returns:
        The extracted section text, or empty string if not found.
    """
    # Collect all start matches across all patterns
    all_starts: list[re.Match[str]] = []
    for pattern in start_patterns:
        all_starts.extend(re.finditer(pattern, text, re.IGNORECASE))

    if not all_starts:
        return ""

    # For each start match, find the end and measure content length.
    # Pick the longest result (the actual section, not a TOC entry).
    best_section = ""
    for start_match in all_starts:
        start_pos = start_match.end()
        end_pos = len(text)

        for pattern in end_patterns:
            end_match = re.search(
                pattern, text[start_pos:], re.IGNORECASE,
            )
            if end_match:
                candidate = start_pos + end_match.start()
                if candidate < end_pos:
                    end_pos = candidate

        section = text[start_pos:end_pos].strip()
        if len(section) > len(best_section):
            best_section = section

    return best_section
