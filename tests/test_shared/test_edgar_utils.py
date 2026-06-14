"""Tests for shared SEC EDGAR utility functions."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from doxa_shared.utils.edgar import (
    _strip_html,
    extract_10k_sections,
    fetch_filing_text,
    fetch_recent_filings,
    resolve_cik,
)

# ---------------------------------------------------------------------------
# Sample data for mocks
# ---------------------------------------------------------------------------

SAMPLE_COMPANY_TICKERS = {
    "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": "789019", "ticker": "MSFT", "title": "Microsoft Corp"},
    "2": {"cik_str": "1018724", "ticker": "AMZN", "title": "Amazon.com Inc"},
}

SAMPLE_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "10-K", "8-K"],
            "filingDate": [
                "2024-11-01",
                "2024-08-15",
                "2023-11-03",
                "2024-06-01",
            ],
            "accessionNumber": [
                "0000320193-24-000123",
                "0000320193-24-000100",
                "0000320193-23-000106",
                "0000320193-24-000050",
            ],
            "primaryDocument": [
                "aapl-20240928.htm",
                "aapl-10q.htm",
                "aapl-20230930.htm",
                "aapl-8k.htm",
            ],
        }
    }
}

SAMPLE_10K_HTML = """
<html><body>
<h2>Item 1A. Risk Factors</h2>
<p>The Company is subject to various regulatory risks that could
affect its business operations and financial results.</p>
<p>Competition in the technology sector remains intense and the
Company must continue to innovate to maintain market position.</p>
<h2>Item 1B. Unresolved Staff Comments</h2>
<p>None.</p>
<h2>Item 3. Legal Proceedings</h2>
<p>The Company is currently involved in various lawsuits and
claims arising in the ordinary course of business.</p>
<h2>Item 4. Mine Safety Disclosures</h2>
<p>Not applicable.</p>
<h2>Item 7. Management's Discussion and Analysis</h2>
<p>Revenue increased 8% year-over-year driven by strong
performance in the services segment.</p>
<h2>Item 7A. Quantitative Disclosures</h2>
</body></html>
"""


# ---------------------------------------------------------------------------
# resolve_cik tests
# ---------------------------------------------------------------------------


class TestResolveCik:
    """Tests for resolve_cik function."""

    def setup_method(self) -> None:
        """Clear CIK cache between tests."""
        import doxa_shared.utils.edgar as edgar_module

        edgar_module._cik_cache = None

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_resolves_known_ticker(self, mock_get: Mock) -> None:
        mock_resp = Mock()
        mock_resp.json.return_value = SAMPLE_COMPANY_TICKERS
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        result = resolve_cik("AAPL")
        assert result == "0000320193"

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_resolves_case_insensitive(self, mock_get: Mock) -> None:
        mock_resp = Mock()
        mock_resp.json.return_value = SAMPLE_COMPANY_TICKERS
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        result = resolve_cik("aapl")
        assert result == "0000320193"

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_returns_none_for_unknown_ticker(
        self, mock_get: Mock,
    ) -> None:
        mock_resp = Mock()
        mock_resp.json.return_value = SAMPLE_COMPANY_TICKERS
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        result = resolve_cik("ZZZZZ")
        assert result is None

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_caches_after_first_call(self, mock_get: Mock) -> None:
        mock_resp = Mock()
        mock_resp.json.return_value = SAMPLE_COMPANY_TICKERS
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        resolve_cik("AAPL")
        resolve_cik("MSFT")
        # Should only call the endpoint once
        assert mock_get.call_count == 1

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_zero_pads_cik_to_10_digits(self, mock_get: Mock) -> None:
        mock_resp = Mock()
        mock_resp.json.return_value = SAMPLE_COMPANY_TICKERS
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        result = resolve_cik("AAPL")
        assert result is not None
        assert len(result) == 10

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_raises_on_network_error(self, mock_get: Mock) -> None:
        import httpx

        mock_get.side_effect = httpx.HTTPError("Connection failed")

        with pytest.raises(httpx.HTTPError):
            resolve_cik("AAPL")


# ---------------------------------------------------------------------------
# fetch_recent_filings tests
# ---------------------------------------------------------------------------


class TestFetchRecentFilings:
    """Tests for fetch_recent_filings function."""

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_filters_10k_filings(self, mock_get: Mock) -> None:
        mock_resp = Mock()
        mock_resp.json.return_value = SAMPLE_SUBMISSIONS
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        result = fetch_recent_filings("0000320193")
        assert len(result) == 2
        assert all(f["form"] == "10-K" for f in result)

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_returns_filing_metadata(self, mock_get: Mock) -> None:
        mock_resp = Mock()
        mock_resp.json.return_value = SAMPLE_SUBMISSIONS
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        result = fetch_recent_filings("0000320193")
        first = result[0]
        assert first["accession_number"] == "0000320193-24-000123"
        assert first["filing_date"] == "2024-11-01"
        assert first["primary_document"] == "aapl-20240928.htm"

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_returns_empty_for_no_matches(
        self, mock_get: Mock,
    ) -> None:
        mock_resp = Mock()
        mock_resp.json.return_value = SAMPLE_SUBMISSIONS
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        result = fetch_recent_filings("0000320193", form_type="20-F")
        assert result == []

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_includes_user_agent_header(
        self, mock_get: Mock,
    ) -> None:
        mock_resp = Mock()
        mock_resp.json.return_value = SAMPLE_SUBMISSIONS
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        fetch_recent_filings("0000320193")
        call_kwargs = mock_get.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert "User-Agent" in headers


# ---------------------------------------------------------------------------
# fetch_filing_text tests
# ---------------------------------------------------------------------------


class TestFetchFilingText:
    """Tests for fetch_filing_text function."""

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_fetches_and_strips_html(self, mock_get: Mock) -> None:
        mock_resp = Mock()
        mock_resp.text = "<html><body><p>Hello world</p></body></html>"
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        result = fetch_filing_text(
            "0000320193",
            "0000320193-24-000123",
            "aapl-20240928.htm",
        )
        assert "Hello world" in result
        assert "<p>" not in result

    @patch("doxa_shared.utils.edgar.httpx.get")
    def test_constructs_correct_url(self, mock_get: Mock) -> None:
        mock_resp = Mock()
        mock_resp.text = "content"
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        fetch_filing_text(
            "0000320193",
            "0000320193-24-000123",
            "aapl-20240928.htm",
        )
        url = mock_get.call_args[0][0]
        assert "320193" in url
        assert "000032019324000123" in url
        assert "aapl-20240928.htm" in url


# ---------------------------------------------------------------------------
# _strip_html tests
# ---------------------------------------------------------------------------


class TestStripHtml:
    """Tests for _strip_html helper."""

    def test_removes_tags(self) -> None:
        assert "Hello" in _strip_html("<b>Hello</b>")
        assert "<b>" not in _strip_html("<b>Hello</b>")

    def test_decodes_entities(self) -> None:
        result = _strip_html("AT&amp;T")
        assert "AT&T" in result

    def test_removes_style_blocks(self) -> None:
        html = "<style>body{color:red}</style>Content"
        result = _strip_html(html)
        assert "color:red" not in result
        assert "Content" in result

    def test_removes_script_blocks(self) -> None:
        html = "<script>alert('x')</script>Content"
        result = _strip_html(html)
        assert "alert" not in result
        assert "Content" in result


# ---------------------------------------------------------------------------
# extract_10k_sections tests
# ---------------------------------------------------------------------------


class TestExtract10kSections:
    """Tests for extract_10k_sections function."""

    def test_extracts_all_three_sections(self) -> None:
        text = _strip_html(SAMPLE_10K_HTML)
        result = extract_10k_sections(text)

        assert "risk_factors" in result
        assert "legal_proceedings" in result
        assert "md_and_a" in result

    def test_risk_factors_contains_content(self) -> None:
        text = _strip_html(SAMPLE_10K_HTML)
        result = extract_10k_sections(text)
        assert "regulatory risks" in result["risk_factors"]

    def test_legal_proceedings_contains_content(self) -> None:
        text = _strip_html(SAMPLE_10K_HTML)
        result = extract_10k_sections(text)
        assert "lawsuits" in result["legal_proceedings"]

    def test_mda_contains_content(self) -> None:
        text = _strip_html(SAMPLE_10K_HTML)
        result = extract_10k_sections(text)
        assert "Revenue" in result["md_and_a"]

    def test_returns_empty_strings_for_missing_sections(self) -> None:
        result = extract_10k_sections("No relevant sections here.")
        assert result["risk_factors"] == ""
        assert result["legal_proceedings"] == ""
        assert result["md_and_a"] == ""

    def test_handles_empty_input(self) -> None:
        result = extract_10k_sections("")
        assert result["risk_factors"] == ""
        assert result["legal_proceedings"] == ""
        assert result["md_and_a"] == ""
