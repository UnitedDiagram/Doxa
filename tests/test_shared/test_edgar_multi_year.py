"""Tests for multi-year SEC EDGAR 10-K fetching functionality."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from doxa_shared.utils.edgar import fetch_multi_year_10k


@pytest.fixture
def mock_filing_metadata() -> list[dict[str, str]]:
    """Return mock filing metadata for 3 years of 10-K filings."""
    return [
        {
            "accession_number": "0000320193-24-000106",
            "filing_date": "2024-11-01",
            "primary_document": "aapl-20240928.htm",
            "form": "10-K",
        },
        {
            "accession_number": "0000320193-23-000077",
            "filing_date": "2023-11-03",
            "primary_document": "aapl-20230930.htm",
            "form": "10-K",
        },
        {
            "accession_number": "0000320193-22-000108",
            "filing_date": "2022-10-28",
            "primary_document": "aapl-20220924.htm",
            "form": "10-K",
        },
    ]


@pytest.fixture
def mock_filing_text() -> str:
    """Return mock 10-K filing HTML content."""
    return """
    <html>
    <body>
    Item 1A - Risk Factors
    We face intense competition from other technology companies.
    Supply chain disruptions could impact our product availability.
    Item 1B - Unresolved Staff Comments
    None.
    </body>
    </html>
    """


@pytest.fixture
def mock_sections() -> dict[str, str]:
    """Return mock extracted sections from 10-K."""
    return {
        "risk_factors": (
            "We face intense competition from other technology companies.\n"
            "Supply chain disruptions could impact our product availability."
        ),
        "legal_proceedings": "",
        "md_and_a": "",
    }


def test_fetch_multi_year_10k_success_3_years(
    mock_filing_metadata: list[dict[str, str]],
    mock_filing_text: str,
    mock_sections: dict[str, str],
) -> None:
    """Test successful fetch of 3 years of 10-K filings."""
    with patch("doxa_shared.utils.edgar.fetch_recent_filings") as mock_fetch, \
         patch("doxa_shared.utils.edgar.fetch_filing_text") as mock_text, \
         patch("doxa_shared.utils.edgar.extract_10k_sections") as mock_extract:

        mock_fetch.return_value = mock_filing_metadata
        mock_text.return_value = mock_filing_text
        mock_extract.return_value = mock_sections

        result = fetch_multi_year_10k(cik="0000320193", years=3)

        # Verify 3 filings returned
        assert len(result) == 3

        # Verify structure of first filing
        assert "year" in result[0]
        assert "filing_date" in result[0]
        assert "risk_factors_text" in result[0]

        # Verify filing dates
        assert result[0]["filing_date"] == "2024-11-01"
        assert result[1]["filing_date"] == "2023-11-03"
        assert result[2]["filing_date"] == "2022-10-28"

        # Verify risk factors extracted
        assert "competition" in result[0]["risk_factors_text"]

        # Verify sorted newest to oldest (already in order)
        assert result[0]["filing_date"] > result[1]["filing_date"]


def test_fetch_multi_year_10k_only_2_years_available(
    mock_filing_metadata: list[dict[str, str]],
    mock_filing_text: str,
    mock_sections: dict[str, str],
) -> None:
    """Test handling when only 2 years of filings available (e.g., recent IPO)."""
    with patch("doxa_shared.utils.edgar.fetch_recent_filings") as mock_fetch, \
         patch("doxa_shared.utils.edgar.fetch_filing_text") as mock_text, \
         patch("doxa_shared.utils.edgar.extract_10k_sections") as mock_extract:

        # Only return 2 filings
        mock_fetch.return_value = mock_filing_metadata[:2]
        mock_text.return_value = mock_filing_text
        mock_extract.return_value = mock_sections

        result = fetch_multi_year_10k(cik="0001234567", years=3)

        # Should return 2 filings (all available)
        assert len(result) == 2
        assert result[0]["filing_date"] == "2024-11-01"
        assert result[1]["filing_date"] == "2023-11-03"


def test_fetch_multi_year_10k_api_failure() -> None:
    """Test graceful handling when SEC EDGAR API fails."""
    with patch("doxa_shared.utils.edgar.fetch_recent_filings") as mock_fetch:
        mock_fetch.side_effect = Exception("SEC API timeout")

        result = fetch_multi_year_10k(cik="0000320193", years=3)

        # Should return empty list on failure
        assert result == []


def test_fetch_multi_year_10k_missing_risk_factors() -> None:
    """Test handling when Risk Factors section is missing from filing."""
    mock_metadata = [
        {
            "accession_number": "0000320193-24-000106",
            "filing_date": "2024-11-01",
            "primary_document": "aapl-20240928.htm",
            "form": "10-K",
        },
    ]

    with patch("doxa_shared.utils.edgar.fetch_recent_filings") as mock_fetch, \
         patch("doxa_shared.utils.edgar.fetch_filing_text") as mock_text, \
         patch("doxa_shared.utils.edgar.extract_10k_sections") as mock_extract:

        mock_fetch.return_value = mock_metadata
        mock_text.return_value = "<html></html>"
        # Empty risk_factors section
        mock_extract.return_value = {
            "risk_factors": "",
            "legal_proceedings": "",
            "md_and_a": "",
        }

        result = fetch_multi_year_10k(cik="0000320193", years=1)

        # Should still return filing, just with empty risk_factors_text
        assert len(result) == 1
        assert result[0]["risk_factors_text"] == ""


def test_fetch_multi_year_10k_year_extraction() -> None:
    """Test that year is correctly extracted from filing_date."""
    mock_metadata = [
        {
            "accession_number": "0000320193-24-000106",
            "filing_date": "2024-11-01",
            "primary_document": "aapl-20240928.htm",
            "form": "10-K",
        },
    ]

    with patch("doxa_shared.utils.edgar.fetch_recent_filings") as mock_fetch, \
         patch("doxa_shared.utils.edgar.fetch_filing_text") as mock_text, \
         patch("doxa_shared.utils.edgar.extract_10k_sections") as mock_extract:

        mock_fetch.return_value = mock_metadata
        mock_text.return_value = "<html></html>"
        mock_extract.return_value = {
            "risk_factors": "Test risk",
            "legal_proceedings": "",
            "md_and_a": "",
        }

        result = fetch_multi_year_10k(cik="0000320193", years=1)

        # Verify year extracted from filing_date
        assert result[0]["year"] == 2024


def test_fetch_multi_year_10k_partial_failure() -> None:
    """Test handling when fetch_recent_filings fails at API level."""
    with patch("doxa_shared.utils.edgar.fetch_recent_filings") as mock_fetch:
        # SEC API fails entirely
        mock_fetch.side_effect = Exception("SEC API timeout")

        result = fetch_multi_year_10k(cik="0000320193", years=3)

        # Should return empty list and log warning
        assert result == []


def test_fetch_multi_year_10k_malformed_filing_date() -> None:
    """Test handling when SEC returns malformed filing_date."""
    mock_metadata = [
        {
            "accession_number": "0000320193-24-000106",
            "filing_date": "INVALID-DATE",
            "primary_document": "aapl-20240928.htm",
            "form": "10-K",
        },
    ]

    with patch("doxa_shared.utils.edgar.fetch_recent_filings") as mock_fetch, \
         patch("doxa_shared.utils.edgar.fetch_filing_text") as mock_text, \
         patch("doxa_shared.utils.edgar.extract_10k_sections") as mock_extract:

        mock_fetch.return_value = mock_metadata
        mock_text.return_value = "<html></html>"
        mock_extract.return_value = {
            "risk_factors": "Test",
            "legal_proceedings": "",
            "md_and_a": "",
        }

        result = fetch_multi_year_10k(cik="0000320193", years=1)

        # Should return empty list when date parsing fails (exception caught)
        assert result == []
