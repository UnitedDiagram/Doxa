"""Verify _strip_title_and_snapshot removes leading title and snapshot."""

from __future__ import annotations

from src.export.pdf_export import _strip_title_and_snapshot


def test_strips_title_and_snapshot_section() -> None:
    """Title block and Snapshot table are removed for the PDF cover."""
    markdown = (
        "# AAPL — Equity Research Note\n"
        "\n"
        "Rating: Buy | 12-Mo Price Target: $180.00 | Date: 2026-05-12\n"
        "\n"
        "## Snapshot\n"
        "\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "| Price | $150 |\n"
        "\n"
        "## I. Investment Summary\n"
        "\n"
        "Body content here.\n"
    )
    result = _strip_title_and_snapshot(markdown)
    assert result.startswith("## I. Investment Summary")
    assert "Snapshot" not in result
    assert "Body content here" in result


def test_preserves_markdown_with_no_title() -> None:
    """Markdown without a leading H1 passes through unchanged."""
    markdown = "## Some Section\n\nContent.\n"
    assert _strip_title_and_snapshot(markdown).rstrip() == markdown.rstrip()


def test_handles_title_without_snapshot() -> None:
    """Title is dropped even if no Snapshot section is present."""
    markdown = (
        "# TICKER\n"
        "\n"
        "Meta line\n"
        "\n"
        "## I. Investment Summary\n"
        "\n"
        "Body.\n"
    )
    result = _strip_title_and_snapshot(markdown)
    assert result.startswith("## I. Investment Summary")
    assert "TICKER" not in result
