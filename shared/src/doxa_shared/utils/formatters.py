"""Number formatting utilities for Doxa reports and UI display.

Provides consistent formatting for financial values across agents
and frontend components. All formatters handle missing data gracefully
by returning "N/A" when values are None or cannot be converted.
"""

from __future__ import annotations

from typing import Any


def fmt_number(value: Any, prefix: str = "", suffix: str = "") -> str:
    """Format a numeric value with thousand separators and optional prefix/suffix.

    Args:
        value: Numeric value to format (int, float, or string).
        prefix: Optional prefix string (e.g., "$" for currency).
        suffix: Optional suffix string (e.g., "B" for billions).

    Returns:
        Formatted string with commas as thousand separators and 2 decimal
        places, or "N/A" if value is None or cannot be converted to float.

    Examples:
        >>> fmt_number(1500000, "$")
        '$1,500,000.00'
        >>> fmt_number(2.5, "", "B")
        '2.50B'
        >>> fmt_number(None)
        'N/A'
    """
    if value is None:
        return "N/A"
    try:
        return f"{prefix}{float(value):,.2f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_pct(value: Any) -> str:
    """Format a decimal ratio as a percentage string.

    Args:
        value: Decimal ratio to format (e.g., 0.15 for 15%).

    Returns:
        Percentage string with 1 decimal place, or "N/A" if value
        is None or cannot be converted to float.

    Examples:
        >>> fmt_pct(0.1523)
        '15.2%'
        >>> fmt_pct(0.05)
        '5.0%'
        >>> fmt_pct(None)
        'N/A'
    """
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_large_number(value: Any, prefix: str = "$") -> str:
    """Format a large numeric value with T/B/M suffix for readability.

    Args:
        value: Numeric value to format (int, float, or string).
        prefix: Optional prefix string (e.g., "$" for currency, "" for shares).

    Returns:
        Formatted string with T/B/M suffix, or "N/A" if value is None or
        cannot be converted to float.

    Examples:
        >>> fmt_large_number(3_870_000_000_000)
        '$3.87T'
        >>> fmt_large_number(117_800_000_000)
        '$117.8B'
        >>> fmt_large_number(435_600_000)
        '$435.6M'
        >>> fmt_large_number(None)
        'N/A'
    """
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if v >= 1e12:
            return f"{prefix}{v / 1e12:.2f}T"
        if v >= 1e9:
            return f"{prefix}{v / 1e9:.1f}B"
        if v >= 1e6:
            return f"{prefix}{v / 1e6:.1f}M"
        return fmt_number(value, prefix)
    except (TypeError, ValueError):
        return "N/A"


def fmt_ratio(value: Any) -> str:
    """Format a ratio with 2 decimal places and 'x' suffix.

    Args:
        value: Ratio value to format (e.g., P/E ratio, equity multiplier).

    Returns:
        Formatted ratio string with 2 decimal places and 'x' suffix,
        or "N/A" if value is None or cannot be converted to float.

    Examples:
        >>> fmt_ratio(15.67)
        '15.67x'
        >>> fmt_ratio(2.3)
        '2.30x'
        >>> fmt_ratio(None)
        'N/A'
    """
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return "N/A"
