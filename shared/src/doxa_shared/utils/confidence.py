"""Standardized confidence scoring utilities for Doxa agents.

Provides reusable functions for data completeness, time-series quality,
cross-agent validation, and weighted confidence aggregation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from doxa_shared.types.state import ResearchState

_THRESHOLDS: list[tuple[int, float]] = [
    (0, 0.0),
    (1, 25.0),
    (3, 50.0),
    (5, 75.0),
    (10, 100.0),
]
"""Piecewise-linear reference points (years, score%) for time-series quality."""


def calculate_data_completeness(
    required_fields: Sequence[str],
    data: dict[str, Any],
) -> float:
    """Score (0-100) based on how many *required_fields* are populated.

    A field counts as populated if it exists in *data* and is not
    ``None``, empty string, empty dict, or empty list.

    Args:
        required_fields: Field names to check.
        data: Dictionary to inspect.

    Returns:
        Completeness percentage (0.0–100.0).
    """
    if not required_fields:
        return 100.0
    present = sum(
        1 for f in required_fields if _is_populated(data.get(f))
    )
    return present / len(required_fields) * 100


def _is_populated(value: Any) -> bool:
    """Return ``True`` if *value* is non-empty and non-None."""
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (dict, list)) and len(value) == 0:
        return False
    return True


def calculate_time_series_quality(
    years_available: int,
    years_expected: int = 10,
) -> float:
    """Score (0-100) penalising missing historical data.

    Uses piecewise-linear interpolation between reference thresholds:
    0Y → 0%, 1Y → 25%, 3Y → 50%, 5Y → 75%, 10Y → 100%.

    Args:
        years_available: Years of data actually available.
        years_expected: Benchmark for full quality (default 10).

    Returns:
        Quality percentage (0.0–100.0).
    """
    if years_available <= 0:
        return 0.0
    if years_available >= years_expected:
        return 100.0

    for i in range(1, len(_THRESHOLDS)):
        y_lo, s_lo = _THRESHOLDS[i - 1]
        y_hi, s_hi = _THRESHOLDS[i]
        if years_available <= y_hi:
            frac = (years_available - y_lo) / (y_hi - y_lo)
            return s_lo + frac * (s_hi - s_lo)

    return 100.0


def cross_validate_agents(
    state: ResearchState,
) -> dict[str, Any]:
    """Check consistency between agent outputs.

    Detects divergences (DCF vs comps, quant vs sentiment) and
    returns warning flags with a confidence penalty.

    Args:
        state: Fully-populated ResearchState.

    Returns:
        Dict with ``flags`` (list[str]) and ``penalty`` (float 0-20).
    """
    flags: list[str] = []
    penalty = 0.0

    val = state.get("valuation_analysis", {})
    dcf = val.get("dcf", {})
    comps = val.get("comps", {})

    dcf_fv = dcf.get("fair_value_per_share")
    implied = comps.get("implied_valuations", {})

    if dcf_fv and isinstance(dcf_fv, (int, float)) and dcf_fv > 0:
        for iv in implied.values():
            if isinstance(iv, (int, float)) and iv > 0:
                div = abs(dcf_fv - iv) / max(dcf_fv, iv)
                if div > 0.5:
                    flags.append(
                        f"DCF vs comps divergence: {div:.0%}"
                    )
                    penalty += 10.0
                break

    quant_signal = (
        state.get("quant_analysis", {}).get("signal", "")
    )
    sentiment = state.get("sentiment_score", 0.0)

    if quant_signal == "BULLISH" and sentiment < -0.3:
        flags.append("Quant bullish but sentiment negative")
        penalty += 5.0
    elif quant_signal == "BEARISH" and sentiment > 0.3:
        flags.append("Quant bearish but sentiment positive")
        penalty += 5.0

    return {"flags": flags, "penalty": min(penalty, 20.0)}


def aggregate_confidence(
    scores: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Weighted average confidence score, clamped to 0-100.

    Args:
        scores: Individual confidence scores (0-100 each).
        weights: Corresponding weights (must sum to ~1.0).

    Returns:
        Weighted confidence score (0.0–100.0).

    Raises:
        ValueError: If lengths differ or weights don't sum to 1.0
            (±0.01 tolerance).
    """
    if len(scores) != len(weights):
        msg = (
            f"scores ({len(scores)}) and weights "
            f"({len(weights)}) must have same length"
        )
        raise ValueError(msg)

    weight_sum = math.fsum(weights)
    if abs(weight_sum - 1.0) > 0.01:
        msg = f"Weights must sum to 1.0 (got {weight_sum:.4f})"
        raise ValueError(msg)

    result = math.fsum(s * w for s, w in zip(scores, weights))
    return max(0.0, min(100.0, result))
