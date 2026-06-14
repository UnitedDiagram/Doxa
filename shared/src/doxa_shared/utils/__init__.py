"""Shared utility functions for Doxa."""

from __future__ import annotations

from doxa_shared.utils.cache import (
    TTL_FINANCIALS,
    TTL_PRICE_DATA,
    TTL_SEC_FILINGS,
    CacheBackend,
    CacheStats,
    InMemoryCache,
    cached_fetch,
    get_cache,
)
from doxa_shared.utils.confidence import (
    aggregate_confidence,
    calculate_data_completeness,
    calculate_time_series_quality,
    cross_validate_agents,
)
from doxa_shared.utils.formatters import fmt_number, fmt_pct, fmt_ratio
from doxa_shared.utils.market_data import df_get, safe_get
from doxa_shared.utils.quant import (
    altman_zone,
    compute_altman_z,
    compute_asset_turnover,
    compute_equity_multiplier,
    compute_profit_margin,
    compute_roe,
    derive_dupont_driver,
)

__all__ = [
    # cache
    "CacheBackend",
    "CacheStats",
    "InMemoryCache",
    "TTL_FINANCIALS",
    "TTL_PRICE_DATA",
    "TTL_SEC_FILINGS",
    "cached_fetch",
    "get_cache",
    # confidence
    "aggregate_confidence",
    "calculate_data_completeness",
    "calculate_time_series_quality",
    "cross_validate_agents",
    # formatters
    "fmt_number",
    "fmt_pct",
    "fmt_ratio",
    # market_data
    "df_get",
    "safe_get",
    # quant
    "altman_zone",
    "compute_altman_z",
    "compute_asset_turnover",
    "compute_equity_multiplier",
    "compute_profit_margin",
    "compute_roe",
    "derive_dupont_driver",
]
