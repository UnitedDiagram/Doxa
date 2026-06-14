"""Quantitative financial analysis utilities for Doxa.

This module provides DuPont analysis and Altman Z-Score calculations
for equity research. All functions handle missing data gracefully by
returning None when required inputs are unavailable.
"""

from __future__ import annotations

from typing import Any


def compute_profit_margin(financials: dict[str, Any]) -> float | None:
    """Calculate net profit margin as Net Income / Total Revenue.

    Args:
        financials: Dictionary containing financial statement data.
            Must include 'net_income' and 'total_revenue' keys.

    Returns:
        Net profit margin as a decimal (e.g., 0.15 for 15%), or None
        if required data is missing or revenue is zero.
    """
    net_income = financials.get("net_income")
    revenue = financials.get("total_revenue")
    if net_income is None or revenue is None or revenue == 0:
        return None
    return float(net_income) / float(revenue)


def compute_asset_turnover(financials: dict[str, Any]) -> float | None:
    """Calculate asset turnover as Total Revenue / Average Total Assets.

    Args:
        financials: Dictionary containing financial statement data.
            Must include 'total_revenue' and 'total_assets' keys.
            Optionally includes 'total_assets_prev' for averaging.

    Returns:
        Asset turnover ratio, or None if required data is missing
        or average assets is zero.
    """
    revenue = financials.get("total_revenue")
    assets = financials.get("total_assets")
    assets_prev = financials.get("total_assets_prev")
    if revenue is None or assets is None:
        return None
    avg_assets = (
        (float(assets) + float(assets_prev)) / 2 if assets_prev else float(assets)
    )
    if avg_assets == 0:
        return None
    return float(revenue) / avg_assets


def compute_equity_multiplier(financials: dict[str, Any]) -> float | None:
    """Calculate equity multiplier as Average Total Assets / Average Equity.

    Args:
        financials: Dictionary containing financial statement data.
            Must include 'total_assets' and 'stockholders_equity' keys.
            Optionally includes prior period values for averaging.

    Returns:
        Equity multiplier (financial leverage), or None if required
        data is missing or average equity is zero.
    """
    assets = financials.get("total_assets")
    assets_prev = financials.get("total_assets_prev")
    equity = financials.get("stockholders_equity")
    equity_prev = financials.get("stockholders_equity_prev")
    if assets is None or equity is None:
        return None
    avg_assets = (
        (float(assets) + float(assets_prev)) / 2 if assets_prev else float(assets)
    )
    avg_equity = (
        (float(equity) + float(equity_prev)) / 2 if equity_prev else float(equity)
    )
    if avg_equity == 0:
        return None
    return avg_assets / avg_equity


def compute_roe(
    profit_margin: float | None,
    asset_turnover: float | None,
    equity_multiplier: float | None,
) -> float | None:
    """Calculate Return on Equity using DuPont 3-factor decomposition.

    ROE = Profit Margin × Asset Turnover × Equity Multiplier

    Args:
        profit_margin: Net profit margin (Net Income / Revenue).
        asset_turnover: Asset turnover ratio (Revenue / Avg Assets).
        equity_multiplier: Equity multiplier (Avg Assets / Avg Equity).

    Returns:
        Return on Equity as a decimal (e.g., 0.20 for 20%), or None
        if any component is missing.
    """
    if profit_margin is None or asset_turnover is None or equity_multiplier is None:
        return None
    return profit_margin * asset_turnover * equity_multiplier


def derive_dupont_driver(
    profit_margin: float | None,
    asset_turnover: float | None,
    equity_multiplier: float | None,
) -> str:
    """Identify the primary driver of ROE based on DuPont components.

    Analyzes the three DuPont factors to determine whether the company's
    ROE is driven by profitability, efficiency, or leverage.

    Args:
        profit_margin: Net profit margin.
        asset_turnover: Asset turnover ratio.
        equity_multiplier: Equity multiplier (financial leverage).

    Returns:
        A descriptive label: "High Profitability", "High Asset Efficiency",
        "High Leverage, Low Margin", "Balanced", or "Insufficient Data".
    """
    if profit_margin is None or asset_turnover is None or equity_multiplier is None:
        return "Insufficient Data"
    if equity_multiplier > 3 and profit_margin < 0.10:
        return "High Leverage, Low Margin"
    if profit_margin > 0.20:
        return "High Profitability"
    if asset_turnover > 1.5:
        return "High Asset Efficiency"
    return "Balanced"


def compute_altman_z(
    market_data: dict[str, Any], financials: dict[str, Any]
) -> float | None:
    """Calculate Altman Z-Score for bankruptcy risk assessment.

    The Altman Z-Score formula (for public manufacturing companies):
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    Where:
    - X1 = Working Capital / Total Assets
    - X2 = Retained Earnings / Total Assets
    - X3 = EBIT / Total Assets
    - X4 = Market Cap / Total Liabilities
    - X5 = Total Revenue / Total Assets

    Args:
        market_data: Dictionary containing 'market_cap'.
        financials: Dictionary containing balance sheet and income
            statement items (working_capital, retained_earnings, ebit,
            total_assets, total_liabilities, total_revenue).

    Returns:
        Altman Z-Score, or None if total_assets is missing or zero.
        Missing component values default to 0.0 in the calculation.
    """
    total_assets = financials.get("total_assets")
    if not total_assets or float(total_assets) == 0:
        return None

    ta = float(total_assets)
    working_capital = financials.get("working_capital")
    retained_earnings = financials.get("retained_earnings")
    ebit = financials.get("ebit")
    market_cap = market_data.get("market_cap")
    total_liabilities = financials.get("total_liabilities")
    total_revenue = financials.get("total_revenue")

    x1 = float(working_capital) / ta if working_capital is not None else 0.0
    x2 = float(retained_earnings) / ta if retained_earnings is not None else 0.0
    x3 = float(ebit) / ta if ebit is not None else 0.0
    x4 = (
        float(market_cap) / float(total_liabilities)
        if (
            market_cap is not None
            and total_liabilities
            and float(total_liabilities) != 0
        )
        else 0.0
    )
    x5 = float(total_revenue) / ta if total_revenue is not None else 0.0

    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


def altman_zone(z: float | None) -> str:
    """Classify Altman Z-Score into bankruptcy risk zones.

    Standard zones for public manufacturing companies:
    - Safe Zone: Z > 2.99 (low bankruptcy risk)
    - Grey Zone: 1.81 < Z ≤ 2.99 (moderate risk)
    - Distress Zone: Z ≤ 1.81 (high bankruptcy risk)

    Args:
        z: Altman Z-Score value.

    Returns:
        Risk zone classification: "Safe", "Grey", "Distress", or
        "Unknown" if z is None.
    """
    if z is None:
        return "Unknown"
    if z > 2.99:
        return "Safe"
    if z > 1.81:
        return "Grey"
    return "Distress"
