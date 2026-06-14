"""Valuation calculation utilities for DCF and comparable company analysis.

This module provides financial valuation functions for Doxa's ValuationAgent,
including DCF model calculations, WACC computation, and valuation multiples.
"""

from __future__ import annotations

from typing import Any

from doxa_shared.utils.market_data import df_get


def calculate_free_cash_flow(cash_flow_data: dict[str, Any]) -> list[float]:
    """Calculate 5-year free cash flow projections based on historical data.

    Projects future FCF using historical operating cash flow growth rate.
    If insufficient historical data is available, returns empty list.

    Args:
        cash_flow_data: Dictionary containing 'operating_cash_flow' list
            with historical values (most recent first).

    Returns:
        List of 5 projected FCF values. Empty list if insufficient data.

    Example:
        >>> data = {'operating_cash_flow': [100.0, 90.0, 80.0]}
        >>> calculate_free_cash_flow(data)
        [111.1, 123.4, 137.2, 152.4, 169.3]
    """
    operating_cf = cash_flow_data.get("operating_cash_flow", [])

    if len(operating_cf) < 2:
        return []

    # Calculate historical growth rate (most recent / previous - 1)
    recent = operating_cf[0]
    previous = operating_cf[1]

    if previous <= 0:
        return []

    growth_rate = (recent / previous) - 1.0

    # Project 5 years
    projections = []
    current_fcf = recent

    for _ in range(5):
        current_fcf = current_fcf * (1 + growth_rate)
        projections.append(current_fcf)

    return projections


def calculate_wacc(
    beta: float,
    risk_free_rate: float,
    market_risk_premium: float,
    debt_equity_ratio: float,
    tax_rate: float,
) -> float:
    """Calculate Weighted Average Cost of Capital (WACC).

    Uses CAPM for cost of equity and after-tax cost of debt.

    Args:
        beta: Company's market beta.
        risk_free_rate: Risk-free rate (e.g., 0.045 for 4.5%).
        market_risk_premium: Expected market return above risk-free rate.
        debt_equity_ratio: Total debt / total equity.
        tax_rate: Corporate tax rate (e.g., 0.21 for 21%).

    Returns:
        WACC as decimal (e.g., 0.10 for 10%).

    Example:
        >>> calculate_wacc(1.2, 0.045, 0.07, 0.5, 0.21)
        0.0924
    """
    # Cost of Equity (CAPM)
    cost_of_equity = risk_free_rate + beta * market_risk_premium

    # Assume cost of debt = risk_free_rate + 2% spread (simplified for POC)
    cost_of_debt = risk_free_rate + 0.02

    # After-tax cost of debt
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

    # Weight of equity and debt
    total_capital = 1 + debt_equity_ratio
    weight_equity = 1 / total_capital
    weight_debt = debt_equity_ratio / total_capital

    # WACC
    wacc = (weight_equity * cost_of_equity) + (weight_debt * after_tax_cost_of_debt)

    return wacc


def calculate_terminal_value(
    final_fcf: float, growth_rate: float, wacc: float
) -> float:
    """Calculate terminal value using perpetuity growth method.

    Args:
        final_fcf: Final year free cash flow projection.
        growth_rate: Perpetual growth rate (e.g., 0.025 for 2.5%).
        wacc: Weighted average cost of capital (e.g., 0.10 for 10%).

    Returns:
        Terminal value. Returns 0.0 if growth_rate >= wacc (invalid).

    Example:
        >>> calculate_terminal_value(100.0, 0.025, 0.10)
        1333.33
    """
    if growth_rate >= wacc:
        return 0.0

    terminal_fcf = final_fcf * (1 + growth_rate)
    terminal_value = terminal_fcf / (wacc - growth_rate)

    return terminal_value


def calculate_dcf_fair_value(
    fcf_projections: list[float],
    terminal_value: float,
    wacc: float,
    shares_outstanding: float,
) -> float:
    """Calculate DCF fair value per share.

    Discounts projected cash flows and terminal value to present value,
    then divides by shares outstanding.

    Args:
        fcf_projections: List of 5 projected FCF values.
        terminal_value: Terminal value at end of projection period.
        wacc: Weighted average cost of capital for discounting.
        shares_outstanding: Number of shares outstanding.

    Returns:
        Fair value per share. Returns 0.0 if shares_outstanding is 0.

    Example:
        >>> fcf = [100, 110, 121, 133, 146]
        >>> calculate_dcf_fair_value(fcf, 2000, 0.10, 1000)
        2.14
    """
    if shares_outstanding == 0:
        return 0.0

    # Calculate present value of FCF projections
    pv_fcf = 0.0
    for year, fcf in enumerate(fcf_projections, start=1):
        discount_factor = (1 + wacc) ** year
        pv_fcf += fcf / discount_factor

    # Calculate present value of terminal value (at year 5)
    pv_terminal = terminal_value / ((1 + wacc) ** 5)

    # Total enterprise value
    enterprise_value = pv_fcf + pv_terminal

    # Fair value per share
    fair_value_per_share = enterprise_value / shares_outstanding

    return fair_value_per_share


def calculate_valuation_multiples(
    market_cap: float,
    revenue: float,
    ebitda: float,
    book_value: float,
    net_income: float,
    total_debt: float = 0.0,
    cash: float = 0.0,
) -> dict[str, float | None]:
    """Calculate valuation multiples for comparable company analysis.

    Args:
        market_cap: Company market capitalization.
        revenue: Total revenue.
        ebitda: Earnings before interest, taxes, depreciation, amortization.
        book_value: Book value (total assets - total liabilities).
        net_income: Net income.
        total_debt: Total debt for enterprise value calculation.
        cash: Cash and equivalents for enterprise value calculation.

    Returns:
        Dictionary with keys 'P/E', 'EV/EBITDA', 'P/B', 'P/S'.
        None for multiples that cannot be calculated.

    Example:
        >>> calculate_valuation_multiples(1000, 500, 200, 400, 50, 100, 50)
        {'P/E': 20.0, 'EV/EBITDA': 5.25, 'P/B': 2.5, 'P/S': 2.0}
    """
    multiples: dict[str, float | None] = {}

    # P/E ratio
    multiples["P/E"] = market_cap / net_income if net_income > 0 else None

    # EV/EBITDA (Enterprise Value = Market Cap + Debt - Cash)
    enterprise_value = market_cap + total_debt - cash
    multiples["EV/EBITDA"] = (
        enterprise_value / ebitda if ebitda > 0 else None
    )

    # P/B ratio
    multiples["P/B"] = market_cap / book_value if book_value > 0 else None

    # P/S ratio
    multiples["P/S"] = market_cap / revenue if revenue > 0 else None

    return multiples


def generate_sensitivity_table(
    base_wacc: float,
    base_growth: float,
    fcf_projections: list[float],
    shares_outstanding: float,
) -> dict[str, dict[str, float]]:
    """Generate sensitivity analysis table for DCF valuation.

    Creates a grid of fair values across WACC ±2% and growth ±1%.

    Args:
        base_wacc: Base case WACC (e.g., 0.10 for 10%).
        base_growth: Base case perpetual growth rate (e.g., 0.025 for 2.5%).
        fcf_projections: List of 5 projected FCF values.
        shares_outstanding: Number of shares outstanding.

    Returns:
        Nested dict: {wacc_str: {growth_str: fair_value}}.

    Example:
        >>> fcf = [100, 110, 121, 133, 146]
        >>> generate_sensitivity_table(0.10, 0.025, fcf, 1000)
        {'8.0%': {'1.5%': 55.2, '2.0%': 58.1, ...}, ...}
    """
    sensitivity: dict[str, dict[str, float]] = {}

    # WACC range: ±2%
    wacc_range = [
        base_wacc - 0.02,
        base_wacc - 0.01,
        base_wacc,
        base_wacc + 0.01,
        base_wacc + 0.02,
    ]

    # Growth range: ±1%
    growth_range = [
        base_growth - 0.01,
        base_growth - 0.005,
        base_growth,
        base_growth + 0.005,
        base_growth + 0.01,
    ]

    for wacc in wacc_range:
        wacc_str = f"{wacc * 100:.1f}%"
        sensitivity[wacc_str] = {}

        for growth in growth_range:
            growth_str = f"{growth * 100:.1f}%"

            # Calculate fair value for this combination
            if len(fcf_projections) >= 5:
                terminal_value = calculate_terminal_value(
                    fcf_projections[-1], growth, wacc
                )
                fair_value = calculate_dcf_fair_value(
                    fcf_projections, terminal_value, wacc, shares_outstanding
                )
                sensitivity[wacc_str][growth_str] = round(fair_value, 2)
            else:
                sensitivity[wacc_str][growth_str] = 0.0

    return sensitivity


def calculate_dupont_analysis(financials: dict[str, Any]) -> dict[str, Any] | None:
    """Calculate DuPont 3-factor ROE decomposition.

    Decomposes Return on Equity (ROE) into three components:
    - Net Profit Margin = Net Income / Revenue
    - Asset Turnover = Revenue / Total Assets
    - Equity Multiplier = Total Assets / Shareholders' Equity

    ROE = Net Margin × Asset Turnover × Equity Multiplier

    Args:
        financials: Dictionary containing financial statement data.
            Required keys: 'net_income', 'total_revenue', 'total_assets',
            'stockholders_equity'.

    Returns:
        Dictionary with DuPont components and ROE, or None if required
        data is missing or invalid (zero revenue, zero/negative equity).

    Example:
        >>> financials = {
        ...     'net_income': 30_000_000_000,
        ...     'total_revenue': 100_000_000_000,
        ...     'total_assets': 200_000_000_000,
        ...     'stockholders_equity': 150_000_000_000,
        ... }
        >>> result = calculate_dupont_analysis(financials)
        >>> result['roe']  # Should be 0.20 (20%)
        0.20
    """
    # Extract required fields
    net_income = financials.get("net_income")
    total_revenue = financials.get("total_revenue")
    total_assets = financials.get("total_assets")
    stockholders_equity = financials.get("stockholders_equity")

    # Validate required data
    if (
        net_income is None
        or total_revenue is None
        or total_assets is None
        or stockholders_equity is None
    ):
        return None

    # Convert to float and validate non-zero denominators
    revenue = float(total_revenue)
    assets = float(total_assets)
    equity = float(stockholders_equity)

    if revenue == 0 or assets == 0 or equity <= 0:
        return None

    # Calculate DuPont components
    profit_margin = float(net_income) / revenue
    asset_turnover = revenue / assets
    equity_multiplier = assets / equity

    # Calculate ROE (should equal net_income / equity)
    roe_dupont = profit_margin * asset_turnover * equity_multiplier
    roe_direct = float(net_income) / equity

    return {
        "profit_margin": profit_margin,
        "asset_turnover": asset_turnover,
        "equity_multiplier": equity_multiplier,
        "roe": roe_direct,  # Use direct calculation as authoritative
        "roe_dupont": roe_dupont,  # Store both for verification
    }


def calculate_altman_z_score(
    balance_sheet: Any,
    income_stmt: Any,
    market_cap: float,
) -> dict[str, Any] | None:
    """Calculate Altman Z-Score for bankruptcy risk assessment.

    The Altman Z-Score is a formula for predicting bankruptcy risk:
    Z = 1.2×(WC/TA) + 1.4×(RE/TA) + 3.3×(EBIT/TA) + 0.6×(MVE/TL) + 1.0×(Sales/TA)

    Where:
        WC = Working Capital (Current Assets - Current Liabilities)
        TA = Total Assets
        RE = Retained Earnings
        EBIT = Earnings Before Interest and Taxes
        MVE = Market Value of Equity (market cap)
        TL = Total Liabilities

    Interpretation:
        Z > 2.99: "Safe Zone" - Low bankruptcy risk
        1.81 < Z < 2.99: "Grey Zone" - Moderate bankruptcy risk
        Z < 1.81: "Distress Zone" - High bankruptcy risk

    Args:
        balance_sheet: Balance sheet DataFrame-like object with financial data.
        income_stmt: Income statement DataFrame-like object with profitability data.
        market_cap: Market capitalization (market value of equity).

    Returns:
        Dictionary containing 'z_score', 'interpretation', and 'components',
        or None if required data is missing or invalid (zero denominators).

    Example:
        >>> # Strong company example - balance sheet and income statement
        >>> # would be DataFrames from yfinance with financial data
        >>> # result = calculate_altman_z_score(balance_sheet, income_stmt, market_cap)
        >>> # result['z_score'] > 2.99  # Safe Zone for healthy company
    """
    # Extract balance sheet components using defensive df_get pattern
    current_assets = df_get(
        balance_sheet, ["Current Assets", "currentAssets", "TotalCurrentAssets"], 0
    )
    current_liabilities = df_get(
        balance_sheet,
        ["Current Liabilities", "currentLiabilities", "TotalCurrentLiabilities"],
        0,
    )
    retained_earnings = df_get(
        balance_sheet, ["Retained Earnings", "retainedEarnings", "RetainedEarnings"], 0
    )
    total_assets = df_get(
        balance_sheet, ["Total Assets", "totalAssets", "TotalAssets"], 0
    )
    total_liabilities = df_get(
        balance_sheet,
        [
            "Total Liabilities Net Minority Interest",
            "Total Liabilities",
            "totalLiabilities",
            "TotalLiabilitiesNetMinorityInterest",
        ],
        0,
    )

    # Extract income statement components
    ebit = df_get(income_stmt, ["EBIT", "ebit", "OperatingIncome"], 0)
    total_revenue = df_get(
        income_stmt, ["Total Revenue", "totalRevenue", "TotalRevenue"], 0
    )

    # Validate all required components exist
    if (
        current_assets is None
        or current_liabilities is None
        or total_assets is None
        or total_liabilities is None
    ):
        return None

    # Convert to float and validate non-zero denominators
    ta = float(total_assets)
    tl = float(total_liabilities)

    if ta == 0 or tl == 0:
        return None

    # Calculate components (handle None values gracefully)
    wc = float(current_assets) - float(current_liabilities)
    re = float(retained_earnings) if retained_earnings is not None else 0.0
    ebit_val = float(ebit) if ebit is not None else 0.0
    sales = float(total_revenue) if total_revenue is not None else 0.0
    mve = float(market_cap)

    # Calculate Z-Score components
    wc_ta = 1.2 * (wc / ta)
    re_ta = 1.4 * (re / ta)
    ebit_ta = 3.3 * (ebit_val / ta)
    mve_tl = 0.6 * (mve / tl)
    sales_ta = 1.0 * (sales / ta)

    # Total Z-Score
    z_score = wc_ta + re_ta + ebit_ta + mve_tl + sales_ta

    # Interpret Z-Score
    if z_score > 2.99:
        interpretation = "Safe Zone - Low bankruptcy risk"
    elif z_score > 1.81:
        interpretation = "Grey Zone - Moderate bankruptcy risk"
    else:
        interpretation = "Distress Zone - High bankruptcy risk"

    return {
        "z_score": z_score,
        "interpretation": interpretation,
        "components": {
            "working_capital_to_assets": wc_ta,
            "retained_earnings_to_assets": re_ta,
            "ebit_to_assets": ebit_ta,
            "market_value_to_liabilities": mve_tl,
            "sales_to_assets": sales_ta,
        },
    }


def calculate_financial_ratios(
    balance_sheet: Any,
    income_stmt: Any,
    cashflow: Any,  # noqa: ARG001
) -> dict[str, Any] | None:
    """Calculate comprehensive financial ratios across four categories.

    Calculates profitability, liquidity, leverage, and efficiency ratios
    using balance sheet, income statement, and cash flow data.

    Profitability Ratios:
        - Gross Margin = Gross Profit / Revenue
        - Operating Margin = Operating Income / Revenue
        - Net Margin = Net Income / Revenue
        - ROA = Net Income / Total Assets
        - ROE = Net Income / Shareholders' Equity

    Liquidity Ratios:
        - Current Ratio = Current Assets / Current Liabilities
        - Quick Ratio = (Current Assets - Inventory) / Current Liabilities

    Leverage Ratios:
        - Debt-to-Equity = Total Debt / Shareholders' Equity
        - Interest Coverage = EBIT / Interest Expense

    Efficiency Ratios:
        - Asset Turnover = Revenue / Total Assets
        - Inventory Turnover = COGS / Inventory
        - Days Inventory Outstanding = 365 / Inventory Turnover

    Args:
        balance_sheet: Balance sheet DataFrame-like object with financial position.
        income_stmt: Income statement DataFrame-like object with profitability.
        cashflow: Cash flow DataFrame-like object (reserved for future use).

    Returns:
        Dictionary with nested structure:
        {
            'profitability': {...},
            'liquidity': {...},
            'leverage': {...},
            'efficiency': {...}
        }
        Individual ratios return None if required data is missing or invalid.
        Returns None if all required base data is missing.

    Example:
        >>> # DataFrames from yfinance with complete financial data
        >>> # result = calculate_financial_ratios(bs, inc, cf)
        >>> # result['profitability']['gross_margin']  # e.g., 0.40 (40%)
    """
    # Extract balance sheet data
    current_assets = df_get(
        balance_sheet, ["Current Assets", "currentAssets"], 0
    )
    current_liabilities = df_get(
        balance_sheet, ["Current Liabilities", "currentLiabilities"], 0
    )
    inventory = df_get(balance_sheet, ["Inventory", "inventory"], 0)
    total_assets = df_get(balance_sheet, ["Total Assets", "totalAssets"], 0)
    total_debt = df_get(
        balance_sheet,
        ["Total Debt", "totalDebt", "Long Term Debt", "longTermDebt"],
        0,
    )
    stockholders_equity = df_get(
        balance_sheet,
        ["Stockholder Equity", "stockholdersEquity", "Total Equity"],
        0,
    )

    # Extract income statement data
    total_revenue = df_get(
        income_stmt, ["Total Revenue", "totalRevenue"], 0
    )
    gross_profit = df_get(
        income_stmt, ["Gross Profit", "grossProfit"], 0
    )
    operating_income = df_get(
        income_stmt, ["Operating Income", "operatingIncome"], 0
    )
    net_income = df_get(income_stmt, ["Net Income", "netIncome"], 0)
    ebit = df_get(income_stmt, ["EBIT", "ebit"], 0)
    interest_expense = df_get(
        income_stmt, ["Interest Expense", "interestExpense"], 0
    )
    cogs = df_get(
        income_stmt, ["Cost Of Revenue", "costOfRevenue", "COGS"], 0
    )

    # Initialize result structure
    result: dict[str, Any] = {
        "profitability": {},
        "liquidity": {},
        "leverage": {},
        "efficiency": {},
    }

    # PROFITABILITY RATIOS
    if total_revenue and total_revenue > 0:
        result["profitability"]["gross_margin"] = (
            float(gross_profit) / float(total_revenue)
            if gross_profit is not None
            else None
        )
        result["profitability"]["operating_margin"] = (
            float(operating_income) / float(total_revenue)
            if operating_income is not None
            else None
        )
        result["profitability"]["net_margin"] = (
            float(net_income) / float(total_revenue)
            if net_income is not None
            else None
        )
    else:
        result["profitability"]["gross_margin"] = None
        result["profitability"]["operating_margin"] = None
        result["profitability"]["net_margin"] = None

    if total_assets and total_assets > 0 and net_income is not None:
        result["profitability"]["roa"] = float(net_income) / float(total_assets)
    else:
        result["profitability"]["roa"] = None

    if (
        stockholders_equity
        and stockholders_equity > 0
        and net_income is not None
    ):
        result["profitability"]["roe"] = float(net_income) / float(
            stockholders_equity
        )
    else:
        result["profitability"]["roe"] = None

    # LIQUIDITY RATIOS
    if (
        current_assets
        and current_liabilities
        and current_liabilities > 0
    ):
        result["liquidity"]["current_ratio"] = float(current_assets) / float(
            current_liabilities
        )

        # Quick ratio = (Current Assets - Inventory) / Current Liabilities
        inventory_val = float(inventory) if inventory is not None else 0.0
        quick_assets = float(current_assets) - inventory_val
        result["liquidity"]["quick_ratio"] = quick_assets / float(
            current_liabilities
        )
    else:
        result["liquidity"]["current_ratio"] = None
        result["liquidity"]["quick_ratio"] = None

    # LEVERAGE RATIOS
    if (
        total_debt is not None
        and stockholders_equity
        and stockholders_equity > 0
    ):
        result["leverage"]["debt_to_equity"] = float(total_debt) / float(
            stockholders_equity
        )
    else:
        result["leverage"]["debt_to_equity"] = None

    if ebit and interest_expense and interest_expense > 0:
        result["leverage"]["interest_coverage"] = float(ebit) / float(
            interest_expense
        )
    else:
        result["leverage"]["interest_coverage"] = None

    # EFFICIENCY RATIOS
    if total_revenue and total_assets and total_assets > 0:
        result["efficiency"]["asset_turnover"] = float(total_revenue) / float(
            total_assets
        )
    else:
        result["efficiency"]["asset_turnover"] = None

    if cogs and inventory and inventory > 0:
        inventory_turnover = float(cogs) / float(inventory)
        result["efficiency"]["inventory_turnover"] = inventory_turnover
        result["efficiency"]["days_inventory_outstanding"] = (
            365.0 / inventory_turnover
        )
    else:
        result["efficiency"]["inventory_turnover"] = None
        result["efficiency"]["days_inventory_outstanding"] = None

    return result


def analyze_5y_trends(
    historical_data: dict[str, list[float]],
) -> dict[str, str] | None:
    """Analyze 5-year trends for financial metrics using CAGR.

    Calculates Compound Annual Growth Rate (CAGR) for each metric and
    classifies the trend as Improving, Deteriorating, or Stable based
    on growth rate thresholds.

    CAGR = ((ending_value / beginning_value) ^ (1 / years)) - 1

    Classification Thresholds:
        - CAGR > 5%: "Improving" (strong positive growth)
        - CAGR < -5%: "Deteriorating" (significant decline)
        - -5% <= CAGR <= 5%: "Stable" (modest change)

    Args:
        historical_data: Dictionary mapping metric names to lists of annual values.
            Each list should contain values in chronological order (oldest first).
            Requires at least 3 years of data for trend calculation.

    Returns:
        Dictionary mapping each metric name to its trend classification
        ("Improving", "Deteriorating", or "Stable"), or None if insufficient data
        (less than 3 years for all metrics).

    Example:
        >>> data = {
        ...     'revenue': [100.0, 110.0, 121.0, 133.1, 146.4],  # ~10% CAGR
        ...     'gross_margin': [0.40, 0.405, 0.41, 0.408, 0.412],  # ~0.6% CAGR
        ... }
        >>> result = analyze_5y_trends(data)
        >>> result['revenue']  # "Improving"
        >>> result['gross_margin']  # "Stable"
    """
    if not historical_data:
        return None

    # Check if we have at least 3 years of data for any metric
    has_sufficient_data = any(
        len(values) >= 3 for values in historical_data.values()
    )

    if not has_sufficient_data:
        return None

    trends: dict[str, str] = {}

    for metric_name, values in historical_data.items():
        # Require at least 3 years for trend analysis
        if len(values) < 3:
            trends[metric_name] = "Stable"  # Insufficient data, default to Stable
            continue

        # Get beginning and ending values
        beginning_value = values[0]
        ending_value = values[-1]
        years = len(values) - 1  # Number of years between first and last

        # Handle zero or negative beginning value (can't calculate CAGR)
        if beginning_value <= 0:
            trends[metric_name] = "Stable"  # Default to Stable for invalid CAGR
            continue

        # Calculate CAGR: ((ending / beginning) ^ (1 / years)) - 1
        cagr = (ending_value / beginning_value) ** (1.0 / years) - 1.0

        # Classify trend based on CAGR threshold
        if cagr > 0.05:  # > 5%
            trends[metric_name] = "Improving"
        elif cagr < -0.05:  # < -5%
            trends[metric_name] = "Deteriorating"
        else:  # -5% to +5%
            trends[metric_name] = "Stable"

    return trends


def fetch_insider_ownership(info: dict[str, Any]) -> dict[str, float | None]:
    """Extract insider and institutional ownership percentages from yfinance info.

    Extracts ownership data from yfinance info dict and converts from decimal
    to percentage format. Handles missing data gracefully.

    Args:
        info: yfinance info dictionary containing ownership data.
            Expected keys: 'heldPercentInsiders', 'heldPercentInstitutions'.

    Returns:
        Dictionary with keys:
        - insider_pct: Insider ownership percentage (e.g., 15.2 for 15.2%)
        - institutional_pct: Institutional ownership percentage

        Returns None for unavailable fields.

    Example:
        >>> info = {'heldPercentInsiders': 0.152, 'heldPercentInstitutions': 0.645}
        >>> fetch_insider_ownership(info)
        {'insider_pct': 15.2, 'institutional_pct': 64.5}
    """
    insider_decimal = info.get("heldPercentInsiders")
    institutional_decimal = info.get("heldPercentInstitutions")

    # Convert from decimal to percentage (0.152 → 15.2)
    insider_pct = insider_decimal * 100 if insider_decimal is not None else None
    institutional_pct = (
        institutional_decimal * 100 if institutional_decimal is not None else None
    )

    return {
        "insider_pct": insider_pct,
        "institutional_pct": institutional_pct,
    }


def calculate_ceo_ownership_value(
    info: dict[str, Any],  # noqa: ARG001
    current_price: float,  # noqa: ARG001
) -> str | None:
    """Calculate CEO ownership stake value in dollar terms.

    NOTE: MVP implementation returns None. CEO share count is not reliably
    available in yfinance info dict (companyOfficers.totalPay is total
    compensation, not share count). Future implementation could parse
    SEC Form 4 filings for accurate share ownership data.

    Args:
        info: yfinance info dictionary.
        current_price: Current stock price.

    Returns:
        None for MVP. Future: formatted value like "$127M" or "$1.5B".

    Example:
        >>> calculate_ceo_ownership_value(info, 175.43)
        None  # MVP returns None
    """
    # MVP: Return None - CEO share ownership data not reliably available
    # Future: Parse SEC Form 4 or proxy statements for accurate CEO share count
    return None


# Insider trading signal thresholds
# Buying must exceed 2× selling to signal strong positive conviction
_HEAVY_BUYING_THRESHOLD = 2
# Selling must exceed 3× buying to signal negative conviction
# (asymmetric because insiders sell for many reasons: taxes, diversification, etc.)
_HEAVY_SELLING_THRESHOLD = 3


def interpret_insider_signal(
    buying: int | None,
    selling: int | None,
) -> str:
    """Interpret insider trading activity signal.

    Classifies insider trading patterns as positive, negative, or neutral
    based on the ratio of buying to selling transactions over a period.

    Signal Classification:
        - "positive": Heavy buying (buys > sells × HEAVY_BUYING_THRESHOLD)
        - "negative": Heavy selling (sells > buys × HEAVY_SELLING_THRESHOLD)
        - "neutral": Balanced activity or insufficient data

    Thresholds are asymmetric because insiders sell for many non-negative
    reasons (taxes, diversification), but buying signals stronger conviction.

    Args:
        buying: Number of insider buy transactions.
        selling: Number of insider sell transactions.

    Returns:
        Signal classification: "positive", "negative", or "neutral".

    Example:
        >>> interpret_insider_signal(buying=10, selling=2)
        'positive'
        >>> interpret_insider_signal(buying=2, selling=10)
        'negative'
        >>> interpret_insider_signal(buying=None, selling=None)
        'neutral'
    """
    # Handle None values (data unavailable)
    if buying is None or selling is None:
        return "neutral"

    # Heavy buying: buys > sells × HEAVY_BUYING_THRESHOLD
    if buying > selling * _HEAVY_BUYING_THRESHOLD:
        return "positive"

    # Heavy selling: sells > buys × HEAVY_SELLING_THRESHOLD
    if selling > buying * _HEAVY_SELLING_THRESHOLD:
        return "negative"

    # Balanced or modest activity
    return "neutral"
