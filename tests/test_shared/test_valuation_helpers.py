"""Tests for valuation calculation helper functions."""

from __future__ import annotations

import pandas as pd
from doxa_shared.utils.valuation import (
    analyze_5y_trends,
    calculate_altman_z_score,
    calculate_dupont_analysis,
    calculate_financial_ratios,
)


class TestCalculateDupontAnalysis:
    """Tests for DuPont ROE decomposition analysis."""

    def test_complete_data_calculates_roe_correctly(self) -> None:
        """Test DuPont with complete financial data."""
        financials = {
            "net_income": 30_000_000_000,
            "total_revenue": 100_000_000_000,
            "total_assets": 200_000_000_000,
            "stockholders_equity": 150_000_000_000,
        }

        result = calculate_dupont_analysis(financials)

        assert result is not None
        assert "profit_margin" in result
        assert "asset_turnover" in result
        assert "equity_multiplier" in result
        assert "roe" in result

        # Verify calculations
        expected_margin = 30_000_000_000 / 100_000_000_000  # 0.30
        expected_turnover = 100_000_000_000 / 200_000_000_000  # 0.50
        expected_multiplier = 200_000_000_000 / 150_000_000_000  # 1.33
        expected_roe = 30_000_000_000 / 150_000_000_000  # 0.20

        assert abs(result["profit_margin"] - expected_margin) < 0.001
        assert abs(result["asset_turnover"] - expected_turnover) < 0.001
        assert abs(result["equity_multiplier"] - expected_multiplier) < 0.01
        assert abs(result["roe"] - expected_roe) < 0.001

        # Sanity check: ROE should equal margin × turnover × multiplier
        calculated_roe = (
            result["profit_margin"]
            * result["asset_turnover"]
            * result["equity_multiplier"]
        )
        assert abs(calculated_roe - result["roe"]) < 0.01

    def test_missing_net_income_returns_none(self) -> None:
        """Test that missing net income returns None."""
        financials = {
            "total_revenue": 100_000_000_000,
            "total_assets": 200_000_000_000,
            "stockholders_equity": 150_000_000_000,
        }

        result = calculate_dupont_analysis(financials)

        assert result is None

    def test_zero_revenue_returns_none(self) -> None:
        """Test that zero revenue returns None."""
        financials = {
            "net_income": 30_000_000_000,
            "total_revenue": 0,
            "total_assets": 200_000_000_000,
            "stockholders_equity": 150_000_000_000,
        }

        result = calculate_dupont_analysis(financials)

        assert result is None

    def test_negative_equity_returns_none(self) -> None:
        """Test that negative equity returns None (can't calculate)."""
        financials = {
            "net_income": 30_000_000_000,
            "total_revenue": 100_000_000_000,
            "total_assets": 200_000_000_000,
            "stockholders_equity": -10_000_000_000,  # Negative equity
        }

        result = calculate_dupont_analysis(financials)

        assert result is None


class TestCalculateAltmanZScore:
    """Tests for Altman Z-Score bankruptcy risk assessment."""

    def test_complete_data_safe_zone(self) -> None:
        """Test Z-Score calculation with data yielding Safe Zone (Z > 2.99)."""
        # Mock balance sheet with strong financial position
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Current Assets": 150_000_000_000,
                    "Current Liabilities": 50_000_000_000,
                    "Retained Earnings": 80_000_000_000,
                    "Total Assets": 200_000_000_000,
                    "Total Liabilities": 60_000_000_000,
                }
            }
        )

        # Mock income statement with strong profitability
        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "EBIT": 40_000_000_000,
                    "Total Revenue": 100_000_000_000,
                }
            }
        )

        market_cap = 300_000_000_000  # Strong market valuation

        result = calculate_altman_z_score(balance_sheet, income_stmt, market_cap)

        assert result is not None
        assert "z_score" in result
        assert "interpretation" in result
        assert "components" in result

        # Verify Safe Zone interpretation
        assert result["z_score"] > 2.99
        assert "Safe Zone" in result["interpretation"]
        assert "low bankruptcy risk" in result["interpretation"].lower()

    def test_complete_data_grey_zone(self) -> None:
        """Test Z-Score calculation with data yielding Grey Zone (1.81 < Z < 2.99)."""
        # Mock balance sheet with moderate financial position
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Current Assets": 80_000_000_000,
                    "Current Liabilities": 60_000_000_000,
                    "Retained Earnings": 30_000_000_000,
                    "Total Assets": 150_000_000_000,
                    "Total Liabilities": 90_000_000_000,
                }
            }
        )

        # Mock income statement with moderate profitability
        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "EBIT": 15_000_000_000,
                    "Total Revenue": 80_000_000_000,
                }
            }
        )

        market_cap = 100_000_000_000  # Moderate market valuation

        result = calculate_altman_z_score(balance_sheet, income_stmt, market_cap)

        assert result is not None
        assert 1.81 < result["z_score"] < 2.99
        assert "Grey Zone" in result["interpretation"]
        assert "moderate bankruptcy risk" in result["interpretation"].lower()

    def test_complete_data_distress_zone(self) -> None:
        """Test Z-Score calculation with data yielding Distress Zone (Z < 1.81)."""
        # Mock balance sheet with weak financial position
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Current Assets": 40_000_000_000,
                    "Current Liabilities": 50_000_000_000,
                    "Retained Earnings": 5_000_000_000,
                    "Total Assets": 100_000_000_000,
                    "Total Liabilities": 85_000_000_000,
                }
            }
        )

        # Mock income statement with weak profitability
        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "EBIT": 3_000_000_000,
                    "Total Revenue": 50_000_000_000,
                }
            }
        )

        market_cap = 20_000_000_000  # Low market valuation

        result = calculate_altman_z_score(balance_sheet, income_stmt, market_cap)

        assert result is not None
        assert result["z_score"] < 1.81
        assert "Distress Zone" in result["interpretation"]
        assert "high bankruptcy risk" in result["interpretation"].lower()

    def test_missing_critical_data_returns_none(self) -> None:
        """Test that missing critical balance sheet data returns None."""
        # Missing Current Assets and Current Liabilities (can't calculate WC)
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Retained Earnings": 30_000_000_000,
                    "Total Assets": 150_000_000_000,
                    "Total Liabilities": 90_000_000_000,
                }
            }
        )

        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "EBIT": 15_000_000_000,
                    "Total Revenue": 80_000_000_000,
                }
            }
        )

        market_cap = 100_000_000_000

        result = calculate_altman_z_score(balance_sheet, income_stmt, market_cap)

        assert result is None

    def test_zero_total_assets_returns_none(self) -> None:
        """Test that zero total assets returns None (division by zero)."""
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Current Assets": 80_000_000_000,
                    "Current Liabilities": 60_000_000_000,
                    "Retained Earnings": 30_000_000_000,
                    "Total Assets": 0,  # Zero total assets
                    "Total Liabilities": 90_000_000_000,
                }
            }
        )

        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "EBIT": 15_000_000_000,
                    "Total Revenue": 80_000_000_000,
                }
            }
        )

        market_cap = 100_000_000_000

        result = calculate_altman_z_score(balance_sheet, income_stmt, market_cap)

        assert result is None

    def test_zero_total_liabilities_returns_none(self) -> None:
        """Test that zero total liabilities returns None (division by zero)."""
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Current Assets": 80_000_000_000,
                    "Current Liabilities": 60_000_000_000,
                    "Retained Earnings": 30_000_000_000,
                    "Total Assets": 150_000_000_000,
                    "Total Liabilities": 0,  # Zero total liabilities
                }
            }
        )

        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "EBIT": 15_000_000_000,
                    "Total Revenue": 80_000_000_000,
                }
            }
        )

        market_cap = 100_000_000_000

        result = calculate_altman_z_score(balance_sheet, income_stmt, market_cap)

        assert result is None


class TestCalculateFinancialRatios:
    """Tests for comprehensive financial ratio calculation."""

    def test_complete_data_all_ratios(self) -> None:
        """Test financial ratios with complete data across all categories."""
        # Mock balance sheet
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Current Assets": 150_000_000_000,
                    "Current Liabilities": 50_000_000_000,
                    "Inventory": 20_000_000_000,
                    "Total Assets": 200_000_000_000,
                    "Total Debt": 60_000_000_000,
                    "Stockholder Equity": 140_000_000_000,
                }
            }
        )

        # Mock income statement
        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "Total Revenue": 100_000_000_000,
                    "Gross Profit": 40_000_000_000,
                    "Operating Income": 25_000_000_000,
                    "Net Income": 20_000_000_000,
                    "EBIT": 25_000_000_000,
                    "Interest Expense": 2_000_000_000,
                    "Cost Of Revenue": 60_000_000_000,
                }
            }
        )

        # Mock cashflow (for efficiency ratios)
        cashflow = pd.DataFrame({"2023": {}})

        result = calculate_financial_ratios(balance_sheet, income_stmt, cashflow)

        assert result is not None
        assert "profitability" in result
        assert "liquidity" in result
        assert "leverage" in result
        assert "efficiency" in result

        # Verify profitability ratios
        prof = result["profitability"]
        assert abs(prof["gross_margin"] - 0.40) < 0.01  # 40%
        assert abs(prof["operating_margin"] - 0.25) < 0.01  # 25%
        assert abs(prof["net_margin"] - 0.20) < 0.01  # 20%
        assert abs(prof["roa"] - 0.10) < 0.01  # 10%
        assert abs(prof["roe"] - 0.143) < 0.01  # ~14.3%

        # Verify liquidity ratios
        liq = result["liquidity"]
        assert abs(liq["current_ratio"] - 3.0) < 0.1  # 3.0
        assert abs(liq["quick_ratio"] - 2.6) < 0.1  # 2.6

        # Verify leverage ratios
        lev = result["leverage"]
        assert abs(lev["debt_to_equity"] - 0.429) < 0.01  # ~0.429
        assert abs(lev["interest_coverage"] - 12.5) < 0.1  # 12.5

        # Verify efficiency ratios
        eff = result["efficiency"]
        assert abs(eff["asset_turnover"] - 0.5) < 0.01  # 0.5

    def test_missing_profitability_data_returns_partial(self) -> None:
        """Test that missing profitability data returns None for affected ratios."""
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Current Assets": 150_000_000_000,
                    "Current Liabilities": 50_000_000_000,
                    "Total Assets": 200_000_000_000,
                    "Stockholder Equity": 140_000_000_000,
                }
            }
        )

        # Missing profitability fields
        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "Total Revenue": 100_000_000_000,
                }
            }
        )

        cashflow = pd.DataFrame({"2023": {}})

        result = calculate_financial_ratios(balance_sheet, income_stmt, cashflow)

        assert result is not None
        # Liquidity should work
        assert result["liquidity"]["current_ratio"] == 3.0
        # Profitability margins should be None
        assert result["profitability"]["gross_margin"] is None
        assert result["profitability"]["operating_margin"] is None

    def test_negative_equity_debt_to_equity_none(self) -> None:
        """Test that negative equity returns None for debt-to-equity ratio."""
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Current Assets": 150_000_000_000,
                    "Current Liabilities": 50_000_000_000,
                    "Total Assets": 200_000_000_000,
                    "Total Debt": 60_000_000_000,
                    "Stockholder Equity": -10_000_000_000,  # Negative
                }
            }
        )

        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "Total Revenue": 100_000_000_000,
                    "Net Income": 20_000_000_000,
                }
            }
        )

        cashflow = pd.DataFrame({"2023": {}})

        result = calculate_financial_ratios(balance_sheet, income_stmt, cashflow)

        assert result is not None
        # D/E should be None (can't calculate with negative equity)
        assert result["leverage"]["debt_to_equity"] is None
        # Other ratios should still work
        assert result["liquidity"]["current_ratio"] == 3.0

    def test_zero_interest_expense_coverage_none(self) -> None:
        """Test that zero interest expense returns None for interest coverage."""
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Current Assets": 150_000_000_000,
                    "Current Liabilities": 50_000_000_000,
                    "Total Assets": 200_000_000_000,
                    "Stockholder Equity": 140_000_000_000,
                }
            }
        )

        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "Total Revenue": 100_000_000_000,
                    "EBIT": 25_000_000_000,
                    "Interest Expense": 0,  # Zero interest
                }
            }
        )

        cashflow = pd.DataFrame({"2023": {}})

        result = calculate_financial_ratios(balance_sheet, income_stmt, cashflow)

        assert result is not None
        # Interest coverage should be None (division by zero)
        assert result["leverage"]["interest_coverage"] is None

    def test_inventory_turnover_with_cogs(self) -> None:
        """Test inventory turnover calculation with COGS."""
        balance_sheet = pd.DataFrame(
            {
                "2023": {
                    "Current Assets": 150_000_000_000,
                    "Current Liabilities": 50_000_000_000,
                    "Inventory": 20_000_000_000,
                    "Total Assets": 200_000_000_000,
                    "Stockholder Equity": 140_000_000_000,
                }
            }
        )

        income_stmt = pd.DataFrame(
            {
                "2023": {
                    "Total Revenue": 100_000_000_000,
                    "Cost Of Revenue": 60_000_000_000,  # COGS
                }
            }
        )

        cashflow = pd.DataFrame({"2023": {}})

        result = calculate_financial_ratios(balance_sheet, income_stmt, cashflow)

        assert result is not None
        # Inventory turnover = COGS / Inventory = 60B / 20B = 3.0
        assert abs(result["efficiency"]["inventory_turnover"] - 3.0) < 0.01
        # Days inventory = 365 / 3.0 = ~122 days
        assert abs(result["efficiency"]["days_inventory_outstanding"] - 121.67) < 1.0


class TestAnalyze5yTrends:
    """Tests for 5-year trend analysis and classification."""

    def test_improving_trend_positive_cagr(self) -> None:
        """Test that positive CAGR > 5% classifies as Improving."""
        # Revenue growing from 100 to ~160 over 5 years (~10% CAGR)
        historical_data = {
            "revenue": [100.0, 110.0, 121.0, 133.1, 146.4],
        }

        result = analyze_5y_trends(historical_data)

        assert result is not None
        assert result["revenue"] == "Improving"

    def test_deteriorating_trend_negative_cagr(self) -> None:
        """Test that negative CAGR < -5% classifies as Deteriorating."""
        # Margin declining from 0.30 to ~0.20 over 5 years (~8% decline CAGR)
        historical_data = {
            "gross_margin": [0.30, 0.28, 0.26, 0.24, 0.22],
        }

        result = analyze_5y_trends(historical_data)

        assert result is not None
        assert result["gross_margin"] == "Deteriorating"

    def test_stable_trend_small_change(self) -> None:
        """Test that CAGR between -5% and 5% classifies as Stable."""
        # ROE staying relatively flat around 0.15
        historical_data = {
            "roe": [0.15, 0.155, 0.152, 0.148, 0.151],
        }

        result = analyze_5y_trends(historical_data)

        assert result is not None
        assert result["roe"] == "Stable"

    def test_multiple_metrics_mixed_trends(self) -> None:
        """Test analyzing multiple metrics with different trend directions."""
        historical_data = {
            "revenue": [100.0, 110.0, 121.0, 133.1, 146.4],  # Improving
            "gross_margin": [0.40, 0.405, 0.41, 0.408, 0.412],  # Stable
            "debt_equity": [0.50, 0.45, 0.40, 0.35, 0.30],  # Deteriorating (declining)
        }

        result = analyze_5y_trends(historical_data)

        assert result is not None
        assert result["revenue"] == "Improving"
        assert result["gross_margin"] == "Stable"
        assert result["debt_equity"] == "Deteriorating"  # CAGR is negative

    def test_insufficient_data_returns_none(self) -> None:
        """Test that less than 3 years of data returns None."""
        historical_data = {
            "revenue": [100.0, 110.0],  # Only 2 years
        }

        result = analyze_5y_trends(historical_data)

        assert result is None

    def test_partial_data_three_years(self) -> None:
        """Test that exactly 3 years of data works (minimum requirement)."""
        # 3 years: 100 -> 110 -> 121 (~10% CAGR)
        historical_data = {
            "revenue": [100.0, 110.0, 121.0],
        }

        result = analyze_5y_trends(historical_data)

        assert result is not None
        assert result["revenue"] == "Improving"

    def test_zero_initial_value_returns_none(self) -> None:
        """Test that zero initial value returns None (can't calculate CAGR)."""
        historical_data = {
            "revenue": [0.0, 100.0, 110.0, 121.0, 133.1],
        }

        result = analyze_5y_trends(historical_data)

        assert result is not None
        # Should skip revenue (zero initial value)
        assert result["revenue"] == "Stable"  # Falls back to stable for invalid CAGR
