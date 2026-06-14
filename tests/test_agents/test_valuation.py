"""Tests for ValuationAgent and shared valuation utilities."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from doxa_shared.utils.valuation import (
    calculate_dcf_fair_value,
    calculate_free_cash_flow,
    calculate_terminal_value,
    calculate_valuation_multiples,
    calculate_wacc,
    generate_sensitivity_table,
)

from src.agents.valuation import ValuationAgent

# ---------------------------------------------------------------------------
# Shared utility unit tests
# ---------------------------------------------------------------------------


class TestCalculateFreeCashFlow:
    """Tests for calculate_free_cash_flow."""

    def test_projects_five_years_from_growing_data(self) -> None:
        data = {"operating_cash_flow": [110.0, 100.0, 90.0]}
        result = calculate_free_cash_flow(data)
        assert len(result) == 5
        # Growth rate = (110/100) - 1 = 10%
        assert pytest.approx(result[0], rel=1e-3) == 121.0
        assert pytest.approx(result[1], rel=1e-3) == 133.1

    def test_returns_empty_for_single_data_point(self) -> None:
        data = {"operating_cash_flow": [100.0]}
        assert calculate_free_cash_flow(data) == []

    def test_returns_empty_for_empty_data(self) -> None:
        assert calculate_free_cash_flow({}) == []

    def test_returns_empty_when_previous_is_zero(self) -> None:
        data = {"operating_cash_flow": [100.0, 0.0]}
        assert calculate_free_cash_flow(data) == []

    def test_returns_empty_when_previous_is_negative(self) -> None:
        data = {"operating_cash_flow": [100.0, -50.0]}
        assert calculate_free_cash_flow(data) == []


class TestCalculateWacc:
    """Tests for calculate_wacc."""

    def test_basic_wacc_calculation(self) -> None:
        wacc = calculate_wacc(
            beta=1.0,
            risk_free_rate=0.045,
            market_risk_premium=0.07,
            debt_equity_ratio=0.5,
            tax_rate=0.21,
        )
        # Cost of equity = 0.045 + 1.0 * 0.07 = 0.115
        # Cost of debt = 0.045 + 0.02 = 0.065, after-tax = 0.065 * 0.79 = 0.05135
        # Weight equity = 1/1.5 = 0.6667, Weight debt = 0.5/1.5 = 0.3333
        # WACC = 0.6667 * 0.115 + 0.3333 * 0.05135
        assert 0.05 < wacc < 0.15

    def test_zero_debt_returns_cost_of_equity(self) -> None:
        wacc = calculate_wacc(
            beta=1.2,
            risk_free_rate=0.045,
            market_risk_premium=0.07,
            debt_equity_ratio=0.0,
            tax_rate=0.21,
        )
        # With zero debt, WACC should equal cost of equity
        expected_cost_of_equity = 0.045 + 1.2 * 0.07
        assert pytest.approx(wacc, rel=1e-6) == expected_cost_of_equity

    def test_high_beta_produces_higher_wacc(self) -> None:
        low_beta = calculate_wacc(1.0, 0.045, 0.07, 0.5, 0.21)
        high_beta = calculate_wacc(2.0, 0.045, 0.07, 0.5, 0.21)
        assert high_beta > low_beta


class TestCalculateTerminalValue:
    """Tests for calculate_terminal_value."""

    def test_basic_calculation(self) -> None:
        tv = calculate_terminal_value(100.0, 0.025, 0.10)
        # TV = 100 * 1.025 / (0.10 - 0.025) = 102.5 / 0.075 = 1366.67
        assert pytest.approx(tv, rel=1e-3) == 1366.67

    def test_returns_zero_when_growth_equals_wacc(self) -> None:
        assert calculate_terminal_value(100.0, 0.10, 0.10) == 0.0

    def test_returns_zero_when_growth_exceeds_wacc(self) -> None:
        assert calculate_terminal_value(100.0, 0.15, 0.10) == 0.0


class TestCalculateDcfFairValue:
    """Tests for calculate_dcf_fair_value."""

    def test_basic_fair_value(self) -> None:
        fcf = [100.0, 110.0, 121.0, 133.1, 146.41]
        fv = calculate_dcf_fair_value(fcf, 2000.0, 0.10, 1000.0)
        assert fv > 0

    def test_returns_zero_for_zero_shares(self) -> None:
        fcf = [100.0, 110.0, 121.0, 133.1, 146.41]
        assert calculate_dcf_fair_value(fcf, 2000.0, 0.10, 0.0) == 0.0

    def test_discounting_reduces_value(self) -> None:
        fcf = [100.0, 100.0, 100.0, 100.0, 100.0]
        # Higher WACC = more discounting = lower fair value
        fv_low_wacc = calculate_dcf_fair_value(fcf, 1000.0, 0.05, 100.0)
        fv_high_wacc = calculate_dcf_fair_value(fcf, 1000.0, 0.15, 100.0)
        assert fv_low_wacc > fv_high_wacc


class TestCalculateValuationMultiples:
    """Tests for calculate_valuation_multiples."""

    def test_all_multiples_calculated(self) -> None:
        result = calculate_valuation_multiples(
            market_cap=1000.0,
            revenue=500.0,
            ebitda=200.0,
            book_value=400.0,
            net_income=50.0,
            total_debt=100.0,
            cash=50.0,
        )
        assert result["P/E"] == pytest.approx(20.0)
        # EV = 1000 + 100 - 50 = 1050; EV/EBITDA = 1050/200 = 5.25
        assert result["EV/EBITDA"] == pytest.approx(5.25)
        assert result["P/B"] == pytest.approx(2.5)
        assert result["P/S"] == pytest.approx(2.0)

    def test_none_for_zero_denominators(self) -> None:
        result = calculate_valuation_multiples(
            market_cap=1000.0,
            revenue=0.0,
            ebitda=0.0,
            book_value=0.0,
            net_income=0.0,
        )
        assert result["P/E"] is None
        assert result["EV/EBITDA"] is None
        assert result["P/B"] is None
        assert result["P/S"] is None

    def test_defaults_to_zero_debt_and_cash(self) -> None:
        result = calculate_valuation_multiples(
            market_cap=1000.0,
            revenue=500.0,
            ebitda=200.0,
            book_value=400.0,
            net_income=50.0,
        )
        # EV = 1000 + 0 - 0 = 1000; EV/EBITDA = 1000/200 = 5.0
        assert result["EV/EBITDA"] == pytest.approx(5.0)


class TestGenerateSensitivityTable:
    """Tests for generate_sensitivity_table."""

    def test_generates_5x5_grid(self) -> None:
        fcf = [100.0, 110.0, 121.0, 133.1, 146.41]
        table = generate_sensitivity_table(
            base_wacc=0.10,
            base_growth=0.025,
            fcf_projections=fcf,
            shares_outstanding=1000.0,
        )
        assert len(table) == 5  # 5 WACC values
        for wacc_key, growth_values in table.items():
            assert len(growth_values) == 5  # 5 growth values each

    def test_lower_wacc_produces_higher_values(self) -> None:
        fcf = [100.0, 110.0, 121.0, 133.1, 146.41]
        table = generate_sensitivity_table(
            base_wacc=0.10,
            base_growth=0.025,
            fcf_projections=fcf,
            shares_outstanding=1000.0,
        )
        # Lower WACC should produce higher fair values
        low_wacc_val = table["8.0%"]["2.5%"]
        high_wacc_val = table["12.0%"]["2.5%"]
        assert low_wacc_val > high_wacc_val

    def test_returns_zeros_for_short_projections(self) -> None:
        fcf = [100.0, 110.0]  # Only 2 projections, need 5
        table = generate_sensitivity_table(
            base_wacc=0.10,
            base_growth=0.025,
            fcf_projections=fcf,
            shares_outstanding=1000.0,
        )
        for growth_values in table.values():
            for val in growth_values.values():
                assert val == 0.0


# ---------------------------------------------------------------------------
# ValuationAgent integration tests
# ---------------------------------------------------------------------------


def _make_yfinance_mock() -> Mock:
    """Create a properly structured yfinance Ticker mock.

    yfinance DataFrames have financial line items as rows (index)
    and dates as columns.
    """
    mock_company = Mock()

    # Cashflow: rows = line items, columns = dates
    mock_company.cashflow = pd.DataFrame(
        {
            "2024-09-30": [120_000_000_000.0],
            "2023-09-30": [100_000_000_000.0],
            "2022-09-30": [90_000_000_000.0],
        },
        index=["Operating Cash Flow"],
    )

    # Income statement
    mock_company.income_stmt = pd.DataFrame(
        {
            "2024-09-30": [25_000_000_000.0, 120_000_000_000.0],
        },
        index=["Tax Provision", "Pretax Income"],
    )

    # Balance sheet: rows = line items, columns = dates
    mock_company.balance_sheet = pd.DataFrame(
        {
            "2024-09-30": [110_000_000_000.0, 70_000_000_000.0],
        },
        index=["Total Debt", "Stockholders Equity"],
    )

    # Company info
    mock_company.info = {
        "beta": 1.2,
        "sharesOutstanding": 15_000_000_000,
        "currentPrice": 190.0,
        "sector": "Technology",
        "marketCap": 2_850_000_000_000,
        "totalRevenue": 390_000_000_000,
        "ebitda": 130_000_000_000,
        "bookValue": 4.5,
        "netIncome": 95_000_000_000,
        "totalDebt": 110_000_000_000,
        "totalCash": 60_000_000_000,
    }

    return mock_company


class TestValuationAgentExecution:
    """Integration tests for ValuationAgent.execute()."""

    def test_basic_execution_populates_all_sections(self) -> None:
        agent = ValuationAgent()

        with patch("src.agents.valuation.yf.Ticker") as mock_ticker:
            mock_company = _make_yfinance_mock()
            mock_ticker.return_value = mock_company

            state: dict = {
                "ticker": "AAPL",
                "errors": [],
                "valuation_analysis": {},
            }
            result = agent.execute(state)

        va = result["valuation_analysis"]
        assert "dcf" in va
        assert "comps" in va
        assert "confidence" in va

    def test_dcf_produces_meaningful_values(self) -> None:
        agent = ValuationAgent()

        with patch("src.agents.valuation.yf.Ticker") as mock_ticker:
            mock_company = _make_yfinance_mock()
            mock_ticker.return_value = mock_company

            state: dict = {
                "ticker": "AAPL",
                "errors": [],
                "valuation_analysis": {},
            }
            result = agent.execute(state)

        dcf = result["valuation_analysis"]["dcf"]
        # DCF should actually produce values (not empty dict)
        assert dcf, "DCF should produce non-empty results"
        assert len(dcf["fcf_projections"]) == 5
        assert dcf["wacc"] > 0
        assert dcf["terminal_value"] > 0
        assert dcf["fair_value_per_share"] > 0
        assert "sensitivity_table" in dcf

    def test_comps_produces_peer_multiples(self) -> None:
        agent = ValuationAgent()

        with patch("src.agents.valuation.yf.Ticker") as mock_ticker:
            mock_company = _make_yfinance_mock()
            mock_ticker.return_value = mock_company

            # All Ticker() calls return the same mock (target + peers)
            mock_ticker.return_value = mock_company

            state: dict = {
                "ticker": "TEST",
                "errors": [],
                "valuation_analysis": {},
            }
            result = agent.execute(state)

        comps = result["valuation_analysis"]["comps"]
        assert comps, "Comps should produce non-empty results"
        assert len(comps["peer_companies"]) > 0
        assert len(comps["peer_multiples"]) > 0
        assert comps["median_multiples"] is not None

    def test_handles_missing_ticker(self) -> None:
        agent = ValuationAgent()

        state: dict = {"errors": [], "valuation_analysis": {}}
        result = agent.execute(state)

        assert len(result["errors"]) > 0
        assert "No ticker" in result["errors"][0]

    def test_error_accumulation_never_raises(self) -> None:
        agent = ValuationAgent()

        with patch("src.agents.valuation.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = Exception("API failure")

            state: dict = {
                "ticker": "INVALID",
                "errors": [],
                "valuation_analysis": {},
            }
            result = agent.execute(state)

        assert len(result["errors"]) > 0
        assert result["valuation_analysis"]["confidence"] == 0.0

    def test_returns_same_state_object(self) -> None:
        agent = ValuationAgent()

        with patch("src.agents.valuation.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = Exception("fail")

            state: dict = {
                "ticker": "TEST",
                "errors": [],
                "valuation_analysis": {},
            }
            result = agent.execute(state)

        assert result is state


class TestConfidenceScoring:
    """Tests for _calculate_confidence."""

    def test_complete_data_gives_100(self) -> None:
        agent = ValuationAgent()

        dcf_data = {
            "fcf_projections": [100, 110, 121, 133, 146],
            "wacc": 0.10,
            "terminal_value": 2000,
        }
        comps_data = {
            "peer_multiples": {
                "A": {"P/E": 25, "EV/EBITDA": 15},
                "B": {"P/E": 30, "EV/EBITDA": 18},
                "C": {"P/E": 22, "EV/EBITDA": 12},
                "D": {"P/E": 20, "EV/EBITDA": 10},
            }
        }
        quant_data = {
            "dupont_analysis": {"roe": 0.20},
            "altman_z_score": {"z_score": 3.5},
            "financial_ratios": {
                "profitability": {
                    "gross_margin": 0.40,
                    "operating_margin": 0.25,
                    "net_margin": 0.15,
                    "roa": 0.10,
                    "roe": 0.20,
                },
                "liquidity": {"current_ratio": 2.0, "quick_ratio": 1.5},
                "leverage": {"debt_to_equity": 0.5, "interest_coverage": 10.0},
                "efficiency": {"asset_turnover": 0.8},
            },
            "trend_analysis": {
                "revenue": "Improving",
                "gross_margin": "Stable",
                "roe": "Improving",
                "debt_equity": "Stable",
                "current_ratio": "Stable",
            },
        }
        assert agent._calculate_confidence(dcf_data, comps_data, quant_data) == 100.0

    def test_partial_data_gives_intermediate(self) -> None:
        agent = ValuationAgent()

        dcf_data = {
            "fcf_projections": [100, 110, 121],
            "wacc": 0.10,
            "terminal_value": 0,
        }
        comps_data = {
            "peer_multiples": {
                "A": {"P/E": 25},
                "B": {"P/E": 30},
            }
        }
        quant_data = {
            "dupont_analysis": None,
            "altman_z_score": None,
            "financial_ratios": None,
            "trend_analysis": None,
        }
        confidence = agent._calculate_confidence(dcf_data, comps_data, quant_data)
        assert 0 < confidence < 100

    def test_no_data_gives_zero(self) -> None:
        agent = ValuationAgent()
        quant_data = {
            "dupont_analysis": None,
            "altman_z_score": None,
            "financial_ratios": None,
            "trend_analysis": None,
        }
        assert agent._calculate_confidence({}, {}, quant_data) == 0.0
