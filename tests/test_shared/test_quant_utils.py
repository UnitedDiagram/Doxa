"""Tests for shared quantitative analysis utility functions."""

from __future__ import annotations

from doxa_shared.utils.quant import (
    altman_zone,
    compute_altman_z,
    compute_asset_turnover,
    compute_equity_multiplier,
    compute_profit_margin,
    compute_roe,
    derive_dupont_driver,
)


class TestComputeProfitMargin:
    """Tests for compute_profit_margin."""

    def test_normal_calculation(self) -> None:
        fin = {"net_income": 30.0, "total_revenue": 100.0}
        assert compute_profit_margin(fin) == 0.3

    def test_returns_none_when_missing_revenue(self) -> None:
        assert compute_profit_margin({"net_income": 30.0}) is None

    def test_returns_none_when_missing_income(self) -> None:
        assert compute_profit_margin({"total_revenue": 100.0}) is None

    def test_returns_none_when_revenue_is_zero(self) -> None:
        fin = {"net_income": 30.0, "total_revenue": 0}
        assert compute_profit_margin(fin) is None


class TestComputeAssetTurnover:
    """Tests for compute_asset_turnover."""

    def test_with_current_and_previous_assets(self) -> None:
        fin = {
            "total_revenue": 200.0,
            "total_assets": 100.0,
            "total_assets_prev": 80.0,
        }
        # avg_assets = (100+80)/2 = 90; turnover = 200/90
        result = compute_asset_turnover(fin)
        assert result is not None
        assert abs(result - 200.0 / 90.0) < 0.001

    def test_with_only_current_assets(self) -> None:
        fin = {"total_revenue": 200.0, "total_assets": 100.0}
        assert compute_asset_turnover(fin) == 2.0

    def test_returns_none_when_missing_revenue(self) -> None:
        assert compute_asset_turnover({"total_assets": 100.0}) is None

    def test_returns_none_when_missing_assets(self) -> None:
        assert compute_asset_turnover({"total_revenue": 200.0}) is None


class TestComputeEquityMultiplier:
    """Tests for compute_equity_multiplier."""

    def test_normal_calculation(self) -> None:
        fin = {"total_assets": 200.0, "stockholders_equity": 100.0}
        assert compute_equity_multiplier(fin) == 2.0

    def test_returns_none_when_equity_is_zero(self) -> None:
        fin = {"total_assets": 200.0, "stockholders_equity": 0}
        assert compute_equity_multiplier(fin) is None

    def test_returns_none_when_missing_assets(self) -> None:
        assert compute_equity_multiplier({"stockholders_equity": 100.0}) is None


class TestComputeRoe:
    """Tests for compute_roe (DuPont 3-factor)."""

    def test_normal_roe(self) -> None:
        # ROE = 0.3 * 2.0 * 2.0 = 1.2
        assert compute_roe(0.3, 2.0, 2.0) == 1.2

    def test_returns_none_with_missing_component(self) -> None:
        assert compute_roe(None, 2.0, 2.0) is None
        assert compute_roe(0.3, None, 2.0) is None
        assert compute_roe(0.3, 2.0, None) is None


class TestDeriveDupontDriver:
    """Tests for derive_dupont_driver."""

    def test_high_profitability(self) -> None:
        assert derive_dupont_driver(0.25, 1.0, 2.0) == "High Profitability"

    def test_high_asset_efficiency(self) -> None:
        assert derive_dupont_driver(0.10, 2.0, 2.0) == "High Asset Efficiency"

    def test_high_leverage_low_margin(self) -> None:
        assert derive_dupont_driver(0.05, 1.0, 4.0) == "High Leverage, Low Margin"

    def test_balanced(self) -> None:
        assert derive_dupont_driver(0.15, 1.2, 2.0) == "Balanced"

    def test_insufficient_data(self) -> None:
        assert derive_dupont_driver(None, 1.0, 2.0) == "Insufficient Data"


class TestComputeAltmanZ:
    """Tests for compute_altman_z."""

    def test_normal_calculation(self) -> None:
        md = {"market_cap": 1000.0}
        fin = {
            "total_assets": 500.0,
            "working_capital": 100.0,
            "retained_earnings": 200.0,
            "ebit": 150.0,
            "total_liabilities": 300.0,
            "total_revenue": 400.0,
        }
        z = compute_altman_z(md, fin)
        assert z is not None
        assert z > 0

    def test_returns_none_when_no_total_assets(self) -> None:
        assert compute_altman_z({"market_cap": 1000.0}, {}) is None

    def test_returns_none_when_total_assets_zero(self) -> None:
        fin = {"total_assets": 0}
        assert compute_altman_z({"market_cap": 1000.0}, fin) is None

    def test_handles_missing_components(self) -> None:
        md = {"market_cap": 1000.0}
        fin = {"total_assets": 500.0}
        z = compute_altman_z(md, fin)
        assert z is not None


class TestAltmanZone:
    """Tests for altman_zone classification."""

    def test_safe_zone(self) -> None:
        assert altman_zone(3.5) == "Safe"

    def test_grey_zone(self) -> None:
        assert altman_zone(2.5) == "Grey"

    def test_distress_zone(self) -> None:
        assert altman_zone(1.5) == "Distress"

    def test_boundary_safe(self) -> None:
        assert altman_zone(3.0) == "Safe"

    def test_boundary_grey(self) -> None:
        assert altman_zone(1.82) == "Grey"

    def test_boundary_distress(self) -> None:
        assert altman_zone(1.81) == "Distress"

    def test_unknown_for_none(self) -> None:
        assert altman_zone(None) == "Unknown"
