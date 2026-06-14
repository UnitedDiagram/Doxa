"""Tests for doxa_shared.utils.confidence module."""

from __future__ import annotations

import pytest
from doxa_shared.types.state import ResearchState, create_initial_state
from doxa_shared.utils.confidence import (
    aggregate_confidence,
    calculate_data_completeness,
    calculate_time_series_quality,
    cross_validate_agents,
)


class TestDataCompleteness:
    """calculate_data_completeness tests."""

    def test_all_present(self) -> None:
        """All populated fields yield 100%."""
        data = {"a": 1, "b": "hello", "c": [1]}
        assert calculate_data_completeness(
            ["a", "b", "c"], data
        ) == 100.0

    def test_half_present(self) -> None:
        """Half populated fields yield 50%."""
        data: dict[str, object] = {"a": 1, "b": None}
        score = calculate_data_completeness(["a", "b"], data)
        assert score == pytest.approx(50.0)

    def test_none_present(self) -> None:
        """No matching fields yield 0%."""
        assert calculate_data_completeness(["x", "y"], {}) == 0.0

    def test_empty_values_not_counted(self) -> None:
        """None, empty str, empty dict/list are unpopulated."""
        data: dict[str, object] = {
            "a": None, "b": "", "c": {}, "d": [],
        }
        score = calculate_data_completeness(
            ["a", "b", "c", "d"], data
        )
        assert score == 0.0

    def test_empty_required_returns_100(self) -> None:
        """No required fields means 100% by default."""
        assert calculate_data_completeness([], {"a": 1}) == 100.0

    def test_whitespace_string_not_counted(self) -> None:
        """Whitespace-only string is unpopulated."""
        assert calculate_data_completeness(
            ["a"], {"a": "  "}
        ) == 0.0

    def test_zero_is_populated(self) -> None:
        """Numeric zero is a valid populated value."""
        assert calculate_data_completeness(
            ["a"], {"a": 0}
        ) == 100.0

    def test_false_is_populated(self) -> None:
        """Boolean False is a valid populated value."""
        assert calculate_data_completeness(
            ["a"], {"a": False}
        ) == 100.0


class TestTimeSeriesQuality:
    """calculate_time_series_quality tests."""

    def test_10y(self) -> None:
        """10 years of data scores 100%."""
        assert calculate_time_series_quality(10) == 100.0

    def test_5y(self) -> None:
        """5 years scores 75%."""
        assert calculate_time_series_quality(5) == 75.0

    def test_3y(self) -> None:
        """3 years scores 50%."""
        assert calculate_time_series_quality(3) == 50.0

    def test_1y(self) -> None:
        """1 year scores 25%."""
        assert calculate_time_series_quality(1) == 25.0

    def test_0y(self) -> None:
        """0 years scores 0%."""
        assert calculate_time_series_quality(0) == 0.0

    def test_negative_years(self) -> None:
        """Negative years scores 0%."""
        assert calculate_time_series_quality(-1) == 0.0

    def test_7y_interpolation(self) -> None:
        """7 years interpolates between 5Y and 10Y thresholds."""
        assert calculate_time_series_quality(7) == pytest.approx(
            85.0
        )

    def test_2y_interpolation(self) -> None:
        """2 years interpolates between 1Y and 3Y thresholds."""
        assert calculate_time_series_quality(2) == pytest.approx(
            37.5
        )

    def test_exceeds_expected(self) -> None:
        """More data than expected still scores 100%."""
        assert calculate_time_series_quality(15) == 100.0

    def test_custom_expected(self) -> None:
        """Custom years_expected changes the 100% threshold."""
        assert (
            calculate_time_series_quality(5, years_expected=5)
            == 100.0
        )


@pytest.fixture()
def base_state() -> ResearchState:
    """Minimal state for cross-validation tests."""
    return create_initial_state("TEST")


class TestCrossValidate:
    """cross_validate_agents tests."""

    def test_consistent_no_flags(
        self, base_state: ResearchState
    ) -> None:
        """Consistent signals produce no flags."""
        base_state["quant_analysis"] = {"signal": "BULLISH"}
        base_state["sentiment_score"] = 0.5
        result = cross_validate_agents(base_state)
        assert result["flags"] == []
        assert result["penalty"] == 0.0

    def test_dcf_comps_divergence(
        self, base_state: ResearchState
    ) -> None:
        """Large DCF vs comps gap triggers a flag."""
        base_state["valuation_analysis"] = {
            "dcf": {"fair_value_per_share": 100.0},
            "comps": {"implied_valuations": {"pe": 300.0}},
        }
        result = cross_validate_agents(base_state)
        assert len(result["flags"]) == 1
        assert "divergence" in result["flags"][0].lower()
        assert result["penalty"] == 10.0

    def test_quant_bullish_sentiment_negative(
        self, base_state: ResearchState
    ) -> None:
        """Bullish quant + negative sentiment triggers a flag."""
        base_state["quant_analysis"] = {"signal": "BULLISH"}
        base_state["sentiment_score"] = -0.5
        result = cross_validate_agents(base_state)
        assert any(
            "bullish" in f.lower() for f in result["flags"]
        )
        assert result["penalty"] == 5.0

    def test_quant_bearish_sentiment_positive(
        self, base_state: ResearchState
    ) -> None:
        """Bearish quant + positive sentiment triggers a flag."""
        base_state["quant_analysis"] = {"signal": "BEARISH"}
        base_state["sentiment_score"] = 0.5
        result = cross_validate_agents(base_state)
        assert any(
            "bearish" in f.lower() for f in result["flags"]
        )
        assert result["penalty"] == 5.0

    def test_missing_data_no_flags(
        self, base_state: ResearchState
    ) -> None:
        """Missing data produces no flags (can't validate)."""
        result = cross_validate_agents(base_state)
        assert result["flags"] == []
        assert result["penalty"] == 0.0

    def test_penalty_capped_at_20(
        self, base_state: ResearchState
    ) -> None:
        """Total penalty is capped at 20 points."""
        base_state["valuation_analysis"] = {
            "dcf": {"fair_value_per_share": 100.0},
            "comps": {"implied_valuations": {"pe": 300.0}},
        }
        base_state["quant_analysis"] = {"signal": "BULLISH"}
        base_state["sentiment_score"] = -0.5
        result = cross_validate_agents(base_state)
        assert result["penalty"] <= 20.0


class TestAggregateConfidence:
    """aggregate_confidence tests."""

    def test_equal_weights_all_100(self) -> None:
        """All scores 100 with equal weights yields 100."""
        assert aggregate_confidence(
            [100, 100, 100], [1 / 3, 1 / 3, 1 / 3]
        ) == pytest.approx(100.0, abs=0.1)

    def test_equal_weights_all_0(self) -> None:
        """All scores 0 yields 0."""
        assert aggregate_confidence(
            [0, 0], [0.5, 0.5]
        ) == 0.0

    def test_mixed_scores(self) -> None:
        """Weighted average of mixed scores."""
        result = aggregate_confidence(
            [100, 50], [0.6, 0.4]
        )
        assert result == pytest.approx(80.0)

    def test_clamped_to_0_100(self) -> None:
        """Result is always in 0-100 range."""
        result = aggregate_confidence([0, 0], [0.5, 0.5])
        assert 0.0 <= result <= 100.0

    def test_length_mismatch_raises(self) -> None:
        """Different-length scores and weights raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            aggregate_confidence([100], [0.5, 0.5])

    def test_weights_not_1_raises(self) -> None:
        """Weights not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            aggregate_confidence([100, 100], [0.5, 0.6])

    def test_weights_tolerance(self) -> None:
        """Weights summing to 1.005 are accepted (within 0.01)."""
        result = aggregate_confidence(
            [80, 60], [0.505, 0.5]
        )
        assert isinstance(result, float)
