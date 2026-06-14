"""Tests for src.orchestrator module."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest
from doxa_shared.types.state import create_initial_state

from src.orchestrator import run_pipeline_parallel


@pytest.fixture()
def base_state():
    """Fresh ResearchState for orchestrator tests."""
    return create_initial_state("TEST")


@pytest.fixture()
def _mock_agents():
    """Patch all 5 agent classes in src.orchestrator."""
    with (
        patch("src.orchestrator.MarketDataAgent") as mda,
        patch("src.orchestrator.ValuationAgent") as va,
        patch("src.orchestrator.RegulatoryAgent") as ra,
        patch("src.orchestrator.SentimentAgent") as sa,
        patch("src.orchestrator.WriterAgent") as wa,
    ):
        mda.return_value.fetch_data.side_effect = lambda s: s
        va.return_value.execute.side_effect = lambda s: s
        ra.return_value.analyze.side_effect = lambda s: s
        sa.return_value.analyze = AsyncMock(
            side_effect=lambda s: s
        )
        wa.return_value.generate_report.side_effect = (
            lambda s: s
        )
        yield {
            "market_data": mda,
            "valuation": va,
            "regulatory": ra,
            "sentiment": sa,
            "writer": wa,
        }


@pytest.mark.usefixtures("_mock_agents")
class TestPipelineBasic:
    """Basic pipeline execution tests."""

    async def test_all_agents_called(
        self,
        base_state,
        _mock_agents,
    ) -> None:
        """Every agent is invoked exactly once."""
        result = await run_pipeline_parallel(base_state)

        mocks = _mock_agents
        mocks["market_data"].return_value.fetch_data.assert_called_once()
        mocks["valuation"].return_value.execute.assert_called_once()
        mocks["regulatory"].return_value.analyze.assert_called_once()
        mocks["sentiment"].return_value.analyze.assert_called_once()
        mocks["writer"].return_value.generate_report.assert_called_once()
        assert result is base_state

    async def test_state_identity_preserved(
        self,
        base_state,
        _mock_agents,
    ) -> None:
        """Pipeline returns the same state object."""
        result = await run_pipeline_parallel(base_state)
        assert result is base_state

    async def test_state_mutation(
        self,
        base_state,
        _mock_agents,
    ) -> None:
        """Agents can mutate state via side effects."""

        def set_market(s):
            s["market_data"] = {"current_price": 150.0}
            return s

        _mock_agents[
            "market_data"
        ].return_value.fetch_data.side_effect = set_market

        result = await run_pipeline_parallel(base_state)
        assert result["market_data"]["current_price"] == 150.0


class TestErrorIsolation:
    """One agent's failure must not crash the pipeline."""

    async def test_parallel_agent_failure(
        self, base_state
    ) -> None:
        """A failing parallel agent appends to errors."""
        with (
            patch("src.orchestrator.MarketDataAgent") as mda,
            patch("src.orchestrator.ValuationAgent") as va,
            patch("src.orchestrator.RegulatoryAgent") as ra,
            patch("src.orchestrator.SentimentAgent") as sa,
            patch("src.orchestrator.WriterAgent") as wa,
        ):
            mda.return_value.fetch_data.side_effect = (
                lambda s: s
            )
            va.return_value.execute.side_effect = RuntimeError(
                "valuation boom"
            )
            ra.return_value.analyze.side_effect = lambda s: s
            sa.return_value.analyze = AsyncMock(
                side_effect=lambda s: s
            )
            wa.return_value.generate_report.side_effect = (
                lambda s: s
            )

            result = await run_pipeline_parallel(base_state)

        assert any(
            "ValuationAgent failed" in e
            for e in result["errors"]
        )
        assert result is base_state

    async def test_stage1_failure_continues(
        self, base_state
    ) -> None:
        """MarketDataAgent failure does not stop later stages."""
        with (
            patch("src.orchestrator.MarketDataAgent") as mda,
            patch("src.orchestrator.ValuationAgent") as va,
            patch("src.orchestrator.RegulatoryAgent") as ra,
            patch("src.orchestrator.SentimentAgent") as sa,
            patch("src.orchestrator.WriterAgent") as wa,
        ):
            mda.return_value.fetch_data.side_effect = (
                RuntimeError("no data")
            )
            va.return_value.execute.side_effect = lambda s: s
            ra.return_value.analyze.side_effect = lambda s: s
            sa.return_value.analyze = AsyncMock(
                side_effect=lambda s: s
            )
            wa.return_value.generate_report.side_effect = (
                lambda s: s
            )

            result = await run_pipeline_parallel(base_state)

        assert any(
            "MarketDataAgent failed" in e
            for e in result["errors"]
        )
        wa.return_value.generate_report.assert_called_once()


class TestParallelExecution:
    """Verify agents actually run concurrently."""

    async def test_parallel_saves_time(
        self, base_state
    ) -> None:
        """Stage-2 agents run in parallel, not sequentially."""
        delay = 0.1

        def slow(s):
            time.sleep(delay)
            return s

        with (
            patch("src.orchestrator.MarketDataAgent") as mda,
            patch("src.orchestrator.ValuationAgent") as va,
            patch("src.orchestrator.RegulatoryAgent") as ra,
            patch("src.orchestrator.SentimentAgent") as sa,
            patch("src.orchestrator.WriterAgent") as wa,
        ):
            mda.return_value.fetch_data.side_effect = (
                lambda s: s
            )
            va.return_value.execute.side_effect = slow
            ra.return_value.analyze.side_effect = slow
            sa.return_value.analyze = AsyncMock(
                side_effect=lambda s: s
            )
            wa.return_value.generate_report.side_effect = (
                lambda s: s
            )

            start = time.monotonic()
            await run_pipeline_parallel(base_state)
            elapsed = time.monotonic() - start

        sequential_stage2 = 2 * delay  # Now only 2 slow agents (not 3)
        assert elapsed < sequential_stage2, (
            f"Took {elapsed:.3f}s — expected less than "
            f"{sequential_stage2:.3f}s (sequential)"
        )
