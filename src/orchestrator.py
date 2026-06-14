"""Parallel agent execution framework for Doxa.

Runs independent agents concurrently via ``asyncio.gather`` and
``asyncio.to_thread``, preserving the error-accumulation semantics
of the sequential pipeline in ``src.main``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

from doxa_shared.utils.tracing import TraceTimer, log_trace

from src.agents.editor import EditorAgent
from src.agents.market_data import MarketDataAgent
from src.agents.regulatory import RegulatoryAgent
from src.agents.sentiment import SentimentAgent
from src.agents.valuation import ValuationAgent
from src.agents.writer import WriterAgent
from src.state import ResearchState

logger = logging.getLogger(__name__)


async def run_pipeline_parallel(
    state: ResearchState,
) -> ResearchState:
    """Run the 6-agent pipeline with parallel execution.

    Dependency graph::

        Stage 1  MarketDataAgent          (sequential)
        Stage 2  Quant | Valuation | Reg  (parallel)
        Stage 3  SentimentAgent           (sequential, async)
        Stage 4  WriterAgent              (sequential)

    All agents mutate *state* in place. Parallel agents write to
    disjoint keys; ``state['errors']`` is protected by a lock.

    Args:
        state: Initialised ResearchState.

    Returns:
        The same state object, enriched by all agents.
    """
    errors_lock = threading.Lock()
    sequential_ms = 0.0

    pipeline = TraceTimer()
    with pipeline:
        # --- Stage 1: MarketData (must run first) ---
        t1 = TraceTimer()
        with t1:
            try:
                state = MarketDataAgent().fetch_data(state)
            except Exception as exc:
                state["errors"].append(
                    f"MarketDataAgent failed: {exc}"
                )
                logger.warning(
                    "MarketDataAgent failed: %s", exc
                )
        log_trace(
            logger,
            "agent_completed",
            agent="MarketDataAgent",
            elapsed_ms=t1.elapsed_ms,
        )
        sequential_ms += t1.elapsed_ms

        # --- Stage 2: parallel block ---
        async def _run_agent(
            name: str,
            fn: Callable[[ResearchState], ResearchState],
        ) -> float:
            at = TraceTimer()
            with at:
                try:
                    await asyncio.to_thread(fn, state)
                except Exception as exc:
                    with errors_lock:
                        state["errors"].append(
                            f"{name} failed: {exc}"
                        )
                    logger.warning("%s failed: %s", name, exc)
            log_trace(
                logger,
                "agent_completed",
                agent=name,
                elapsed_ms=at.elapsed_ms,
            )
            return at.elapsed_ms

        par = TraceTimer()
        with par:
            stage2 = await asyncio.gather(
                _run_agent(
                    "ValuationAgent",
                    ValuationAgent().execute,
                ),
                _run_agent(
                    "RegulatoryAgent",
                    RegulatoryAgent().analyze,
                ),
            )
        sequential_ms += sum(stage2)
        log_trace(
            logger,
            "parallel_block_completed",
            elapsed_ms=par.elapsed_ms,
            sequential_estimate_ms=sum(stage2),
        )

        # --- Stage 3: Sentiment (already async) ---
        t3 = TraceTimer()
        with t3:
            try:
                state = await SentimentAgent().analyze(state)
            except Exception as exc:
                state["errors"].append(
                    f"SentimentAgent failed: {exc}"
                )
                logger.warning(
                    "SentimentAgent failed: %s", exc
                )
        log_trace(
            logger,
            "agent_completed",
            agent="SentimentAgent",
            elapsed_ms=t3.elapsed_ms,
        )
        sequential_ms += t3.elapsed_ms

        # --- Stage 4: Writer ---
        t4 = TraceTimer()
        with t4:
            try:
                await asyncio.to_thread(
                    WriterAgent().generate_report, state
                )
            except Exception as exc:
                state["errors"].append(
                    f"WriterAgent failed: {exc}"
                )
                logger.warning("WriterAgent failed: %s", exc)
        log_trace(
            logger,
            "agent_completed",
            agent="WriterAgent",
            elapsed_ms=t4.elapsed_ms,
        )
        sequential_ms += t4.elapsed_ms

        # --- Stage 5: Editor (async) ---
        t5 = TraceTimer()
        with t5:
            try:
                state = await EditorAgent().analyze(state)
            except Exception as exc:
                state["errors"].append(
                    f"EditorAgent failed: {exc}"
                )
                logger.warning(
                    "EditorAgent failed: %s", exc
                )
        log_trace(
            logger,
            "agent_completed",
            agent="EditorAgent",
            elapsed_ms=t5.elapsed_ms,
        )
        sequential_ms += t5.elapsed_ms

    logger.info(
        "Pipeline: %.0fms total (%.0fms sequential est, "
        "%.0fms saved)",
        pipeline.elapsed_ms,
        sequential_ms,
        sequential_ms - pipeline.elapsed_ms,
    )
    return state
