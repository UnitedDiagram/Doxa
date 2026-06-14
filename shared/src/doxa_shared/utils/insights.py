"""Shared utility for posting cross-domain insights to the insights board.

Each agent calls ``post_insight()`` to record a signal on the shared board,
which downstream agents (WriterAgent, EditorAgent) read for cross-domain
intelligence during report generation.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from doxa_shared.types.state import ResearchState
from doxa_shared.utils.tracing import log_trace

logger = logging.getLogger(__name__)


def post_insight(
    state: ResearchState,
    agent: str,
    category: str,
    signal: str,
    confidence: float,
) -> None:
    """Append a cross-domain insight to the shared insights board.

    Validates confidence is in [0.0, 1.0], clamping out-of-range values with
    a warning. Appends a dict conforming to the insight schema and logs a
    structured trace. Never raises — on any error, logs a warning and returns.

    Args:
        state: Shared ResearchState; ``insights_board`` list is mutated.
        agent: Name of the posting agent (e.g. ``"MarketDataAgent"``).
        category: Domain category (e.g. ``"volume"``, ``"regulatory"``).
        signal: Human-readable signal description.
        confidence: Confidence score in [0.0, 1.0].
    """
    try:
        if confidence < 0.0 or confidence > 1.0:
            logger.warning(
                "post_insight: confidence %.3f out of [0,1] for %s/%s — clamping",
                confidence,
                agent,
                category,
            )
            confidence = max(0.0, min(1.0, confidence))

        insight: dict[str, Any] = {
            "agent": agent,
            "category": category,
            "signal": signal,
            "confidence": confidence,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        state["insights_board"].append(insight)

        log_trace(
            logger,
            "insight_posted",
            agent=agent,
            category=category,
            signal=signal,
            confidence=confidence,
        )
    except Exception as exc:
        logger.warning("post_insight failed for %s/%s: %s", agent, category, exc)
