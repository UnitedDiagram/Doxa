"""Structured JSON trace logging for Doxa agents.

Provides lightweight utilities for emitting JSON-structured log messages
compatible with CloudWatch and similar log aggregators. This is the POC
foundation for production structlog logging (Epic 3).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from types import TracebackType
from typing import Any


def log_trace(
    logger: logging.Logger,
    event: str,
    **kwargs: Any,
) -> None:
    """Log a structured JSON trace event.

    Formats the event and kwargs as a JSON object with an automatic
    ISO 8601 timestamp. Uses WARNING level for error events,
    INFO for everything else.

    Args:
        logger: The logger instance to write to.
        event: Event name (e.g. "agent_started", "api_call").
        **kwargs: Additional fields to include in the JSON payload.
    """
    payload: dict[str, Any] = {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        **kwargs,
    }
    level = logging.WARNING if "error" in event else logging.INFO
    logger.log(level, json.dumps(payload, default=str))


class TraceTimer:
    """Context manager that measures elapsed wall-clock time.

    Uses ``time.monotonic()`` for clock-adjustment immunity.

    Example::

        timer = TraceTimer()
        with timer:
            do_work()
        print(f"Took {timer.elapsed_ms:.0f}ms")
    """

    def __init__(self) -> None:
        """Initialize timer with zero elapsed time."""
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> TraceTimer:
        """Record start time."""
        self._start = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Calculate elapsed time in milliseconds."""
        self.elapsed_ms = (time.monotonic() - self._start) * 1000
