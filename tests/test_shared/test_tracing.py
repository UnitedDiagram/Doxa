"""Tests for shared tracing utility functions."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from doxa_shared.utils.tracing import TraceTimer, log_trace


class TestLogTrace:
    """Tests for log_trace function."""

    def test_outputs_valid_json(self, caplog: Any) -> None:
        """Trace output must be parseable JSON."""
        test_logger = logging.getLogger("test.trace")
        with caplog.at_level(logging.DEBUG, logger="test.trace"):
            log_trace(test_logger, "test_event", ticker="AAPL")
        record = caplog.records[-1]
        parsed = json.loads(record.message)
        assert parsed["event"] == "test_event"
        assert parsed["ticker"] == "AAPL"

    def test_includes_timestamp(self, caplog: Any) -> None:
        """Every trace must include an ISO 8601 timestamp."""
        test_logger = logging.getLogger("test.trace")
        with caplog.at_level(logging.DEBUG, logger="test.trace"):
            log_trace(test_logger, "test_event")
        parsed = json.loads(caplog.records[-1].message)
        assert "timestamp" in parsed
        assert "T" in parsed["timestamp"]  # ISO 8601 has a T separator

    def test_info_level_for_normal_events(self, caplog: Any) -> None:
        """Normal events should log at INFO level."""
        test_logger = logging.getLogger("test.trace")
        with caplog.at_level(logging.DEBUG, logger="test.trace"):
            log_trace(test_logger, "agent_started", agent="Test")
        assert caplog.records[-1].levelno == logging.INFO

    def test_warning_level_for_error_events(self, caplog: Any) -> None:
        """Events with 'error' in the name should log at WARNING."""
        test_logger = logging.getLogger("test.trace")
        with caplog.at_level(logging.DEBUG, logger="test.trace"):
            log_trace(test_logger, "agent_error", msg="fail")
        assert caplog.records[-1].levelno == logging.WARNING

    def test_passes_extra_kwargs(self, caplog: Any) -> None:
        """All kwargs should appear in the JSON payload."""
        test_logger = logging.getLogger("test.trace")
        with caplog.at_level(logging.DEBUG, logger="test.trace"):
            log_trace(
                test_logger,
                "api_call",
                source="yfinance.fast_info",
                ticker="NVDA",
                fields_requested=["price", "cap"],
            )
        parsed = json.loads(caplog.records[-1].message)
        assert parsed["source"] == "yfinance.fast_info"
        assert parsed["fields_requested"] == ["price", "cap"]


class TestTraceTimer:
    """Tests for TraceTimer context manager."""

    def test_measures_elapsed_time(self) -> None:
        """TraceTimer should measure non-negative elapsed time."""
        timer = TraceTimer()
        with timer:
            time.sleep(0.01)
        assert timer.elapsed_ms >= 0

    def test_elapsed_is_reasonable(self) -> None:
        """Elapsed time should be roughly correct."""
        timer = TraceTimer()
        with timer:
            time.sleep(0.05)
        # Should be at least 40ms (allow some slack)
        assert timer.elapsed_ms >= 40

    def test_elapsed_before_use_is_zero(self) -> None:
        """Elapsed should be 0 before the context manager runs."""
        timer = TraceTimer()
        assert timer.elapsed_ms == 0.0
