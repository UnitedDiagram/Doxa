"""EditorAgent — distills research reports to high-signal insights using Claude."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from anthropic import AsyncAnthropic
from doxa_shared.prompts.editor import EDITORIAL_PROMPT

from src.config import ANTHROPIC_API_KEY
from src.state import ResearchState

logger = logging.getLogger(__name__)


class EditorAgent:
    """Performs editorial distillation on the final research report.

    Reduces page count to 30-50 pages, surfaces non-obvious risks,
    identifies thesis drift, and removes redundancies.
    """

    async def analyze(self, state: ResearchState) -> ResearchState:
        """Run editorial distillation and update state.

        Args:
            state: A ResearchState with ``final_report`` populated.

        Returns:
            The same state dict with ``final_report`` updated in-place.
        """
        ticker = state["ticker"]
        logger.info("EditorAgent distilling report for %s", ticker)

        report_content = state.get("final_report", "")
        if not report_content:
            logger.warning("No report found for %s; EditorAgent skipping", ticker)
            return state

        if not ANTHROPIC_API_KEY:
            msg = "ANTHROPIC_API_KEY not set; EditorAgent skipping distillation"
            logger.warning(msg)
            state["errors"].append(msg)
            return state

        # Format insights board for prompt injection
        insights_board_text = _format_insights_for_editor(
            state.get("insights_board", [])
        )

        edited_report, rationale, api_success = await _call_claude_async(
            ticker, report_content, insights_board=insights_board_text
        )

        if api_success and edited_report:
            state["final_report"] = edited_report
            logger.info("EditorAgent successfully distilled report for %s", ticker)
        else:
            logger.warning("EditorAgent failed to distill report for %s", ticker)

        # Add provenance metadata
        if "provenance_metadata" not in state:
            state["provenance_metadata"] = {}
        state["provenance_metadata"]["editor"] = {
            "agent": "EditorAgent",
            "source": "Claude editorial analysis",
            "timestamp": datetime.now(UTC).isoformat(),
            "rationale": rationale or "API call failed",
        }

        return state


async def _call_claude_async(
    ticker: str, report_content: str, target_pages: int = 40,
    *, insights_board: str = "",
) -> tuple[str, str, bool]:
    """Call Claude async to distill the research report.

    Args:
        ticker: The stock ticker symbol.
        report_content: The full Markdown report to distill.
        target_pages: The target length in simulated pages.
        insights_board: Pre-formatted insights board string for prompt injection.

    Returns:
        A (edited_report, rationale, api_success) tuple.
    """
    prompt = EDITORIAL_PROMPT.format(
        ticker=ticker,
        report_content=report_content,
        target_pages=target_pages,
        insights_board=insights_board or "No cross-domain insights available.",
    )

    try:
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        async with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=8192,  # Reports can be long
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = await stream.get_final_message()

        raw_text = ""
        for block in message.content:
            if block.type == "text":
                raw_text = block.text.strip()
                break

        if not raw_text:
            logger.warning("Claude returned no text block for %s", ticker)
            return "", "", False

        # Strip markdown code fences if Claude wraps JSON in ```json ... ```
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        # Try JSON parse; fall back to treating raw_text as the report directly.
        # Embedding large Markdown inside JSON can produce parse errors when the
        # report contains unescaped special characters.
        try:
            parsed = json.loads(raw_text)
            edited_report = str(parsed.get("edited_report", ""))
            rationale = str(parsed.get("rationale", ""))
        except json.JSONDecodeError:
            logger.warning(
                "EditorAgent JSON parse failed for %s; using raw text as report",
                ticker,
            )
            edited_report = raw_text
            rationale = ""

        return edited_report, rationale, True

    except Exception as exc:
        logger.warning("Claude editorial call failed for %s: %s", ticker, exc)
        return "", "", False


def _format_insights_for_editor(insights: list[dict[str, Any]]) -> str:
    """Format insights board as concise bullet list for the editor prompt.

    Args:
        insights: List of insight dicts from state['insights_board'].

    Returns:
        Formatted string, or 'No cross-domain insights available.' if empty.
    """
    if not insights:
        return "No cross-domain insights available."
    lines: list[str] = []
    for insight in insights:
        agent = insight.get("agent", "")
        cat = insight.get("category", "")
        signal = insight.get("signal", "")
        conf = insight.get("confidence", 0.0)
        lines.append(f"- [{agent}/{cat} | {conf:.0%}] {signal}")
    return "\n".join(lines)
