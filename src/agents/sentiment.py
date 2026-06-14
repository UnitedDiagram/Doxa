"""SentimentAgent — scores real news headlines using Claude AI."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from anthropic import AsyncAnthropic
from doxa_shared.prompts.sentiment import SENTIMENT_PROMPT
from doxa_shared.prompts.sentiment_enhanced import SENTIMENT_ENHANCED_PROMPT
from doxa_shared.utils.insights import post_insight
from doxa_shared.utils.sentiment_data import fetch_alternative_data
from doxa_shared.utils.tracing import log_trace

from src.config import ANTHROPIC_API_KEY, configure_logging
from src.state import ResearchState, create_initial_state

logger = logging.getLogger(__name__)


class SentimentAgent:
    """Analyzes real news headlines for sentiment using Claude.

    Uses the news already fetched by MarketDataAgent from state["news"].
    Falls back to score=0.0 if the API call fails or no headlines are present.
    """

    async def analyze(self, state: ResearchState) -> ResearchState:
        """Run sentiment analysis and update state.

        Args:
            state: A ResearchState with ``ticker`` and ``news`` populated.

        Returns:
            The same state dict with sentiment_score and sentiment_rationale set.
        """
        ticker = state["ticker"]

        # Task 2.7: Fetch alternative data
        state["alternative_data"] = fetch_alternative_data(ticker)

        headlines = [item["headline"] for item in state["news"] if item.get("headline")]

        if not headlines:
            logger.warning(
                "No headlines for %s; sentiment stays 0.0",
                ticker,
            )
            _post_sentiment_insights(state)
            return state

        if not ANTHROPIC_API_KEY:
            msg = "ANTHROPIC_API_KEY not set; sentiment score stays 0.0"
            logger.warning(msg)
            state["errors"].append(msg)
            return state

        (
            score, rationale, contradictions, parsed_json, api_success
        ) = await _call_claude_async(
            ticker,
            headlines,
            alternative_data=state.get("alternative_data"),
            regulatory_analysis=state.get("regulatory_analysis"),
        )
        if not api_success:
            msg = (
                f"SentimentAgent: Claude API call failed for {ticker}; "
                "sentiment score stays 0.0"
            )
            logger.warning(msg)
            state["errors"].append(msg)
            confidence, missing_fields, reason = _calculate_enhanced_confidence(
                parsed_json, api_success, state.get("alternative_data")
            )
            state["sentiment_confidence"] = confidence
            state["sentiment_confidence_details"] = {
                "score": confidence,
                "missing_fields": missing_fields,
                "reason": reason,
            }
            _post_sentiment_insights(state)
            return state

        state["sentiment_score"] = score
        state["sentiment_rationale"] = rationale
        if contradictions and "alternative_data" in state:
            state["alternative_data"]["contradictions"] = contradictions

        # Calculate and store confidence
        confidence, missing_fields, reason = _calculate_enhanced_confidence(
            parsed_json, api_success, state.get("alternative_data")
        )
        state["sentiment_confidence"] = confidence
        state["sentiment_confidence_details"] = {
            "score": confidence,
            "missing_fields": missing_fields,
            "reason": reason,
        }

        log_trace(
            logger,
            "confidence_calculated",
            agent="SentimentAgent",
            confidence=confidence,
            missing_fields=missing_fields,
            reason=reason,
        )

        logger.info(
            "Sentiment for %s: score=%.2f, confidence=%.0f%% (%d headlines)",
            ticker,
            score,
            confidence,
            len(headlines),
        )

        # Add provenance metadata
        if "provenance_metadata" not in state:
            state["provenance_metadata"] = {}
        state["provenance_metadata"]["sentiment"] = {
            "agent": "SentimentAgent",
            "source": "Claude analysis",
            "timestamp": datetime.now(UTC).isoformat(),
            "confidence": confidence,
            "headline_count": len(state.get("news", [])),
        }

        _post_sentiment_insights(state)

        return state


def _calculate_enhanced_confidence(
    parsed_json: dict[str, Any],
    api_call_succeeded: bool,
    alternative_data: dict[str, Any] | None = None,
) -> tuple[float, list[str], str]:
    """Calculate enhanced confidence score based on data availability and API success.

    Scoring logic:
    - Base score: 50
    - +10 for insider data
    - +10 for short interest
    - +10 for options flow
    - +10 for social sentiment
    - +10 for Claude's successfully parsed analysis

    Args:
        parsed_json: The parsed JSON response from Claude API.
        api_call_succeeded: Whether the API call succeeded.
        alternative_data: Optional alternative market data.

    Returns:
        A tuple of (score, missing_fields, reason).
    """
    score = 50.0
    missing_fields = []
    reasons = ["Base score: 50"]

    if alternative_data:
        if alternative_data.get("insider_trading"):
            score += 10
            reasons.append("+10 insider data")
        else:
            missing_fields.append("insider_trading")

        if alternative_data.get("short_interest"):
            score += 10
            reasons.append("+10 short interest")
        else:
            missing_fields.append("short_interest")

        if alternative_data.get("options_flow"):
            score += 10
            reasons.append("+10 options flow")
        else:
            missing_fields.append("options_flow")

        if alternative_data.get("social_media"):
            score += 10
            reasons.append("+10 social sentiment")
        else:
            missing_fields.append("social_media")

    if api_call_succeeded and parsed_json:
        # Check if basic fields are present
        basic_fields = ["score", "rationale", "key_catalyst"]
        missing_basic = [f for f in basic_fields if f not in parsed_json]
        if not missing_basic:
            score += 10
            reasons.append("+10 Claude analysis success")
        else:
            missing_fields.extend(missing_basic)
    else:
        missing_fields.append("claude_analysis")

    return score, missing_fields, ", ".join(reasons)


async def _call_claude_async(
    ticker: str,
    headlines: list[str],
    alternative_data: dict[str, Any] | None = None,
    regulatory_analysis: dict[str, Any] | None = None,
) -> tuple[float, str, list[str], dict[str, Any], bool]:
    """Call Claude async to score sentiment with enhanced context.

    Args:
        ticker: The stock ticker symbol.
        headlines: List of headline strings to evaluate.
        alternative_data: Optional alternative market data.
        regulatory_analysis: Optional regulatory/management context.

    Returns:
        A (score, rationale, contradictions, parsed_json, api_success) tuple.
        Returns (0.0, "", [], {}, False) on any failure.
    """
    numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))

    if alternative_data:
        prompt = SENTIMENT_ENHANCED_PROMPT.format(
            ticker=ticker,
            headlines=numbered,
            insider_trading=alternative_data.get("insider_trading", "N/A"),
            short_interest=alternative_data.get("short_interest", "N/A"),
            options_flow=alternative_data.get("options_flow", "N/A"),
            social_media=alternative_data.get("social_media", "N/A"),
            risk_factors=regulatory_analysis.get("risk_factors", "N/A")
            if regulatory_analysis
            else "N/A",
            management_signal=regulatory_analysis.get("signal", "N/A")
            if regulatory_analysis
            else "N/A",
        )
    else:
        prompt = SENTIMENT_PROMPT.format(ticker=ticker, headlines=numbered)

    try:
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        async with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=2048,
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
            return 0.0, "", [], {}, False

        # Strip markdown code fences if Claude wraps JSON in ```json ... ```
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        parsed = json.loads(raw_text)
        score_value = parsed.get("score", 0.0)
        # Handle null values explicitly
        if score_value is None:
            score_value = 0.0
        score = float(score_value)
        score = max(-1.0, min(1.0, score))
        rationale = str(parsed.get("rationale", ""))
        key_catalyst = str(parsed.get("key_catalyst", ""))
        contradictions = parsed.get("contradictions", [])
        if not isinstance(contradictions, list):
            contradictions = [str(contradictions)]

        if key_catalyst:
            rationale = f"{rationale} Key catalyst: {key_catalyst}"
        return score, rationale, contradictions, parsed, True

    except Exception as exc:
        logger.warning("Claude sentiment call failed for %s: %s", ticker, exc)
        return 0.0, "", [], {}, False


def _post_sentiment_insights(state: ResearchState) -> None:
    """Post cross-domain sentiment signals to the insights board.

    Reads alternative_data and sentiment_score from state and posts insights
    for insider selling, short interest spikes, and sentiment contradictions.
    Appends to state['errors'] on failure; never raises.

    Args:
        state: ResearchState with alternative_data and sentiment_score set.
    """
    try:
        ticker = state.get("ticker", "")
        alt = state.get("alternative_data") or {}

        # Insider selling cluster
        insider = alt.get("insider_trading")
        if insider and isinstance(insider, dict):
            net_sells = insider.get("net_shares_sold", 0) or 0
            net_buys = insider.get("net_shares_bought", 0) or 0
            if net_sells > net_buys and net_sells > 0:
                post_insight(
                    state,
                    agent="SentimentAgent",
                    category="insider_activity",
                    signal=(
                        f"{ticker} net insider selling: "
                        f"{net_sells:,} shares sold vs {net_buys:,} bought"
                    ),
                    confidence=0.8,
                )

        # Rising short interest
        short = alt.get("short_interest")
        if short and isinstance(short, dict):
            short_pct = short.get("short_interest_pct", 0) or 0
            if short_pct >= 10.0:  # >= 10% short interest is elevated
                post_insight(
                    state,
                    agent="SentimentAgent",
                    category="short_interest",
                    signal=(
                        f"{ticker} elevated short interest at {short_pct:.1f}% "
                        f"of float — bearish positioning signal"
                    ),
                    confidence=0.75,
                )

        # Sentiment contradictions
        contradictions = alt.get("contradictions") or []
        if isinstance(contradictions, list) and contradictions:
            for contradiction in contradictions[:2]:  # Cap at 2
                signal_text = (
                    str(contradiction)[:150]
                    if not isinstance(contradiction, str)
                    else contradiction[:150]
                )
                post_insight(
                    state,
                    agent="SentimentAgent",
                    category="sentiment_contradiction",
                    signal=f"{ticker} sentiment contradiction: {signal_text}",
                    confidence=0.7,
                )

        # Strong negative sentiment score
        sent_score = state.get("sentiment_score", 0.0)
        if sent_score <= -0.5:
            post_insight(
                state,
                agent="SentimentAgent",
                category="sentiment_contradiction",
                signal=(
                    f"{ticker} strongly negative news sentiment score: "
                    f"{sent_score:.2f} (range -1.0 to 1.0)"
                ),
                confidence=0.7,
            )

    except Exception as exc:
        msg = f"_post_sentiment_insights failed: {exc}"
        logger.warning(msg)
        state["errors"].append(msg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _async_main() -> None:
    """Async entry point for CLI testing."""
    configure_logging()
    state = create_initial_state("NVDA")
    # Provide some sample headlines for standalone testing
    state["news"] = [
        {"headline": "NVDA announces record quarterly earnings beating all estimates"},
        {"headline": "NVDA faces supply chain concerns amid rising demand"},
        {"headline": "NVDA partners with major cloud providers for AI infrastructure"},
    ]
    agent = SentimentAgent()
    result = await agent.analyze(state)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(_async_main())
