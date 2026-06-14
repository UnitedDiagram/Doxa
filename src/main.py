"""Doxa orchestrator — runs all agents in sequence with human input."""

from __future__ import annotations

import asyncio
import logging

from src.agents.editor import EditorAgent
from src.agents.market_data import MarketDataAgent
from src.agents.regulatory import RegulatoryAgent
from src.agents.sentiment import SentimentAgent
from src.agents.valuation import ValuationAgent
from src.agents.writer import WriterAgent
from src.config import configure_logging
from src.state import create_initial_state

logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the full Doxa research pipeline with 5 agents.

    Pipeline order:
        1. MarketDataAgent — fetch price/financial data
        2. ValuationAgent — DCF, comps, and quantitative analysis
        3. RegulatoryAgent — SEC EDGAR analysis
        4. SentimentAgent — news sentiment scoring
        5. Human input — analyst subjective notes
        6. WriterAgent — generate final Markdown report
        7. EditorAgent — high-signal distillation
    """
    configure_logging()

    ticker = input("Enter ticker symbol: ").strip()
    state = create_initial_state(ticker)

    # Phase 1: Market data
    print(f"\n[1/6] Fetching market data for {state['ticker']}...")
    state = MarketDataAgent().fetch_data(state)

    # Phase 2: Valuation (includes quantitative analysis)
    print("[2/6] Running valuation and quantitative analysis...")
    state = ValuationAgent().execute(state)

    # Phase 3: Regulatory
    print("[3/6] Analyzing regulatory filings (SEC EDGAR)...")
    state = RegulatoryAgent().analyze(state)

    # Phase 4: Sentiment
    print("[4/6] Analyzing news sentiment...")
    state = await SentimentAgent().analyze(state)

    # Human-in-the-loop
    valuation = state.get("valuation_analysis", {})
    confidence = valuation.get("confidence", 0)
    print(f"\n--- Data gathered for {state['ticker']} ---")
    print(f"Valuation Confidence: {confidence:.0f}%")
    print(f"Sentiment           : {state['sentiment_score']:+.2f}")
    notes = input("\nAmeya, what's your subjective take on this stock?\n> ")
    state["human_notes"] = notes

    # Phase 5: Generate report
    print("\n[5/6] Generating comprehensive report...")
    state = WriterAgent().generate_report(state)

    # Phase 6: Distillation
    print("[6/6] Distilling report for high-signal insights...")
    state = await EditorAgent().analyze(state)

    print("\n" + state["final_report"])

    if state["errors"]:
        print(f"\n{len(state['errors'])} error(s) during processing:")
        for err in state["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    asyncio.run(main())
