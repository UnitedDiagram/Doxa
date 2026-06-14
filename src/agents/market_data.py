"""MarketDataAgent — "The Scout" that fetches market data via yfinance."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import yfinance as yf
from doxa_shared.constants.yfinance import (
    FAST_INFO_LAST_PRICE,
    FAST_INFO_MARKET_CAP,
    FAST_INFO_YEAR_HIGH,
    FAST_INFO_YEAR_LOW,
    INFO_NET_INCOME,
    INFO_TOTAL_CASH,
    INFO_TOTAL_DEBT,
    INFO_TOTAL_REVENUE,
)
from doxa_shared.utils.cache import get_cache
from doxa_shared.utils.confidence import calculate_data_completeness
from doxa_shared.utils.insights import post_insight
from doxa_shared.utils.market_data import (
    analyze_volume_patterns,
    calculate_52week_range,
    calculate_beta,
    df_get,
    fetch_5y_price_history,
    fetch_dividend_history,
    fetch_split_history,
    safe_get,
)
from doxa_shared.utils.tracing import TraceTimer, log_trace

from src.config import (
    HISTORY_PERIOD,
    SENTIMENT_MAX_HEADLINES,
    configure_logging,
)
from src.state import ResearchState, create_initial_state

logger = logging.getLogger(__name__)


class MarketDataAgent:
    """Fetches current market data and financials for a given ticker.

    Attributes:
        history_period: The lookback period for price history (default from env).
    """

    def __init__(self, history_period: str = HISTORY_PERIOD) -> None:
        self.history_period = history_period

    def fetch_data(self, state: ResearchState) -> ResearchState:
        """Populate market_data and financials on the shared state.

        Args:
            state: A ResearchState with at least ``ticker`` set.

        Returns:
            The same state dict, mutated with market data and financials.

        Raises:
            ValueError: If the ticker in state is empty.
        """
        ticker_symbol = state["ticker"]
        _validate_ticker(ticker_symbol)

        log_trace(
            logger, "agent_started",
            agent="MarketDataAgent", ticker=ticker_symbol,
        )
        logger.info("Fetching data for %s", ticker_symbol)
        yf_ticker = yf.Ticker(ticker_symbol)

        timer = TraceTimer()
        with timer:
            _extract_market_data(yf_ticker, state)
            _extract_price_history(
                yf_ticker, state, self.history_period,
            )
            _extract_financials(yf_ticker, state)
            _extract_news(
                yf_ticker, state, SENTIMENT_MAX_HEADLINES,
            )
            _extract_institutional_depth(yf_ticker, state)
            try:
                _post_market_insights(state)
            except Exception as _exc:
                logger.warning("_post_market_insights call failed: %s", _exc)

        logger.info(
            "Finished fetching data for %s (%d errors)",
            ticker_symbol,
            len(state["errors"]),
        )
        log_trace(
            logger, "agent_completed",
            agent="MarketDataAgent",
            ticker=ticker_symbol,
            execution_time_ms=round(timer.elapsed_ms, 1),
            errors_count=len(state["errors"]),
        )

        # Add provenance metadata with confidence scoring
        if "provenance_metadata" not in state:
            state["provenance_metadata"] = {}

        # Calculate confidence based on data completeness
        required_fields = [
            "current_price",
            "market_cap",
            "historical_prices_5y",
            "beta",
        ]
        confidence = calculate_data_completeness(
            required_fields=required_fields,
            data=state["market_data"],
        )

        state["provenance_metadata"]["market_data"] = {
            "agent": "MarketDataAgent",
            "source": "yfinance",
            "timestamp": datetime.now(UTC).isoformat(),
            "confidence": confidence,
            "data_fields": list(state["market_data"].keys()),
        }

        return state


# ---------------------------------------------------------------------------
# Private helpers (each <50 lines, single responsibility)
# ---------------------------------------------------------------------------


def _validate_ticker(ticker: str) -> None:
    """Raise ValueError if ticker is empty or whitespace."""
    if not ticker.strip():
        raise ValueError("Ticker must not be empty")


def _extract_market_data(yf_ticker: yf.Ticker, state: ResearchState) -> None:
    """Extract current price, market cap, and 52-week range from fast_info."""
    fields = [
        FAST_INFO_LAST_PRICE, FAST_INFO_MARKET_CAP,
        FAST_INFO_YEAR_HIGH, FAST_INFO_YEAR_LOW,
    ]
    log_trace(
        logger, "api_call",
        source="yfinance.fast_info",
        ticker=state["ticker"],
        fields_requested=fields,
    )
    try:
        fi: Any = yf_ticker.fast_info
        state["market_data"]["current_price"] = safe_get(fi, FAST_INFO_LAST_PRICE)
        state["market_data"]["market_cap"] = safe_get(fi, FAST_INFO_MARKET_CAP)
        state["market_data"]["fifty_two_week_high"] = safe_get(fi, FAST_INFO_YEAR_HIGH)
        state["market_data"]["fifty_two_week_low"] = safe_get(fi, FAST_INFO_YEAR_LOW)
        logger.debug("Market data extracted for %s", state["ticker"])
    except Exception as exc:
        msg = f"Failed to extract market data: {exc}"
        logger.warning(msg)
        state["errors"].append(msg)
        log_trace(
            logger, "agent_error",
            agent="MarketDataAgent",
            error_type=type(exc).__name__,
            error_message=str(exc),
            data_source="yfinance.fast_info",
        )


def _extract_price_history(
    yf_ticker: yf.Ticker, state: ResearchState, period: str
) -> None:
    """Fetch historical closing prices and store as list[float]."""
    log_trace(
        logger, "api_call",
        source="yfinance.history",
        ticker=state["ticker"],
        period=period,
    )
    try:
        hist = yf_ticker.history(period=period)
        if hist.empty:
            msg = f"No price history returned for {state['ticker']}"
            logger.warning(msg)
            state["errors"].append(msg)
            return
        data_points = len(hist)
        state["market_data"]["price_history"] = hist["Close"].tolist()
        state["market_data"]["price_history_period"] = period
        log_trace(
            logger, "api_call",
            source="yfinance.history",
            ticker=state["ticker"],
            period=period,
            data_points=data_points,
        )
        logger.debug(
            "Price history: %d data points for %s",
            data_points,
            state["ticker"],
        )
    except Exception as exc:
        msg = f"Failed to extract price history: {exc}"
        logger.warning(msg)
        state["errors"].append(msg)
        log_trace(
            logger, "agent_error",
            agent="MarketDataAgent",
            error_type=type(exc).__name__,
            error_message=str(exc),
            data_source="yfinance.history",
        )


def _extract_financials(yf_ticker: yf.Ticker, state: ResearchState) -> None:
    """Extract financial fundamentals from ticker.info and financial statements."""
    fields = [
        INFO_TOTAL_REVENUE, INFO_NET_INCOME,
        INFO_TOTAL_CASH, INFO_TOTAL_DEBT,
    ]
    log_trace(
        logger, "api_call",
        source="yfinance.info",
        ticker=state["ticker"],
        fields_requested=fields,
    )
    try:
        info: dict[str, Any] = yf_ticker.info
        state["financials"]["total_revenue"] = info.get(INFO_TOTAL_REVENUE)
        state["financials"]["net_income"] = info.get(INFO_NET_INCOME)
        state["financials"]["total_cash"] = info.get(INFO_TOTAL_CASH)
        state["financials"]["total_debt"] = info.get(INFO_TOTAL_DEBT)
        logger.debug("Financials extracted for %s", state["ticker"])
    except Exception as exc:
        msg = f"Failed to extract financials: {exc}"
        logger.warning(msg)
        state["errors"].append(msg)
        log_trace(
            logger, "agent_error",
            agent="MarketDataAgent",
            error_type=type(exc).__name__,
            error_message=str(exc),
            data_source="yfinance.info",
        )

    _extract_statements(yf_ticker, state)


def _extract_statements(yf_ticker: yf.Ticker, state: ResearchState) -> None:
    """Extract full financial statements for DuPont and Z-Score analysis."""
    fin = state["financials"]

    # --- Income Statement ---
    log_trace(
        logger, "api_call",
        source="yfinance.financials",
        ticker=state["ticker"],
    )
    try:
        inc = yf_ticker.financials
        if inc is not None and not inc.empty:
            fin["total_revenue"] = (
                fin.get("total_revenue")
                or df_get(inc, ["Total Revenue"], 0)
            )
            fin["net_income"] = (
                fin.get("net_income")
                or df_get(inc, ["Net Income"], 0)
            )
            fin["ebit"] = df_get(inc, ["EBIT", "Ebit", "Operating Income"], 0)
    except Exception as exc:
        logger.warning(
            "Failed to extract income statement for %s: %s",
            state["ticker"], exc,
        )
        log_trace(
            logger, "agent_error",
            agent="MarketDataAgent",
            error_type=type(exc).__name__,
            error_message=str(exc),
            data_source="yfinance.financials",
        )

    # --- Balance Sheet ---
    log_trace(
        logger, "api_call",
        source="yfinance.balance_sheet",
        ticker=state["ticker"],
    )
    try:
        bs = yf_ticker.balance_sheet
        if bs is not None and not bs.empty:
            fin["total_assets"] = df_get(bs, ["Total Assets"], 0)
            fin["total_assets_prev"] = df_get(bs, ["Total Assets"], 1)
            fin["total_liabilities"] = df_get(
                bs,
                ["Total Liabilities Net Minority Interest",
                 "Total Liab", "Total Liabilities"],
                0,
            )
            equity_labels = [
                "Stockholders Equity",
                "Total Stockholder Equity",
                "Common Stock Equity",
            ]
            fin["stockholders_equity"] = df_get(
                bs, equity_labels, 0,
            )
            fin["stockholders_equity_prev"] = df_get(
                bs, equity_labels, 1,
            )
            fin["retained_earnings"] = df_get(
                bs, ["Retained Earnings", "Retained Earnings Accumulated Deficit"], 0
            )
            current_assets = df_get(bs, ["Current Assets", "Total Current Assets"], 0)
            current_liabilities = df_get(
                bs,
                ["Current Liabilities",
                 "Total Current Liabilities",
                 "Current Liabilities And Short Term Debt"],
                0,
            )
            if (
                current_assets is not None
                and current_liabilities is not None
            ):
                fin["working_capital"] = (
                    float(current_assets)
                    - float(current_liabilities)
                )
            else:
                fin["working_capital"] = None
    except Exception as exc:
        logger.warning(
            "Failed to extract balance sheet for %s: %s",
            state["ticker"], exc,
        )
        log_trace(
            logger, "agent_error",
            agent="MarketDataAgent",
            error_type=type(exc).__name__,
            error_message=str(exc),
            data_source="yfinance.balance_sheet",
        )

    # --- Cash Flow Statement ---
    log_trace(
        logger, "api_call",
        source="yfinance.cashflow",
        ticker=state["ticker"],
    )
    try:
        cf = yf_ticker.cashflow
        if cf is not None and not cf.empty:
            cf_labels = [
                "Operating Cash Flow",
                "Total Cash From Operating Activities",
                "Cash From Operations",
                "Cash Flows From Used In Operating Activities",
            ]
            fin["operating_cash_flow"] = df_get(
                cf, cf_labels, 0,
            )
    except Exception as exc:
        logger.warning(
            "Failed to extract cash flow for %s: %s",
            state["ticker"], exc,
        )
        log_trace(
            logger, "agent_error",
            agent="MarketDataAgent",
            error_type=type(exc).__name__,
            error_message=str(exc),
            data_source="yfinance.cashflow",
        )

    logger.debug("Financial statements extracted for %s", state["ticker"])


def _extract_news(
    yf_ticker: yf.Ticker, state: ResearchState, max_headlines: int
) -> None:
    """Fetch recent news headlines from yfinance and store in state."""
    log_trace(
        logger, "api_call",
        source="yfinance.news",
        ticker=state["ticker"],
    )
    try:
        items: list[Any] = yf_ticker.news or []
        state["news"] = [
            {
                "headline": a.get("content", {}).get("title", ""),
                "url": a.get("canonicalUrl", {}).get("url", ""),
                "publisher": a.get("content", {})
                    .get("provider", {})
                    .get("displayName", ""),
            }
            for a in items[:max_headlines]
            if a.get("content", {}).get("title")
        ]
        log_trace(
            logger, "api_call",
            source="yfinance.news",
            ticker=state["ticker"],
            headlines_count=len(state["news"]),
        )
        logger.debug(
            "News extracted for %s: %d headlines",
            state["ticker"],
            len(state["news"]),
        )
    except Exception as exc:
        msg = f"Failed to extract news: {exc}"
        logger.warning(msg)
        state["errors"].append(msg)
        log_trace(
            logger, "agent_error",
            agent="MarketDataAgent",
            error_type=type(exc).__name__,
            error_message=str(exc),
            data_source="yfinance.news",
        )


def _extract_institutional_depth(yf_ticker: yf.Ticker, state: ResearchState) -> None:
    """Extract institutional-grade market data depth.

    Includes 5y history, dividends, splits, beta, ownership, etc.
    """
    cache = get_cache()

    # Fetch 5-year price history with caching
    log_trace(
        logger, "api_call",
        source="yfinance.history",
        ticker=state["ticker"],
        period="5y",
    )
    prices_5y = fetch_5y_price_history(yf_ticker, cache)
    state["market_data"]["historical_prices_5y"] = prices_5y

    # Fetch dividend history
    log_trace(
        logger, "api_call",
        source="yfinance.dividends",
        ticker=state["ticker"],
    )
    dividend_hist = fetch_dividend_history(yf_ticker, years=5)
    state["market_data"]["dividend_history"] = dividend_hist

    # Fetch split history
    log_trace(
        logger, "api_call",
        source="yfinance.splits",
        ticker=state["ticker"],
    )
    split_hist = fetch_split_history(yf_ticker, years=5)
    state["market_data"]["split_history"] = split_hist

    # Analyze volume patterns (requires 5y price data)
    if prices_5y is not None:
        volume_analysis = analyze_volume_patterns(prices_5y)
        state["market_data"]["volume_analysis"] = volume_analysis
    else:
        state["market_data"]["volume_analysis"] = None

    # Calculate 52-week range from daily Close prices.
    # Note: differs from fast_info fifty_two_week_high/low which use intraday prices.
    current_price = state["market_data"].get("current_price")
    if current_price:
        week_52_range = calculate_52week_range(yf_ticker, current_price)
        state["market_data"]["week_52_range"] = week_52_range
    else:
        state["market_data"]["week_52_range"] = None

    # Calculate beta vs S&P 500
    log_trace(
        logger, "api_call",
        source="yfinance.history",
        ticker=state["ticker"],
        period="3y",
        purpose="beta_calculation",
    )
    beta = calculate_beta(yf_ticker)
    state["market_data"]["beta"] = beta

    # Extract market cap, shares outstanding, float, institutional ownership
    try:
        info = yf_ticker.info

        # Only set market_cap from info if fast_info didn't provide one (H3)
        if not state["market_data"].get("market_cap"):
            state["market_data"]["market_cap"] = info.get("marketCap")
        state["market_data"]["shares_outstanding"] = info.get("sharesOutstanding")
        state["market_data"]["company_name"] = (
            info.get("longName") or info.get("shortName")
        )
        state["market_data"]["enterprise_value"] = info.get("enterpriseValue")
        # trailingAnnualDividendYield is a fraction (e.g. 0.004) across
        # yfinance versions; "dividendYield" changed units in 0.2.50+.
        state["market_data"]["dividend_yield"] = info.get(
            "trailingAnnualDividendYield"
        )
        float_shares = info.get("floatShares")
        shares_outstanding = info.get("sharesOutstanding")

        # Calculate float percentage
        if float_shares and shares_outstanding and shares_outstanding > 0:
            state["market_data"]["float_pct"] = (
                (float_shares / shares_outstanding) * 100
            )
        else:
            state["market_data"]["float_pct"] = None

        # Convert fraction (0-1) to percentage (0-100) for consistency
        held_pct = info.get("heldPercentInstitutions")
        state["market_data"]["institutional_ownership_pct"] = (
            held_pct * 100 if held_pct is not None else None
        )

        # Extract peer comparison context (sector/industry + comparable metrics)
        state["market_data"]["peer_comparison"] = (
            _extract_peer_context(info)
        )

        logger.debug("Institutional depth data extracted for %s", state["ticker"])

    except Exception as exc:
        msg = f"Failed to extract institutional ownership data: {exc}"
        logger.warning(msg)
        state["errors"].append(msg)
        log_trace(
            logger, "agent_error",
            agent="MarketDataAgent",
            error_type=type(exc).__name__,
            error_message=str(exc),
            data_source="yfinance.info",
        )


def _extract_peer_context(info: dict[str, Any]) -> dict[str, Any] | None:
    """Extract sector/industry context and stock's comparable metrics.

    Args:
        info: yfinance ticker.info dict.

    Returns:
        Dict with sector, industry, and stock_metrics for peer comparison,
        or None if no sector/industry data available.
    """
    sector = info.get("sector")
    industry = info.get("industry")
    if not sector and not industry:
        return None
    return {
        "sector": sector,
        "industry": industry,
        "stock_metrics": {
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
        },
    }


def _post_market_insights(state: ResearchState) -> None:
    """Post cross-domain market signals to the insights board.

    Reads volume_analysis, week_52_range, and price_history from market_data
    and posts high-signal observations for downstream agent consumption.
    Appends to state['errors'] on failure; never raises.

    Args:
        state: ResearchState with market_data populated.
    """
    try:
        md = state.get("market_data", {})
        ticker = state.get("ticker", "")

        # Volume spike signal
        vol = md.get("volume_analysis") or {}
        if vol.get("spike_detected"):
            ratio = vol.get("spike_ratio", 0)
            post_insight(
                state,
                agent="MarketDataAgent",
                category="volume",
                signal=(
                    f"Unusual volume spike: {ratio:.1f}x "
                    f"30-day average for {ticker}"
                ),
                confidence=0.8,
            )

        # 52-week high/low breach signal
        week_range = md.get("week_52_range") or {}
        current = md.get("current_price")
        high_52 = week_range.get("high")
        low_52 = week_range.get("low")
        if current and high_52 and high_52 > 0:
            if (high_52 - current) / high_52 <= 0.02:
                post_insight(
                    state,
                    agent="MarketDataAgent",
                    category="price_action",
                    signal=(
                        f"{ticker} trading within 2% of "
                        f"52-week high (${high_52:.2f})"
                    ),
                    confidence=0.75,
                )
        if current and low_52 and low_52 > 0:
            if (current - low_52) / low_52 <= 0.02:
                post_insight(
                    state,
                    agent="MarketDataAgent",
                    category="price_action",
                    signal=f"{ticker} trading within 2% of 52-week low (${low_52:.2f})",
                    confidence=0.75,
                )

        # Significant drawdown from price history
        history = md.get("price_history") or []
        if len(history) >= 20:
            recent_high = max(history[-20:])
            current_val = history[-1]
            if recent_high > 0:
                drawdown = (recent_high - current_val) / recent_high
                if drawdown >= 0.15:
                    post_insight(
                        state,
                        agent="MarketDataAgent",
                        category="technical",
                        signal=(
                            f"{ticker} down {drawdown*100:.0f}% from "
                            f"20-day rolling high (${recent_high:.2f})"
                        ),
                        confidence=0.7,
                    )
    except Exception as exc:
        msg = f"_post_market_insights failed: {exc}"
        logger.warning(msg)
        state["errors"].append(msg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configure_logging()
    initial_state = create_initial_state("NVDA")
    agent = MarketDataAgent()
    result = agent.fetch_data(initial_state)
    print(json.dumps(result, indent=2, default=str))
