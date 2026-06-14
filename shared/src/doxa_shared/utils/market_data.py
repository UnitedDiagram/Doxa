"""Market data utility functions for Doxa."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def df_get(df: Any, row_labels: list[str], col_index: int) -> float | None:
    """Try multiple row label variants and return the value at col_index.

    This function handles yfinance's inconsistent row labeling by trying multiple
    variations of row labels (e.g., "Total Revenue", "totalRevenue", "TotalRevenue")
    until a match is found. Returns None if no valid value is found.

    Args:
        df: DataFrame-like object with labeled rows and columns.
        row_labels: List of row label variants to try (in priority order).
        col_index: Column index to extract the value from.

    Returns:
        Float value at the specified position, or None if not found or invalid.
    """
    for label in row_labels:
        try:
            if label in df.index:
                col = df.iloc[:, col_index]
                val = df.loc[label, col.name]
                if val is not None and str(val) not in ("nan", "None", "<NA>"):
                    return float(val)
        except Exception:
            continue
    return None


def safe_get(obj: Any, attr: str) -> Any:
    """Get an attribute from an object, returning None on failure.

    This function safely extracts attributes from yfinance objects like fast_info
    that may raise exceptions or return invalid data.

    Args:
        obj: Object to extract attribute from (typically yfinance fast_info).
        attr: Attribute name to retrieve.

    Returns:
        Attribute value if successful, None otherwise.
    """
    try:
        return getattr(obj, attr, None)
    except Exception:
        return None


def fetch_5y_price_history(
    ticker: Any,
    cache: Any = None,
) -> Any:
    """Fetch 5-year daily price and volume data with defensive parsing.

    Args:
        ticker: yfinance Ticker object.
        cache: Optional cache backend for caching historical data.

    Returns:
        DataFrame with 5-year OHLCV data, or None if fetch fails.
    """
    try:
        if cache is not None:
            # Import here to avoid circular dependency
            from doxa_shared.utils.cache import TTL_PRICE_DATA, cached_fetch

            prices_5y = cached_fetch(
                cache,
                key=f"{ticker.ticker}:prices:5y",
                fetcher=lambda: ticker.history(period="5y"),
                ttl_seconds=TTL_PRICE_DATA,
            )
        else:
            prices_5y = ticker.history(period="5y")

        if prices_5y is None or prices_5y.empty:
            logger.warning("No 5-year price history returned for %s", ticker.ticker)
            return None

        # Verify expected columns exist (defensive yfinance parsing)
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in prices_5y.columns:
                logger.warning(
                    "Missing column %s in 5y price history for %s",
                    col,
                    ticker.ticker,
                )
                return None

        return prices_5y

    except Exception as e:
        logger.warning("Failed to fetch 5y price history for %s: %s", ticker.ticker, e)
        return None


def fetch_dividend_history(
    ticker: Any,
    years: int = 5,
) -> dict[str, Any] | None:
    """Fetch dividend history with yield and growth analysis.

    Args:
        ticker: yfinance Ticker object.
        years: Number of years to look back (default: 5).

    Returns:
        Dict with total_dividends, yield, growth_rate, payment_count,
        or None if no dividends.
    """
    try:
        from datetime import UTC, datetime, timedelta

        dividends = ticker.dividends
        if dividends is None or dividends.empty:
            return None

        # Filter to last N years (tz-aware to match yfinance indices)
        cutoff_date = datetime.now(tz=UTC) - timedelta(days=years * 365)
        if hasattr(dividends.index, "tz") and dividends.index.tz is None:
            cutoff_date = cutoff_date.replace(tzinfo=None)
        dividends_filtered = dividends[dividends.index >= cutoff_date]

        if dividends_filtered.empty:
            return None

        total_dividends = float(dividends_filtered.sum())
        payment_count = len(dividends_filtered)

        # Calculate dividend yield (annual dividends / current price)
        current_price = safe_get(ticker.fast_info, "last_price")
        dividend_yield = (
            (total_dividends / years / current_price) * 100 if current_price else None
        )

        # Calculate CAGR if we have >1 year of data
        growth_rate = None
        if payment_count > 1:
            first_year_div = dividends_filtered.iloc[0]
            last_year_div = dividends_filtered.iloc[-1]
            if first_year_div > 0:
                growth_rate = (
                    ((last_year_div / first_year_div) ** (1 / years) - 1) * 100
                )

        return {
            "total_dividends": total_dividends,
            "yield": dividend_yield,
            "growth_rate": growth_rate,
            "payment_count": payment_count,
        }

    except Exception as e:
        logger.warning("Failed to fetch dividend history for %s: %s", ticker.ticker, e)
        return None


def fetch_split_history(
    ticker: Any,
    years: int = 5,
) -> list[dict[str, Any]] | None:
    """Fetch stock split history with dates and ratios.

    Args:
        ticker: yfinance Ticker object.
        years: Number of years to look back (default: 5).

    Returns:
        List of dicts with date, ratio, impact, or None if no splits.
    """
    try:
        from datetime import UTC, datetime, timedelta

        splits = ticker.splits
        if splits is None or splits.empty:
            return None

        # Filter to last N years (tz-aware to match yfinance indices)
        cutoff_date = datetime.now(tz=UTC) - timedelta(days=years * 365)
        if hasattr(splits.index, "tz") and splits.index.tz is None:
            cutoff_date = cutoff_date.replace(tzinfo=None)
        splits_filtered = splits[splits.index >= cutoff_date]

        if splits_filtered.empty:
            return None

        split_history = []
        for date, ratio in splits_filtered.items():
            from fractions import Fraction

            frac = Fraction(float(ratio)).limit_denominator(100)
            if ratio >= 1:
                ratio_str = f"{frac.numerator}-for-{frac.denominator} split"
            else:
                ratio_str = (
                    f"{frac.numerator}-for-{frac.denominator} reverse split"
                )
            split_history.append({
                "date": str(date.date()),
                "ratio": ratio_str,
                "impact": f"Price adjusted by {ratio}x",
            })

        return split_history if split_history else None

    except Exception as e:
        logger.warning("Failed to fetch split history for %s: %s", ticker.ticker, e)
        return None


def analyze_volume_patterns(
    prices_df: Any,
) -> dict[str, Any] | None:
    """Calculate average volume and detect unusual activity (>2x average).

    Args:
        prices_df: DataFrame with Volume column.

    Returns:
        Dict with avg_volume, current_volume, unusual_activity_detected, volume_ratio,
        or None on error.
    """
    try:
        # Handle missing Volume column defensively
        if "Volume" not in prices_df.columns:
            logger.warning("Volume column missing in price DataFrame")
            return None

        if prices_df.empty:
            return None

        avg_volume = float(prices_df["Volume"].mean())
        current_volume = float(prices_df["Volume"].iloc[-1])

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0
        unusual_activity_detected = volume_ratio > 2.0

        return {
            "avg_volume": avg_volume,
            "current_volume": current_volume,
            "unusual_activity_detected": unusual_activity_detected,
            "volume_ratio": volume_ratio,
        }

    except Exception as e:
        logger.warning("Failed to analyze volume patterns: %s", e)
        return None


def calculate_52week_range(
    ticker: Any,
    current_price: float,
) -> dict[str, Any] | None:
    """Calculate 52-week high/low and current price position.

    Args:
        ticker: yfinance Ticker object.
        current_price: Current stock price.

    Returns:
        Dict with week_52_high, week_52_low, current_position_pct, or None on error.
    """
    try:
        hist_1y = ticker.history(period="1y")
        if hist_1y is None or hist_1y.empty:
            logger.warning("No 1-year history for 52-week range calculation")
            return None

        if "Close" not in hist_1y.columns:
            logger.warning("Close column missing in 1y price history")
            return None

        week_52_high = float(hist_1y["Close"].max())
        week_52_low = float(hist_1y["Close"].min())

        # Calculate position: (current - low) / (high - low) * 100
        if week_52_high == week_52_low:
            current_position_pct = 50.0  # Edge case: flat price
        else:
            current_position_pct = (
                (current_price - week_52_low) / (week_52_high - week_52_low) * 100
            )

        return {
            "week_52_high": week_52_high,
            "week_52_low": week_52_low,
            "current_position_pct": current_position_pct,
        }

    except Exception as e:
        logger.warning("Failed to calculate 52-week range: %s", e)
        return None


def calculate_beta(
    ticker: Any,
    market_ticker: str = "^GSPC",
) -> float | None:
    """Calculate beta coefficient vs S&P 500.

    Beta > 1: more volatile than market
    Beta < 1: less volatile than market
    Beta = 1: same volatility as market

    Args:
        ticker: yfinance Ticker object for the stock.
        market_ticker: Market index ticker (default: ^GSPC for S&P 500).

    Returns:
        Beta coefficient, or None if insufficient data (requires 252+ trading days).
    """
    try:
        import yfinance as yf

        # Fetch 3-year daily data for both ticker and market
        stock_hist = ticker.history(period="3y")
        market = yf.Ticker(market_ticker)
        market_hist = market.history(period="3y")

        if (
            stock_hist is None
            or stock_hist.empty
            or market_hist is None
            or market_hist.empty
        ):
            logger.warning("Insufficient price data for beta calculation")
            return None

        # Require at least 252 trading days (1 year)
        if len(stock_hist) < 252 or len(market_hist) < 252:
            logger.warning(
                "Insufficient data points for beta: stock=%d, market=%d (need 252+)",
                len(stock_hist),
                len(market_hist),
            )
            return None

        # Calculate daily returns
        stock_returns = stock_hist["Close"].pct_change().dropna()
        market_returns = market_hist["Close"].pct_change().dropna()

        # Align indices (only use overlapping dates)
        aligned_stock, aligned_market = stock_returns.align(
            market_returns, join="inner"
        )

        if len(aligned_stock) < 252:
            logger.warning("Insufficient overlapping data for beta calculation")
            return None

        # Beta = covariance(stock, market) / variance(market)
        covariance = aligned_stock.cov(aligned_market)
        market_variance = aligned_market.var()

        if market_variance == 0:
            logger.warning("Market variance is zero, cannot calculate beta")
            return None

        beta = covariance / market_variance
        return float(beta)

    except Exception as e:
        logger.warning("Failed to calculate beta: %s", e)
        return None
