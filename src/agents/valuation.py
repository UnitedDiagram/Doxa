"""ValuationAgent for Doxa - DCF and comparable company analysis.

This agent generates institutional-grade valuation analysis including:
- DCF (Discounted Cash Flow) models with 5-year projections
- Comparable company analysis with peer multiples
- Confidence scoring based on data completeness
"""

from __future__ import annotations

import logging
import statistics
from datetime import UTC, datetime
from typing import Any

import yfinance as yf
from doxa_shared.types.state import ResearchState
from doxa_shared.utils.insights import post_insight
from doxa_shared.utils.market_data import df_get
from doxa_shared.utils.valuation import (
    analyze_5y_trends,
    calculate_altman_z_score,
    calculate_dcf_fair_value,
    calculate_dupont_analysis,
    calculate_financial_ratios,
    calculate_terminal_value,
    calculate_valuation_multiples,
    calculate_wacc,
    generate_sensitivity_table,
)

logger = logging.getLogger(__name__)


class ValuationAgent:
    """Agent for generating DCF valuations and comparable company analysis."""

    def execute(self, state: ResearchState) -> ResearchState:
        """Execute valuation analysis and update state.

        Generates DCF model and comparable company analysis, writing results
        to state['valuation_analysis']. Never raises exceptions - errors are
        appended to state['errors'] list.

        Args:
            state: ResearchState dict containing 'ticker' key.

        Returns:
            Updated ResearchState with 'valuation_analysis' populated.
        """
        ticker = state.get("ticker")
        if not ticker:
            error_msg = "ValuationAgent: No ticker provided in state"
            logger.warning(error_msg)
            state["errors"].append(error_msg)
            return state

        logger.info("ValuationAgent starting for %s", ticker)

        # Initialize valuation analysis structure
        valuation_analysis: dict[str, Any] = {
            "dcf": {},
            "comps": {},
            "dupont_analysis": {},
            "altman_z_score": {},
            "financial_ratios": {},
            "trend_analysis": {},
            "confidence": 0.0,
        }

        try:
            # Fetch company data once to avoid redundant API calls
            company = yf.Ticker(ticker)
            info = company.info

            # DCF Analysis
            dcf_result = self._calculate_dcf(company, info, ticker, state)
            valuation_analysis["dcf"] = dcf_result

            # Comparable Company Analysis
            comps_result = self._calculate_comps(info, ticker, state)
            valuation_analysis["comps"] = comps_result

            # Quantitative Analysis (absorbed from QuantAgent)
            quant_result = self._calculate_quantitative_analysis(
                company, info, ticker, state
            )
            valuation_analysis["dupont_analysis"] = quant_result.get(
                "dupont_analysis"
            )
            valuation_analysis["altman_z_score"] = quant_result.get(
                "altman_z_score"
            )
            valuation_analysis["financial_ratios"] = quant_result.get(
                "financial_ratios"
            )
            valuation_analysis["trend_analysis"] = quant_result.get(
                "trend_analysis"
            )

            # Confidence Scoring (includes quant components)
            confidence = self._calculate_confidence(
                dcf_result, comps_result, quant_result
            )
            valuation_analysis["confidence"] = confidence

            logger.info(
                "ValuationAgent completed for %s (confidence: %.1f%%)",
                ticker,
                confidence,
            )

        except Exception as e:
            error_msg = f"ValuationAgent: Unexpected error: {e}"
            logger.warning(error_msg, exc_info=True)
            state["errors"].append(error_msg)
            valuation_analysis["confidence"] = 0.0

        state["valuation_analysis"] = valuation_analysis
        state["quant_analysis"] = self._derive_quant_summary(state, valuation_analysis)

        # Add provenance metadata
        if "provenance_metadata" not in state:
            state["provenance_metadata"] = {}
        state["provenance_metadata"]["valuation"] = {
            "agent": "ValuationAgent",
            "source": "DCF model + quantitative analysis",
            "timestamp": datetime.now(UTC).isoformat(),
            "confidence": valuation_analysis.get("confidence", 0.0),
            "components": [
                "dcf",
                "comps",
                "dupont",
                "z_score",
                "ratios",
                "trends",
            ],
        }

        _post_valuation_insights(state)

        return state

    def _derive_quant_summary(
        self,
        state: ResearchState,
        valuation_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Derive flat quant_analysis summary from nested valuation_analysis.

        Bridges the deprecated quant_analysis field that WriterAgent reads.
        """
        dcf = valuation_analysis.get("dcf") or {}
        dupont = valuation_analysis.get("dupont_analysis") or {}
        z = valuation_analysis.get("altman_z_score") or {}

        # Signal from DCF upside/downside
        upside = dcf.get("upside_downside_pct", 0) or 0
        if upside > 15:
            signal = "BULLISH"
        elif upside < -15:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        # Altman Z-Score fields
        altman_z: float | None = z.get("z_score") if isinstance(z, dict) else None
        interpretation: str = z.get("interpretation", "") if isinstance(z, dict) else ""
        if "Safe" in interpretation:
            altman_zone = "Safe"
        elif "Grey" in interpretation:
            altman_zone = "Grey"
        elif "Distress" in interpretation:
            altman_zone = "Distress"
        else:
            altman_zone = ""

        # DuPont fields
        profit_margin: float | None = dupont.get("profit_margin")
        asset_turnover: float | None = dupont.get("asset_turnover")
        equity_multiplier: float | None = dupont.get("equity_multiplier")
        roe: float | None = dupont.get("roe")

        # Classify dominant DuPont driver
        if profit_margin is not None and profit_margin > 0.20:
            dupont_driver = "High Profitability"
        elif asset_turnover is not None and asset_turnover > 1.0:
            dupont_driver = "Asset Efficiency"
        elif equity_multiplier is not None and equity_multiplier > 3.0:
            dupont_driver = "High Leverage"
        elif dupont:
            dupont_driver = "Balanced Returns"
        else:
            dupont_driver = ""

        md = state.get("market_data", {})
        pe_ratio = md.get("pe_trailing") or (
            md.get("peer_comparison") or {}
        ).get("stock_metrics", {}).get("pe_trailing")

        return {
            "signal": signal,
            "altman_z": altman_z,
            "altman_zone": altman_zone,
            "pe_ratio": pe_ratio,
            "confidence": valuation_analysis.get("confidence", 0.0),
            "dupont_driver": dupont_driver,
            "profit_margin": profit_margin,
            "roe": roe,
            "asset_turnover": asset_turnover,
            "equity_multiplier": equity_multiplier,
        }

    def _calculate_dcf(
        self,
        company: yf.Ticker,
        info: dict[str, Any],
        ticker: str,
        state: ResearchState,
    ) -> dict[str, Any]:
        """Calculate DCF valuation model.

        Fetches financial data, calculates FCF projections, WACC, terminal value,
        and fair value per share. Returns empty dict on failure.
        """
        dcf_data: dict[str, Any] = {}

        try:
            # Fetch financial statements
            cashflow = company.cashflow
            income_stmt = company.income_stmt
            balance_sheet = company.balance_sheet

            # Extract operating cash flow (historical)
            operating_cf_values: list[float] = []
            if not cashflow.empty and "Operating Cash Flow" in cashflow.index:
                operating_cf_series = cashflow.loc["Operating Cash Flow"]
                # Convert to list, most recent first
                operating_cf_values = operating_cf_series.dropna().tolist()

            if len(operating_cf_values) < 2:
                error_msg = (
                    f"ValuationAgent: Insufficient cash flow data for {ticker}"
                )
                logger.warning(error_msg)
                state["errors"].append(error_msg)
                return {}

            # Calculate historical growth rate for multi-stage projection
            recent_cf = operating_cf_values[0]
            previous_cf = operating_cf_values[1]
            historical_growth = (
                (recent_cf / previous_cf) - 1.0 if previous_cf > 0 else 0.15
            )

            # Multi-stage growth assumptions
            high_growth = max(min(historical_growth, 0.20), 0.10)  # 10-20%
            transition_growth = (high_growth + 0.025) / 2  # Fade to terminal
            terminal_growth = 0.025  # 2.5% perpetual

            # Project FCF with multi-stage growth
            fcf_projections = []
            current_fcf = recent_cf

            # Years 1-3: High growth phase
            for _ in range(3):
                current_fcf = current_fcf * (1 + high_growth)
                fcf_projections.append(current_fcf)

            # Years 4-5: Transition phase
            for _ in range(2):
                current_fcf = current_fcf * (1 + transition_growth)
                fcf_projections.append(current_fcf)

            if len(fcf_projections) != 5:
                error_msg = (
                    f"ValuationAgent: Failed to project FCF for {ticker}"
                )
                logger.warning(error_msg)
                state["errors"].append(error_msg)
                return {}

            # Store growth assumptions for transparency
            growth_assumptions = {
                "years_1_3": round(high_growth * 100, 1),  # As percentage
                "years_4_5": round(transition_growth * 100, 1),
                "terminal": round(terminal_growth * 100, 1),
                "historical_growth": round(historical_growth * 100, 1),
            }

            # Extract WACC inputs
            beta = info.get("beta", 1.0)
            shares_outstanding = info.get("sharesOutstanding", 0)

            # Get debt and equity from balance sheet
            total_debt = 0.0
            total_equity = 0.0
            if not balance_sheet.empty:
                if "Total Debt" in balance_sheet.index:
                    total_debt = balance_sheet.loc["Total Debt"].iloc[0]
                if "Stockholders Equity" in balance_sheet.index:
                    total_equity = (
                        balance_sheet.loc["Stockholders Equity"].iloc[0]
                    )

            # Calculate debt/equity ratio
            debt_equity_ratio = (
                total_debt / total_equity if total_equity > 0 else 0.5
            )

            # Get tax rate (approximate)
            tax_rate = 0.21  # Default US corporate tax rate
            if (
                not income_stmt.empty
                and "Tax Provision" in income_stmt.index
                and "Pretax Income" in income_stmt.index
            ):
                tax_provision = income_stmt.loc["Tax Provision"].iloc[0]
                pretax_income = income_stmt.loc["Pretax Income"].iloc[0]
                if pretax_income > 0:
                    tax_rate = abs(tax_provision / pretax_income)

            # Calculate WACC (POC constants: 4.5% risk-free, 7% market premium)
            wacc = calculate_wacc(
                beta=beta,
                risk_free_rate=0.045,
                market_risk_premium=0.07,
                debt_equity_ratio=debt_equity_ratio,
                tax_rate=tax_rate,
            )

            # Calculate terminal value (2.5% perpetual growth)
            terminal_value = calculate_terminal_value(
                final_fcf=fcf_projections[-1],
                growth_rate=0.025,
                wacc=wacc,
            )

            # Calculate fair value per share
            if shares_outstanding == 0:
                error_msg = (
                    f"ValuationAgent: No shares outstanding for {ticker}"
                )
                logger.warning(error_msg)
                state["errors"].append(error_msg)
                return {}

            fair_value_per_share = calculate_dcf_fair_value(
                fcf_projections=fcf_projections,
                terminal_value=terminal_value,
                wacc=wacc,
                shares_outstanding=shares_outstanding,
            )

            # Get current price
            current_price = info.get("currentPrice", 0)
            if current_price == 0:
                current_price = info.get("regularMarketPrice", 0)

            # Calculate upside/downside
            upside_downside_pct = (
                (
                    (fair_value_per_share - current_price)
                    / current_price
                    * 100
                )
                if current_price > 0
                else 0.0
            )

            # Generate sensitivity analysis table
            sensitivity_table = generate_sensitivity_table(
                base_wacc=wacc,
                base_growth=0.025,
                fcf_projections=fcf_projections,
                shares_outstanding=shares_outstanding,
            )

            # Build DCF result
            dcf_data = {
                "fcf_projections": fcf_projections,
                "terminal_value": terminal_value,
                "wacc": wacc,
                "fair_value_per_share": round(fair_value_per_share, 2),
                "current_price": round(current_price, 2),
                "upside_downside_pct": round(upside_downside_pct, 1),
                "sensitivity_table": sensitivity_table,
                "growth_assumptions": growth_assumptions,
            }

            logger.info(
                "DCF calculation complete for %s: "
                "Fair value $%.2f vs $%.2f",
                ticker,
                fair_value_per_share,
                current_price,
            )

        except Exception as e:
            error_msg = f"ValuationAgent DCF: {e}"
            logger.warning(error_msg, exc_info=True)
            state["errors"].append(error_msg)
            return {}

        return dcf_data

    def _calculate_comps(
        self,
        info: dict[str, Any],
        ticker: str,
        state: ResearchState,
    ) -> dict[str, Any]:
        """Calculate comparable company analysis.

        Identifies peer companies, calculates multiples, and determines
        implied valuation based on peer medians.
        """
        comps_data: dict[str, Any] = {}

        try:
            # Use sector (broad) not industry (too specific) for peer matching
            sector = info.get("sector", "Unknown")

            # Hardcoded peer lists by sector (POC simplification)
            peer_map = {
                "Technology": ["AAPL", "MSFT", "GOOGL", "META"],
                "Financial Services": ["JPM", "BAC", "WFC", "C"],
                "Healthcare": ["JNJ", "PFE", "UNH", "ABBV"],
                "Consumer Cyclical": ["AMZN", "HD", "NKE", "SBUX"],
                "Consumer Defensive": ["WMT", "PG", "KO", "PEP"],
                "Energy": ["XOM", "CVX", "COP", "SLB"],
                "Industrials": ["BA", "CAT", "GE", "HON"],
                "Communication Services": ["GOOGL", "META", "DIS", "NFLX"],
                "Real Estate": ["AMT", "PLD", "CCI", "EQIX"],
                "Basic Materials": ["LIN", "APD", "ECL", "DD"],
                "Utilities": ["NEE", "DUK", "SO", "D"],
                "Unknown": ["SPY"],  # Market ETF as fallback
            }

            # Select peers (exclude ticker itself if present)
            peers = peer_map.get(sector, peer_map["Unknown"])
            peers = [p for p in peers if p != ticker][:6]  # Max 6 peers

            if not peers:
                error_msg = f"ValuationAgent: No peers found for {ticker}"
                logger.warning(error_msg)
                state["errors"].append(error_msg)
                return {}

            # Calculate multiples for each peer
            peer_multiples: dict[str, dict[str, float | None]] = {}
            for peer_ticker in peers:
                try:
                    peer = yf.Ticker(peer_ticker)
                    peer_info = peer.info

                    market_cap = peer_info.get("marketCap", 0)
                    revenue = peer_info.get("totalRevenue", 0)
                    ebitda = peer_info.get("ebitda", 0)
                    book_value = peer_info.get(
                        "bookValue", 0
                    ) * peer_info.get("sharesOutstanding", 0)
                    net_income = peer_info.get("netIncomeToCommon", 0)
                    peer_debt = peer_info.get("totalDebt", 0)
                    peer_cash = peer_info.get("totalCash", 0)

                    multiples = calculate_valuation_multiples(
                        market_cap=market_cap,
                        revenue=revenue,
                        ebitda=ebitda,
                        book_value=book_value,
                        net_income=net_income,
                        total_debt=peer_debt,
                        cash=peer_cash,
                    )

                    peer_multiples[peer_ticker] = multiples
                    logger.debug("Calculated multiples for %s", peer_ticker)

                except Exception as e:
                    error_msg = (
                        f"ValuationAgent: Failed to fetch {peer_ticker}: {e}"
                    )
                    logger.warning(error_msg)
                    state["errors"].append(error_msg)

            if not peer_multiples:
                error_msg = (
                    f"ValuationAgent: No peer data available for {ticker}"
                )
                logger.warning(error_msg)
                state["errors"].append(error_msg)
                return {}

            # Calculate median and mean multiples
            median_multiples: dict[str, float | None] = {}
            mean_multiples: dict[str, float | None] = {}

            for multiple_name in ["P/E", "EV/EBITDA", "P/B", "P/S"]:
                # Filter out None values and ensure float type
                values: list[float] = []
                for m in peer_multiples.values():
                    val = m.get(multiple_name)
                    if val is not None:
                        values.append(float(val))

                if values:
                    median_multiples[multiple_name] = statistics.median(values)
                    mean_multiples[multiple_name] = statistics.mean(values)
                else:
                    median_multiples[multiple_name] = None
                    mean_multiples[multiple_name] = None

            # Calculate implied valuations for target company
            target_market_cap = info.get("marketCap", 0)
            target_revenue = info.get("totalRevenue", 0)
            target_ebitda = info.get("ebitda", 0)
            target_book_value = info.get("bookValue", 0) * info.get(
                "sharesOutstanding", 0
            )
            target_net_income = info.get("netIncomeToCommon", 0)
            target_debt = info.get("totalDebt", 0)
            target_cash = info.get("totalCash", 0)

            implied_valuations: dict[str, float | None] = {}

            if median_multiples.get("P/E") and target_net_income > 0:
                implied_valuations["P/E"] = (
                    median_multiples["P/E"] * target_net_income
                )

            if median_multiples.get("EV/EBITDA") and target_ebitda > 0:
                implied_valuations["EV/EBITDA"] = (
                    median_multiples["EV/EBITDA"] * target_ebitda
                )

            if median_multiples.get("P/B") and target_book_value > 0:
                implied_valuations["P/B"] = (
                    median_multiples["P/B"] * target_book_value
                )

            if median_multiples.get("P/S") and target_revenue > 0:
                implied_valuations["P/S"] = (
                    median_multiples["P/S"] * target_revenue
                )

            # Calculate target company multiples
            target_multiples = calculate_valuation_multiples(
                market_cap=target_market_cap,
                revenue=target_revenue,
                ebitda=target_ebitda,
                book_value=target_book_value,
                net_income=target_net_income,
                total_debt=target_debt,
                cash=target_cash,
            )

            # Calculate premium/discount vs peers
            premium_discount_pct: dict[str, float | None] = {}

            for multiple_name in ["P/E", "EV/EBITDA", "P/B", "P/S"]:
                target_val = target_multiples.get(multiple_name)
                peer_median = median_multiples.get(multiple_name)

                if target_val and peer_median and peer_median > 0:
                    premium_discount_pct[multiple_name] = (
                        (target_val - peer_median) / peer_median * 100
                    )
                else:
                    premium_discount_pct[multiple_name] = None

            # Generate valuation justification based on fundamentals
            valuation_justification = self._generate_valuation_justification(
                premium_discount_pct, info, state
            )

            # Build comps result
            comps_data = {
                "peer_companies": peers,
                "peer_multiples": peer_multiples,
                "median_multiples": median_multiples,
                "mean_multiples": mean_multiples,
                "implied_valuations": implied_valuations,
                "premium_discount_pct": premium_discount_pct,
                "valuation_justification": valuation_justification,
            }

            logger.info(
                "Comps analysis complete for %s with %d peers",
                ticker,
                len(peers),
            )

        except Exception as e:
            error_msg = f"ValuationAgent Comps: {e}"
            logger.warning(error_msg, exc_info=True)
            state["errors"].append(error_msg)
            return {}

        return comps_data

    def _calculate_quantitative_analysis(
        self,
        company: yf.Ticker,
        info: dict[str, Any],
        ticker: str,
        state: ResearchState,
    ) -> dict[str, Any]:
        """Calculate quantitative financial analysis (absorbed from QuantAgent).

        Performs DuPont analysis, Altman Z-Score, financial ratios,
        and 5-year trend analysis. Returns dict with all results.
        """
        quant_data: dict[str, Any] = {
            "dupont_analysis": None,
            "altman_z_score": None,
            "financial_ratios": None,
            "trend_analysis": None,
        }

        try:
            # Fetch financial statements
            balance_sheet = company.balance_sheet
            income_stmt = company.income_stmt
            cashflow = company.cashflow

            # DuPont Analysis
            try:
                if not balance_sheet.empty and not income_stmt.empty:
                    financials = {
                        "net_income": df_get(income_stmt, ["Net Income"], 0),
                        "total_revenue": df_get(income_stmt, ["Total Revenue"], 0),
                        "total_assets": df_get(balance_sheet, ["Total Assets"], 0),
                        "stockholders_equity": df_get(
                            balance_sheet,
                            ["Stockholders Equity", "Common Stock Equity"],
                            0,
                        ),
                    }
                    dupont = calculate_dupont_analysis(financials)
                    quant_data["dupont_analysis"] = dupont
            except Exception as e:
                error_msg = f"ValuationAgent DuPont: {e}"
                logger.warning(error_msg)
                state["errors"].append(error_msg)

            # Altman Z-Score
            try:
                if not balance_sheet.empty and not income_stmt.empty:
                    market_cap = info.get("marketCap", 0)
                    z_score = calculate_altman_z_score(
                        balance_sheet, income_stmt, market_cap
                    )
                    quant_data["altman_z_score"] = z_score
            except Exception as e:
                error_msg = f"ValuationAgent Z-Score: {e}"
                logger.warning(error_msg)
                state["errors"].append(error_msg)

            # Financial Ratios
            try:
                if (
                    not balance_sheet.empty
                    and not income_stmt.empty
                    and not cashflow.empty
                ):
                    ratios = calculate_financial_ratios(
                        balance_sheet, income_stmt, cashflow
                    )
                    quant_data["financial_ratios"] = ratios
            except Exception as e:
                error_msg = f"ValuationAgent Ratios: {e}"
                logger.warning(error_msg)
                state["errors"].append(error_msg)

            # Trend Analysis (5-year historical data)
            try:
                if not income_stmt.empty and not balance_sheet.empty:
                    # Collect 5-year historical data for key metrics
                    historical_data: dict[str, list[float]] = {}

                    # Revenue trend
                    if "Total Revenue" in income_stmt.index:
                        revenue_series = income_stmt.loc["Total Revenue"]
                        historical_data["revenue"] = (
                            revenue_series.dropna().tolist()[::-1]
                        )  # Reverse to oldest first

                    # Margin trends (if data available)
                    if (
                        "Gross Profit" in income_stmt.index
                        and "Total Revenue" in income_stmt.index
                    ):
                        gross_profit = income_stmt.loc["Gross Profit"]
                        revenue = income_stmt.loc["Total Revenue"]
                        margins = (gross_profit / revenue).dropna()
                        if len(margins) > 0:
                            historical_data["gross_margin"] = margins.tolist()[
                                ::-1
                            ]

                    # ROE trend (if we have net income and equity)
                    if (
                        "Net Income" in income_stmt.index
                        and "Stockholders Equity" in balance_sheet.index
                    ):
                        net_income = income_stmt.loc["Net Income"]
                        equity = balance_sheet.loc["Stockholders Equity"]
                        # Align dates
                        common_dates = net_income.index.intersection(
                            equity.index
                        )
                        if len(common_dates) > 0:
                            roe_series = (
                                net_income.loc[common_dates]
                                / equity.loc[common_dates]
                            )
                            historical_data["roe"] = (
                                roe_series.dropna().tolist()[::-1]
                            )

                    # Analyze trends
                    if historical_data:
                        trends = analyze_5y_trends(historical_data)
                        quant_data["trend_analysis"] = trends

            except Exception as e:
                error_msg = f"ValuationAgent Trends: {e}"
                logger.warning(error_msg)
                state["errors"].append(error_msg)

            logger.info(
                "Quantitative analysis complete for %s", ticker
            )

        except Exception as e:
            error_msg = f"ValuationAgent Quant: Unexpected error: {e}"
            logger.warning(error_msg, exc_info=True)
            state["errors"].append(error_msg)

        return quant_data

    def _generate_valuation_justification(
        self,
        premium_discount_pct: dict[str, float | None],
        info: dict[str, Any],
        state: ResearchState,
    ) -> str:
        """Generate justification text for premium/discount vs peers.

        Analyzes company fundamentals (margins, ROE, growth) from state
        to explain valuation premium or discount relative to peers.
        """
        # Get average premium/discount across multiples
        valid_premiums = [
            v for v in premium_discount_pct.values() if v is not None
        ]
        if not valid_premiums:
            return "Insufficient data to justify valuation"

        avg_premium = sum(valid_premiums) / len(valid_premiums)

        # Extract fundamentals for justification
        valuation_analysis = state.get("valuation_analysis", {})
        financial_ratios = valuation_analysis.get("financial_ratios")

        # Build justification based on premium/discount direction
        if avg_premium > 10:  # Premium valuation
            reasons = []

            # Check profitability
            if financial_ratios and financial_ratios.get("profitability"):
                roe = financial_ratios["profitability"].get("roe")
                net_margin = financial_ratios["profitability"].get("net_margin")

                if roe and roe > 0.15:  # ROE > 15%
                    reasons.append(
                        f"{round(roe * 100, 1)}% ROE (above market average)"
                    )
                if net_margin and net_margin > 0.15:  # Net margin > 15%
                    reasons.append(
                        f"{round(net_margin * 100, 1)}% net margin"
                    )

            # Check growth trends
            trend_analysis = valuation_analysis.get("trend_analysis")
            if trend_analysis:
                if trend_analysis.get("revenue") == "Improving":
                    reasons.append("strong revenue growth")
                if trend_analysis.get("gross_margin") == "Improving":
                    reasons.append("expanding margins")

            if reasons:
                return (
                    f"+{round(avg_premium, 1)}% premium justified by "
                    f"{', '.join(reasons)}"
                )
            else:
                return (
                    f"+{round(avg_premium, 1)}% premium "
                    f"(fundamentals analysis pending)"
                )

        elif avg_premium < -10:  # Discount valuation
            reasons = []

            # Check for weaknesses
            if financial_ratios:
                if financial_ratios.get("leverage"):
                    debt_equity = financial_ratios["leverage"].get(
                        "debt_to_equity"
                    )
                    if debt_equity and debt_equity > 1.0:  # High leverage
                        reasons.append(
                            f"high leverage (D/E: {round(debt_equity, 2)})"
                        )

                if financial_ratios.get("profitability"):
                    net_margin = financial_ratios["profitability"].get(
                        "net_margin"
                    )
                    if net_margin and net_margin < 0.05:  # Low margins
                        reasons.append("low profitability")

            # Check negative trends
            trend_analysis = valuation_analysis.get("trend_analysis")
            if trend_analysis:
                if trend_analysis.get("revenue") == "Deteriorating":
                    reasons.append("declining revenue")
                if trend_analysis.get("gross_margin") == "Deteriorating":
                    reasons.append("margin compression")

            if reasons:
                return (
                    f"{round(avg_premium, 1)}% discount reflects "
                    f"{', '.join(reasons)}"
                )
            else:
                return (
                    f"{round(avg_premium, 1)}% discount "
                    f"(fundamentals analysis pending)"
                )

        else:  # Fair valuation (-10% to +10%)
            return "Fair valuation relative to peers"

    def _calculate_confidence(
        self,
        dcf_data: dict[str, Any],
        comps_data: dict[str, Any],
        quant_data: dict[str, Any],
    ) -> float:
        """Calculate confidence score based on data completeness.

        New weights (with quant integration):
        - DCF: 40% (reduced from 50%)
        - Comps: 30% (reduced from 50%)
        - Quant Analysis: 20% (NEW)
        - Trend Analysis: 10% (NEW)

        Returns score 0-100.
        """
        dcf_confidence = 0.0
        comps_confidence = 0.0
        quant_confidence = 0.0
        trend_confidence = 0.0

        # DCF confidence (40% weight)
        if dcf_data:
            fcf_proj = dcf_data.get("fcf_projections", [])
            wacc = dcf_data.get("wacc", 0)
            terminal_val = dcf_data.get("terminal_value", 0)

            if len(fcf_proj) == 5 and wacc > 0 and terminal_val > 0:
                dcf_confidence = 100.0
            elif len(fcf_proj) >= 3 and wacc > 0:
                dcf_confidence = 75.0
            elif len(fcf_proj) >= 2:
                dcf_confidence = 50.0
            elif len(fcf_proj) >= 1:
                dcf_confidence = 25.0

        # Comps confidence (30% weight)
        if comps_data:
            peer_multiples = comps_data.get("peer_multiples", {})

            # Count peers with complete data
            complete_peers = sum(
                1
                for p_multiples in peer_multiples.values()
                if any(v is not None for v in p_multiples.values())
            )

            if complete_peers >= 4:
                comps_confidence = 100.0
            elif complete_peers == 3:
                comps_confidence = 75.0
            elif complete_peers == 2:
                comps_confidence = 50.0
            elif complete_peers >= 1:
                comps_confidence = 25.0

        # Quant Analysis confidence (20% weight)
        if quant_data:
            # Count available quant components
            components_available = 0.0
            components_total = 3.0  # DuPont, Z-Score, Ratios

            if quant_data.get("dupont_analysis"):
                components_available += 1

            if quant_data.get("altman_z_score"):
                components_available += 1

            if quant_data.get("financial_ratios"):
                # Check if we have at least 8 ratios
                ratios = quant_data["financial_ratios"]
                ratio_count = sum(
                    1
                    for category in ratios.values()
                    for val in category.values()
                    if val is not None
                )
                if ratio_count >= 8:
                    components_available += 1.0
                elif ratio_count >= 5:
                    components_available += 0.75
                else:
                    components_available += 0.5

            # Score based on component completeness
            if components_total > 0:
                quant_confidence = (
                    components_available / components_total
                ) * 100.0

        # Trend Analysis confidence (10% weight)
        if quant_data and quant_data.get("trend_analysis"):
            trends = quant_data["trend_analysis"]
            metric_count = len(trends)

            if metric_count >= 5:  # 5+ metrics
                trend_confidence = 100.0
            elif metric_count >= 3:  # 3-4 metrics
                trend_confidence = 75.0
            elif metric_count >= 2:  # 2 metrics
                trend_confidence = 50.0
            elif metric_count >= 1:  # 1 metric
                trend_confidence = 25.0

        # Weighted average
        total_confidence = (
            (dcf_confidence * 0.4)
            + (comps_confidence * 0.3)
            + (quant_confidence * 0.2)
            + (trend_confidence * 0.1)
        )

        return round(total_confidence, 1)


def _post_valuation_insights(state: ResearchState) -> None:
    """Post cross-domain valuation signals to the insights board.

    Reads valuation_analysis from state and posts insights for margin
    compression, Altman Z distress, and low/declining ROE.
    Appends to state['errors'] on failure; never raises.

    Args:
        state: ResearchState with valuation_analysis populated.
    """
    try:
        ticker = state.get("ticker", "")
        val = state.get("valuation_analysis", {})

        # Altman Z-Score distress zone
        altman = val.get("altman_z_score") or {}
        altman_z = altman.get("z_score")
        altman_interp = altman.get("interpretation", "")
        if "Distress" in altman_interp and altman_z is not None:
            post_insight(
                state,
                agent="ValuationAgent",
                category="leverage",
                signal=(
                    f"{ticker} in Altman Z-Score distress zone "
                    f"(Z={altman_z:.2f} < 1.81) — elevated bankruptcy risk"
                ),
                confidence=0.9,
            )

        # Low / declining ROE (from DuPont analysis)
        dupont = val.get("dupont_analysis") or {}
        roe = dupont.get("roe")
        if roe is not None and roe < 0.05:
            post_insight(
                state,
                agent="ValuationAgent",
                category="profitability",
                signal=(
                    f"{ticker} ROE of {roe*100:.1f}% below 5% threshold "
                    f"— weak capital efficiency"
                ),
                confidence=0.75,
            )

        # Margin compression via trend analysis
        trend = val.get("trend_analysis") or {}
        if trend.get("gross_margin") == "Deteriorating":
            post_insight(
                state,
                agent="ValuationAgent",
                category="margin",
                signal=f"{ticker} gross margin on multi-year deteriorating trend",
                confidence=0.8,
            )

        # Large DCF discount (undervaluation or fundamental concern)
        dcf = val.get("dcf") or {}
        upside_pct = dcf.get("upside_downside_pct", 0)
        if upside_pct <= -20:
            post_insight(
                state,
                agent="ValuationAgent",
                category="valuation",
                signal=(
                    f"{ticker} DCF fair value {upside_pct:+.0f}% vs current price "
                    f"— potential overvaluation concern"
                ),
                confidence=0.7,
            )

    except Exception as exc:
        msg = f"_post_valuation_insights failed: {exc}"
        logger.warning(msg)
        state["errors"].append(msg)

