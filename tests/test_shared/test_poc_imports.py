"""Tests verifying POC agents import from doxa_shared package."""

from __future__ import annotations

import importlib


class TestPocImportsFromShared:
    """Verify all POC agents import utilities from shared package, not local."""

    def test_market_data_imports_utils_from_shared(self) -> None:
        mod = importlib.import_module("src.agents.market_data")
        # df_get and safe_get should come from doxa_shared
        assert mod.df_get.__module__ == "doxa_shared.utils.market_data"
        assert mod.safe_get.__module__ == "doxa_shared.utils.market_data"

    def test_market_data_imports_constants_from_shared(self) -> None:
        mod = importlib.import_module("src.agents.market_data")
        # Constants should come from doxa_shared.constants.yfinance
        assert hasattr(mod, "FAST_INFO_LAST_PRICE")

    def test_valuation_imports_from_shared(self) -> None:
        """Verify ValuationAgent imports quant utilities from shared."""
        mod = importlib.import_module("src.agents.valuation")
        src = importlib.import_module("doxa_shared.utils.valuation")
        assert mod.calculate_dupont_analysis is src.calculate_dupont_analysis
        assert mod.calculate_altman_z_score is src.calculate_altman_z_score

    def test_sentiment_imports_from_shared(self) -> None:
        mod = importlib.import_module("src.agents.sentiment")
        src = importlib.import_module("doxa_shared.prompts.sentiment")
        assert mod.SENTIMENT_PROMPT is src.SENTIMENT_PROMPT

    def test_writer_imports_from_shared(self) -> None:
        mod = importlib.import_module("src.agents.writer")
        src_fmt = importlib.import_module("doxa_shared.utils.formatters")
        src_prompt = importlib.import_module("doxa_shared.prompts.writer")
        assert mod.fmt_number is src_fmt.fmt_number
        assert mod.NARRATIVE_PROMPT is src_prompt.NARRATIVE_PROMPT

    def test_state_reexports_from_shared(self) -> None:
        mod = importlib.import_module("src.state")
        src = importlib.import_module("doxa_shared.types.state")
        assert mod.ResearchState is src.ResearchState
        assert mod.create_initial_state is src.create_initial_state
