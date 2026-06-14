# CLAUDE.md

Guidance for AI coding assistants (and humans) working on this repo.

## Project: Doxa

AI-powered equity research pipeline. Takes a stock ticker, runs it through a
6-agent analysis pipeline, and produces a high-signal equity research report
focused on non-obvious risks, thesis drift, and mispriced assumptions.

### Architecture

**Two-package monorepo:**

- `shared/` — `doxa-shared` package (types, constants, prompts, pure utility functions)
  - `types/state.py` — `ResearchState` TypedDict and `create_initial_state()`
  - `constants/yfinance.py` — yfinance field mappings
  - `prompts/` — Claude prompt templates (sentiment, regulatory, writer, editor)
  - `utils/` — pure computation helpers (quant, valuation, market_data, edgar, formatters)
- `src/` — main `doxa` package (agents + orchestration)
  - `agents/` — six agents, each transforms `ResearchState`:
    1. `MarketDataAgent` — fetches price/financial data via yfinance
    2. `ValuationAgent` — DCF model, comps, DuPont analysis, Altman Z-Score, ratio analysis
    3. `RegulatoryAgent` — SEC EDGAR 10-K filings via Claude
    4. `SentimentAgent` — alternative data (insider trading, short interest, sentiment contradictions) via Claude; data feed is mocked
    5. `WriterAgent` — generates comprehensive Markdown research report via Claude
    6. `EditorAgent` — distills to high-signal content
  - `export/` — PDF rendering of reports via WeasyPrint
  - `main.py` — CLI orchestrator
  - `config.py` — env vars + logging setup
  - `state.py` — re-exports from shared
- `app.py` — Streamlit web UI (three-stage flow: input → summary tabs → report)
- `tests/` — pytest suite

**Data flow:** All agents read and return `ResearchState` (a TypedDict). Pipeline is sequential — each agent enriches the shared state dict. State is mutated in place, never recreated. Agents also post findings to a shared insights board for cross-domain intelligence.

### Tech Stack & Version Constraints

- **Python 3.12+** (required — PEP 695 type parameter syntax throughout)
- **anthropic >= 0.40.0** — Messages API (earlier versions incompatible)
- **yfinance >= 0.2.36** — financial market data (API is unstable, see gotchas below)
- **streamlit >= 1.35** — web UI (session state persistence fix)
- **httpx >= 0.27** — SEC EDGAR API calls
- **weasyprint >= 60.0** — PDF export (needs native pango libraries)
- **mypy >= 1.8, ruff >= 0.2, pytest >= 8.0** — dev tooling

### Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ./shared && pip install -e ".[dev]"

# Run
streamlit run app.py          # Web UI
python -m src.main            # CLI

# Quality (all must pass before commit)
ruff check . --fix            # Auto-fix lint issues
ruff check .                  # Verify zero errors
mypy src/ shared/             # Type check (strict mode)
pytest tests/                 # Run tests
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Claude API key |
| `DOXA_LOG_LEVEL` | `INFO` | Logging level |
| `DOXA_HISTORY_PERIOD` | `1y` | yfinance price history window |
| `DOXA_MAX_HEADLINES` | `10` | Max headlines for sentiment analysis |

Never commit `.env` files. Use `.env.example` as template. `.env` must live in project root (python-dotenv loads from there only).

## Critical Rules

**Agent pattern:**
- Single public method per agent that takes and returns `ResearchState`
- Agents are stateless — all state lives in the ResearchState dict
- Never create a new state dict — modify and return the one passed in
- All state fields initialized in `create_initial_state()` — don't add new fields ad-hoc
- Business logic belongs in `shared/utils/`; agents are thin orchestration wrappers

**Error handling:**
- On data fetch/API failures: append to `state['errors']`, log `logger.warning()`, continue
- NEVER raise exceptions on data failures — pipeline must complete with partial data
- Only raise for programmer errors (ValueError for invalid input)
- Wrap ALL external API calls (yfinance, anthropic) in try/except

**Python conventions:**
- `from __future__ import annotations` at top of every module
- Absolute imports only: `from src.module import X` (never relative)
- Type hints required on all function signatures — mypy strict, no `Any` without justification
- Google-style docstrings on all public functions/classes
- `logging` module, not `print()` (except `__main__` CLI entry points)
- Private helpers: underscore prefix, under 50 lines, single responsibility
- Ruff rules: E, F, W, I, D, UP — line length 88

**Naming:**
- Variables/functions: `snake_case` | Classes: `PascalCase` | Constants: `UPPER_SNAKE_CASE`
- Finance abbreviations OK: `bs`, `cf`, `inc`, `ta`, `roe`, `pe`, `ebit`
- No single-letter variables (except `i`, `j`, `df`, `e`)

## Working Style

- Minimum code that solves the problem — no speculative features, abstractions, or configurability
- Touch only what the task requires; match existing style; don't refactor or "improve" adjacent code
- Remove imports/variables your changes made unused; leave pre-existing dead code alone (mention it instead)
- Verify before declaring done: run the quality gates above

## yfinance Gotchas

- API is unstable — data structures change between versions without warning
- Use `fast_info` for current quotes, `info` for fundamentals
- Missing data is common — always check for None
- **DataFrame column names vary by ticker/region** — NEVER assume exact labels
  - Use the `_df_get()` pattern with multiple label attempts (required)
- News format changes — defensive parsing required
- Caches data locally — clear cache in tests or mock at import level

## Testing Rules

- Mock ALL external API calls (yfinance, anthropic) — never real calls in tests
- Mock the `anthropic.Anthropic()` client factory, not individual calls
- Arrange-Act-Assert pattern, one assertion focus per test
- Test error accumulation: verify errors append to `state['errors']`, no exceptions raised
- Test state identity: verify same state object is returned (not a new one)
- Test file naming: `tests/test_<module>/test_<function>_<scenario>.py`
