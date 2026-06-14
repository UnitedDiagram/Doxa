# Contributing to Doxa

Thanks for your interest in contributing!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ./shared
pip install -e ".[dev]"
cp .env.example .env   # add your own ANTHROPIC_API_KEY
```

## Quality gates

All three must pass before a PR is merged (CI enforces them):

```bash
ruff check .          # lint (rules: E, F, W, I, D, UP — line length 88)
mypy src/ shared/     # strict type checking
pytest tests/         # test suite
```

## Code conventions

- `from __future__ import annotations` at the top of every module
- Absolute imports only (`from src.module import X`, never relative)
- Type hints on all function signatures — mypy strict, no `Any` without
  justification
- Google-style docstrings on public functions and classes
- Use `logging`, not `print()` (except `__main__` CLI entry points)
- Private helpers: underscore prefix, under 50 lines, single responsibility

## Agent pattern

Each agent exposes a single public method that takes and returns
`ResearchState`:

- Agents are stateless — all state lives in the `ResearchState` dict
- Never create a new state dict — modify and return the one passed in
- All state fields are initialized in `create_initial_state()` — don't add
  fields ad-hoc
- Business logic belongs in `shared/src/doxa_shared/utils/`; agents are
  thin orchestration wrappers

## Error handling

- On data-fetch/API failures: append to `state["errors"]`, log a warning,
  and continue — the pipeline must complete with partial data
- Never raise exceptions on data failures; only raise for programmer errors
  (e.g. `ValueError` for invalid input)
- Wrap all external API calls (yfinance, anthropic, httpx) in try/except

## Testing rules

- Mock **all** external API calls — no real network calls in tests
- Mock the `anthropic.Anthropic()` client factory, not individual calls
- Arrange-Act-Assert, one assertion focus per test
- Test error accumulation: errors append to `state["errors"]`, no exceptions
- Test state identity: the same state object is returned, not a new one

## Pull requests

1. Fork and create a feature branch
2. Make your change, keeping diffs surgical — don't reformat or refactor
   unrelated code
3. Add or update tests
4. Ensure the quality gates pass
5. Open a PR with a clear description of the problem and the change
