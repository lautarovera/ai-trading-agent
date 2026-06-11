# AI Trading Agent

An AI-driven **paper trading** agent for company stocks and ETFs that combines three signal sources, executes simulated trades, and exposes a watch-only dashboard.

> ⚠️ **Paper trading only.** No real money is involved anywhere in this codebase. The broker is a SQLite-backed simulation; switching to real money would be an explicit, separate development phase.

## Architecture

```
yfinance market data ──► Strategies ──► Ensemble ──► Risk manager ──► Paper broker (SQLite)
                          │                                                │
                          ├─ Technical (SMA cross, RSI, MACD)              ▼
                          ├─ ML (gradient boosting, 5-day return)    Streamlit dashboard
                          └─ LLM analyst (Claude, optional)          (watch-only)
```

- **Technical strategy** — mechanical price-series analysis: SMA 20/50 crossover, RSI mean reversion, MACD momentum, averaged with agreement-based confidence.
- **ML strategy** — per-symbol `GradientBoostingRegressor` predicting the 5-day forward return from lagged returns, volatility, RSI/MACD and volume features. Confidence comes from walk-forward validation accuracy.
- **LLM analyst** — Claude (`claude-haiku-4-5`) receives a compact price-action summary and returns a structured stance (score, confidence, rationale). Views are cached per symbol per trading day to minimize token spend. Enabled only when `ANTHROPIC_API_KEY` is set; the agent works fine without it.
- **Ensemble** — confidence- and weight-scaled average of all signals; BUY/SELL thresholds in `config.yaml`.
- **Risk manager** — max 15% of equity per symbol, 5% cash reserve, 8% stop-loss, 25% take-profit trim, max trades per cycle.
- **Paper broker** — wallet, positions, trades, equity history and signal log persisted in `data/paper_broker.db`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies are declared in `pyproject.toml`:

```sh
uv sync
export ANTHROPIC_API_KEY=...   # optional, enables the LLM analyst
```

## Usage

```sh
# One evaluation cycle (fetch data, generate signals, paper-trade)
uv run python run_agent.py --once

# Continuous loop (every 30 min by default, see config.yaml)
uv run python run_agent.py

# Watch-only dashboard (wallet, holdings, equity curve, trades, signals)
uv run streamlit run dashboard.py

# Historical validation of the technical+ML ensemble vs SPY buy-and-hold
uv run python backtest.py --days 250
```

## Configuration

Everything lives in [config.yaml](config.yaml): universe of symbols, starting cash, strategy weights, ensemble thresholds, risk limits, and the loop interval.

## Roadmap

- [x] v0: watch-only dashboard, automated paper trading
- [ ] User intervention (manual orders, strategy toggles) from the dashboard
- [ ] Richer fundamentals/news inputs for the LLM analyst
- [ ] Live paper-broker integration (e.g. Alpaca paper API)
