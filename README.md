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

## Brokers

Two interchangeable backends, selected via `portfolio.broker` in `config.yaml`:

- `local` (default) — SQLite simulation, instant idealized fills.
- `alpaca` — routes orders to the [Alpaca](https://alpaca.markets) **paper** trading API (realistic fills, market-hours enforcement). Needs free paper keys in `.env`:
  ```
  ALPACA_API_KEY=...
  ALPACA_SECRET_KEY=...
  ```
  The client is hard-pinned to the paper endpoint; live trading is deliberately not wired up. A `max_order_value` guardrail rejects oversized orders.

### Activating Alpaca paper trading

1. Sign up free at <https://app.alpaca.markets> — paper trading needs no funding, no residency checks, no broker approval (works from the EU).
2. Generate **paper** API keys (dashboard → "Paper" toggle → API Keys) and add them to `.env` in the project root:
   ```
   ALPACA_API_KEY=PK...
   ALPACA_SECRET_KEY=...
   ```
3. Set `broker: alpaca` under `portfolio:` in `config.yaml`, then restart the agent:
   ```sh
   systemctl --user restart trading-agent   # or just re-run run_agent.py
   ```

The Alpaca paper account starts with $100k virtual cash, and Alpaca's own web dashboard gives an independent view of positions and fills to cross-check the agent.

## Roadmap

- [x] v0: watch-only dashboard, automated paper trading
- [x] Alpaca paper-broker integration
- [ ] User intervention (manual orders, strategy toggles) from the dashboard
- [ ] Richer fundamentals/news inputs for the LLM analyst
- [ ] Live trading with guardrails (explicit opt-in, small budget)
