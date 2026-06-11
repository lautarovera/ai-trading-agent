"""Quick historical validation of the signal ensemble (technical + ML).

Replays the last N trading days bar by bar against the same decision logic the
live agent uses (LLM strategy excluded — too slow/costly to replay).

Usage:
    uv run python backtest.py [--days 250] [--symbols AAPL MSFT SPY]
"""

from __future__ import annotations

import argparse
import logging

from trading_agent.config import load_config
from trading_agent.data import MarketData
from trading_agent.ensemble import Ensemble
from trading_agent.strategies import MLStrategy, TechnicalStrategy

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=250)
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config()
    symbols = args.symbols or cfg["universe"]["symbols"]
    data = MarketData(period="5y", interval="1d",
                      cache_dir=cfg["data"]["cache_dir"], cache_ttl_minutes=120)
    s = cfg["strategies"]
    weights = {name: conf["weight"] for name, conf in s.items()}
    ensemble = Ensemble(weights, cfg["ensemble"]["buy_threshold"],
                        cfg["ensemble"]["sell_threshold"])

    cash = cfg["portfolio"]["starting_cash"]
    positions: dict[str, float] = {}
    histories = {}
    for sym in symbols:
        try:
            histories[sym] = data.history(sym)
        except Exception as e:
            print(f"skip {sym}: {e}")
    symbols = list(histories)
    n_bars = min(len(h) for h in histories.values())
    start = n_bars - args.days
    if start < 300:
        raise SystemExit("Not enough history for the requested window")

    # ML models are trained once on data before the backtest window (no lookahead)
    ml = MLStrategy(min_train_rows=s["ml"]["min_train_rows"])
    tech = TechnicalStrategy()
    for sym in symbols:
        ml.evaluate(sym, histories[sym].iloc[:start])

    initial_equity = cash
    for i in range(start, n_bars):
        prices = {sym: float(h["Close"].iloc[i]) for sym, h in histories.items()}
        for sym in symbols:
            window = histories[sym].iloc[: i + 1]
            signals = [sig for strat in (tech, ml)
                       if (sig := strat.evaluate(sym, window)) is not None]
            if not signals:
                continue
            d = ensemble.decide(sym, signals)
            price = prices[sym]
            if d.action == "BUY" and cash > price:
                budget = min(cash, (cash + sum(
                    positions.get(x, 0) * prices[x] for x in symbols)) * 0.15)
                qty = int(budget / price)
                if qty > 0:
                    positions[sym] = positions.get(sym, 0) + qty
                    cash -= qty * price
            elif d.action == "SELL" and positions.get(sym, 0) > 0:
                cash += positions.pop(sym) * price

    final_prices = {sym: float(h["Close"].iloc[n_bars - 1]) for sym, h in histories.items()}
    equity = cash + sum(q * final_prices[sym] for sym, q in positions.items())

    # Buy-and-hold SPY benchmark over the same window
    bench = None
    if "SPY" in histories:
        spy = histories["SPY"]["Close"]
        bench = float(spy.iloc[n_bars - 1] / spy.iloc[start] - 1)

    print(f"\nBacktest over last {args.days} trading days ({len(symbols)} symbols)")
    print(f"  Final equity : {equity:,.2f}  (start {initial_equity:,.2f})")
    print(f"  Total return : {equity / initial_equity - 1:+.2%}")
    if bench is not None:
        print(f"  SPY buy&hold : {bench:+.2%}")


if __name__ == "__main__":
    main()
