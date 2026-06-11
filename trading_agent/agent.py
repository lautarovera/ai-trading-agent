"""Orchestrator: fetch data -> collect signals -> ensemble -> risk -> paper trade."""

from __future__ import annotations

import logging

from .broker import PaperBroker
from .config import load_config
from .data import MarketData
from .ensemble import Decision, Ensemble
from .risk import RiskManager
from .strategies import LLMAnalystStrategy, MLStrategy, TechnicalStrategy

log = logging.getLogger(__name__)


class TradingAgent:
    def __init__(self, config_path: str | None = None):
        cfg = load_config(config_path)
        self.cfg = cfg
        self.symbols: list[str] = cfg["universe"]["symbols"]

        d = cfg["data"]
        self.data = MarketData(cache_dir=d["cache_dir"],
                               cache_ttl_minutes=d["cache_ttl_minutes"],
                               period=d["history_period"], interval=d["interval"])

        p = cfg["portfolio"]
        self.broker = PaperBroker(db_path=p["db_path"], starting_cash=p["starting_cash"])

        s = cfg["strategies"]
        self.strategies = []
        if s["technical"]["enabled"]:
            self.strategies.append(TechnicalStrategy())
        if s["ml"]["enabled"]:
            self.strategies.append(MLStrategy(min_train_rows=s["ml"]["min_train_rows"]))
        if s["llm"]["enabled"]:
            llm = LLMAnalystStrategy(model=s["llm"]["model"])
            if llm.enabled:
                self.strategies.append(llm)

        weights = {name: conf["weight"] for name, conf in s.items()}
        e = cfg["ensemble"]
        self.ensemble = Ensemble(weights, e["buy_threshold"], e["sell_threshold"])
        self.risk = RiskManager(**cfg["risk"])

    def run_cycle(self) -> list[Decision]:
        """One full evaluation cycle. Returns the decisions made."""
        prices = self.data.prices(self.symbols)

        # 1) Risk exits first (stop-loss / take-profit)
        for sym, qty, reason in self.risk.forced_exits(self.broker, prices):
            if self.broker.sell(sym, qty, prices[sym], reason):
                log.info("RISK EXIT %s x%.0f @ %.2f — %s", sym, qty, prices[sym], reason)

        # 2) Evaluate every symbol
        decisions: list[Decision] = []
        for sym in self.symbols:
            if sym not in prices:
                continue
            try:
                history = self.data.history(sym)
            except Exception as e:
                log.warning("Skipping %s: %s", sym, e)
                continue
            signals = []
            for strat in self.strategies:
                sig = strat.evaluate(sym, history)
                if sig is not None:
                    signals.append(sig)
                    self.broker.log_signal(sig)
            if signals:
                decisions.append(self.ensemble.decide(sym, signals))

        # 3) Execute: strongest convictions first, bounded per cycle
        decisions.sort(key=lambda d: abs(d.combined_score), reverse=True)
        trades_done = 0
        for d in decisions:
            if trades_done >= self.risk.max_trades_per_cycle:
                break
            price = prices[d.symbol]
            if d.action == "BUY":
                qty = self.risk.buy_qty(self.broker, d.symbol, price,
                                        d.combined_score, prices)
                if qty > 0 and self.broker.buy(d.symbol, qty, price, d.rationale):
                    log.info("BUY %s x%.0f @ %.2f (score %+.2f)",
                             d.symbol, qty, price, d.combined_score)
                    trades_done += 1
            elif d.action == "SELL":
                pos = self.broker.positions().get(d.symbol)
                if pos and self.broker.sell(d.symbol, pos["qty"], price, d.rationale):
                    log.info("SELL %s x%.0f @ %.2f (score %+.2f)",
                             d.symbol, pos["qty"], price, d.combined_score)
                    trades_done += 1

        equity = self.broker.snapshot_equity(prices)
        log.info("Cycle done: equity=%.2f cash=%.2f positions=%d",
                 equity, self.broker.cash, len(self.broker.positions()))
        return decisions
