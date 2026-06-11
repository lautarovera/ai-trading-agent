"""Alpaca broker adapter — PAPER endpoint only.

Routes orders to Alpaca's paper trading API while reusing the local SQLite
store (inherited from PaperBroker) for the trade log, equity history and
signal log that the dashboard reads.

Live trading is intentionally not wired up: the client is hard-pinned to
paper=True. Going live is a deliberate, separate change.

Requires ALPACA_API_KEY and ALPACA_SECRET_KEY (paper keys from
https://app.alpaca.markets, free account).
"""

from __future__ import annotations

import logging
import os

from .broker import PaperBroker

log = logging.getLogger(__name__)


class AlpacaPaperBroker(PaperBroker):
    def __init__(self, db_path: str = "data/alpaca_log.db", starting_cash: float = 0.0,
                 max_order_value: float = 20_000.0):
        # Local DB is used only for logging (trades/equity/signals); cash and
        # positions live on Alpaca's side.
        super().__init__(db_path=db_path, starting_cash=starting_cash)
        self.max_order_value = max_order_value

        from alpaca.trading.client import TradingClient

        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Create free paper "
                "keys at https://app.alpaca.markets and add them to .env"
            )
        self.client = TradingClient(key, secret, paper=True)
        acct = self.client.get_account()
        log.info("Connected to Alpaca PAPER account %s (equity %s)",
                 acct.account_number, acct.equity)

    # -- state comes from Alpaca ---------------------------------------------
    @property
    def cash(self) -> float:
        return float(self.client.get_account().cash)

    def positions(self) -> dict[str, dict]:
        return {
            p.symbol: {"qty": float(p.qty), "avg_price": float(p.avg_entry_price)}
            for p in self.client.get_all_positions()
        }

    def equity(self, prices: dict[str, float] | None = None) -> float:
        return float(self.client.get_account().equity)

    # -- orders go to Alpaca, then get logged locally --------------------------
    def _submit(self, symbol: str, qty: float, side, price: float,
                rationale: str) -> bool:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if qty <= 0:
            return False
        if qty * price > self.max_order_value:
            log.warning("Order rejected by guardrail: %s %s x%.2f exceeds "
                        "max_order_value %.0f", side, symbol, qty, self.max_order_value)
            return False
        try:
            self.client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY))
        except Exception as e:
            log.warning("Alpaca order failed: %s %s x%.2f — %s", side, symbol, qty, e)
            return False
        side_str = "BUY" if side == OrderSide.BUY else "SELL"
        with self.conn:
            self.conn.execute(
                "INSERT INTO trades (ts, symbol, side, qty, price, rationale) "
                "VALUES (datetime('now'), ?, ?, ?, ?, ?)",
                (symbol, side_str, qty, price, rationale))
        return True

    def buy(self, symbol: str, qty: float, price: float, rationale: str = "") -> bool:
        from alpaca.trading.enums import OrderSide
        if qty * price > self.cash:
            return False
        return self._submit(symbol, qty, OrderSide.BUY, price, rationale)

    def sell(self, symbol: str, qty: float, price: float, rationale: str = "") -> bool:
        from alpaca.trading.enums import OrderSide
        pos = self.positions().get(symbol)
        if pos is None:
            return False
        return self._submit(symbol, min(qty, pos["qty"]), OrderSide.SELL,
                            price, rationale)
