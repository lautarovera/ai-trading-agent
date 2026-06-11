"""Risk management: position sizing, stop-loss / take-profit, exposure caps."""

from __future__ import annotations

import logging

from .broker import PaperBroker

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, max_position_pct: float = 0.15, cash_reserve_pct: float = 0.05,
                 stop_loss_pct: float = 0.08, take_profit_pct: float = 0.25,
                 max_trades_per_cycle: int = 5):
        self.max_position_pct = max_position_pct
        self.cash_reserve_pct = cash_reserve_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_trades_per_cycle = max_trades_per_cycle

    def buy_qty(self, broker: PaperBroker, symbol: str, price: float,
                score: float, prices: dict[str, float]) -> float:
        """Shares to buy: score-scaled target position, capped by limits."""
        equity = broker.equity(prices)
        pos = broker.positions().get(symbol)
        current_value = pos["qty"] * price if pos else 0.0

        target_value = equity * self.max_position_pct * min(1.0, abs(score))
        room = target_value - current_value
        spendable = broker.cash - equity * self.cash_reserve_pct
        budget = min(room, spendable)
        if budget < price:  # not enough for one whole share
            return 0.0
        return float(int(budget / price))

    def forced_exits(self, broker: PaperBroker,
                     prices: dict[str, float]) -> list[tuple[str, float, str]]:
        """Stop-loss / take-profit exits as (symbol, qty, reason)."""
        exits = []
        for sym, pos in broker.positions().items():
            price = prices.get(sym)
            if not price:
                continue
            change = price / pos["avg_price"] - 1
            if change <= -self.stop_loss_pct:
                exits.append((sym, pos["qty"],
                              f"Stop-loss: {change:.1%} from avg cost {pos['avg_price']:.2f}"))
            elif change >= self.take_profit_pct:
                # Trim half on take-profit, let the rest run
                exits.append((sym, pos["qty"] / 2,
                              f"Take-profit trim: {change:+.1%} from avg cost"))
        return exits
