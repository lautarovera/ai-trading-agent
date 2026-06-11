"""Combine per-strategy signals into one decision score per symbol."""

from __future__ import annotations

from dataclasses import dataclass

from .strategies.base import Signal


@dataclass
class Decision:
    symbol: str
    combined_score: float  # -1 .. +1
    action: str            # BUY | SELL | HOLD
    signals: list[Signal]

    @property
    def rationale(self) -> str:
        return " | ".join(f"[{s.strategy}] {s.rationale}" for s in self.signals)


class Ensemble:
    def __init__(self, weights: dict[str, float],
                 buy_threshold: float = 0.2, sell_threshold: float = -0.2):
        self.weights = weights
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def decide(self, symbol: str, signals: list[Signal]) -> Decision:
        weighted_sum = 0.0
        weight_total = 0.0
        for s in signals:
            w = self.weights.get(s.strategy, 1.0) * s.confidence
            weighted_sum += s.score * w
            weight_total += w
        score = weighted_sum / weight_total if weight_total > 0 else 0.0

        if score >= self.buy_threshold:
            action = "BUY"
        elif score <= self.sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"
        return Decision(symbol, score, action, signals)
