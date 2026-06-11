"""Mechanical price-series strategies: SMA crossover, RSI, MACD momentum.

The three sub-signals are averaged into one technical Signal per symbol.
"""

from __future__ import annotations

import pandas as pd

from .base import Signal, Strategy


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal_line = line.ewm(span=9, adjust=False).mean()
    return line, signal_line


class TechnicalStrategy(Strategy):
    name = "technical"

    def evaluate(self, symbol: str, history: pd.DataFrame) -> Signal | None:
        if len(history) < 60:
            return None
        close = history["Close"]
        notes = []

        # 1) SMA 20/50 crossover with distance-based strength
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        sma_score = max(-1.0, min(1.0, (sma20 - sma50) / sma50 * 10))
        notes.append(f"SMA20 {'above' if sma_score > 0 else 'below'} SMA50 ({sma_score:+.2f})")

        # 2) RSI mean reversion: oversold -> buy, overbought -> sell
        r = float(rsi(close).iloc[-1])
        if r < 30:
            rsi_score = (30 - r) / 30
        elif r > 70:
            rsi_score = -(r - 70) / 30
        else:
            rsi_score = 0.0
        notes.append(f"RSI={r:.0f} ({rsi_score:+.2f})")

        # 3) MACD momentum: histogram normalized by price
        line, sig = macd(close)
        hist = float(line.iloc[-1] - sig.iloc[-1])
        macd_score = max(-1.0, min(1.0, hist / float(close.iloc[-1]) * 100))
        notes.append(f"MACD hist={hist:+.2f} ({macd_score:+.2f})")

        score = (sma_score + rsi_score + macd_score) / 3
        # Agreement between indicators raises confidence
        signs = [s for s in (sma_score, rsi_score, macd_score) if abs(s) > 0.05]
        agreement = (
            sum(1 for s in signs if (s > 0) == (score > 0)) / len(signs) if signs else 0.0
        )
        confidence = 0.3 + 0.6 * agreement

        return Signal(symbol, self.name, score, confidence, "; ".join(notes))
