"""LLM analyst strategy: asks Claude for a directional stance on a symbol.

Requires ANTHROPIC_API_KEY; the strategy silently disables itself otherwise.
The model receives a compact summary of recent price action and returns a
structured stance validated against a Pydantic schema.

Views are cached per (symbol, last bar date): the input summary is built from
daily bars, so re-querying within the same trading day would send identical
input and waste tokens. The cache persists to disk across restarts.
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd
from pydantic import BaseModel

from ..config import resolve_path
from .base import Signal, Strategy

log = logging.getLogger(__name__)


class AnalystView(BaseModel):
    score: float        # -1.0 strong sell .. +1.0 strong buy
    confidence: float   # 0.0 .. 1.0
    rationale: str


SYSTEM_PROMPT = (
    "You are a disciplined quantitative equity analyst evaluating stocks and ETFs "
    "for a paper-trading portfolio with a 1-4 week horizon. You receive a summary "
    "of recent price action and must output a directional stance.\n"
    "- score: -1.0 (strong sell) to +1.0 (strong buy); use 0 when there is no edge.\n"
    "- confidence: 0.0 to 1.0; be conservative — price summaries alone rarely "
    "justify confidence above 0.6.\n"
    "- rationale: one or two sentences.\n"
    "Avoid overtrading: small or ambiguous setups deserve scores near 0."
)


def summarize_history(symbol: str, history: pd.DataFrame) -> str:
    close = history["Close"]
    last = float(close.iloc[-1])

    def chg(days: int) -> str:
        if len(close) <= days:
            return "n/a"
        return f"{(last / float(close.iloc[-days - 1]) - 1):+.1%}"

    vol30 = float(close.pct_change().rolling(30).std().iloc[-1]) * (252 ** 0.5)
    hi52 = float(close.tail(252).max())
    lo52 = float(close.tail(252).min())
    return (
        f"Symbol: {symbol}\n"
        f"Last close: {last:.2f}\n"
        f"Returns: 1w {chg(5)}, 1m {chg(21)}, 3m {chg(63)}, 6m {chg(126)}, 1y {chg(252)}\n"
        f"Annualized 30d volatility: {vol30:.0%}\n"
        f"52-week range: {lo52:.2f} - {hi52:.2f} "
        f"(now {((last - lo52) / (hi52 - lo52) * 100) if hi52 > lo52 else 50:.0f}% of range)"
    )


class LLMAnalystStrategy(Strategy):
    name = "llm"

    def __init__(self, model: str = "claude-haiku-4-5",
                 cache_path: str = "data/cache/llm_views.json"):
        self.model = model
        self._cache_path = resolve_path(cache_path)
        self._cache: dict[str, dict] = {}
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except Exception:
                log.warning("Could not read LLM view cache; starting fresh")
        self._client = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except Exception as e:
                log.warning("Anthropic client init failed, LLM strategy disabled: %s", e)
        else:
            log.info("ANTHROPIC_API_KEY not set; LLM analyst strategy disabled")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _save_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache))
        except Exception as e:
            log.warning("Could not persist LLM view cache: %s", e)

    def evaluate(self, symbol: str, history: pd.DataFrame) -> Signal | None:
        if not self.enabled:
            return None
        # One query per symbol per trading day — same-day inputs are identical
        key = f"{symbol}:{history.index[-1].date().isoformat()}"
        cached = self._cache.get(key)
        if cached is not None:
            return Signal(symbol, self.name, cached["score"], cached["confidence"],
                          cached["rationale"] + " (cached)")
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": summarize_history(symbol, history)}],
                output_format=AnalystView,
            )
            view: AnalystView = response.parsed_output
            self._cache[key] = view.model_dump()
            self._save_cache()
            return Signal(symbol, self.name, view.score, view.confidence, view.rationale)
        except Exception as e:
            log.warning("LLM analyst failed for %s: %s", symbol, e)
            return None
