"""Market data fetching with a small on-disk cache (yfinance backend)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import resolve_path

log = logging.getLogger(__name__)


class MarketData:
    def __init__(self, cache_dir: str = "data/cache", cache_ttl_minutes: int = 30,
                 period: str = "2y", interval: str = "1d"):
        self.cache_dir = resolve_path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl_minutes * 60
        self.period = period
        self.interval = interval

    def _cache_file(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}_{self.period}_{self.interval}.parquet"

    def history(self, symbol: str) -> pd.DataFrame:
        """Return OHLCV daily history for a symbol (cached)."""
        cache = self._cache_file(symbol)
        if cache.exists() and (time.time() - cache.stat().st_mtime) < self.cache_ttl:
            return pd.read_parquet(cache)

        df = yf.Ticker(symbol).history(period=self.period, interval=self.interval,
                                       auto_adjust=True)
        if df.empty:
            if cache.exists():
                log.warning("Fetch failed for %s; using stale cache", symbol)
                return pd.read_parquet(cache)
            raise RuntimeError(f"No market data returned for {symbol}")

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.to_parquet(cache)
        return df

    def last_price(self, symbol: str) -> float:
        return float(self.history(symbol)["Close"].iloc[-1])

    def prices(self, symbols: list[str]) -> dict[str, float]:
        out = {}
        for s in symbols:
            try:
                out[s] = self.last_price(s)
            except Exception as e:
                log.warning("Could not get price for %s: %s", s, e)
        return out
