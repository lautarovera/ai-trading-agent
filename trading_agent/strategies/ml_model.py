"""ML prediction strategy: gradient boosting on engineered price features.

Per symbol, trains a GradientBoostingRegressor to predict the next 5-day
return from lagged returns, volatility, RSI, MACD and volume features.
Models are trained once per process and reused across cycles.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .base import Signal, Strategy
from .technical import macd, rsi

log = logging.getLogger(__name__)

HORIZON = 5  # predict 5-trading-day forward return
SCORE_SCALE = 0.04  # a predicted +4% move over the horizon maps to score 1.0


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    close, volume = history["Close"], history["Volume"]
    ret = close.pct_change()
    feats = pd.DataFrame(index=history.index)
    for lag in (1, 2, 3, 5, 10):
        feats[f"ret_{lag}d"] = close.pct_change(lag)
    feats["vol_10d"] = ret.rolling(10).std()
    feats["vol_30d"] = ret.rolling(30).std()
    feats["rsi"] = rsi(close) / 100
    line, sig = macd(close)
    feats["macd_hist"] = (line - sig) / close
    feats["sma20_dist"] = close / close.rolling(20).mean() - 1
    feats["sma50_dist"] = close / close.rolling(50).mean() - 1
    feats["volume_ratio"] = volume / volume.rolling(20).mean()
    return feats


class MLStrategy(Strategy):
    name = "ml"

    def __init__(self, min_train_rows: int = 250):
        self.min_train_rows = min_train_rows
        self._models: dict[str, tuple[GradientBoostingRegressor, float]] = {}

    def _train(self, symbol: str, history: pd.DataFrame):
        feats = build_features(history)
        target = history["Close"].pct_change(HORIZON).shift(-HORIZON)
        data = feats.assign(target=target).dropna()
        if len(data) < self.min_train_rows:
            return None

        X, y = data.drop(columns="target"), data["target"]
        # Walk-forward split: last 20% as validation to estimate confidence
        split = int(len(data) * 0.8)
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8,
            random_state=42,
        )
        model.fit(X.iloc[:split], y.iloc[:split])
        pred_val = model.predict(X.iloc[split:])
        # Directional hit rate on validation as the confidence proxy
        hits = np.mean(np.sign(pred_val) == np.sign(y.iloc[split:]))
        # Refit on the full set for live prediction
        model.fit(X, y)
        self._models[symbol] = (model, float(hits))
        log.info("Trained ML model for %s (val directional accuracy %.0f%%)",
                 symbol, hits * 100)
        return self._models[symbol]

    def evaluate(self, symbol: str, history: pd.DataFrame) -> Signal | None:
        entry = self._models.get(symbol) or self._train(symbol, history)
        if entry is None:
            return None
        model, hit_rate = entry

        feats = build_features(history).dropna()
        if feats.empty:
            return None
        pred = float(model.predict(feats.iloc[[-1]])[0])
        score = max(-1.0, min(1.0, pred / SCORE_SCALE))
        # Hit rate near 0.5 is noise; scale confidence by edge over coin flip
        confidence = max(0.0, min(1.0, (hit_rate - 0.5) * 4))
        rationale = (f"GBM predicts {pred:+.2%} over {HORIZON}d "
                     f"(val accuracy {hit_rate:.0%})")
        return Signal(symbol, self.name, score, confidence, rationale)
