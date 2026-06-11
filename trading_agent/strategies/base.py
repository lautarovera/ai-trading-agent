"""Strategy interface and signal model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    """A directional opinion on one symbol.

    score: -1.0 (strong sell) .. +1.0 (strong buy)
    confidence: 0.0 .. 1.0, scales the signal's weight in the ensemble
    """
    symbol: str
    strategy: str
    score: float
    confidence: float
    rationale: str

    def __post_init__(self):
        self.score = max(-1.0, min(1.0, float(self.score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def evaluate(self, symbol: str, history: pd.DataFrame) -> Signal | None:
        """Return a Signal for the symbol, or None if no opinion."""
