"""
Strategy interface. Live runner and backtest engine both call .evaluate().
This is the single point of truth that fixes the live/backtest drift.

Skeleton — no implementation yet.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from ..core.events import Signal


@dataclass
class MarketState:
    """Everything a strategy needs at one decision point."""
    symbol: str
    bars_entry: pd.DataFrame              # OHLCV at entry timeframe (e.g. 15m)
    bars_trend: pd.DataFrame              # OHLCV at trend timeframe (e.g. 1h)
    bars_higher: pd.DataFrame | None = None   # Optional higher TF (e.g. 4h) — used by 4H trend gate


class Strategy(ABC):
    name: str
    required_indicators: list[str]

    @abstractmethod
    def evaluate(self, state: MarketState) -> Signal | None:
        """Return a Signal (any severity) or None.

        IMPORTANT: this method must be deterministic and free of I/O.
        Same input -> same output. The backtest depends on this.
        """
