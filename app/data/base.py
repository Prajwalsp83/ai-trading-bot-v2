"""
MarketDataProvider interface. All data sources implement this.

Skeleton — no implementation yet.
"""
from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,         # "15m", "1h", ...
        lookback_bars: int,
    ) -> pd.DataFrame:
        """Return DataFrame indexed by UTC timestamp with columns:
        Open, High, Low, Close, Volume."""

    @abstractmethod
    def subscribe_live(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[pd.DataFrame], None],
    ) -> None:
        """Stream new bars to callback. Polling providers may simulate this."""

    @abstractmethod
    def is_market_open(self, symbol: str) -> bool:
        """Used to skip processing during closed sessions."""
