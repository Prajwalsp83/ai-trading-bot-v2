"""
yfinance-backed provider. Polls; not true real-time.
Used for backtest historical fetch and current live mode.

Skeleton — no implementation yet.
"""
from typing import Callable

import pandas as pd

from .base import MarketDataProvider


class YFinanceProvider(MarketDataProvider):
    def __init__(self, poll_seconds: int = 60) -> None:
        self.poll_seconds = poll_seconds
        # TODO: cache last bar per (symbol, timeframe) so we know when a bar closed.

    def get_ohlcv(self, symbol: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        """TODO: map timeframe -> yf interval/period; download; normalise columns."""
        raise NotImplementedError

    def subscribe_live(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[pd.DataFrame], None],
    ) -> None:
        """TODO: spawn a thread that polls every poll_seconds and fires callback
        only when a new bar closes (not on every poll)."""
        raise NotImplementedError

    def is_market_open(self, symbol: str) -> bool:
        """TODO: GC=F is ~24h with daily break; FX has session calendar."""
        raise NotImplementedError
