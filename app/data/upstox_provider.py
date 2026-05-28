"""
Upstox v2 market data: historical candles + websocket live feed.

MCX gold instruments (continuous front-month, rolled monthly):
  - GOLDM (mini, 100g)  -> instrument_key like "MCX_FO|GOLDM25MAYFUT"
  - GOLD  (full, 1kg)
  - GOLDPETAL (1g)

Skeleton — no implementation yet.
"""
from typing import Callable

import pandas as pd

from .base import MarketDataProvider


class UpstoxProvider(MarketDataProvider):
    def __init__(self, broker) -> None:
        """Reuses the authenticated UpstoxBroker for API access.
        Keeps OAuth + token refresh in one place."""
        self.broker = broker

    def get_ohlcv(self, symbol: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        """TODO:
          - resolve current front-month contract for `symbol` (e.g. GOLDM -> GOLDM25MAYFUT)
          - map timeframe ("15m"->"15minute", "1h"->"30minute"x2 or "60minute")
            Note: Upstox doesn't expose 60m directly — aggregate from 30m or 15m.
          - GET /v2/historical-candle/{instrument_key}/{interval}/{to}/{from}
          - return DataFrame Open/High/Low/Close/Volume indexed UTC.
        """
        raise NotImplementedError

    def subscribe_live(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[pd.DataFrame], None],
    ) -> None:
        """TODO:
          - websocket: wss://api.upstox.com/v2/feed/market-data-feed
          - subscribe to {instrument_key} for full mode (LTP + OHLC tick)
          - aggregate ticks into candles at requested timeframe
          - on bar close: callback(latest_bars_df)
        """
        raise NotImplementedError

    def is_market_open(self, symbol: str) -> bool:
        """TODO: MCX gold hours: 09:00-23:30 IST Mon-Fri, with afternoon
        break. Check via GET /v2/market/timings or cached schedule."""
        raise NotImplementedError

    def front_month_contract(self, base_symbol: str) -> str:
        """TODO: GOLDM -> current front-month expiry instrument_key.
        Auto-roll logic: switch to next month 3 trading days before expiry."""
        raise NotImplementedError
