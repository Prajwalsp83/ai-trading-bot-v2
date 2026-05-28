"""
Trend indicators. Pure functions, no state.
"""
from __future__ import annotations

import pandas as pd


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average via pandas .ewm."""
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def ema_slope(ema_series: pd.Series, lookback: int = 5) -> pd.Series:
    """Slope of EMA over `lookback` bars (per-bar rise).
    Positive = rising trend, negative = falling. Used for early-warning."""
    return (ema_series - ema_series.shift(lookback)) / lookback


def stack_aligned_long(ema_fast: pd.Series, ema_slow: pd.Series) -> pd.Series:
    """True where fast > slow (bullish stack)."""
    return ema_fast > ema_slow


def stack_aligned_short(ema_fast: pd.Series, ema_slow: pd.Series) -> pd.Series:
    """True where fast < slow (bearish stack)."""
    return ema_fast < ema_slow
