"""
Volatility indicators.
"""
from __future__ import annotations

import pandas as pd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — Wilder's smoothing."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's smoothing == EMA with alpha = 1/period
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def atr_percentile(atr_series: pd.Series, lookback: int = 100) -> pd.Series:
    """Rolling percentile rank of ATR (0–1). >0.7 == high-vol regime."""
    return atr_series.rolling(lookback, min_periods=lookback // 2).rank(pct=True)


def expanding_volatility(atr_series: pd.Series, lookback: int = 3) -> pd.Series:
    """True where ATR has risen `lookback` bars in a row — pre-breakout cue."""
    rising = (atr_series.diff() > 0)
    return rising.rolling(lookback).sum() == lookback
