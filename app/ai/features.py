"""
Feature engineering for ML layers. Pure functions on OHLCV DataFrames.

Skeleton — no implementation yet.
"""
import pandas as pd


def returns(close: pd.Series, n: int = 1) -> pd.Series:
    """TODO: log returns over n bars."""
    raise NotImplementedError


def realized_vol(close: pd.Series, lookback: int = 20) -> pd.Series:
    """TODO: rolling std of log returns."""
    raise NotImplementedError


def ema_distance_pct(close: pd.Series, ema_series: pd.Series) -> pd.Series:
    """TODO: (close - ema) / ema."""
    raise NotImplementedError


def trend_strength(close: pd.Series, ema_fast: pd.Series, ema_slow: pd.Series) -> pd.Series:
    """TODO: |ema_fast - ema_slow| / ATR — magnitude-normalised stack distance."""
    raise NotImplementedError


def build_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """TODO: assemble all features into one frame for training/inference."""
    raise NotImplementedError
