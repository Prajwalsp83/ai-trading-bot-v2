"""
Performance metrics. All work on the trades DataFrame produced by the engine.

Skeleton — no implementation yet.
"""
import pandas as pd


def total_return(trades: pd.DataFrame, starting_equity: float) -> float:
    raise NotImplementedError


def sharpe_ratio(equity_curve: pd.Series, periods_per_year: int = 252 * 26) -> float:
    """TODO: 26 ~ 6.5h * 4 (15m bars). Adjust per timeframe."""
    raise NotImplementedError


def sortino_ratio(equity_curve: pd.Series, periods_per_year: int = 252 * 26) -> float:
    raise NotImplementedError


def max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """TODO: return (max_dd_pct, duration_bars)."""
    raise NotImplementedError


def expectancy(trades: pd.DataFrame) -> float:
    """TODO: avg(R) over all trades."""
    raise NotImplementedError


def profit_factor(trades: pd.DataFrame) -> float:
    raise NotImplementedError


def hit_rate(trades: pd.DataFrame) -> float:
    raise NotImplementedError


def summary(trades: pd.DataFrame, equity_curve: pd.Series, starting_equity: float) -> dict:
    """TODO: bundle everything for the dashboard."""
    raise NotImplementedError
