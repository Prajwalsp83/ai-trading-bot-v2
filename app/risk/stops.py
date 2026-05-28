"""
SL / TP / trailing rules. All ATR-based.

  initial SL  = entry -/+ k_sl * ATR
  initial TP  = entry +/- k_tp * ATR        (k_tp >= k_sl gives RR >= 1)
  breakeven   = once price moves 1R in our favour
  trail       = after 2R, trail by k_trail * ATR
  time exit   = close if open > max_bars and price within +/- 0.3R of entry

Skeleton — no implementation yet.
"""
from dataclasses import dataclass

from ..core.events import Position


@dataclass
class StopParams:
    k_sl: float = 1.5
    k_tp: float = 3.0
    k_trail: float = 1.0
    breakeven_after_r: float = 1.0
    trail_after_r: float = 2.0
    max_bars: int = 96   # ~1 day on 15m


def initial_sl_tp(side: str, entry: float, atr: float, p: StopParams) -> tuple[float, float]:
    """TODO: return (sl, tp) for BUY or SELL."""
    raise NotImplementedError


def update_trailing(position: Position, last_price: float, atr: float, p: StopParams) -> float | None:
    """Return new SL price, or None if no change.

    TODO:
      - compute current R = (price - entry) / |entry - sl|
      - if R >= breakeven_after_r and SL still beyond entry: move to entry
      - if R >= trail_after_r: trail by k_trail * ATR
    """
    raise NotImplementedError


def time_stop_triggered(position: Position, bars_open: int, last_price: float, p: StopParams) -> bool:
    """TODO: bars_open >= max_bars and price near entry."""
    raise NotImplementedError
