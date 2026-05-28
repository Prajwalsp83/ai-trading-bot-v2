"""
Fair Value Gap (FVG) detection.

An FVG is a 3-bar imbalance: a price range that the middle bar 'skipped'.

  Bullish FVG: bar[i].low > bar[i-2].high
      gap range = [bar[i-2].high, bar[i].low]
      meaning: price moved up so fast that this band was never traded by candles 1 and 3.

  Bearish FVG: bar[i].high < bar[i-2].low
      gap range = [bar[i].high, bar[i-2].low]

Mitigation: an FVG is "mitigated" (closed) once a future bar wicks into its range.
SMC traders watch FVGs as zones where price often retraces to before continuing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class FVG:
    side: Literal["bull", "bear"]
    top: float          # upper bound of the gap
    bottom: float       # lower bound
    created_idx: int    # bar index where the FVG was completed (the 3rd bar)
    created_ts: pd.Timestamp
    mitigated: bool = False
    mitigated_idx: int | None = None
    mitigated_ts: pd.Timestamp | None = None

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


def find_fvgs(df: pd.DataFrame, max_age_bars: int | None = None) -> list[FVG]:
    """Detect all FVGs in a DataFrame and mark which are mitigated by later bars.

    Args:
        df: OHLCV DataFrame, time-indexed.
        max_age_bars: if set, drop FVGs older than this (otherwise return all).
                      Default None = return all detected.
    """
    if len(df) < 3:
        return []

    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index

    fvgs: list[FVG] = []

    for i in range(2, len(df)):
        # Bullish FVG: high[i-2] < low[i]
        if highs[i - 2] < lows[i]:
            fvgs.append(FVG(
                side="bull",
                top=float(lows[i]),
                bottom=float(highs[i - 2]),
                created_idx=i,
                created_ts=idx[i],
            ))
        # Bearish FVG: low[i-2] > high[i]
        if lows[i - 2] > highs[i]:
            fvgs.append(FVG(
                side="bear",
                top=float(lows[i - 2]),
                bottom=float(highs[i]),
                created_idx=i,
                created_ts=idx[i],
            ))

    # Mitigation pass: walk forward, mark FVGs that got wicked into
    for fvg in fvgs:
        for j in range(fvg.created_idx + 1, len(df)):
            bar_high = highs[j]
            bar_low = lows[j]
            # FVG range overlaps with bar's [low, high]?
            if bar_low <= fvg.top and bar_high >= fvg.bottom:
                fvg.mitigated = True
                fvg.mitigated_idx = j
                fvg.mitigated_ts = idx[j]
                break

    if max_age_bars is not None:
        cutoff = len(df) - max_age_bars
        fvgs = [f for f in fvgs if f.created_idx >= cutoff]

    return fvgs


def open_fvgs(fvgs: list[FVG], side: Literal["bull", "bear"] | None = None) -> list[FVG]:
    """Filter to only unmitigated FVGs, optionally one side."""
    out = [f for f in fvgs if not f.mitigated]
    if side is not None:
        out = [f for f in out if f.side == side]
    return out


def fvg_at_price(fvgs: list[FVG], price: float, side: Literal["bull", "bear"] | None = None) -> FVG | None:
    """Return the first open FVG that the given price sits inside (most recent first)."""
    candidates = open_fvgs(fvgs, side=side)
    # Most recent first
    candidates.sort(key=lambda f: f.created_idx, reverse=True)
    for f in candidates:
        if f.contains(price):
            return f
    return None
