"""
Order Block (OB) detection.

Concept: institutions accumulate/distribute orders at a specific candle before a
strong, impulsive move. That candle's high/low becomes a zone — when price
retraces to it later, those orders often 'pay attention'.

Working definition used here:
  Bullish OB = the LAST bearish (down) candle before a 3+-bar impulsive UP move
                that culminates in a BOS to the upside.
                Zone = [OB.low, OB.high].
  Bearish OB = mirror.

  Mitigated = price has since wicked into the OB's zone.

Detection requires we know where BOS events happened (-> structure module).

This is one of many possible OB definitions. The choice trades off recall (more
zones, more trades) vs precision (fewer but more reliable zones).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ..indicators.structure import StructureEvent


@dataclass
class OrderBlock:
    side: Literal["bull", "bear"]
    top: float
    bottom: float
    created_idx: int           # the OB candle's index
    created_ts: pd.Timestamp
    impulse_idx: int           # index of the BOS that confirmed this OB
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


def find_order_blocks(df: pd.DataFrame, events: list[StructureEvent],
                       min_impulse_bars: int = 3) -> list[OrderBlock]:
    """Find OBs by walking back from each BOS to the last opposite-direction candle.

    Args:
        df: OHLCV time-indexed DataFrame.
        events: structure events from app.indicators.structure.find_structure_events()
        min_impulse_bars: how many bars of one-sided movement is "impulsive". Higher
                          = stricter (only big moves create OBs).
    """
    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index

    obs: list[OrderBlock] = []

    for ev in events:
        # We're interested only in BOS events (continuation), not CHoCH? Both work — keep both.
        bos_idx = ev.idx
        if ev.side == "up":
            # walk back from BOS, find last bearish candle (close < open) before the impulse
            # also verify the impulse has at least `min_impulse_bars` of bullish bars after the OB
            ob_idx = None
            for j in range(bos_idx - 1, max(-1, bos_idx - 30), -1):
                if closes[j] < opens[j]:  # bearish
                    ob_idx = j
                    break
            if ob_idx is None:
                continue
            # require N bullish bars between ob_idx and bos_idx
            bullish_bars = sum(1 for k in range(ob_idx + 1, bos_idx + 1) if closes[k] > opens[k])
            if bullish_bars < min_impulse_bars:
                continue
            obs.append(OrderBlock(
                side="bull",
                top=float(highs[ob_idx]),
                bottom=float(lows[ob_idx]),
                created_idx=ob_idx,
                created_ts=idx[ob_idx],
                impulse_idx=bos_idx,
            ))
        elif ev.side == "down":
            ob_idx = None
            for j in range(bos_idx - 1, max(-1, bos_idx - 30), -1):
                if closes[j] > opens[j]:  # bullish
                    ob_idx = j
                    break
            if ob_idx is None:
                continue
            bearish_bars = sum(1 for k in range(ob_idx + 1, bos_idx + 1) if closes[k] < opens[k])
            if bearish_bars < min_impulse_bars:
                continue
            obs.append(OrderBlock(
                side="bear",
                top=float(highs[ob_idx]),
                bottom=float(lows[ob_idx]),
                created_idx=ob_idx,
                created_ts=idx[ob_idx],
                impulse_idx=bos_idx,
            ))

    # Mitigation pass
    for ob in obs:
        for j in range(ob.impulse_idx + 1, len(df)):
            if lows[j] <= ob.top and highs[j] >= ob.bottom:
                ob.mitigated = True
                ob.mitigated_idx = j
                ob.mitigated_ts = idx[j]
                break

    return obs


def open_order_blocks(obs: list[OrderBlock], side: Literal["bull", "bear"] | None = None) -> list[OrderBlock]:
    out = [o for o in obs if not o.mitigated]
    if side is not None:
        out = [o for o in out if o.side == side]
    return out
