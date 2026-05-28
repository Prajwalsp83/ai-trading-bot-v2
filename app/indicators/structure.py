"""
Market structure: swing highs/lows, BOS (Break of Structure), CHoCH (Change of Character).

Same module is used on any timeframe (1H for HTF bias, 15m for LTF confirmation).

Definitions (these matter — different choices give different trades):

  Swing high at index i: high[i] > high[i-pivot..i-1] AND high[i] > high[i+1..i+pivot]
    Confirmed `pivot` bars after the swing forms.
    Default pivot=2 (2 bars on each side). Stricter pivot=3 catches only larger swings.

  BOS (bullish): close above the most recent unbroken swing high, while last
                 confirmed structure direction was UP (HH-HL pattern).
                 -> trend continuation upward.
  BOS (bearish): mirror.

  CHoCH (bullish): close above the most recent unbroken swing high, while last
                   confirmed structure direction was DOWN (LH-LL).
                   -> structural shift from down to up.
  CHoCH (bearish): mirror.

Stateless: each call takes a DataFrame and returns the current state.
This keeps live and backtest behaviour identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


Direction = Literal["up", "down", "none"]


@dataclass
class Swing:
    idx: int            # row index in the source DataFrame
    ts: pd.Timestamp
    price: float
    kind: Literal["high", "low"]


@dataclass
class StructureEvent:
    idx: int
    ts: pd.Timestamp
    kind: Literal["BOS", "CHoCH"]
    side: Direction     # "up" or "down"
    broken_swing_price: float
    close: float


@dataclass
class StructureSnapshot:
    """Result of analyse_structure() — everything a strategy needs to decide bias."""
    swings: list[Swing]
    events: list[StructureEvent]
    current_bias: Direction           # last confirmed BOS/CHoCH direction
    last_swing_high: Swing | None
    last_swing_low: Swing | None
    dealing_range_high: float | None  # for premium/discount
    dealing_range_low: float | None


def find_swings(df: pd.DataFrame, pivot: int = 2) -> list[Swing]:
    """Fractal-based swing detection.
    Returns all swings in chronological order. The most recent `pivot` bars
    cannot have a confirmed swing yet (need future bars to confirm).
    """
    swings: list[Swing] = []
    if len(df) < 2 * pivot + 1:
        return swings

    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index

    for i in range(pivot, len(df) - pivot):
        window_h = highs[i - pivot:i + pivot + 1]
        window_l = lows[i - pivot:i + pivot + 1]
        if highs[i] == window_h.max() and (window_h.argmax() == pivot):
            swings.append(Swing(idx=i, ts=idx[i], price=float(highs[i]), kind="high"))
        if lows[i] == window_l.min() and (window_l.argmin() == pivot):
            swings.append(Swing(idx=i, ts=idx[i], price=float(lows[i]), kind="low"))

    swings.sort(key=lambda s: s.idx)
    return swings


def find_structure_events(df: pd.DataFrame, swings: list[Swing]) -> list[StructureEvent]:
    """Walk forward through bars, finding BOS and CHoCH events.

    Logic:
      Maintain current_bias starting at "none".
      For each bar after a swing forms:
        - If close > unbroken_swing_high:
            event = BOS  if current_bias == "up"
            event = CHoCH if current_bias == "down" or "none"
            current_bias becomes "up"
            the broken swing high is consumed (no longer "unbroken")
        - mirror for swing lows
    """
    events: list[StructureEvent] = []
    if not swings:
        return events

    closes = df["Close"].values
    idx = df.index

    bias: Direction = "none"
    unbroken_highs: list[Swing] = []
    unbroken_lows: list[Swing] = []
    next_swing = 0

    for i in range(len(df)):
        # Add swings confirmed by this bar
        while next_swing < len(swings) and swings[next_swing].idx <= i:
            s = swings[next_swing]
            if s.kind == "high":
                unbroken_highs.append(s)
            else:
                unbroken_lows.append(s)
            next_swing += 1

        c = closes[i]

        # Bullish: close above most recent unbroken swing high (before this bar)
        broken_h = None
        for sh in reversed(unbroken_highs):
            if sh.idx >= i:
                continue
            if c > sh.price:
                broken_h = sh
                break
        if broken_h is not None:
            kind = "BOS" if bias == "up" else "CHoCH"
            events.append(StructureEvent(
                idx=i, ts=idx[i], kind=kind, side="up",
                broken_swing_price=broken_h.price, close=float(c),
            ))
            bias = "up"
            unbroken_highs = [sh for sh in unbroken_highs if sh.idx > broken_h.idx]

        # Bearish
        broken_l = None
        for sl in reversed(unbroken_lows):
            if sl.idx >= i:
                continue
            if c < sl.price:
                broken_l = sl
                break
        if broken_l is not None:
            kind = "BOS" if bias == "down" else "CHoCH"
            events.append(StructureEvent(
                idx=i, ts=idx[i], kind=kind, side="down",
                broken_swing_price=broken_l.price, close=float(c),
            ))
            bias = "down"
            unbroken_lows = [sl for sl in unbroken_lows if sl.idx > broken_l.idx]

    return events


def analyse_structure(df: pd.DataFrame, pivot: int = 2) -> StructureSnapshot:
    """Top-level call: run swing + event detection and summarise current state."""
    swings = find_swings(df, pivot=pivot)
    events = find_structure_events(df, swings)

    bias: Direction = events[-1].side if events else "none"

    last_swing_high = next((s for s in reversed(swings) if s.kind == "high"), None)
    last_swing_low = next((s for s in reversed(swings) if s.kind == "low"), None)

    LOOKBACK = 8
    recent = swings[-LOOKBACK:]
    if recent:
        dr_high = max((s.price for s in recent if s.kind == "high"), default=None)
        dr_low = min((s.price for s in recent if s.kind == "low"), default=None)
    else:
        dr_high = dr_low = None

    return StructureSnapshot(
        swings=swings,
        events=events,
        current_bias=bias,
        last_swing_high=last_swing_high,
        last_swing_low=last_swing_low,
        dealing_range_high=dr_high,
        dealing_range_low=dr_low,
    )


def equilibrium(snap: StructureSnapshot) -> float | None:
    """50% mid of dealing range. Above = premium (good for shorts), below = discount (good for longs)."""
    if snap.dealing_range_high is None or snap.dealing_range_low is None:
        return None
    return (snap.dealing_range_high + snap.dealing_range_low) / 2.0


def is_discount(price: float, snap: StructureSnapshot) -> bool:
    eq = equilibrium(snap)
    return eq is not None and price <= eq


def is_premium(price: float, snap: StructureSnapshot) -> bool:
    eq = equilibrium(snap)
    return eq is not None and price >= eq
