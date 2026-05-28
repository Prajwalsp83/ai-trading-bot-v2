"""
Breakout trend-following strategy.

Same rules in live and backtest — that's the whole point of the v2 redesign.

BUY READY when ALL of:
  - 15m EMA fast > slow
  - 1H  EMA fast > slow            (higher-TF trend filter)
  - Current bar high > previous bar high   (breakout)
  - ATR >= atr_min                 (volatility filter)

SELL READY = mirror.

Early-warning severities:
  - WATCHLIST       : 1H stack aligned alone (trend is forming on higher TF)
  - BREAKOUT_WATCH  : 3 of 4 conditions true (entry is one bar away)
  - BUY_READY/SELL_READY: all 4 true (execute)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from ..core.events import Severity, Signal
from ..indicators.trend import ema
from ..indicators.volatility import atr, atr_percentile
from .base import MarketState, Strategy


@dataclass
class BreakoutTrendParams:
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    atr_min: float = 5.0

    # Filters added after first backtest showed too many false breakouts.
    # atr_pct_min: skip when current ATR is in the bottom N pct of last 100 bars.
    #              0.0 = off, 0.3 = require top 70% volatility.
    # min_trend_strength: skip when |ema_fast - ema_slow| / ema_slow < threshold.
    #              0.0 = off, 0.005 = require 0.5% EMA separation.
    atr_pct_min: float = 0.0
    min_trend_strength: float = 0.0

    # P4: 4H trend gate. When True and MarketState.bars_higher is provided,
    # entries require 4H EMA fast/slow alignment in the trade direction.
    use_higher_tf_gate: bool = False


class BreakoutTrendStrategy(Strategy):
    name = "breakout_trend"
    required_indicators = ["ema", "atr"]

    def __init__(self, params: BreakoutTrendParams | None = None) -> None:
        self.params = params or BreakoutTrendParams()

    # Public: identical signature for live + backtest.
    def evaluate(self, state: MarketState) -> Signal | None:
        p = self.params
        be = state.bars_entry       # entry timeframe (15m)
        bt = state.bars_trend       # trend timeframe (1H)

        # Need enough history for the slow EMA to be defined.
        if len(be) < p.ema_slow + 2 or len(bt) < p.ema_slow + 2:
            return None

        # ---- indicators on entry TF ----
        ema_f_e = ema(be["Close"], p.ema_fast)
        ema_s_e = ema(be["Close"], p.ema_slow)
        atr_e = atr(be["High"], be["Low"], be["Close"], p.atr_period)

        # ---- indicators on trend TF ----
        ema_f_t = ema(bt["Close"], p.ema_fast)
        ema_s_t = ema(bt["Close"], p.ema_slow)

        # ---- pick the latest *closed* bar (the last row) ----
        last = be.iloc[-1]
        prev = be.iloc[-2]
        current_high = last["High"]
        current_low = last["Low"]
        prev_high = prev["High"]
        prev_low = prev["Low"]
        price = last["Close"]
        atr_val = atr_e.iloc[-1]
        ts = _bar_timestamp(be.index[-1])

        # NaN guard
        if pd.isna(atr_val) or pd.isna(ema_f_e.iloc[-1]) or pd.isna(ema_s_e.iloc[-1]) \
                or pd.isna(ema_f_t.iloc[-1]) or pd.isna(ema_s_t.iloc[-1]):
            return None

        # ---- new filters: skip low-vol / trendless markets ----
        # Why-skipped flag travels in Signal.extras for the daily summary.
        rejection_reason: str | None = None
        if p.atr_pct_min > 0:
            atr_pct = atr_percentile(atr_e, lookback=100).iloc[-1]
            if pd.isna(atr_pct) or atr_pct < p.atr_pct_min:
                rejection_reason = "atr_pct_too_low"
        if rejection_reason is None and p.min_trend_strength > 0:
            trend_strength_e = abs(ema_f_e.iloc[-1] - ema_s_e.iloc[-1]) / ema_s_e.iloc[-1]
            if trend_strength_e < p.min_trend_strength:
                rejection_reason = "trend_strength_too_low"

        # ---- P4: 4H higher-TF gate ----
        h4_up = h4_dn = True
        if p.use_higher_tf_gate and state.bars_higher is not None \
                and len(state.bars_higher) >= p.ema_slow + 2:
            bh = state.bars_higher
            ema_f_h = ema(bh["Close"], p.ema_fast)
            ema_s_h = ema(bh["Close"], p.ema_slow)
            if not (pd.isna(ema_f_h.iloc[-1]) or pd.isna(ema_s_h.iloc[-1])):
                h4_up = ema_f_h.iloc[-1] > ema_s_h.iloc[-1]
                h4_dn = ema_f_h.iloc[-1] < ema_s_h.iloc[-1]

        if rejection_reason is not None:
            return Signal(
                ts=ts, symbol=state.symbol, side="NONE",
                severity=Severity.WATCHLIST,
                price=price, atr=atr_val,
                reason=f"skipped: {rejection_reason}",
                extras={"rejection_reason": rejection_reason, "skipped": True},
            )

        # ---- 4 conditions per side ----
        long_cond = {
            "15m_stack_up": ema_f_e.iloc[-1] > ema_s_e.iloc[-1],
            "1h_stack_up":  ema_f_t.iloc[-1] > ema_s_t.iloc[-1],
            "breakout_up":  current_high > prev_high,
            "atr_ok":       atr_val >= p.atr_min,
            "4h_stack_up":  h4_up,
        }
        short_cond = {
            "15m_stack_dn": ema_f_e.iloc[-1] < ema_s_e.iloc[-1],
            "1h_stack_dn":  ema_f_t.iloc[-1] < ema_s_t.iloc[-1],
            "breakout_dn":  current_low < prev_low,
            "atr_ok":       atr_val >= p.atr_min,
            "4h_stack_dn":  h4_dn,
        }
        long_n = sum(long_cond.values())
        short_n = sum(short_cond.values())
        n_total = len(long_cond)  # 5 with 4H gate row, but always counted equally

        # ---- decide severity ----
        # Strongest signal wins. Long takes priority over short on ties (rare).
        if long_n == n_total:
            return Signal(
                ts=ts, symbol=state.symbol, side="BUY",
                severity=Severity.BUY_READY,
                price=price, atr=atr_val,
                reason=f"All {n_total} long conditions met",
                extras={"conditions": long_cond},
            )
        if short_n == n_total:
            return Signal(
                ts=ts, symbol=state.symbol, side="SELL",
                severity=Severity.SELL_READY,
                price=price, atr=atr_val,
                reason=f"All {n_total} short conditions met",
                extras={"conditions": short_cond},
            )
        if long_n == n_total - 1:
            # Identify which condition failed (for rejection telemetry)
            missing = [k for k, v in long_cond.items() if not v]
            return Signal(
                ts=ts, symbol=state.symbol, side="BUY",
                severity=Severity.BREAKOUT_WATCH,
                price=price, atr=atr_val,
                reason=f"{long_n} of {n_total} long conditions met — entry near",
                extras={"conditions": long_cond, "missing": missing},
            )
        if short_n == n_total - 1:
            missing = [k for k, v in short_cond.items() if not v]
            return Signal(
                ts=ts, symbol=state.symbol, side="SELL",
                severity=Severity.BREAKOUT_WATCH,
                price=price, atr=atr_val,
                reason=f"{short_n} of {n_total} short conditions met — entry near",
                extras={"conditions": short_cond, "missing": missing},
            )
        # Higher-TF trend alone -> WATCHLIST (lowest-priority alert)
        if long_cond["1h_stack_up"] and not short_cond["1h_stack_dn"]:
            return Signal(
                ts=ts, symbol=state.symbol, side="BUY",
                severity=Severity.WATCHLIST,
                price=price, atr=atr_val,
                reason="1H trend up, awaiting 15m alignment",
            )
        if short_cond["1h_stack_dn"]:
            return Signal(
                ts=ts, symbol=state.symbol, side="SELL",
                severity=Severity.WATCHLIST,
                price=price, atr=atr_val,
                reason="1H trend down, awaiting 15m alignment",
            )
        return None


def _bar_timestamp(idx_value) -> datetime:
    """Normalize a pandas index value to a tz-aware datetime."""
    if isinstance(idx_value, pd.Timestamp):
        ts = idx_value.to_pydatetime()
    else:
        ts = idx_value
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts
