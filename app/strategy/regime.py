"""
Regime detection — ADX + ATR percentile router.

Gold cycles between two clean regimes that demand opposite strategies:

  TREND  (ADX >= 25 and EMA stack aligned): breakout-trend works, mean
         reversion gets steamrolled. Strong DXY/yields move days, NFP
         continuation moves, FOMC follow-through.

  CHOP   (ADX <= 20, or EMA stack disorganised): SMC mean-reversion at
         premium/discount POIs works, breakout gets faked.

  TRANSITION (20 < ADX < 25): in between. Strategies should down-size or stay
         out. This is where most retail bots get chopped up.

  HIGH_VOL: ATR percentile > 0.95. Even trending bots should reduce risk —
         spreads widen, slippage explodes. Optional gate.

ADX is computed Wilder-style. EMA stack uses 50/200 on the trend timeframe.

The router returns a Regime *and* per-strategy weights. mt5_live.py reads
the breakout weight; mt5_smc.py reads the SMC weight. A weight of 0 means
"this strategy refuses to trade in the current regime".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Regime(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CHOP = "chop"
    TRANSITION = "transition"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


@dataclass
class RegimeParams:
    adx_period: int = 14
    adx_trend_min: float = 25.0           # >= this = trending
    adx_chop_max: float = 20.0            # <= this = chop
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    atr_pct_lookback: int = 100
    high_vol_pct: float = 0.95            # ATR percentile above this -> HIGH_VOL flag
    # Per-strategy weights for each regime. mt5_live/mt5_smc multiply their
    # base risk by these. Keeps regime-aware sizing in *config*, not in code.
    weights: dict = field(default_factory=lambda: {
        Regime.TREND_UP.value:   {"breakout": 1.0, "smc": 0.3},
        Regime.TREND_DOWN.value: {"breakout": 1.0, "smc": 0.3},
        Regime.CHOP.value:       {"breakout": 0.0, "smc": 1.0},
        Regime.TRANSITION.value: {"breakout": 0.5, "smc": 0.5},
        Regime.HIGH_VOL.value:   {"breakout": 0.5, "smc": 0.5},
        Regime.UNKNOWN.value:    {"breakout": 0.0, "smc": 0.0},
    })


@dataclass
class RegimeSnapshot:
    regime: Regime
    adx: float
    di_plus: float
    di_minus: float
    ema_fast: float
    ema_slow: float
    atr: float
    atr_pct_rank: float
    note: str
    weight_breakout: float
    weight_smc: float


# ============================ ADX (Wilder) ===========================

def _wilder_ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (ADX, +DI, -DI). All Wilder-smoothed."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)

    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

    atr_w = _wilder_ema(tr, period)
    plus_di = 100 * _wilder_ema(plus_dm, period) / atr_w.replace(0, pd.NA)
    minus_di = 100 * _wilder_ema(minus_dm, period) / atr_w.replace(0, pd.NA)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)) * 100
    adx_series = _wilder_ema(dx.fillna(0), period)
    return adx_series, plus_di.fillna(0), minus_di.fillna(0)


# =============================== Classify ============================

def classify_regime(bars_trend: pd.DataFrame,
                    params: RegimeParams | None = None) -> RegimeSnapshot:
    """Classify the trend timeframe (typically 1H or 4H) into a regime."""
    p = params or RegimeParams()
    if len(bars_trend) < max(p.ema_slow, p.adx_period) + 5:
        return _unknown(p, "not_enough_history")

    high, low, close = bars_trend["High"], bars_trend["Low"], bars_trend["Close"]
    ema_f = close.ewm(span=p.ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=p.ema_slow, adjust=False).mean()

    # ATR + percentile for HIGH_VOL flag
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr_s = _wilder_ema(tr, p.atr_period)
    atr_pct = atr_s.rolling(p.atr_pct_lookback).rank(pct=True)

    adx_s, di_p, di_m = adx(high, low, close, p.adx_period)

    if pd.isna(adx_s.iloc[-1]) or pd.isna(ema_f.iloc[-1]) or pd.isna(ema_s.iloc[-1]):
        return _unknown(p, "indicator_nan")

    adx_val = float(adx_s.iloc[-1])
    di_p_val = float(di_p.iloc[-1])
    di_m_val = float(di_m.iloc[-1])
    ef = float(ema_f.iloc[-1])
    es = float(ema_s.iloc[-1])
    atr_val = float(atr_s.iloc[-1])
    atr_rank = float(atr_pct.iloc[-1]) if not pd.isna(atr_pct.iloc[-1]) else 0.5

    # High vol always wins — it's a meta-state we surface even on top of trend.
    if atr_rank >= p.high_vol_pct:
        regime = Regime.HIGH_VOL
        note = f"atr_pct={atr_rank:.2f} >= {p.high_vol_pct}"
    elif adx_val >= p.adx_trend_min and ef > es and di_p_val > di_m_val:
        regime = Regime.TREND_UP
        note = f"adx={adx_val:.1f} +DI>{di_m_val:.1f} ema50>ema200"
    elif adx_val >= p.adx_trend_min and ef < es and di_m_val > di_p_val:
        regime = Regime.TREND_DOWN
        note = f"adx={adx_val:.1f} -DI>{di_p_val:.1f} ema50<ema200"
    elif adx_val <= p.adx_chop_max:
        regime = Regime.CHOP
        note = f"adx={adx_val:.1f} <= {p.adx_chop_max}"
    else:
        regime = Regime.TRANSITION
        note = f"adx={adx_val:.1f} between {p.adx_chop_max}-{p.adx_trend_min}"

    weights = p.weights.get(regime.value, {"breakout": 0.5, "smc": 0.5})
    return RegimeSnapshot(
        regime=regime, adx=adx_val, di_plus=di_p_val, di_minus=di_m_val,
        ema_fast=ef, ema_slow=es, atr=atr_val, atr_pct_rank=atr_rank,
        note=note,
        weight_breakout=float(weights.get("breakout", 0.0)),
        weight_smc=float(weights.get("smc", 0.0)),
    )


def _unknown(p: RegimeParams, why: str) -> RegimeSnapshot:
    w = p.weights.get(Regime.UNKNOWN.value, {"breakout": 0.0, "smc": 0.0})
    return RegimeSnapshot(
        regime=Regime.UNKNOWN, adx=0.0, di_plus=0.0, di_minus=0.0,
        ema_fast=0.0, ema_slow=0.0, atr=0.0, atr_pct_rank=0.0,
        note=why,
        weight_breakout=float(w.get("breakout", 0.0)),
        weight_smc=float(w.get("smc", 0.0)),
    )


def regime_weight_for(snapshot: RegimeSnapshot, strategy: str) -> float:
    """Convenience: 'breakout' or 'smc' -> 0..1 risk multiplier."""
    if strategy == "breakout":
        return snapshot.weight_breakout
    if strategy == "smc":
        return snapshot.weight_smc
    return 0.0
