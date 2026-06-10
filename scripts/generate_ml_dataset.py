"""
Phase 5a — ML dataset generation.

Walks N months of historical XAUUSD bars (15m + 1H + 4H), simulates trading
with both strategies (breakout + SMC), and records:

    features   : 30+ market state inputs at signal time
    label      : 1 (WIN) or 0 (LOSS) — based on whether TP or SL hit first
                 within the next 100 bars (~25 hours on 15m)
    metadata   : entry/SL/TP prices, side, strategy, ts

Saved to data/ml_dataset.parquet for training (Phase 5b).

We use yfinance (`GC=F`) for the prototype since MT5 history requires
the terminal to be running. The features are timeframe-agnostic — they'll
transfer to MT5's GOLD.i# in live deployment.

Run:
    cd v2
    python scripts/generate_ml_dataset.py --months 6
    python scripts/generate_ml_dataset.py --months 12 --pair GC=F
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf


HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


# =============================== CONFIG ==============================
# Strategy params — match the live bots after the recent loosening
BREAKOUT_PARAMS = {
    "ema_fast": 50, "ema_slow": 200, "atr_period": 14,
    "atr_min": 10.0, "atr_pct_min": 0.25, "k_sl": 1.5, "k_tp": 2.5,
}
SMC_PARAMS = {
    "htf_pivot": 2, "ltf_pivot": 2, "min_impulse_bars": 3,
    "poi_freshness_bars": 60, "min_poi_score": 2,
    "sl_buffer_atr_frac": 0.25, "require_ltf_choch": False, "min_rr": 1.5,
    "atr_period": 14, "max_structure_lookback_bars": 300,
}

FORWARD_BARS = 100        # ~25 hours on 15m — max time to wait for SL/TP
WARMUP_BARS = 250         # need at least this much history before first signal

IST = timezone(timedelta(hours=5, minutes=30))


# =========================== INDICATORS =============================
def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def atr_percentile(atr_series: pd.Series, lookback: int = 100) -> pd.Series:
    return atr_series.rolling(lookback).rank(pct=True)


def _wilder_ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn.clip(lower=0)
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr_w = _wilder_ema(tr, period)
    plus_di = 100 * _wilder_ema(plus_dm, period) / atr_w.replace(0, np.nan)
    minus_di = 100 * _wilder_ema(minus_dm, period) / atr_w.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return _wilder_ema(dx.fillna(0), period), plus_di.fillna(0), minus_di.fillna(0)


# ====================== PRECOMPUTED INDICATORS =======================
# Bottleneck of v2 was recomputing indicators on growing slices per-bar
# (O(n^2) total). We now compute everything ONCE up front and index by row.
class IndicatorCache:
    """Compute all required indicators once across the full 15m and 1h series."""
    def __init__(self, df15: pd.DataFrame, df1h: pd.DataFrame):
        self.df15 = df15
        self.df1h = df1h

        h, l, c, o = df15["High"], df15["Low"], df15["Close"], df15["Open"]
        # 15m indicators
        self.ema_f_15 = ema(c, BREAKOUT_PARAMS["ema_fast"])
        self.ema_s_15 = ema(c, BREAKOUT_PARAMS["ema_slow"])
        self.atr_14  = atr(h, l, c, BREAKOUT_PARAMS["atr_period"])
        self.atr_5   = atr(h, l, c, 5)
        self.atr_50  = atr(h, l, c, 50)
        self.atr_pct = atr_percentile(self.atr_14, 100)
        self.adx_v, self.di_p, self.di_m = adx(h, l, c, 14)
        # Rolling stats for new features
        self.sma_20  = c.rolling(20).mean()
        self.std_20  = c.rolling(20).std()
        # Vol of vol
        self.atr_std_20 = self.atr_14.rolling(20).std()
        # 5-bar / 20-bar high-low ranges
        self.high_5  = h.rolling(5).max()
        self.low_5   = l.rolling(5).min()
        self.high_20 = h.rolling(20).max()
        self.low_20  = l.rolling(20).min()
        # Open/High/Low/Close as arrays for fast indexing
        self.o, self.h, self.l, self.c = o.values, h.values, l.values, c.values
        # Previous bar shifted closes for returns
        self.close_arr = c.values
        # Bullish/bearish bar bool arrays
        self._bull = (c > o).values
        self._bear = (c < o).values

        # 1h indicators
        h1, l1, c1 = df1h["High"], df1h["Low"], df1h["Close"]
        self.ema_f_1h = ema(c1, 50)
        self.ema_s_1h = ema(c1, 200)
        # For 1h alignment, build an index lookup: for each 15m ts, the last 1h ts <= it
        h1_idx = df1h.index
        # searchsorted to map 15m timestamps to 1h positions
        self._1h_positions = np.searchsorted(h1_idx.values, df15.index.values, side="right") - 1
        self._1h_positions = np.clip(self._1h_positions, 0, len(h1_idx) - 1)

    def get_1h_pos(self, idx_15: int) -> int:
        """Return the 1h bar index corresponding to the given 15m row."""
        return int(self._1h_positions[idx_15])

    def streak_at(self, idx: int, lookback: int = 10) -> int:
        """Compute consec bull/bear streak ending at idx using cached bool arrays."""
        s = 0
        for k in range(idx, max(-1, idx - lookback), -1):
            if self._bull[k]:
                if s < 0: break
                s += 1
            elif self._bear[k]:
                if s > 0: break
                s -= 1
            else:
                break
        return s

    def build_smc(self, pivot: int = 2) -> None:
        """Precompute all 1H swings + a per-bar bias array.
        Run once after construction if SMC signals are needed."""
        h, l = self.df1h["High"].values, self.df1h["Low"].values
        n = len(self.df1h)
        swings: list[tuple[int, float, str]] = []
        for i in range(pivot, n - pivot):
            wh = h[i - pivot:i + pivot + 1]
            wl = l[i - pivot:i + pivot + 1]
            if h[i] == wh.max() and wh.argmax() == pivot:
                swings.append((i, float(h[i]), "high"))
            if l[i] == wl.min() and wl.argmin() == pivot:
                swings.append((i, float(l[i]), "low"))
        swings.sort()
        self.htf_swings_all: list[tuple[int, float, str]] = swings

        # Walk forward computing bias once across full series
        closes = self.df1h["Close"].values
        bias_by_idx = ["none"] * n
        ub_h: list[tuple[int, float, str]] = []
        ub_l: list[tuple[int, float, str]] = []
        next_swing = 0
        bias = "none"
        for i in range(n):
            while next_swing < len(swings) and swings[next_swing][0] <= i:
                s = swings[next_swing]
                if s[2] == "high": ub_h.append(s)
                else: ub_l.append(s)
                next_swing += 1
            c = closes[i]
            for sh in reversed(ub_h):
                if sh[0] >= i: continue
                if c > sh[1]:
                    bias = "up"
                    ub_h = [s for s in ub_h if s[0] > sh[0]]
                    break
            for sl in reversed(ub_l):
                if sl[0] >= i: continue
                if c < sl[1]:
                    bias = "down"
                    ub_l = [s for s in ub_l if s[0] > sl[0]]
                    break
            bias_by_idx[i] = bias
        self.htf_bias_at: list[str] = bias_by_idx


# ============================== SESSIONS =============================
def in_session(ts_utc: pd.Timestamp) -> str:
    """Return session label for the IST hour at this UTC ts."""
    ts_ist = ts_utc.tz_convert(IST) if ts_utc.tz else ts_utc.tz_localize("UTC").tz_convert(IST)
    h, m = ts_ist.hour, ts_ist.minute
    t = h * 60 + m
    if 12 * 60 + 30 <= t <= 16 * 60 + 30: return "London"
    if 18 * 60      <= t <= 21 * 60:      return "NY_overlap"
    if 21 * 60      <= t <= 23 * 60 + 30: return "NY_afternoon"
    return "outside"


# ============================ FEATURES ==============================
def extract_features(cache: "IndicatorCache", idx: int,
                     signal_side: str, strategy: str) -> dict:
    """Build feature vector at 15m bar `idx` using precomputed IndicatorCache.
    O(1) per call — all heavy indicator math already done."""
    if idx < WARMUP_BARS:
        return {}

    bar_ts = cache.df15.index[idx]
    last_open  = cache.o[idx]
    last_high  = cache.h[idx]
    last_low   = cache.l[idx]
    last_close = cache.c[idx]
    bar_range  = last_high - last_low

    # Lookups (NaN-safe)
    def _g(arr, i, default=0.0):
        v = arr.iloc[i] if hasattr(arr, "iloc") else arr[i]
        return float(v) if v == v else default   # NaN check

    ema_f_v   = _g(cache.ema_f_15, idx)
    ema_s_v   = _g(cache.ema_s_15, idx)
    atr_v     = _g(cache.atr_14, idx, 1.0)
    atr_pct_v = _g(cache.atr_pct, idx, 0.5)
    adx_v     = _g(cache.adx_v, idx)
    di_p_v    = _g(cache.di_p, idx)
    di_m_v    = _g(cache.di_m, idx)
    atr_5_v   = _g(cache.atr_5, idx)
    atr_50_v  = _g(cache.atr_50, idx)
    sma_20_v  = _g(cache.sma_20, idx, last_close)
    std_20_v  = _g(cache.std_20, idx)
    vol_of_vol = _g(cache.atr_std_20, idx)
    high_5_v  = _g(cache.high_5, idx, last_high)
    low_5_v   = _g(cache.low_5, idx, last_low)
    high_20_v = _g(cache.high_20, idx, last_high)
    low_20_v  = _g(cache.low_20, idx, last_low)

    # 1H values aligned to this 15m timestamp
    i1h = cache.get_1h_pos(idx)
    ema_f_h_v = _g(cache.ema_f_1h, i1h)
    ema_s_h_v = _g(cache.ema_s_1h, i1h)

    # Bollinger
    bb_range = std_20_v * 4 if std_20_v > 0 else 0.0
    bb_pos = (last_close - sma_20_v) / (std_20_v * 2) if std_20_v > 0 else 0.0

    # Breakout strength using prev bar
    prev_high = cache.h[idx - 1] if idx > 0 else last_high
    prev_low  = cache.l[idx - 1] if idx > 0 else last_low
    breakout_strength = max((last_close - prev_high) / atr_v if atr_v > 0 else 0.0,
                             (prev_low - last_close) / atr_v if atr_v > 0 else 0.0)

    feats = {
        "ts_iso": bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts),
        "strategy": strategy,
        "side": signal_side,
        "close": float(last_close),

        # Time
        "hour_utc": bar_ts.hour if hasattr(bar_ts, "hour") else 0,
        "dow":      bar_ts.dayofweek if hasattr(bar_ts, "dayofweek") else 0,
        "session":  in_session(bar_ts) if hasattr(bar_ts, "tz") else "unknown",

        # Volatility (original)
        "atr_raw":     atr_v,
        "atr_pct":     atr_pct_v,
        "range_5":     float(high_5_v - low_5_v),
        "range_20":    float(high_20_v - low_20_v),

        # Trend / momentum (15m)
        "ema_dist_15": (ema_f_v - ema_s_v) / ema_s_v if ema_s_v else 0.0,
        "px_vs_ema50": (last_close - ema_f_v) / ema_f_v if ema_f_v else 0.0,
        "adx":         adx_v,
        "di_plus":     di_p_v,
        "di_minus":    di_m_v,

        # Trend (1H)
        "ema_dist_1h":   (ema_f_h_v - ema_s_h_v) / ema_s_h_v if ema_s_h_v else 0.0,
        "px_vs_ema50_1h": (last_close - ema_f_h_v) / ema_f_h_v if ema_f_h_v else 0.0,

        # Recent price action
        "ret_1":  (last_close / cache.c[idx - 1] - 1) if idx > 0 and cache.c[idx - 1] else 0.0,
        "ret_5":  (last_close / cache.c[idx - 5] - 1) if idx > 5 and cache.c[idx - 5] else 0.0,
        "ret_20": (last_close / cache.c[idx - 20] - 1) if idx > 20 and cache.c[idx - 20] else 0.0,
        "high_5_dist":  float((high_5_v - last_close) / last_close) if last_close else 0.0,
        "low_5_dist":   float((last_close - low_5_v) / last_close) if last_close else 0.0,
        "high_20_dist": float((high_20_v - last_close) / last_close) if last_close else 0.0,
        "low_20_dist":  float((last_close - low_20_v) / last_close) if last_close else 0.0,

        # === NEW FEATURES (Phase 5a v2) ===
        "atr_5_50_ratio": atr_5_v / atr_50_v if atr_50_v > 0 else 1.0,

        "body_pct":       abs(last_close - last_open) / bar_range if bar_range > 0 else 0.0,
        "upper_wick_pct": (last_high - max(last_open, last_close)) / bar_range if bar_range > 0 else 0.0,
        "lower_wick_pct": (min(last_open, last_close) - last_low) / bar_range if bar_range > 0 else 0.0,
        "range_to_atr":   bar_range / atr_v if atr_v > 0 else 1.0,

        "dist_from_ema200":    (last_close - ema_s_v) / ema_s_v if ema_s_v else 0.0,
        "dist_from_ema200_1h": (last_close - ema_s_h_v) / ema_s_h_v if ema_s_h_v else 0.0,

        "bb_position":  float(bb_pos),
        "bb_width_pct": float(bb_range / sma_20_v) if sma_20_v > 0 else 0.0,

        "roc_10":          (last_close / cache.c[idx - 10] - 1) if idx > 10 and cache.c[idx - 10] else 0.0,
        "consec_streak":   cache.streak_at(idx, lookback=10),
        "vol_of_vol":      vol_of_vol,
        "breakout_strength": float(breakout_strength),

        # Regime
        "regime": _regime_tag(adx_v, ema_f_v > ema_s_v),
    }
    return feats


def _regime_tag(adx_val: float, ema_up: bool) -> str:
    if adx_val >= 25 and ema_up:  return "trend_up"
    if adx_val >= 25 and not ema_up: return "trend_down"
    if adx_val <= 20: return "chop"
    return "transition"


# ============================ STRATEGIES =============================
def breakout_signal(cache: "IndicatorCache", idx: int):
    """O(1) breakout signal using precomputed indicators."""
    if idx < WARMUP_BARS or idx < 2:
        return None

    atr_val = cache.atr_14.iloc[idx]
    ema_f_e = cache.ema_f_15.iloc[idx]
    ema_s_e = cache.ema_s_15.iloc[idx]
    if pd.isna(atr_val) or pd.isna(ema_f_e) or pd.isna(ema_s_e):
        return None

    i1h = cache.get_1h_pos(idx)
    ema_f_t = cache.ema_f_1h.iloc[i1h]
    ema_s_t = cache.ema_s_1h.iloc[i1h]
    if pd.isna(ema_f_t) or pd.isna(ema_s_t):
        return None

    ap = cache.atr_pct.iloc[idx]
    if pd.isna(ap) or ap < BREAKOUT_PARAMS["atr_pct_min"]:
        return None
    if atr_val < BREAKOUT_PARAMS["atr_min"]:
        return None

    last_high  = cache.h[idx]
    last_low   = cache.l[idx]
    last_close = cache.c[idx]
    prev_high  = cache.h[idx - 1]
    prev_low   = cache.l[idx - 1]

    long_cond = (ema_f_e > ema_s_e and ema_f_t > ema_s_t and last_high > prev_high)
    short_cond = (ema_f_e < ema_s_e and ema_f_t < ema_s_t and last_low < prev_low)

    atr_v = float(atr_val)
    if long_cond:
        entry = float(last_close)
        return ("BUY", entry, entry - 1.5 * atr_v, entry + 2.5 * atr_v)
    if short_cond:
        entry = float(last_close)
        return ("SELL", entry, entry + 1.5 * atr_v, entry - 2.5 * atr_v)
    return None


# ----- SMC subset (simplified for dataset gen; matches strategy logic) -----
def _swings(df: pd.DataFrame, pivot: int = 2) -> list[tuple[int, float, str]]:
    """Return list of (idx, price, 'high'|'low') for swings."""
    n = len(df)
    if n < 2 * pivot + 1: return []
    h, l = df["High"].values, df["Low"].values
    out = []
    for i in range(pivot, n - pivot):
        wh, wl = h[i - pivot:i + pivot + 1], l[i - pivot:i + pivot + 1]
        if h[i] == wh.max() and wh.argmax() == pivot:
            out.append((i, float(h[i]), "high"))
        if l[i] == wl.min() and wl.argmin() == pivot:
            out.append((i, float(l[i]), "low"))
    return sorted(out)


def _bias_from_swings(df: pd.DataFrame, swings: list) -> str:
    """Walk closes vs unbroken swings — last direction wins."""
    closes = df["Close"].values
    bias = "none"
    ub_h = [s for s in swings if s[2] == "high"]
    ub_l = [s for s in swings if s[2] == "low"]
    for i in range(len(df)):
        c = closes[i]
        for sh in reversed([s for s in ub_h if s[0] < i]):
            if c > sh[1]:
                bias = "up"
                ub_h = [s for s in ub_h if s[0] > sh[0]]
                break
        for sl in reversed([s for s in ub_l if s[0] < i]):
            if c < sl[1]:
                bias = "down"
                ub_l = [s for s in ub_l if s[0] > sl[0]]
                break
    return bias


def smc_signal(cache: "IndicatorCache", idx: int):
    """O(log n) SMC signal — uses precomputed all-time swings on the 1H series.
    Filters to current lookback window via index, no recomputation."""
    if idx < WARMUP_BARS:
        return None

    atr_val = cache.atr_14.iloc[idx]
    if pd.isna(atr_val) or atr_val <= 0:
        return None
    atr_val = float(atr_val)

    i1h = cache.get_1h_pos(idx)
    if i1h < 60:
        return None

    # Use precomputed all-time 1H swings (built once in cache.build_smc())
    # Filter to those within current lookback window
    lookback_start = max(0, i1h - SMC_PARAMS["max_structure_lookback_bars"])
    swings_h = [s for s in cache.htf_swings_all if lookback_start <= s[0] <= i1h]
    if not swings_h:
        return None

    # Determine current bias from the most recent break of structure in the window
    bias = cache.htf_bias_at[i1h]
    if bias == "none":
        return None

    price = float(cache.c[idx])

    if bias == "up":
        candidates = [s for s in swings_h if s[2] == "low" and price - 1.5 * atr_val <= s[1] <= price]
        if not candidates: return None
        poi_price = max(candidates, key=lambda s: s[0])[1]
        sl = poi_price - SMC_PARAMS["sl_buffer_atr_frac"] * atr_val
        future = [s for s in swings_h if s[2] == "high" and s[1] > price]
        tp = future[0][1] if future else price + 2.5 * atr_val
        risk = abs(price - sl); reward = abs(tp - price)
        if risk <= 0 or reward / risk < SMC_PARAMS["min_rr"]:
            return None
        return ("BUY", price, sl, tp)
    else:
        candidates = [s for s in swings_h if s[2] == "high" and price <= s[1] <= price + 1.5 * atr_val]
        if not candidates: return None
        poi_price = max(candidates, key=lambda s: s[0])[1]
        sl = poi_price + SMC_PARAMS["sl_buffer_atr_frac"] * atr_val
        future = [s for s in swings_h if s[2] == "low" and s[1] < price]
        tp = future[0][1] if future else price - 2.5 * atr_val
        risk = abs(price - sl); reward = abs(tp - price)
        if risk <= 0 or reward / risk < SMC_PARAMS["min_rr"]:
            return None
        return ("SELL", price, sl, tp)


# ======================= FORWARD SIMULATION ==========================
def label_outcome(df15: pd.DataFrame, entry_idx: int, side: str,
                   sl: float, tp: float, forward_bars: int = FORWARD_BARS) -> tuple[int, str, int]:
    """Walk forward, return (label, reason, bars_held).
        label 1 = WIN (TP hit first), 0 = LOSS (SL first), -1 = TIMEOUT."""
    end = min(entry_idx + 1 + forward_bars, len(df15))
    for j in range(entry_idx + 1, end):
        bar = df15.iloc[j]
        h, l = float(bar["High"]), float(bar["Low"])
        if side == "BUY":
            hit_tp = h >= tp
            hit_sl = l <= sl
        else:
            hit_tp = l <= tp
            hit_sl = h >= sl
        # Conservative: if both hit in same bar, assume SL hit first (worse case)
        if hit_sl: return (0, "SL", j - entry_idx)
        if hit_tp: return (1, "TP", j - entry_idx)
    return (-1, "TIMEOUT", end - entry_idx - 1)


# =============================== DATA ================================
def fetch_bars_yfinance(pair: str, months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull (15m, 1h) bars from yfinance for the last `months` months."""
    print(f"Fetching {pair} bars from yfinance (months={months})...")
    if months <= 2:
        df15 = yf.download(pair, period=f"{months * 30}d", interval="15m", progress=False)
    else:
        # yfinance caps 15m at 60d — fall back to 1h as both
        df1h_full = yf.download(pair, period=f"{months * 30}d", interval="1h", progress=False)
        df15 = df1h_full.copy()
        print(f"  (using 1h bars as both 15m and 1h since yfinance caps 15m at 60d)")
    df1h = yf.download(pair, period=f"{months * 30}d", interval="1h", progress=False)

    for df in (df15, df1h):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    return df15, df1h


def fetch_bars_mt5(symbol: str, months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull (15m, 1h) bars directly from MT5 via the MetaTrader5 package.
    Requires running on VPS where MT5 is installed and the bot's .env is loaded
    with MT5_PATH / MT5_LOGIN / MT5_PASSWORD / MT5_SERVER."""
    try:
        import MetaTrader5 as mt5
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
        import os
    except ImportError as e:
        raise RuntimeError(f"MT5 path requires VPS — install MetaTrader5 + dotenv. ({e})")

    path = os.getenv("MT5_PATH")
    if path and os.getenv("MT5_LOGIN"):
        ok = mt5.initialize(
            path=path,
            login=int(os.getenv("MT5_LOGIN")),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            timeout=60000,
        )
    else:
        ok = mt5.initialize()
    if not ok:
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")

    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"symbol_select({symbol}) failed: {mt5.last_error()}")

    # months -> bars
    bars_per_day_15m = 96     # 24 * 4
    bars_per_day_1h = 24
    n_15m = months * 30 * bars_per_day_15m
    n_1h  = months * 30 * bars_per_day_1h

    print(f"Fetching {symbol} bars from MT5 (months={months}: {n_15m} 15m + {n_1h} 1h)...")
    # start_pos=1 drops the in-progress (forming) bar so features/ATR are
    # computed only from completed bars -- matches what the live bot and the
    # backtest see, avoiding look-ahead in the training labels.
    r15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, n_15m)
    r1h = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, n_1h)
    mt5.shutdown()

    if r15 is None or len(r15) == 0:
        raise RuntimeError(f"No M15 bars returned for {symbol}")
    if r1h is None or len(r1h) == 0:
        raise RuntimeError(f"No H1 bars returned for {symbol}")

    def _to_df(rates):
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "tick_volume": "Volume",
        })
        return df[["Open", "High", "Low", "Close", "Volume"]]

    return _to_df(r15), _to_df(r1h)


def fetch_bars(source: str, pair: str, months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch to yfinance or MT5 based on --source flag."""
    if source == "mt5":
        return fetch_bars_mt5(pair, months)
    return fetch_bars_yfinance(pair, months)


# =============================== MAIN ================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=6)
    p.add_argument("--pair", default="GC=F",
                   help="Symbol — yfinance: 'GC=F'. MT5: 'GOLD.i#' (XM) or your broker's symbol.")
    p.add_argument("--source", choices=["yfinance", "mt5"], default="yfinance",
                   help="Data source. 'mt5' requires running on VPS with MT5 installed.")
    p.add_argument("--out", default=str(HERE / "data" / "ml_dataset.parquet"))
    args = p.parse_args()

    df15, df1h = fetch_bars(args.source, args.pair, args.months)
    print(f"  fetched {len(df15)} 15m bars, {len(df1h)} 1h bars "
          f"(span: {df15.index[0]} -> {df15.index[-1]})")

    # Precompute ALL indicators + SMC structure ONCE
    print("  precomputing indicators + SMC swings...")
    cache = IndicatorCache(df15, df1h)
    cache.build_smc(pivot=SMC_PARAMS["htf_pivot"])
    print(f"  done. {len(cache.htf_swings_all)} 1H swings cached. "
          f"Walking {len(df15) - WARMUP_BARS - FORWARD_BARS} bars...")

    rows = []
    skipped_active = 0
    import time as _t
    t0 = _t.time()

    for i in range(WARMUP_BARS, len(df15) - FORWARD_BARS):
        # === Breakout signal ===
        sig = breakout_signal(cache, i)
        if sig is not None:
            side, entry, sl, tp = sig
            label, reason, held = label_outcome(df15, i, side, sl, tp)
            if label >= 0:
                feats = extract_features(cache, i, side, "breakout")
                if feats:
                    feats.update({
                        "entry": entry, "sl": sl, "tp": tp,
                        "label": label, "exit_reason": reason, "bars_held": held,
                        "rr_target": abs(tp - entry) / abs(entry - sl),
                    })
                    rows.append(feats)
            else:
                skipped_active += 1

        # === SMC signal ===
        sig = smc_signal(cache, i)
        if sig is not None:
            side, entry, sl, tp = sig
            label, reason, held = label_outcome(df15, i, side, sl, tp)
            if label >= 0:
                feats = extract_features(cache, i, side, "smc")
                if feats:
                    feats.update({
                        "entry": entry, "sl": sl, "tp": tp,
                        "label": label, "exit_reason": reason, "bars_held": held,
                        "rr_target": abs(tp - entry) / abs(entry - sl),
                    })
                    rows.append(feats)
            else:
                skipped_active += 1

        if i % 500 == 0 and i > WARMUP_BARS:
            elapsed = _t.time() - t0
            print(f"  bar {i}/{len(df15)}  rows so far: {len(rows)}  ({elapsed:.1f}s elapsed)")

    if not rows:
        print("ERROR: no labeled rows generated. Either no signals fired, or all timed out.")
        return 1

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path)
    except Exception:
        # Parquet needs pyarrow; fall back to CSV
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)

    # Summary
    print(f"\n=== Dataset saved: {out_path} ===")
    print(f"Total rows:    {len(df)}")
    print(f"Skipped (TO):  {skipped_active}")
    print(f"Breakout rows: {(df['strategy'] == 'breakout').sum()}")
    print(f"SMC rows:      {(df['strategy'] == 'smc').sum()}")
    print(f"WIN rate:      {df['label'].mean() * 100:.1f}% overall")
    print(f"  breakout:    {df[df['strategy']=='breakout']['label'].mean()*100:.1f}%")
    print(f"  smc:         {df[df['strategy']=='smc']['label'].mean()*100:.1f}%")
    print(f"\nFeatures: {list(df.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
