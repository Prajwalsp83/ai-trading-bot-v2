"""
Phase B.2 — Standalone strategy evaluation.

Pure functions. No MT5 dependency. No globals. Takes OHLCV DataFrames + a
params dataclass. Returns a signal dict or None.

Both live bots (mt5_live.py, mt5_smc.py) AND the backtest engine call these
functions, so live and backtest behavior cannot drift.

Public API:
  evaluate_breakout(df15, df1h, df4h, params) -> dict | None
  evaluate_smc(df15, df1h, params) -> dict | None

Signal dict shape (same as live bot expects):
  {
    "severity":  "BUY_READY" | "SELL_READY" | "BREAKOUT_WATCH" | "WATCHLIST" | "SKIPPED",
    "side":      "BUY" | "SELL" | None,
    "price":     float,
    "atr":       float,
    "reason":    str,
    "rejection_reason": str (only for SKIPPED),
    # SMC-only:
    "sl_suggested": float,
    "tp_suggested": float,
    "rr": float,
    "poi_score": int,
    "htf_bias": str,
  }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# ============================== PARAMS ===============================
@dataclass
class BreakoutSignalParams:
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    atr_min: float = 10.0
    atr_pct_min: float = 0.25
    min_trend_strength: float = 0.0
    use_4h_trend_gate: bool = False
    k_sl: float = 1.5
    k_tp: float = 2.5


@dataclass
class SMCSignalParams:
    htf_pivot: int = 2
    ltf_pivot: int = 2
    min_impulse_bars: int = 3
    poi_freshness_bars: int = 60
    min_poi_score: int = 2
    sl_buffer_atr_frac: float = 0.25
    require_ltf_choch: bool = False
    min_rr: float = 1.5
    atr_period: int = 14
    max_structure_lookback_bars: int = 300


# ============================== INDICATORS ==========================
def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def atr_percentile(s: pd.Series, lookback: int = 100) -> pd.Series:
    return s.rolling(lookback).rank(pct=True)


# ============================== BREAKOUT ============================
def evaluate_breakout(df15: pd.DataFrame, df1h: pd.DataFrame,
                      df4h: pd.DataFrame | None,
                      params: BreakoutSignalParams) -> dict | None:
    """Breakout-trend signal evaluator.

    Conditions (long; mirror for short):
      1. 15m ema_fast > ema_slow
      2. 1H ema_fast > ema_slow
      3. Current bar high > prev bar high (the breakout)
      4. ATR >= atr_min (vol high enough)
      5. (Optional, if use_4h_trend_gate) 4H ema_fast > ema_slow

    Pre-gate: ATR percentile >= atr_pct_min (skip if low-vol regime)
    Output severity: BUY_READY/SELL_READY (all met), BREAKOUT_WATCH (n-1 met),
                     WATCHLIST (1H aligned only), SKIPPED, None
    """
    p = params
    if len(df15) < p.ema_slow + 2 or len(df1h) < p.ema_slow + 2:
        return None

    ema_f_e = ema(df15["Close"], p.ema_fast)
    ema_s_e = ema(df15["Close"], p.ema_slow)
    atr_e = atr(df15["High"], df15["Low"], df15["Close"], p.atr_period)

    ema_f_t = ema(df1h["Close"], p.ema_fast)
    ema_s_t = ema(df1h["Close"], p.ema_slow)

    last = df15.iloc[-1]
    prev = df15.iloc[-2]
    price = float(last["Close"])
    atr_val = float(atr_e.iloc[-1])

    if pd.isna(atr_val) or any(pd.isna(s.iloc[-1]) for s in
                                (ema_f_e, ema_s_e, ema_f_t, ema_s_t)):
        return None

    # === Pre-gates ===
    rejection = None
    if p.atr_pct_min > 0:
        ap = atr_percentile(atr_e, lookback=100).iloc[-1]
        if pd.isna(ap) or ap < p.atr_pct_min:
            rejection = "atr_pct_too_low"
    if rejection is None and p.min_trend_strength > 0:
        ts = abs(ema_f_e.iloc[-1] - ema_s_e.iloc[-1]) / ema_s_e.iloc[-1]
        if ts < p.min_trend_strength:
            rejection = "trend_strength_too_low"

    # === 4H trend gate (optional) ===
    h4_up = h4_dn = True
    if p.use_4h_trend_gate and df4h is not None and len(df4h) >= p.ema_slow + 2:
        ema_f_h = ema(df4h["Close"], p.ema_fast)
        ema_s_h = ema(df4h["Close"], p.ema_slow)
        if not (pd.isna(ema_f_h.iloc[-1]) or pd.isna(ema_s_h.iloc[-1])):
            h4_up = bool(ema_f_h.iloc[-1] > ema_s_h.iloc[-1])
            h4_dn = bool(ema_f_h.iloc[-1] < ema_s_h.iloc[-1])

    if rejection is not None:
        return {"severity": "SKIPPED", "side": None, "price": price, "atr": atr_val,
                "reason": f"skipped: {rejection}", "rejection_reason": rejection}

    long_cond = {
        "15m_stack_up": bool(ema_f_e.iloc[-1] > ema_s_e.iloc[-1]),
        "1h_stack_up":  bool(ema_f_t.iloc[-1] > ema_s_t.iloc[-1]),
        "breakout_up":  bool(last["High"] > prev["High"]),
        "atr_ok":       bool(atr_val >= p.atr_min),
    }
    short_cond = {
        "15m_stack_dn": bool(ema_f_e.iloc[-1] < ema_s_e.iloc[-1]),
        "1h_stack_dn":  bool(ema_f_t.iloc[-1] < ema_s_t.iloc[-1]),
        "breakout_dn":  bool(last["Low"] < prev["Low"]),
        "atr_ok":       bool(atr_val >= p.atr_min),
    }
    if p.use_4h_trend_gate:
        long_cond["4h_stack_up"] = h4_up
        short_cond["4h_stack_dn"] = h4_dn

    long_n = sum(long_cond.values())
    short_n = sum(short_cond.values())
    n_total = len(long_cond)

    if long_n == n_total:
        return {"severity": "BUY_READY", "side": "BUY", "price": price, "atr": atr_val,
                "reason": f"All {n_total} long conditions met", "conditions": long_cond}
    if short_n == n_total:
        return {"severity": "SELL_READY", "side": "SELL", "price": price, "atr": atr_val,
                "reason": f"All {n_total} short conditions met", "conditions": short_cond}
    if long_n == n_total - 1:
        return {"severity": "BREAKOUT_WATCH", "side": "BUY", "price": price, "atr": atr_val,
                "reason": f"{long_n} of {n_total} long conditions met", "conditions": long_cond}
    if short_n == n_total - 1:
        return {"severity": "BREAKOUT_WATCH", "side": "SELL", "price": price, "atr": atr_val,
                "reason": f"{short_n} of {n_total} short conditions met", "conditions": short_cond}
    if long_cond["1h_stack_up"]:
        return {"severity": "WATCHLIST", "side": "BUY", "price": price, "atr": atr_val,
                "reason": "1H trend up, awaiting 15m alignment"}
    if short_cond["1h_stack_dn"]:
        return {"severity": "WATCHLIST", "side": "SELL", "price": price, "atr": atr_val,
                "reason": "1H trend down, awaiting 15m alignment"}
    return None


# ============================== SMC internals =======================
def _smc_swings(df: pd.DataFrame, pivot: int = 2):
    if len(df) < 2 * pivot + 1:
        return []
    h = df["High"].values
    l = df["Low"].values
    idx = df.index
    swings = []
    for i in range(pivot, len(df) - pivot):
        wh = h[i - pivot:i + pivot + 1]
        wl = l[i - pivot:i + pivot + 1]
        if h[i] == wh.max() and wh.argmax() == pivot:
            swings.append((i, idx[i], float(h[i]), "high"))
        if l[i] == wl.min() and wl.argmin() == pivot:
            swings.append((i, idx[i], float(l[i]), "low"))
    swings.sort(key=lambda s: s[0])
    return swings


def _smc_events(df: pd.DataFrame, swings):
    """Return list of (idx, ts, kind, side, broken_price, close) for BOS/CHoCH events."""
    events = []
    if not swings:
        return events
    closes = df["Close"].values
    idx = df.index
    bias = "none"
    ub_h, ub_l = [], []
    nxt = 0
    for i in range(len(df)):
        while nxt < len(swings) and swings[nxt][0] <= i:
            s = swings[nxt]
            (ub_h if s[3] == "high" else ub_l).append(s)
            nxt += 1
        c = closes[i]
        broken_h = None
        for sh in reversed(ub_h):
            if sh[0] >= i: continue
            if c > sh[2]:
                broken_h = sh; break
        if broken_h is not None:
            kind = "BOS" if bias == "up" else "CHoCH"
            events.append((i, idx[i], kind, "up", broken_h[2], float(c)))
            bias = "up"
            ub_h = [sh for sh in ub_h if sh[0] > broken_h[0]]
        broken_l = None
        for sl in reversed(ub_l):
            if sl[0] >= i: continue
            if c < sl[2]:
                broken_l = sl; break
        if broken_l is not None:
            kind = "BOS" if bias == "down" else "CHoCH"
            events.append((i, idx[i], kind, "down", broken_l[2], float(c)))
            bias = "down"
            ub_l = [sl for sl in ub_l if sl[0] > broken_l[0]]
    return events


def _smc_fvgs(df: pd.DataFrame, max_age_bars: int | None = None):
    """Return list of FVGs: (side, top, bottom, created_idx, mitigated)."""
    if len(df) < 3:
        return []
    h = df["High"].values
    l = df["Low"].values
    fvgs = []
    for i in range(2, len(df)):
        if h[i - 2] < l[i]:
            fvgs.append({"side": "bull", "top": float(l[i]), "bottom": float(h[i - 2]),
                         "created_idx": i, "mitigated": False})
        if l[i - 2] > h[i]:
            fvgs.append({"side": "bear", "top": float(l[i - 2]), "bottom": float(h[i]),
                         "created_idx": i, "mitigated": False})
    for f in fvgs:
        for j in range(f["created_idx"] + 1, len(df)):
            if l[j] <= f["top"] and h[j] >= f["bottom"]:
                f["mitigated"] = True
                break
    if max_age_bars is not None:
        cutoff = len(df) - max_age_bars
        fvgs = [f for f in fvgs if f["created_idx"] >= cutoff]
    return fvgs


def _smc_obs(df: pd.DataFrame, events, min_impulse_bars: int = 3):
    """Return list of order blocks: (side, top, bottom, created_idx, impulse_idx, mitigated)."""
    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    obs = []
    for ev in events:
        bos_idx = ev[0]
        side = ev[3]
        if side == "up":
            ob_idx = None
            for j in range(bos_idx - 1, max(-1, bos_idx - 30), -1):
                if closes[j] < opens[j]:
                    ob_idx = j; break
            if ob_idx is None: continue
            bullish = sum(1 for k in range(ob_idx + 1, bos_idx + 1) if closes[k] > opens[k])
            if bullish < min_impulse_bars: continue
            obs.append({"side": "bull", "top": float(highs[ob_idx]), "bottom": float(lows[ob_idx]),
                        "created_idx": ob_idx, "impulse_idx": bos_idx, "mitigated": False})
        else:
            ob_idx = None
            for j in range(bos_idx - 1, max(-1, bos_idx - 30), -1):
                if closes[j] > opens[j]:
                    ob_idx = j; break
            if ob_idx is None: continue
            bearish = sum(1 for k in range(ob_idx + 1, bos_idx + 1) if closes[k] < opens[k])
            if bearish < min_impulse_bars: continue
            obs.append({"side": "bear", "top": float(highs[ob_idx]), "bottom": float(lows[ob_idx]),
                        "created_idx": ob_idx, "impulse_idx": bos_idx, "mitigated": False})
    for ob in obs:
        for j in range(ob["impulse_idx"] + 1, len(df)):
            if lows[j] <= ob["top"] and highs[j] >= ob["bottom"]:
                ob["mitigated"] = True
                break
    return obs


def _smc_pois(snap_swings, obs, fvgs, current_idx: int, atr_val: float,
              freshness_bars: int, min_zone_atr_frac: float = 0.3):
    """Build scored POIs (Points of Interest) from OB+FVG confluences + standalones."""
    # Dealing range from last 8 swings
    recent = snap_swings[-8:] if len(snap_swings) >= 8 else snap_swings
    dr_high = max((s[2] for s in recent if s[3] == "high"), default=None)
    dr_low = min((s[2] for s in recent if s[3] == "low"), default=None)
    eq = (dr_high + dr_low) / 2.0 if (dr_high is not None and dr_low is not None) else None

    def _overlap(t1, b1, t2, b2):
        t, b = min(t1, t2), max(b1, b2)
        return (t, b) if t > b else None

    pois = []
    used_obs = set()
    # OB+FVG confluences
    for i_ob, ob in enumerate(obs):
        if ob["mitigated"]: continue
        for fvg in fvgs:
            if fvg["mitigated"] or fvg["side"] != ob["side"]: continue
            ov = _overlap(ob["top"], ob["bottom"], fvg["top"], fvg["bottom"])
            if ov is None: continue
            top, bot = ov
            score = 2; reasons = ["OB+FVG"]
            mid = (top + bot) / 2
            if ob["side"] == "bull" and eq is not None and mid <= eq:
                score += 1; reasons.append("discount")
            if ob["side"] == "bear" and eq is not None and mid >= eq:
                score += 1; reasons.append("premium")
            if (current_idx - max(ob["created_idx"], fvg["created_idx"])) <= freshness_bars:
                score += 1; reasons.append("fresh")
            if atr_val > 0 and (top - bot) >= min_zone_atr_frac * atr_val:
                score += 1; reasons.append("width_ok")
            pois.append({"side": ob["side"], "top": top, "bottom": bot, "score": score,
                         "reasons": reasons,
                         "created_idx": max(ob["created_idx"], fvg["created_idx"])})
            used_obs.add(i_ob)
    # Standalone OBs
    for i_ob, ob in enumerate(obs):
        if ob["mitigated"] or i_ob in used_obs: continue
        score = 1; reasons = ["OB_only"]
        mid = (ob["top"] + ob["bottom"]) / 2
        if ob["side"] == "bull" and eq is not None and mid <= eq:
            score += 1; reasons.append("discount")
        if ob["side"] == "bear" and eq is not None and mid >= eq:
            score += 1; reasons.append("premium")
        if (current_idx - ob["created_idx"]) <= freshness_bars:
            score += 1; reasons.append("fresh")
        if atr_val > 0 and (ob["top"] - ob["bottom"]) >= min_zone_atr_frac * atr_val:
            score += 1; reasons.append("width_ok")
        pois.append({"side": ob["side"], "top": ob["top"], "bottom": ob["bottom"],
                     "score": score, "reasons": reasons, "created_idx": ob["created_idx"]})
    pois.sort(key=lambda p: (p["score"], p["created_idx"]), reverse=True)
    return pois, eq


def evaluate_smc(df15: pd.DataFrame, df1h: pd.DataFrame,
                 params: SMCSignalParams) -> dict | None:
    """SMC strategy: HTF bias from 1H structure -> POIs -> 15m mitigation -> entry."""
    p = params
    be = df15.iloc[-p.max_structure_lookback_bars:] if len(df15) > p.max_structure_lookback_bars else df15
    bt = df1h.iloc[-p.max_structure_lookback_bars:] if len(df1h) > p.max_structure_lookback_bars else df1h
    if len(be) < 60 or len(bt) < 60:
        return None

    swings = _smc_swings(bt, pivot=p.htf_pivot)
    events = _smc_events(bt, swings)
    if not events:
        return None
    bias = events[-1][3]   # last BOS/CHoCH side
    if bias == "none":
        return None

    obs = _smc_obs(bt, events, min_impulse_bars=p.min_impulse_bars)
    fvgs = _smc_fvgs(bt, max_age_bars=200)
    atr_htf = atr(bt["High"], bt["Low"], bt["Close"], p.atr_period)
    atr_htf_val = float(atr_htf.iloc[-1]) if not pd.isna(atr_htf.iloc[-1]) else 0.0

    pois, _eq = _smc_pois(swings, obs, fvgs, current_idx=len(bt) - 1,
                          atr_val=atr_htf_val, freshness_bars=p.poi_freshness_bars)
    side_str = "bull" if bias == "up" else "bear"
    directional = [poi for poi in pois if poi["side"] == side_str]
    if not directional:
        return {"severity": "WATCHLIST", "side": "BUY" if side_str == "bull" else "SELL",
                "price": float(be.iloc[-1]["Close"]), "atr": 0.0,
                "reason": f"HTF {bias} but no directional POIs"}
    good = [poi for poi in directional if poi["score"] >= p.min_poi_score]
    if not good:
        return {"severity": "WATCHLIST", "side": "BUY" if side_str == "bull" else "SELL",
                "price": float(be.iloc[-1]["Close"]), "atr": 0.0,
                "reason": f"POIs exist but max_score<{p.min_poi_score}"}

    last15 = be.iloc[-1]
    price = float(last15["Close"])
    in_poi = [poi for poi in good if poi["bottom"] <= price <= poi["top"]]
    atr_ltf = atr(be["High"], be["Low"], be["Close"], p.atr_period)
    atr_ltf_val = float(atr_ltf.iloc[-1]) if not pd.isna(atr_ltf.iloc[-1]) else 0.0

    if not in_poi:
        nearest = min(good, key=lambda poi: abs(price - (poi["top"] + poi["bottom"]) / 2))
        nearest_mid = (nearest["top"] + nearest["bottom"]) / 2
        if atr_ltf_val > 0 and abs(price - nearest_mid) <= 1.5 * atr_ltf_val:
            return {"severity": "BREAKOUT_WATCH",
                    "side": "BUY" if side_str == "bull" else "SELL",
                    "price": price, "atr": atr_ltf_val,
                    "reason": f"approaching POI score={nearest['score']} ({','.join(nearest['reasons'])})"}
        return {"severity": "WATCHLIST", "side": "BUY" if side_str == "bull" else "SELL",
                "price": price, "atr": atr_ltf_val,
                "reason": f"HTF {bias}: waiting for POI mitigation"}

    active = max(in_poi, key=lambda poi: poi["score"])

    # LTF CHoCH confirmation
    if p.require_ltf_choch:
        ltf_swings = _smc_swings(be, pivot=p.ltf_pivot)
        ltf_events = _smc_events(be, ltf_swings)
        recent_events = [e for e in ltf_events if e[0] >= len(be) - 10]
        confirmed = (recent_events and recent_events[-1][3] == bias
                     and recent_events[-1][2] in ("CHoCH", "BOS"))
        if not confirmed:
            return {"severity": "BREAKOUT_WATCH",
                    "side": "BUY" if side_str == "bull" else "SELL",
                    "price": price, "atr": atr_ltf_val,
                    "reason": f"in POI score={active['score']}, awaiting 15m CHoCH/BOS"}

    # SL/TP
    buf = p.sl_buffer_atr_frac * atr_ltf_val
    if bias == "up":
        sl = active["bottom"] - buf
        future = [s for s in swings if s[3] == "high" and s[2] > price]
        tp = future[0][2] if future else price + 2.5 * atr_ltf_val
        side = "BUY"
    else:
        sl = active["top"] + buf
        future = [s for s in swings if s[3] == "low" and s[2] < price]
        tp = future[0][2] if future else price - 2.5 * atr_ltf_val
        side = "SELL"

    risk = abs(price - sl)
    reward = abs(tp - price)
    if risk <= 0 or (reward / risk) < p.min_rr:
        return {"severity": "SKIPPED", "side": side, "price": price, "atr": atr_ltf_val,
                "reason": f"rr_too_low ({reward/max(risk,1e-9):.2f})",
                "rejection_reason": "rr_too_low"}

    return {
        "severity": "BUY_READY" if side == "BUY" else "SELL_READY",
        "side": side, "price": price, "atr": atr_ltf_val,
        "reason": (f"POI mitigated + 15m bias-confirm in {bias}, "
                   f"score={active['score']} ({','.join(active['reasons'])})"),
        "sl_suggested": sl, "tp_suggested": tp,
        "rr": reward / risk, "poi_score": active["score"], "htf_bias": bias,
    }
