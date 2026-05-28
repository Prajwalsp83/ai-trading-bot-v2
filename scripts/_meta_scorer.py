"""
Phase 5e — Live ML scoring for the bots.

Loads the trained model (models/meta_labeler.pkl + .meta.json) and scores
signals at live time. Bots call score_signal_live() right before firing a
trade. In shadow mode (default), the bot trades regardless of ML; the
decision is logged for analysis. Once shadow performance is validated, flip
ML_SHADOW_MODE=false in .env and ML can veto trades.

Graceful failure: if the model isn't present (e.g. first-time deploy before
training), score_signal_live() returns ScoreResult(probability=None,
would_trade=True, note='no_model') — bot trades normally.

This module mirrors the feature extraction in generate_ml_dataset.py so
live features match training features exactly.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================== CONFIG ==============================
HERE = Path(__file__).resolve().parent.parent          # project root
MODEL_PATH = HERE / "models" / "meta_labeler.pkl"
META_PATH = HERE / "models" / "meta_labeler.meta.json"

IST = timezone(timedelta(hours=5, minutes=30))


# ============================== STATE ===============================
_MODEL = None        # lazy loaded
_META = None
_LOAD_ERR = None
_MODEL_MTIME = None  # for hot-reload


def _is_shadow_mode() -> bool:
    """Default: shadow (True). Bots ignore ML veto. Flip via env var when ready."""
    return os.getenv("ML_SHADOW_MODE", "true").lower() in ("true", "1", "yes", "on")


def _load_model_if_needed():
    """Lazy + hot-reload: re-loads if the .pkl file mtime changes."""
    global _MODEL, _META, _LOAD_ERR, _MODEL_MTIME
    if not MODEL_PATH.exists():
        _MODEL, _META = None, None
        _LOAD_ERR = "model_not_found"
        return
    try:
        mtime = MODEL_PATH.stat().st_mtime
        if _MODEL is not None and _MODEL_MTIME == mtime:
            return  # already loaded, file unchanged
        with open(MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)
        with open(META_PATH, "r") as f:
            _META = json.load(f)
        _MODEL_MTIME = mtime
        _LOAD_ERR = None
        print(f"[meta_scorer] loaded model "
              f"(framework={_META.get('framework', '?')}, "
              f"val_auc={_META.get('val_auc', 0):.4f}, "
              f"threshold={_META.get('chosen_threshold', 0):.2f})", flush=True)
    except Exception as e:
        _MODEL, _META = None, None
        _LOAD_ERR = f"load_error: {type(e).__name__}: {e}"
        print(f"[meta_scorer] {_LOAD_ERR}", flush=True)


# ============================== RESULT ==============================
@dataclass
class ScoreResult:
    """Returned by score_signal_live().

    Bot uses `would_trade` to decide whether to fire (only when not in shadow).
    `probability`, `threshold`, `note` always populated for logging."""
    probability: float | None        # P(WIN) from model, None if no model
    threshold: float | None          # threshold above which would_trade=True
    would_trade: bool                # True if prob >= threshold (or no model)
    shadow_mode: bool                # whether the bot will actually respect this decision
    note: str                        # human-readable explanation

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scored_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return d


# ============================== FEATURES ============================
def _in_session(ts_utc) -> str:
    """Return session label — must match generator's in_session()."""
    if not hasattr(ts_utc, "tz_convert"):
        ts_utc = pd.Timestamp(ts_utc).tz_localize("UTC") if not getattr(ts_utc, "tz", None) else ts_utc
    ts_ist = ts_utc.tz_convert(IST) if ts_utc.tz else ts_utc.tz_localize("UTC").tz_convert(IST)
    h, m = ts_ist.hour, ts_ist.minute
    t = h * 60 + m
    if 12 * 60 + 30 <= t <= 16 * 60 + 30: return "London"
    if 18 * 60 <= t <= 21 * 60: return "NY_overlap"
    if 21 * 60 <= t <= 23 * 60 + 30: return "NY_afternoon"
    return "outside"


def _regime_tag(adx_val: float, ema_up: bool) -> str:
    if adx_val >= 25 and ema_up: return "trend_up"
    if adx_val >= 25 and not ema_up: return "trend_down"
    if adx_val <= 20: return "chop"
    return "transition"


def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _atr(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx_di(high, low, close, period=14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn.clip(lower=0)
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.fillna(0).ewm(alpha=1 / period, adjust=False).mean(), plus_di.fillna(0), minus_di.fillna(0)


def extract_features_live(df15: pd.DataFrame, df1h: pd.DataFrame,
                           side: str, strategy: str,
                           rr_target: float | None = None) -> dict:
    """Build the same feature vector as training (39 cols).

    df15 and df1h: time-indexed OHLCV. Last row = current bar.
    side: BUY or SELL. strategy: breakout or smc.
    rr_target: optional; if None, computed from k_sl/k_tp standard ratios."""
    if len(df15) < 250 or len(df1h) < 30:
        return {}

    # Indicators on full series — call extracts last row only
    close15 = df15["Close"]
    high15 = df15["High"]
    low15 = df15["Low"]
    open15 = df15["Open"]
    ema_f = _ema(close15, 50)
    ema_s = _ema(close15, 200)
    atr_14 = _atr(high15, low15, close15, 14)
    atr_pct = atr_14.rolling(100).rank(pct=True)
    adx_v, di_p, di_m = _adx_di(high15, low15, close15, 14)
    atr_5 = _atr(high15, low15, close15, 5)
    atr_50 = _atr(high15, low15, close15, 50)
    sma_20 = close15.rolling(20).mean()
    std_20 = close15.rolling(20).std()
    vol_of_vol_s = atr_14.rolling(20).std()

    close1h = df1h["Close"]
    ema_f_1h = _ema(close1h, 50)
    ema_s_1h = _ema(close1h, 200)

    last_idx = len(df15) - 1
    bar_ts = df15.index[-1]
    last_open = float(open15.iloc[-1])
    last_high = float(high15.iloc[-1])
    last_low = float(low15.iloc[-1])
    last_close = float(close15.iloc[-1])
    prev_high = float(high15.iloc[-2]) if last_idx >= 1 else last_high
    prev_low = float(low15.iloc[-2]) if last_idx >= 1 else last_low
    bar_range = last_high - last_low

    def _v(s, i=-1, default=0.0):
        v = s.iloc[i]
        return float(v) if v == v else default  # NaN check

    ema_f_v = _v(ema_f)
    ema_s_v = _v(ema_s)
    atr_v = _v(atr_14, default=1.0)
    atr_pct_v = _v(atr_pct, default=0.5)
    adx_val = _v(adx_v)
    di_p_v = _v(di_p)
    di_m_v = _v(di_m)
    atr_5_v = _v(atr_5)
    atr_50_v = _v(atr_50)
    sma_20_v = _v(sma_20, default=last_close)
    std_20_v = _v(std_20)
    vol_of_vol = _v(vol_of_vol_s)
    ema_f_h_v = _v(ema_f_1h)
    ema_s_h_v = _v(ema_s_1h)

    # Streak
    streak = 0
    for k in range(last_idx, max(-1, last_idx - 11), -1):
        if close15.iloc[k] > open15.iloc[k]:
            if streak < 0: break
            streak += 1
        elif close15.iloc[k] < open15.iloc[k]:
            if streak > 0: break
            streak -= 1
        else:
            break

    # Rolling 5/20 highs/lows
    high_5 = float(high15.iloc[-5:].max()) if last_idx >= 4 else last_high
    low_5 = float(low15.iloc[-5:].min()) if last_idx >= 4 else last_low
    high_20 = float(high15.iloc[-20:].max()) if last_idx >= 19 else last_high
    low_20 = float(low15.iloc[-20:].min()) if last_idx >= 19 else last_low

    bb_pos = (last_close - sma_20_v) / (std_20_v * 2) if std_20_v > 0 else 0.0
    bb_range = std_20_v * 4 if std_20_v > 0 else 0.0
    breakout_strength = max(
        (last_close - prev_high) / atr_v if atr_v > 0 else 0.0,
        (prev_low - last_close) / atr_v if atr_v > 0 else 0.0,
    )

    if rr_target is None:
        rr_target = 1.67 if strategy == "breakout" else 1.50

    feats = {
        "strategy": strategy,
        "side": side,
        "hour_utc": bar_ts.hour if hasattr(bar_ts, "hour") else 0,
        "dow": bar_ts.dayofweek if hasattr(bar_ts, "dayofweek") else 0,
        "session": _in_session(bar_ts),
        "atr_raw": atr_v,
        "atr_pct": atr_pct_v,
        "range_5": high_5 - low_5,
        "range_20": high_20 - low_20,
        "ema_dist_15": (ema_f_v - ema_s_v) / ema_s_v if ema_s_v else 0.0,
        "px_vs_ema50": (last_close - ema_f_v) / ema_f_v if ema_f_v else 0.0,
        "adx": adx_val,
        "di_plus": di_p_v,
        "di_minus": di_m_v,
        "ema_dist_1h": (ema_f_h_v - ema_s_h_v) / ema_s_h_v if ema_s_h_v else 0.0,
        "px_vs_ema50_1h": (last_close - ema_f_h_v) / ema_f_h_v if ema_f_h_v else 0.0,
        "ret_1": float(close15.pct_change().iloc[-1]) if last_idx >= 1 else 0.0,
        "ret_5": (last_close / close15.iloc[-5] - 1) if last_idx >= 5 else 0.0,
        "ret_20": (last_close / close15.iloc[-20] - 1) if last_idx >= 20 else 0.0,
        "high_5_dist": (high_5 - last_close) / last_close if last_close else 0.0,
        "low_5_dist": (last_close - low_5) / last_close if last_close else 0.0,
        "high_20_dist": (high_20 - last_close) / last_close if last_close else 0.0,
        "low_20_dist": (last_close - low_20) / last_close if last_close else 0.0,
        "atr_5_50_ratio": atr_5_v / atr_50_v if atr_50_v > 0 else 1.0,
        "body_pct": abs(last_close - last_open) / bar_range if bar_range > 0 else 0.0,
        "upper_wick_pct": (last_high - max(last_open, last_close)) / bar_range if bar_range > 0 else 0.0,
        "lower_wick_pct": (min(last_open, last_close) - last_low) / bar_range if bar_range > 0 else 0.0,
        "range_to_atr": bar_range / atr_v if atr_v > 0 else 1.0,
        "dist_from_ema200": (last_close - ema_s_v) / ema_s_v if ema_s_v else 0.0,
        "dist_from_ema200_1h": (last_close - ema_s_h_v) / ema_s_h_v if ema_s_h_v else 0.0,
        "bb_position": float(bb_pos),
        "bb_width_pct": float(bb_range / sma_20_v) if sma_20_v > 0 else 0.0,
        "roc_10": (last_close / close15.iloc[-10] - 1) if last_idx >= 10 else 0.0,
        "consec_streak": int(streak),
        "vol_of_vol": vol_of_vol,
        "breakout_strength": float(breakout_strength),
        "regime": _regime_tag(adx_val, ema_f_v > ema_s_v),
        "rr_target": float(rr_target),
    }
    return feats


# ============================== SCORE ===============================
def score_signal_live(df15: pd.DataFrame, df1h: pd.DataFrame,
                       side: str, strategy: str,
                       rr_target: float | None = None) -> ScoreResult:
    """Score a signal. Bot calls this right before firing.

    Returns ScoreResult — bot uses .would_trade to gate live entries, but in
    shadow mode (default) trades regardless. The full result should always
    be logged via _journal.record_signal extras for offline analysis."""
    _load_model_if_needed()
    shadow = _is_shadow_mode()

    if _MODEL is None:
        return ScoreResult(
            probability=None, threshold=None,
            would_trade=True,                  # no model = no veto
            shadow_mode=shadow,
            note=f"no_model: {_LOAD_ERR or 'unknown'}",
        )

    feats = extract_features_live(df15, df1h, side, strategy, rr_target)
    if not feats:
        return ScoreResult(
            probability=None, threshold=_META.get("chosen_threshold"),
            would_trade=True,                  # broken features = no veto
            shadow_mode=shadow,
            note="not_enough_history",
        )

    # Build feature row in the exact training order
    feature_order = _META.get("features", [])
    row = {k: feats.get(k) for k in feature_order}
    X = pd.DataFrame([row])

    try:
        prob = float(_MODEL.predict(X)[0])
    except Exception as e:
        return ScoreResult(
            probability=None, threshold=_META.get("chosen_threshold"),
            would_trade=True,
            shadow_mode=shadow,
            note=f"predict_error: {type(e).__name__}: {str(e)[:100]}",
        )

    thr = float(_META.get("chosen_threshold", 0.5))
    would = prob >= thr
    note = f"prob={prob:.3f} threshold={thr:.3f} {'FIRE' if would else 'VETO'}"

    return ScoreResult(
        probability=prob, threshold=thr,
        would_trade=bool(would),
        shadow_mode=shadow,
        note=note,
    )
