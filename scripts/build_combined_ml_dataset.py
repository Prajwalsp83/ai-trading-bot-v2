"""
Phase G.1 — Build combined ML training dataset from backtest trade logs.

For every trade in every *_trades.parquet under data/backtests/:
  - Load OHLC history (M15 + H1) up to the trade's open_time (no lookahead)
  - Extract features at entry: regime, RSI, ATR%, EMA distance, vol, session, etc
  - Get label from exit_reason: WIN if TP, LOSS otherwise (skip TIMEOUT)
  - Tag with strategy_name so the model can learn per-strategy bias

Output:
  data/ml_dataset_combined.parquet — N labeled samples ready for training.

Run:
    python scripts/build_combined_ml_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))

from _strategies import ema, atr, atr_percentile, rsi


IST = pd.Timedelta(hours=5, minutes=30)


def _session(ts: pd.Timestamp) -> str:
    ist = ts + IST
    h, m = ist.hour, ist.minute
    t = h * 60 + m
    if 12 * 60 + 30 <= t <= 16 * 60 + 30: return "London"
    if 18 * 60 <= t <= 21 * 60: return "NY_overlap"
    if 21 * 60 <= t <= 23 * 60 + 30: return "NY_afternoon"
    return "outside"


def _wilder_ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high, low, close, period=14):
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
    return (_wilder_ema(dx.fillna(0), period).iloc[-1],
            plus_di.fillna(0).iloc[-1],
            minus_di.fillna(0).iloc[-1])


def _regime(adx_v: float, ema_up: bool) -> str:
    if adx_v >= 25 and ema_up: return "trend_up"
    if adx_v >= 25 and not ema_up: return "trend_down"
    if adx_v <= 20: return "chop"
    return "transition"


def extract_features_for_trade(df15: pd.DataFrame, df1h: pd.DataFrame,
                                open_time: pd.Timestamp) -> dict | None:
    """Extract features at the moment the trade opened (no lookahead)."""
    # Window 15m bars to ONLY those at or before open_time
    df15_slice = df15[df15.index <= open_time].tail(500)
    df1h_slice = df1h[df1h.index <= open_time].tail(300)
    if len(df15_slice) < 250 or len(df1h_slice) < 30:
        return None

    close = df15_slice["Close"]
    high = df15_slice["High"]
    low = df15_slice["Low"]
    last_close = float(close.iloc[-1])

    ema_f = ema(close, 50)
    ema_s = ema(close, 200)
    atr_v = float(atr(high, low, close, 14).iloc[-1])
    atr_pct = float(atr_percentile(atr(high, low, close, 14), 100).iloc[-1])
    adx_v, dip, dim = _adx(high, low, close, 14)
    rsi_v = float(rsi(close, 14).iloc[-1])

    # 1H view
    close1h = df1h_slice["Close"]
    ema_f_1h = ema(close1h, 50)
    ema_s_1h = ema(close1h, 200)

    ema_f_v = float(ema_f.iloc[-1])
    ema_s_v = float(ema_s.iloc[-1])
    ema_f_h_v = float(ema_f_1h.iloc[-1])
    ema_s_h_v = float(ema_s_1h.iloc[-1])

    # Bar shape / recent action
    last_open_px = float(df15_slice["Open"].iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    bar_range = last_high - last_low

    return {
        # Time
        "hour_utc": int(open_time.hour),
        "dow": int(open_time.dayofweek),
        "session": _session(open_time),

        # Volatility
        "atr_raw": atr_v,
        "atr_pct": atr_pct if not np.isnan(atr_pct) else 0.5,
        "range_to_atr": bar_range / atr_v if atr_v > 0 else 1.0,

        # Trend / momentum
        "ema_dist_15": (ema_f_v - ema_s_v) / ema_s_v if ema_s_v else 0.0,
        "px_vs_ema50": (last_close - ema_f_v) / ema_f_v if ema_f_v else 0.0,
        "px_vs_ema200": (last_close - ema_s_v) / ema_s_v if ema_s_v else 0.0,
        "ema_dist_1h": (ema_f_h_v - ema_s_h_v) / ema_s_h_v if ema_s_h_v else 0.0,
        "px_vs_ema50_1h": (last_close - ema_f_h_v) / ema_f_h_v if ema_f_h_v else 0.0,
        "adx": adx_v if not np.isnan(adx_v) else 0.0,
        "di_plus": dip if not np.isnan(dip) else 0.0,
        "di_minus": dim if not np.isnan(dim) else 0.0,
        "rsi": rsi_v if not np.isnan(rsi_v) else 50.0,

        # Recent action
        "ret_1":  float(close.pct_change().iloc[-1]) if len(close) > 1 else 0.0,
        "ret_5":  float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0.0,
        "ret_20": float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0.0,

        # Regime tag (matches live regime classifier)
        "regime": _regime(adx_v if not np.isnan(adx_v) else 0.0, ema_f_v > ema_s_v),
    }


def main() -> int:
    backtest_dir = HERE / "data" / "backtests"
    history_dir = HERE / "data" / "history"

    df15 = pd.read_parquet(history_dir / "GOLD_i_M15.parquet")
    df1h = pd.read_parquet(history_dir / "GOLD_i_H1.parquet")
    df15.index = pd.to_datetime(df15.index, utc=True)
    df1h.index = pd.to_datetime(df1h.index, utc=True)
    print(f"History loaded: M15={len(df15):,}, H1={len(df1h):,}")

    # Discover trade logs (skip sweeps — too duplicative with baseline)
    trade_files = sorted(backtest_dir.rglob("*_trades.parquet"))
    print(f"\nFound {len(trade_files)} trade log file(s):")
    for tf in trade_files:
        print(f"  - {tf.relative_to(HERE)}")

    rows = []
    skipped_no_label = skipped_no_features = 0

    for tf in trade_files:
        # Strategy name from filename
        name = tf.stem.replace("_trades", "")
        # Map filename patterns to strategy buckets
        if name.startswith("breakout"):
            strategy = "breakout"
        elif "sweep_aggressive_all" in name:
            strategy = "smc_aggressive"
        elif name.startswith("sweep_baseline") or name.startswith("sweep_rr") or \
             name.startswith("sweep_score"):
            strategy = "smc_baseline"
        elif name.startswith("smc"):
            strategy = "smc"
        elif "liquidity_sweep" in name:
            strategy = "liquidity_sweep"
        elif "mean_reversion" in name:
            strategy = "mean_reversion"
        else:
            strategy = name

        try:
            t = pd.read_parquet(tf)
        except Exception as e:
            print(f"  skip {tf.name}: {e}")
            continue
        if len(t) == 0:
            continue
        t["open_time"] = pd.to_datetime(t["open_time"], utc=True)
        t["close_time"] = pd.to_datetime(t["close_time"], utc=True)

        n_before = len(rows)
        for _, tr in t.iterrows():
            # Label: WIN if TP, LOSS if SL. Drop OTHER/TIMEOUT.
            reason = str(tr.get("exit_reason", ""))
            if reason == "TP":
                label = 1
            elif reason == "SL":
                label = 0
            else:
                skipped_no_label += 1
                continue

            feats = extract_features_for_trade(df15, df1h, tr["open_time"])
            if feats is None:
                skipped_no_features += 1
                continue

            feats.update({
                "strategy": strategy,
                "side": tr.get("side", "BUY"),
                "rr_at_entry": float(tr.get("rr", tr.get("r_realised", 0)) if tr.get("rr") else 1.67),
                "label": label,
            })
            rows.append(feats)
        print(f"  {tf.name}: added {len(rows) - n_before} samples")

    if not rows:
        print("ERROR: no samples produced")
        return 1

    df = pd.DataFrame(rows)
    out = HERE / "data" / "ml_dataset_combined.parquet"
    df.to_parquet(out)

    print(f"\n=== Combined dataset saved: {out} ===")
    print(f"  Samples: {len(df):,}  (skipped: no_label={skipped_no_label}, no_feats={skipped_no_features})")
    print(f"  WIN rate overall: {df['label'].mean()*100:.1f}%")
    print(f"\n  By strategy:")
    by_strat = df.groupby("strategy").agg(
        n=("label", "size"), win_rate=("label", "mean")
    ).sort_values("n", ascending=False)
    print(by_strat.to_string())
    print(f"\n  By regime:")
    print(df.groupby("regime")["label"].agg(["size", "mean"]).to_string())
    print(f"\n  Features: {[c for c in df.columns if c != 'label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
