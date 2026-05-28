"""
Parameter sweep over breakout_trend — FAST version.

Why this is faster than v1:
  - Indicators are computed ONCE per (ema_fast, ema_slow) pair, not per bar.
  - Inner loop uses numpy arrays instead of pandas iloc (~50x faster).
  - Grid trimmed to the parameters that actually move the needle.

Trade-off: this script duplicates the breakout decision logic from
app/strategy/breakout_trend.py for speed. Both must stay in sync — when
you change the strategy, also update _bar_signal() below. The live bot
uses the class version (slow but correct); the sweep uses this fast
version for research only.

Score formula (deliberately punishes big drawdowns):
    score = (return_pct × win_rate) / max(max_dd_pct, 1.0)
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from app.indicators.trend import ema                              # noqa: E402
from app.indicators.volatility import atr, atr_percentile         # noqa: E402

warnings.filterwarnings("ignore")

SYMBOL = "GC=F"
INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.02
WARMUP_BARS = 220

# ---- grid (108 combos; runs in ~30s) ----
# EMAs fixed to the classic 50/200 since sweeping them is a separate concern.
GRID_EMA_PAIRS = [(50, 200)]
GRID = {
    "atr_min":            [5.0, 10.0],
    "k_sl":               [1.0, 1.5, 2.0],
    "k_tp":               [2.5, 3.0, 4.0],
    "atr_pct_min":        [0.0, 0.3, 0.5],
    "min_trend_strength": [0.0, 0.005, 0.010],
}


@dataclass
class RunResult:
    params: dict
    trades: int
    wins: int
    win_rate: float
    avg_r: float
    total_return_pct: float
    max_dd_pct: float
    profit_factor: float
    score: float


def fetch_once() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Downloading {SYMBOL} 15m and 1h…")
    df15 = yf.download(SYMBOL, interval="15m", period="60d", auto_adjust=True, progress=False)
    df1h = yf.download(SYMBOL, interval="1h", period="730d", auto_adjust=True, progress=False)
    for df in (df15, df1h):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    return df15.dropna(), df1h.dropna()


def precompute(df15: pd.DataFrame, df1h: pd.DataFrame,
               ema_fast: int, ema_slow: int) -> dict:
    """Compute indicator series once for this (ema_fast, ema_slow) pair.
    Returns a dict of numpy arrays for the inner loop to index into."""
    close15 = df15["Close"].to_numpy(dtype=float)
    high15 = df15["High"].to_numpy(dtype=float)
    low15 = df15["Low"].to_numpy(dtype=float)

    ema_f_e = ema(df15["Close"], ema_fast).to_numpy(dtype=float)
    ema_s_e = ema(df15["Close"], ema_slow).to_numpy(dtype=float)
    atr_e_s = atr(df15["High"], df15["Low"], df15["Close"], 14)
    atr_e = atr_e_s.to_numpy(dtype=float)
    atr_pct_e = atr_percentile(atr_e_s, 100).to_numpy(dtype=float)

    # trend strength on 15m TF: normalised EMA separation
    with np.errstate(divide="ignore", invalid="ignore"):
        trend_str = np.abs(ema_f_e - ema_s_e) / ema_s_e

    ema_f_t = ema(df1h["Close"], ema_fast).to_numpy(dtype=float)
    ema_s_t = ema(df1h["Close"], ema_slow).to_numpy(dtype=float)

    # For each 15m timestamp, the latest 1h bar at-or-before it.
    m_idx = df15.index.values.astype("datetime64[ns]")
    h_idx = df1h.index.values.astype("datetime64[ns]")
    h_pos_for_m = np.searchsorted(h_idx, m_idx, side="right") - 1

    return {
        "close15": close15, "high15": high15, "low15": low15,
        "ema_f_e": ema_f_e, "ema_s_e": ema_s_e,
        "atr_e": atr_e, "atr_pct_e": atr_pct_e, "trend_str": trend_str,
        "ema_f_t": ema_f_t, "ema_s_t": ema_s_t,
        "h_pos_for_m": h_pos_for_m,
    }


def run_combo(pre: dict, atr_min: float, k_sl: float, k_tp: float,
              atr_pct_min: float, min_trend_strength: float) -> RunResult:
    """Bar-by-bar paper backtest using precomputed indicators."""
    n = len(pre["close15"])
    close15 = pre["close15"]
    high15 = pre["high15"]
    low15 = pre["low15"]
    ema_f_e = pre["ema_f_e"]
    ema_s_e = pre["ema_s_e"]
    atr_e = pre["atr_e"]
    atr_pct_e = pre["atr_pct_e"]
    trend_str = pre["trend_str"]
    ema_f_t = pre["ema_f_t"]
    ema_s_t = pre["ema_s_t"]
    h_for_m = pre["h_pos_for_m"]

    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    pnls: list[float] = []
    rs: list[float] = []
    wins = 0

    pos_side = None        # "BUY" / "SELL" / None
    pos_entry = pos_sl = pos_tp = pos_qty = 0.0

    for i in range(WARMUP_BARS, n):
        # ---- manage open position ----
        if pos_side is not None:
            h = high15[i]; l = low15[i]
            exit_price = None
            if pos_side == "BUY":
                if l <= pos_sl:   exit_price = pos_sl
                elif h >= pos_tp: exit_price = pos_tp
            else:
                if h >= pos_sl:   exit_price = pos_sl
                elif l <= pos_tp: exit_price = pos_tp
            if exit_price is not None:
                pnl = ((exit_price - pos_entry) if pos_side == "BUY"
                        else (pos_entry - exit_price)) * pos_qty
                stop_dist = abs(pos_entry - pos_sl)
                r = pnl / (stop_dist * pos_qty) if stop_dist > 0 else 0.0
                equity += pnl
                pnls.append(pnl); rs.append(r)
                if pnl > 0: wins += 1
                pos_side = None

        if equity > peak: peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd: max_dd = dd

        # ---- look for new entry ----
        if pos_side is None and i > 0:
            a = atr_e[i]
            if np.isnan(a) or a < atr_min: continue
            if atr_pct_min > 0:
                if np.isnan(atr_pct_e[i]) or atr_pct_e[i] < atr_pct_min: continue
            if min_trend_strength > 0:
                if np.isnan(trend_str[i]) or trend_str[i] < min_trend_strength: continue
            ef = ema_f_e[i]; es = ema_s_e[i]
            if np.isnan(ef) or np.isnan(es): continue

            j = h_for_m[i]
            if j < 0: continue
            eft = ema_f_t[j]; est = ema_s_t[j]
            if np.isnan(eft) or np.isnan(est): continue

            prev_high = high15[i - 1]
            prev_low = low15[i - 1]
            cur_high = high15[i]
            cur_low = low15[i]

            long_ready = (ef > es) and (eft > est) and (cur_high > prev_high)
            short_ready = (ef < es) and (eft < est) and (cur_low < prev_low)

            entry = close15[i]
            if long_ready:
                sl = entry - k_sl * a
                tp = entry + k_tp * a
                stop_dist = entry - sl
                if stop_dist > 0:
                    pos_side = "BUY"; pos_entry = entry; pos_sl = sl; pos_tp = tp
                    pos_qty = (equity * RISK_PER_TRADE) / stop_dist
            elif short_ready:
                sl = entry + k_sl * a
                tp = entry - k_tp * a
                stop_dist = sl - entry
                if stop_dist > 0:
                    pos_side = "SELL"; pos_entry = entry; pos_sl = sl; pos_tp = tp
                    pos_qty = (equity * RISK_PER_TRADE) / stop_dist

    # ---- stats ----
    n_tr = len(pnls)
    win_rate = (wins / n_tr * 100) if n_tr else 0.0
    avg_r = (sum(rs) / n_tr) if n_tr else 0.0
    ret_pct = (equity / INITIAL_EQUITY - 1) * 100
    win_pnl = sum(x for x in pnls if x > 0)
    loss_pnl = abs(sum(x for x in pnls if x < 0))
    pf = (win_pnl / loss_pnl) if loss_pnl > 0 else float("inf")
    dd_pct = max_dd * 100
    score = (ret_pct * win_rate) / max(dd_pct, 1.0)

    return RunResult(
        params={}, trades=n_tr, wins=wins, win_rate=win_rate, avg_r=avg_r,
        total_return_pct=ret_pct, max_dd_pct=dd_pct,
        profit_factor=pf, score=score,
    )


def main():
    df15, df1h = fetch_once()
    print(f"  15m bars: {len(df15)}   1h bars: {len(df1h)}")

    keys = list(GRID.keys())
    inner_combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    total = len(GRID_EMA_PAIRS) * len(inner_combos)
    print(f"Running {total} parameter combinations (precomputed, numpy)…\n")

    t0 = time.time()
    results: list[RunResult] = []
    done = 0
    for ema_fast, ema_slow in GRID_EMA_PAIRS:
        pre = precompute(df15, df1h, ema_fast, ema_slow)
        for combo in inner_combos:
            try:
                r = run_combo(pre, combo["atr_min"], combo["k_sl"], combo["k_tp"],
                              combo["atr_pct_min"], combo["min_trend_strength"])
                r.params = {
                    "ema_fast": ema_fast, "ema_slow": ema_slow, **combo,
                }
                results.append(r)
            except Exception as e:
                print(f"  combo failed: {e}")
            done += 1
            if done % 25 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"  {done}/{total} done  ({elapsed:.1f}s elapsed)")

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # need >= 10 trades to be meaningful
    viable = [r for r in results if r.trades >= 10]
    viable.sort(key=lambda r: r.score, reverse=True)

    print("\n" + "=" * 88)
    print(" TOP 10 PARAMETER COMBINATIONS (score = return × win_rate ÷ max_dd)")
    print("=" * 88)
    header = f" {'#':>2}  {'ret%':>7} {'dd%':>6} {'win%':>5} {'PF':>5} {'avgR':>6} {'trades':>7} {'score':>7}"
    print(header)
    print(" " + "-" * (len(header) - 1))

    for i, r in enumerate(viable[:10], 1):
        print(f" {i:>2}  {r.total_return_pct:>+6.2f}% {r.max_dd_pct:>5.1f}% "
              f"{r.win_rate:>4.1f}% {r.profit_factor:>4.2f} "
              f"{r.avg_r:>+5.2f} {r.trades:>7d} {r.score:>7.1f}")
        p = r.params
        print(f"      atr_min={p['atr_min']}  ema={p['ema_fast']}/{p['ema_slow']}  "
              f"k_sl={p['k_sl']}  k_tp={p['k_tp']}  "
              f"atr_pct_min={p['atr_pct_min']}  trend_str={p['min_trend_strength']}")

    # full results CSV
    df = pd.DataFrame([{
        **r.params,
        "trades": r.trades, "win_rate": r.win_rate, "avg_r": r.avg_r,
        "total_return_pct": r.total_return_pct, "max_dd_pct": r.max_dd_pct,
        "profit_factor": r.profit_factor, "score": r.score,
    } for r in results])
    df = df.sort_values("score", ascending=False)
    out = HERE / "data" / "sweep_results.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Full results saved -> {out}")

    if viable:
        best = viable[0]
        print(f"\n  Best config:")
        for k, v in best.params.items():
            print(f"    {k:18s}: {v}")
        print(f"\n  Expected: {best.total_return_pct:+.1f}% return, "
              f"{best.max_dd_pct:.1f}% max DD over 60d, "
              f"{best.trades} trades")
        if best.max_dd_pct > 20:
            print(f"\n  WARNING: best DD ({best.max_dd_pct:.1f}%) still high for small account.")
            print(f"  Drop risk_per_trade in config.yaml from 0.02 to 0.01 for safety.")


if __name__ == "__main__":
    main()
