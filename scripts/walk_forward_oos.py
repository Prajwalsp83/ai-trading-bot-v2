"""
PHASE 3 - True out-of-sample walk-forward optimization.

The old walk_forward.py bins trades from ONE full-history backtest whose params
(aggressive_all) were chosen by sweeping that same history. Everything is
in-sample, so it cannot reveal over-fitting. This script does the honest thing:

  1. Split history into rolling (train, test) window pairs.
  2. On the TRAIN window only, run the param sweep and pick the winner by an
     in-sample metric (default: total R).
  3. Apply that winner's params to the immediately-following TEST window.
  4. Concatenate the TEST (out-of-sample) trade segments across all windows.
  5. Report OOS metrics (total R, PF, max DD in R, win rate) per window and in
     aggregate, side by side with the in-sample winner metrics so degradation
     is visible.

No look-ahead: each test window is scored with params chosen only from data
that precedes it. A warmup buffer of past bars before each window start keeps
indicators (EMA-200 on H1, ATR percentile) warm without leaking future data.

Reuses the existing engine, the sweep's variant grid, and walk_forward's
R-based metric helpers -- no duplicated strategy logic.

Run on the VPS (history parquets live there):
    python scripts/walk_forward_oos.py
    python scripts/walk_forward_oos.py --train-months 12 --test-months 6
    python scripts/walk_forward_oos.py --select-metric profit_factor

Outputs:
    data/backtests/walk_forward_oos_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))

from _backtest_engine import BacktestEngine, BacktestParams, CostModel, SymbolSpecs
from _strategies import evaluate_smc
from sweep_smc import variants as smc_variants, _load_history, _trim
from walk_forward import _window_metrics, _max_drawdown_r

WARMUP_BUFFER_DAYS = 30   # past bars fed before each window for indicator warmup


# ============================ CORE ==================================
def _slice_for_window(df15: pd.DataFrame, df1h: pd.DataFrame,
                      win_start: pd.Timestamp, win_end: pd.Timestamp):
    """Slice df15/df1h to [win_start - buffer, win_end] and return the slices
    plus the count of M15 warmup bars (those strictly before win_start), so the
    engine begins evaluating exactly at win_start with warm indicators."""
    buf_start = win_start - pd.Timedelta(days=WARMUP_BUFFER_DAYS)
    df15_s = df15[(df15.index >= buf_start) & (df15.index <= win_end)]
    df1h_s = df1h[(df1h.index >= buf_start) & (df1h.index <= win_end)]
    warmup = int((df15_s.index < win_start).sum())
    return df15_s, df1h_s, warmup


def _run_window(df15, df1h, specs, strategy_fn, params,
                win_start, win_end, equity, risk_pct, poll_every) -> pd.DataFrame:
    """Run one strategy config over one window. Returns the trades whose entry
    (open_time) falls inside [win_start, win_end] -- the warmup-buffer trades
    are excluded so each window's results are clean."""
    df15_s, df1h_s, warmup = _slice_for_window(df15, df1h, win_start, win_end)
    if len(df15_s) <= warmup + 5:
        return pd.DataFrame()
    engine = BacktestEngine(specs, cost=CostModel(), params=BacktestParams(
        starting_equity=equity, risk_per_trade_pct=risk_pct,
        warmup_bars=max(warmup, 1), poll_every_bars=poll_every,
    ))
    res = engine.run(df15_s, df1h_s, None, strategy_fn, params, "smc")
    t = res.trades
    if len(t) == 0:
        return t
    t = t.copy()
    t["open_time"] = pd.to_datetime(t["open_time"], utc=True)
    return t[(t["open_time"] >= win_start) & (t["open_time"] < win_end)].reset_index(drop=True)


def _select_value(metrics: dict, key: str) -> float:
    """Pull a comparable scalar from a _window_metrics dict (PF 'inf' -> large)."""
    v = metrics.get(key, 0.0)
    if v == "inf":
        return float("inf")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def walk_forward_oos(df15, df1h, specs, strategy_fn, variant_list, *,
                     train_months: int, test_months: int,
                     equity: float, risk_pct: float, poll_every: int,
                     select_metric: str = "total_r", min_is_trades: int = 5) -> dict:
    """Rolling-window OOS walk-forward. variant_list: list of (name, params)."""
    first = df15.index[0]
    last = df15.index[-1]
    # First test window starts after one full train window of history.
    test_start = (pd.Timestamp(year=first.year, month=first.month, day=1, tz="UTC")
                  + pd.DateOffset(months=train_months))

    windows = []
    cur = test_start
    while cur < last:
        nxt = cur + pd.DateOffset(months=test_months)
        windows.append((cur, min(nxt, last)))
        cur = nxt

    per_window = []
    oos_frames = []
    for test_s, test_e in windows:
        train_s = test_s - pd.DateOffset(months=train_months)
        train_e = test_s

        # --- in-sample: score every variant on the train window, pick winner ---
        is_results = []
        for name, params in variant_list:
            tr = _run_window(df15, df1h, specs, strategy_fn, params,
                             train_s, train_e, equity, risk_pct, poll_every)
            m = _window_metrics(tr)
            is_results.append((name, params, m))
        # qualified = enough in-sample trades to be meaningful
        qualified = [r for r in is_results if r[2]["trades"] >= min_is_trades]
        pool = qualified or is_results
        winner = max(pool, key=lambda r: _select_value(r[2], select_metric))
        win_name, win_params, is_metrics = winner

        # --- out-of-sample: apply winner to the test window ---
        oos_tr = _run_window(df15, df1h, specs, strategy_fn, win_params,
                             test_s, test_e, equity, risk_pct, poll_every)
        oos_metrics = _window_metrics(oos_tr)
        if len(oos_tr):
            oos_tr = oos_tr.copy()
            oos_tr["wf_window_start"] = test_s.strftime("%Y-%m-%d")
            oos_tr["wf_selected_variant"] = win_name
            oos_frames.append(oos_tr)

        per_window.append({
            "train_start": train_s.strftime("%Y-%m-%d"),
            "test_start": test_s.strftime("%Y-%m-%d"),
            "test_end": test_e.strftime("%Y-%m-%d"),
            "selected_variant": win_name,
            "select_metric": select_metric,
            "in_sample": is_metrics,
            "out_of_sample": oos_metrics,
        })
        print(f"  {test_s.strftime('%Y-%m-%d')} -> {test_e.strftime('%Y-%m-%d')}  "
              f"pick={win_name:<16} "
              f"IS totalR={is_metrics['total_r']:+.2f} (n={is_metrics['trades']})  "
              f"OOS totalR={oos_metrics['total_r']:+.2f} (n={oos_metrics['trades']})  "
              f"OOS PF={oos_metrics['profit_factor']}", flush=True)

    # --- aggregate OOS (the honest, concatenated out-of-sample track record) ---
    oos_all = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    oos_overall = _window_metrics(oos_all)
    # aggregate in-sample (sum of winners' train-window metrics) for comparison
    is_total_r = round(sum(w["in_sample"]["total_r"] for w in per_window), 3)
    n_pos = sum(1 for w in per_window if w["out_of_sample"]["total_r"] > 0)

    return {
        "config": {
            "train_months": train_months, "test_months": test_months,
            "select_metric": select_metric, "min_is_trades": min_is_trades,
            "equity": equity, "risk_pct": risk_pct, "poll_every": poll_every,
            "warmup_buffer_days": WARMUP_BUFFER_DAYS,
        },
        "n_windows": len(per_window),
        "positive_oos_windows": n_pos,
        "in_sample_total_r": is_total_r,
        "out_of_sample_overall": oos_overall,
        "per_window": per_window,
    }


# ============================ MAIN ==================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history-dir", default=str(HERE / "data" / "history"))
    p.add_argument("--years", type=float, default=None)
    p.add_argument("--train-months", type=int, default=12)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--equity", type=float, default=10000.0)
    p.add_argument("--risk-pct", type=float, default=0.01)
    p.add_argument("--poll-every", type=int, default=4)
    p.add_argument("--select-metric", default="total_r",
                   choices=["total_r", "profit_factor", "sharpe_per_trade", "avg_r"])
    p.add_argument("--out",
                   default=str(HERE / "data" / "backtests" / "walk_forward_oos_report.json"))
    args = p.parse_args()

    df15, df1h, specs = _load_history(Path(args.history_dir))
    df15 = _trim(df15, args.years)
    df1h = _trim(df1h, args.years)

    print("=== SMC walk-forward (out-of-sample) ===")
    print(f"  data: 15m={len(df15):,}, 1h={len(df1h):,}")
    print(f"  span: {df15.index[0]} -> {df15.index[-1]}")
    print(f"  train={args.train_months}mo  test={args.test_months}mo  "
          f"select_by={args.select_metric}")
    print(f"  variants: {[v[0] for v in smc_variants()]}")
    print()

    t0 = time.time()
    report = walk_forward_oos(
        df15, df1h, specs, evaluate_smc, smc_variants(),
        train_months=args.train_months, test_months=args.test_months,
        equity=args.equity, risk_pct=args.risk_pct, poll_every=args.poll_every,
        select_metric=args.select_metric,
    )
    report["elapsed_sec"] = round(time.time() - t0, 1)

    oos = report["out_of_sample_overall"]
    print("\n" + "=" * 78)
    print("=== IN-SAMPLE vs OUT-OF-SAMPLE ===")
    print("=" * 78)
    print(f"  windows:                 {report['n_windows']}")
    print(f"  positive OOS windows:    {report['positive_oos_windows']}/{report['n_windows']}")
    print(f"  IN-SAMPLE  total R:      {report['in_sample_total_r']:+.2f}  "
          f"(sum of per-window winners on their train windows)")
    print(f"  OUT-OF-SAMPLE total R:   {oos['total_r']:+.2f}")
    print(f"  OOS trades:              {oos['trades']}")
    print(f"  OOS win rate:            {oos['win_rate']*100:.1f}%")
    print(f"  OOS profit factor:       {oos['profit_factor']}")
    print(f"  OOS max DD (R):          {oos['max_dd_r']:.2f}")
    print(f"  OOS avg R / trade:       {oos['avg_r']:+.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved OOS walk-forward report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
