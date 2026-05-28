"""
Phase C.1 — Walk-forward analysis on existing trade logs.

We've already produced full trade logs for baseline + aggressive_all SMC
configs. This script bins those trades into rolling 6-month windows and
computes per-window metrics in R-space (independent of equity compounding
path), so we can see honestly:

  - Is the edge consistent across all windows, or concentrated in 1-2?
  - How does the strategy hold up in the worst window vs the best?
  - What % of windows are net-positive?

Why R-based instead of $ P&L:
  R-multiples (= profit/risk per trade) don't compound. They give us
  apples-to-apples per-window comparison. $ P&L would skew toward later
  windows because position size grows with equity.

Run:
    python scripts/walk_forward.py
    python scripts/walk_forward.py --window-months 3   # tighter windows
    python scripts/walk_forward.py --trades-glob "data/backtests/smc_sweep/sweep_*_trades.parquet"

Outputs:
    data/backtests/walk_forward_report.json
    Prints per-strategy per-window table + aggregate summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent


# ============================ HELPERS ===============================
def _max_drawdown_r(cum_r: pd.Series) -> float:
    """Max peak-to-trough drop in cumulative R."""
    if len(cum_r) == 0:
        return 0.0
    running_max = cum_r.expanding().max()
    drawdowns = running_max - cum_r
    return float(drawdowns.max())


def _window_metrics(trades: pd.DataFrame) -> dict:
    """R-based metrics for a slice of trades."""
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "total_r": 0.0, "avg_r": 0.0,
                "profit_factor": 0.0, "sharpe_per_trade": 0.0,
                "max_dd_r": 0.0, "longest_loss_streak": 0}

    r = trades["r_realised"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    win_rate = len(wins) / n if n else 0.0
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    sharpe = (r.mean() / r.std()) if (len(r) > 1 and r.std() > 0) else 0.0

    # Streak
    streak = 0; longest = 0
    for v in r:
        if v <= 0:
            streak += 1; longest = max(longest, streak)
        else:
            streak = 0

    return {
        "trades": n,
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(win_rate, 4),
        "total_r": round(float(r.sum()), 3),
        "avg_r": round(float(r.mean()), 3),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "sharpe_per_trade": round(sharpe, 3),
        "max_dd_r": round(_max_drawdown_r(r.cumsum()), 3),
        "longest_loss_streak": longest,
    }


def _make_windows(start: pd.Timestamp, end: pd.Timestamp, months: int):
    """Generate (window_start, window_end) tuples covering [start, end]."""
    windows = []
    cur = start
    while cur < end:
        nxt = cur + pd.DateOffset(months=months)
        windows.append((cur, min(nxt, end)))
        cur = nxt
    return windows


# ============================== MAIN ================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trades-glob",
                   default=str(HERE / "data" / "backtests" / "**" / "*_trades.parquet"),
                   help="Glob for trade-log parquet files to analyze")
    p.add_argument("--window-months", type=int, default=6,
                   help="Window size for walk-forward (default: 6 months)")
    p.add_argument("--out",
                   default=str(HERE / "data" / "backtests" / "walk_forward_report.json"))
    args = p.parse_args()

    # Discover trade log files
    paths = sorted(glob.glob(args.trades_glob, recursive=True))
    if not paths:
        print(f"ERROR: no trade logs found matching {args.trades_glob}", file=sys.stderr)
        return 1

    print(f"Found {len(paths)} trade log file(s):")
    for p_ in paths:
        print(f"  - {Path(p_).name}")
    print()

    all_reports = {}
    for path in paths:
        name = _strategy_name_from_path(path)
        try:
            tdf = pd.read_parquet(path)
        except Exception as e:
            print(f"  skip {Path(path).name}: {e}")
            continue
        if len(tdf) == 0:
            print(f"  skip {Path(path).name}: no trades")
            continue
        # Ensure datetime
        tdf["open_time"] = pd.to_datetime(tdf["open_time"], utc=True)
        tdf["close_time"] = pd.to_datetime(tdf["close_time"], utc=True)
        tdf = tdf.sort_values("open_time").reset_index(drop=True)

        overall = _window_metrics(tdf)
        first_ts = tdf["open_time"].iloc[0]
        last_ts = tdf["close_time"].iloc[-1]
        # Round first_ts down to month start for cleaner windows
        first_aligned = pd.Timestamp(year=first_ts.year, month=first_ts.month, day=1,
                                      tz="UTC")
        windows = _make_windows(first_aligned, last_ts, args.window_months)

        per_window = []
        for ws, we in windows:
            slc = tdf[(tdf["open_time"] >= ws) & (tdf["open_time"] < we)]
            m = _window_metrics(slc)
            m["window_start"] = ws.strftime("%Y-%m-%d")
            m["window_end"] = we.strftime("%Y-%m-%d")
            per_window.append(m)

        n_windows = len(per_window)
        positive_windows = sum(1 for w in per_window if w["total_r"] > 0)
        positive_pct = round(positive_windows / n_windows * 100, 1) if n_windows else 0
        mean_sharpe = round(
            sum(w["sharpe_per_trade"] for w in per_window if w["trades"] > 0) /
            max(1, sum(1 for w in per_window if w["trades"] > 0)), 3)
        mean_pf_vals = [w["profit_factor"] for w in per_window
                        if isinstance(w["profit_factor"], (int, float)) and w["trades"] > 0]
        mean_pf = round(sum(mean_pf_vals) / len(mean_pf_vals), 2) if mean_pf_vals else 0
        total_r = round(sum(w["total_r"] for w in per_window), 3)
        worst_r = min((w["total_r"] for w in per_window), default=0)
        best_r = max((w["total_r"] for w in per_window), default=0)

        all_reports[name] = {
            "file": path,
            "n_trades_total": len(tdf),
            "span_first": str(first_ts), "span_last": str(last_ts),
            "overall": overall,
            "per_window": per_window,
            "summary": {
                "windows": n_windows,
                "positive_windows": positive_windows,
                "positive_window_pct": positive_pct,
                "mean_sharpe_per_trade": mean_sharpe,
                "mean_profit_factor": mean_pf,
                "total_r_across_windows": total_r,
                "best_window_r": best_r,
                "worst_window_r": worst_r,
            },
        }

    # ===== Print per-strategy tables =====
    for name, rpt in all_reports.items():
        print(f"\n{'='*100}")
        print(f"=== {name} ({rpt['n_trades_total']} trades, {rpt['span_first'][:10]} -> {rpt['span_last'][:10]}) ===")
        print(f"{'='*100}")
        print(f"  OVERALL: trades={rpt['overall']['trades']}  "
              f"WR={rpt['overall']['win_rate']*100:.1f}%  "
              f"PF={rpt['overall']['profit_factor']}  "
              f"avgR={rpt['overall']['avg_r']:+.3f}  "
              f"totalR={rpt['overall']['total_r']:+.2f}  "
              f"Sharpe/trade={rpt['overall']['sharpe_per_trade']:+.3f}  "
              f"maxDD={rpt['overall']['max_dd_r']:.2f}R")
        print()
        print(f"  {'window':<22} {'trades':>6} {'WR':>6} {'PF':>6} {'avgR':>7} "
              f"{'totalR':>8} {'maxDD':>8} {'sharpe':>7} {'streak':>7}")
        for w in rpt["per_window"]:
            pf = w["profit_factor"]
            pf_str = f"{pf}" if isinstance(pf, str) else f"{pf:.2f}"
            print(f"  {w['window_start']} -> {w['window_end'][-5:]:<6} "
                  f"{w['trades']:>6} "
                  f"{w['win_rate']*100:>5.1f}% "
                  f"{pf_str:>6} "
                  f"{w['avg_r']:>+7.3f} "
                  f"{w['total_r']:>+8.3f} "
                  f"{w['max_dd_r']:>8.3f} "
                  f"{w['sharpe_per_trade']:>+7.3f} "
                  f"{w['longest_loss_streak']:>7}")
        s = rpt["summary"]
        print()
        print(f"  AGGREGATE: {s['positive_windows']}/{s['windows']} positive windows "
              f"({s['positive_window_pct']}%)  "
              f"mean_sharpe={s['mean_sharpe_per_trade']:+.3f}  "
              f"mean_PF={s['mean_profit_factor']}  "
              f"totalR={s['total_r_across_windows']:+.2f}  "
              f"best={s['best_window_r']:+.2f}R worst={s['worst_window_r']:+.2f}R")

    # ===== Save combined report =====
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n\nSaved combined walk-forward report: {out_path}")
    return 0


def _strategy_name_from_path(p: str) -> str:
    """Build a readable name from the parquet filename."""
    base = Path(p).stem.replace("_trades", "")
    return base


if __name__ == "__main__":
    sys.exit(main())
