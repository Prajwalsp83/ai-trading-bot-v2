"""
Phase D.alt — Auto-generate QuantStats HTML tearsheet from a backtest.

Reads:
  data/backtests/<strategy>_<tag>_equity.parquet   (per-bar equity curve)
  data/backtests/<strategy>_<tag>_trades.parquet   (closed trades)
  data/history/GOLD_i_M15.parquet                  (for buy-and-hold benchmark)

Produces:
  reports/tearsheet_<strategy>_<tag>.html          (full institutional report)
  reports/tearsheet_<strategy>_<tag>_metrics.csv   (metrics-only CSV for archive)

Run:
    pip install quantstats matplotlib seaborn
    python scripts/generate_tearsheet.py --equity data/backtests/sweep_aggressive_all_equity.parquet
    python scripts/generate_tearsheet.py --all   # process every backtest in data/backtests/

What QuantStats shows:
  - Cumulative returns vs benchmark
  - Drawdown plot + recovery periods
  - Monthly returns heatmap
  - Distribution of returns (histogram + QQ plot)
  - Underwater equity (time spent in drawdown)
  - Top 5 drawdowns table
  - 50+ performance metrics (Sharpe, Sortino, Calmar, Omega, Tail ratio, etc)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent.parent


# ============================== LOADERS =============================
def _load_equity(path: Path) -> pd.Series:
    """Return daily-end equity series, tz-naive (QuantStats requirement)."""
    df = pd.read_parquet(path)
    if "ts" in df.columns:
        df = df.set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True)
    if "equity" not in df.columns:
        raise ValueError(f"no 'equity' column in {path}")
    # Resample to daily end-of-day equity
    daily = df["equity"].resample("1D").last().dropna()
    # Make tz-naive — QuantStats chokes on tz-aware
    daily.index = daily.index.tz_localize(None)
    return daily


def _equity_to_returns(equity: pd.Series, starting_equity: float | None = None) -> pd.Series:
    """Convert equity series to daily returns. Returns are pct_change."""
    if starting_equity is not None and equity.iloc[0] != starting_equity:
        # Prepend the starting point so day 1 has a return
        first_ts = equity.index[0] - pd.Timedelta(days=1)
        equity = pd.concat([pd.Series([starting_equity], index=[first_ts]), equity])
    returns = equity.pct_change().dropna()
    return returns


def _load_benchmark_returns(history_path: Path,
                             start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Buy-and-hold gold = close-to-close pct change over backtest window."""
    df = pd.read_parquet(history_path)
    df.index = pd.to_datetime(df.index, utc=True)
    # Trim to backtest window
    df = df.loc[start:end]
    daily_px = df["Close"].resample("1D").last().dropna()
    daily_px.index = daily_px.index.tz_localize(None)
    return daily_px.pct_change().dropna()


# ============================== GENERATION ==========================
def _generate_one(equity_path: Path, history_path: Path | None,
                   reports_dir: Path, strategy_name: str | None = None) -> Path | None:
    """Generate a single tearsheet. Returns output HTML path or None on error."""
    try:
        import quantstats as qs
    except ImportError:
        print("ERROR: quantstats not installed. Run: pip install quantstats", file=sys.stderr)
        sys.exit(1)

    equity = _load_equity(equity_path)
    if len(equity) < 10:
        print(f"  skip {equity_path.name}: only {len(equity)} daily points")
        return None

    returns = equity.pct_change().dropna()
    if len(returns) < 10:
        print(f"  skip {equity_path.name}: only {len(returns)} return points")
        return None

    # Try to load benchmark
    benchmark = None
    if history_path and history_path.exists():
        try:
            benchmark = _load_benchmark_returns(
                history_path,
                equity.index[0].tz_localize("UTC") if equity.index[0].tz is None else equity.index[0],
                equity.index[-1].tz_localize("UTC") if equity.index[-1].tz is None else equity.index[-1],
            )
            # Align lengths — QuantStats wants matching indices
            benchmark = benchmark.reindex(returns.index, method="ffill").dropna()
            if len(benchmark) < len(returns) * 0.5:
                print(f"  benchmark misaligned, dropping")
                benchmark = None
        except Exception as e:
            print(f"  benchmark load failed ({e}); proceeding without")
            benchmark = None

    # Inferred name
    name = strategy_name or equity_path.stem.replace("_equity", "")
    out_html = reports_dir / f"tearsheet_{name}.html"
    out_csv = reports_dir / f"tearsheet_{name}_metrics.csv"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ===== Generate HTML report =====
    title = f"{name} | {equity.index[0].date()} → {equity.index[-1].date()}"
    print(f"  generating {out_html.name}...")
    try:
        qs.reports.html(
            returns=returns,
            benchmark=benchmark,
            output=str(out_html),
            title=title,
            download_filename=out_html.name,
        )
    except Exception as e:
        # Fallback — generate without benchmark if benchmark causes issues
        print(f"  HTML with benchmark failed ({e}), retrying without benchmark")
        qs.reports.html(returns=returns, output=str(out_html),
                         title=title, download_filename=out_html.name)

    # ===== Metrics CSV (separate, for archive / diff over time) =====
    try:
        metrics = qs.reports.metrics(returns, mode="full", display=False)
        if isinstance(metrics, pd.DataFrame):
            metrics.to_csv(out_csv)
        else:
            with open(out_csv, "w") as f:
                f.write(str(metrics))
    except Exception as e:
        print(f"  metrics CSV failed (non-fatal): {e}")

    # ===== Compact summary to stdout =====
    # Newer QuantStats sometimes returns Series instead of scalar (when given a
    # date range). Wrap each stat in _scalar() to coerce safely.
    def _scalar(v) -> float:
        try:
            if hasattr(v, "item"):
                return float(v.item() if v.size == 1 else v.iloc[-1])
            if hasattr(v, "iloc"):
                return float(v.iloc[-1])
            return float(v)
        except Exception:
            return 0.0

    print(f"  -> {out_html}")
    try:
        print(f"     period:        {equity.index[0].date()} -> {equity.index[-1].date()} "
              f"({len(returns)} days)")
        print(f"     final equity:  ${equity.iloc[-1]:,.2f} (start ${equity.iloc[0]:,.2f})")
        print(f"     total return:  {(equity.iloc[-1]/equity.iloc[0] - 1)*100:+.2f}%")
        print(f"     CAGR:          {_scalar(qs.stats.cagr(returns))*100:+.2f}%")
        print(f"     Sharpe:        {_scalar(qs.stats.sharpe(returns)):+.2f}")
        print(f"     Sortino:       {_scalar(qs.stats.sortino(returns)):+.2f}")
        print(f"     max DD:        {_scalar(qs.stats.max_drawdown(returns))*100:+.2f}%")
        print(f"     calmar:        {_scalar(qs.stats.calmar(returns)):+.2f}")
        print(f"     win days:      {(returns > 0).sum()}/{len(returns)} "
              f"({(returns > 0).mean()*100:.1f}%)")
    except Exception as e:
        print(f"     (metric extract failed: {type(e).__name__}: {e})")
    return out_html


# ============================== MAIN ================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--equity", default=None,
                   help="Path to a specific *_equity.parquet file")
    p.add_argument("--all", action="store_true",
                   help="Process every *_equity.parquet under data/backtests/")
    p.add_argument("--history",
                   default=str(HERE / "data" / "history" / "GOLD_i_M15.parquet"),
                   help="Path to benchmark price history")
    p.add_argument("--reports-dir", default=str(HERE / "reports"))
    args = p.parse_args()

    history_path = Path(args.history) if args.history else None
    reports_dir = Path(args.reports_dir)

    if args.all:
        # Find all equity parquet files recursively
        equity_files = sorted(
            (HERE / "data" / "backtests").rglob("*_equity.parquet"))
        if not equity_files:
            print("No *_equity.parquet files found under data/backtests/", file=sys.stderr)
            return 1
        print(f"Processing {len(equity_files)} equity file(s)...")
        for ep in equity_files:
            print(f"\n=== {ep.relative_to(HERE)} ===")
            _generate_one(ep, history_path, reports_dir)
    elif args.equity:
        ep = Path(args.equity)
        if not ep.exists():
            print(f"ERROR: {ep} not found", file=sys.stderr)
            return 1
        print(f"=== {ep.name} ===")
        _generate_one(ep, history_path, reports_dir)
    else:
        # Default: process the most recent backtest of each strategy
        candidates = {}
        for ep in sorted((HERE / "data" / "backtests").rglob("*_equity.parquet")):
            # Group by strategy name from filename
            name = ep.stem.replace("_equity", "")
            base_strategy = name.split("_")[0]
            if base_strategy == "sweep":
                base_strategy = "sweep_" + name.split("_")[1] if len(name.split("_")) > 1 else name
            candidates[base_strategy] = ep   # latest wins because sorted
        if not candidates:
            print("No backtests found. Specify --equity <path> or run a backtest first.")
            return 1
        print(f"Processing latest of each strategy ({len(candidates)} files)...")
        for strat, ep in candidates.items():
            print(f"\n=== {strat} ({ep.name}) ===")
            _generate_one(ep, history_path, reports_dir)

    print(f"\n\nDone. Open the HTML files in {reports_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
