"""
Phase B.5 — SMC parameter sweep.

Runs the SMC backtest with multiple param variations side-by-side.
Outputs comparison table + per-variation full summary.

Goal: see if loosening the strict params (min_poi_score, min_rr, etc) gets
us more trades while preserving the edge — OR if it destroys the edge.

Honest hypothesis: loosening usually dilutes edge. Real edge concentrates at
high-conviction setups. We test this empirically rather than guessing.

Run on VPS:
    python scripts/sweep_smc.py
    python scripts/sweep_smc.py --years 3      # last 3 years only (faster)
    python scripts/sweep_smc.py --poll-every 4 # match the main backtest

Each variation takes ~10 min on VPS (4yrs M15). 5 variations ~ 50 min.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))

from _backtest_engine import BacktestEngine, BacktestParams, CostModel, SymbolSpecs
from _strategies import evaluate_smc, SMCSignalParams


# ============================== VARIANTS ============================
# Each variant overrides specific fields on the baseline SMCSignalParams.
# Keep this list short — each variant costs ~10 min on VPS.
def variants() -> list[tuple[str, SMCSignalParams]]:
    base = dict(
        htf_pivot=2, ltf_pivot=2, min_impulse_bars=3,
        poi_freshness_bars=60, min_poi_score=2,
        sl_buffer_atr_frac=0.25, require_ltf_choch=False,
        min_rr=1.5, atr_period=14, max_structure_lookback_bars=300,
    )
    return [
        ("baseline",          SMCSignalParams(**base)),
        ("score_1",           SMCSignalParams(**{**base, "min_poi_score": 1})),
        ("rr_1.0",            SMCSignalParams(**{**base, "min_rr": 1.0})),
        ("score_1_rr_1.0",    SMCSignalParams(**{**base, "min_poi_score": 1, "min_rr": 1.0})),
        ("aggressive_all",    SMCSignalParams(**{
            **base, "min_poi_score": 1, "min_rr": 1.0,
            "htf_pivot": 1, "min_impulse_bars": 2,
            "poi_freshness_bars": 120,
        })),
    ]


# ============================== HELPERS =============================
def _load_history(history_dir: Path):
    sym = "GOLD_i"
    df15 = pd.read_parquet(history_dir / f"{sym}_M15.parquet")
    df1h = pd.read_parquet(history_dir / f"{sym}_H1.parquet")
    specs = SymbolSpecs.from_json(history_dir / f"{sym}_specs.json")
    return df15, df1h, specs


def _trim(df: pd.DataFrame, years: float | None) -> pd.DataFrame:
    if years is None: return df
    cutoff = df.index[-1] - pd.Timedelta(days=int(years * 365))
    return df[df.index >= cutoff]


def _run_variant(name: str, params: SMCSignalParams,
                 df15, df1h, specs, equity: float, risk_pct: float,
                 poll_every: int) -> dict:
    print(f"\n--- Variant: {name} ---")
    print(f"  params: min_poi_score={params.min_poi_score} "
          f"min_rr={params.min_rr} htf_pivot={params.htf_pivot} "
          f"min_impulse={params.min_impulse_bars} "
          f"poi_fresh={params.poi_freshness_bars} "
          f"require_choch={params.require_ltf_choch}")

    engine = BacktestEngine(
        specs,
        params=BacktestParams(
            starting_equity=equity, risk_per_trade_pct=risk_pct,
            warmup_bars=250, poll_every_bars=poll_every,
        ),
    )
    t0 = time.time()
    result = engine.run(df15, df1h, None, evaluate_smc, params, "smc")
    elapsed = time.time() - t0
    summary = result.summary()

    # Persist variant results
    out_dir = HERE / "data" / "backtests" / "smc_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"sweep_{name}"
    result.trades.to_parquet(out_dir / f"{tag}_trades.parquet")
    result.equity.to_parquet(out_dir / f"{tag}_equity.parquet")
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump({"variant": name, "params": asdict(params),
                   "elapsed_sec": elapsed, "metrics": summary},
                  f, indent=2, default=str)

    print(f"  elapsed: {elapsed:.0f}s, trades: {summary.get('trades', 0)}")
    return {"variant": name, "params": params, "summary": summary, "elapsed": elapsed}


# ============================== MAIN ================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history-dir", default=str(HERE / "data" / "history"))
    p.add_argument("--years", type=float, default=None)
    p.add_argument("--equity", type=float, default=10000.0)
    p.add_argument("--risk-pct", type=float, default=0.01)
    p.add_argument("--poll-every", type=int, default=4)
    p.add_argument("--variants", nargs="*", default=None,
                   help="Subset of variant names to run (default: all)")
    args = p.parse_args()

    df15, df1h, specs = _load_history(Path(args.history_dir))
    df15 = _trim(df15, args.years)
    df1h = _trim(df1h, args.years)
    print(f"=== SMC parameter sweep ===")
    print(f"  data: 15m={len(df15):,}, 1h={len(df1h):,}")
    print(f"  span: {df15.index[0]} -> {df15.index[-1]}")
    print(f"  starting equity: ${args.equity:,.0f}")
    print(f"  risk per trade: {args.risk_pct*100:.2f}%")
    print(f"  poll every: {args.poll_every} bars")

    all_variants = variants()
    if args.variants:
        all_variants = [v for v in all_variants if v[0] in args.variants]
    print(f"  variants to run: {[v[0] for v in all_variants]}")
    print(f"  estimated total time: ~{len(all_variants) * 10} min")

    results = []
    overall_t0 = time.time()
    for name, params in all_variants:
        results.append(_run_variant(
            name, params, df15, df1h, specs,
            args.equity, args.risk_pct, args.poll_every,
        ))

    total_elapsed = time.time() - overall_t0

    # ===== Comparison table =====
    print(f"\n\n========== SMC SWEEP RESULTS (total {total_elapsed:.0f}s) ==========")
    header = f"{'variant':<20} {'trades':>6} {'WR':>6} {'PF':>6} {'avg_R':>7} " \
             f"{'PnL$':>10} {'PnL%':>7} {'DD%':>6} {'Sharpe':>7} {'loss_strk':>9}"
    print(header)
    print("-" * len(header))
    # Sort by net PnL descending
    results.sort(key=lambda r: r["summary"].get("net_pnl_usd", 0) or 0, reverse=True)
    for r in results:
        s = r["summary"]
        n = s.get("trades", 0)
        if n == 0:
            print(f"{r['variant']:<20} {0:>6}    -      -        -          -       -      -      -        -")
            continue
        print(f"{r['variant']:<20} "
              f"{n:>6} "
              f"{s.get('win_rate_pct', 0):>5.1f}% "
              f"{str(s.get('profit_factor', 0)):>6} "
              f"{s.get('avg_r', 0):>+7.3f} "
              f"${s.get('net_pnl_usd', 0):>+9.2f} "
              f"{s.get('net_pnl_pct', 0):>+6.1f}% "
              f"{s.get('max_dd_pct', 0):>5.2f}% "
              f"{s.get('sharpe_annualized', 0):>+7.2f} "
              f"{s.get('longest_losing_streak', 0):>9}")

    # Save combined summary
    out = HERE / "data" / "backtests" / "smc_sweep" / "sweep_comparison.json"
    with open(out, "w") as f:
        json.dump([{
            "variant": r["variant"],
            "params": asdict(r["params"]),
            "metrics": r["summary"],
            "elapsed_sec": r["elapsed"],
        } for r in results], f, indent=2, default=str)
    print(f"\nCombined summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
