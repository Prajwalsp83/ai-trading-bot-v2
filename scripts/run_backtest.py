"""
Phase B.4 — Backtest orchestrator.

Loads multi-year MT5 history (from fetch_mt5_history.py), runs both strategies
(or just one via --strategy), saves trade log + equity curve, prints summary.

Run on VPS or wherever the parquet files live:
    python scripts/run_backtest.py --strategy both
    python scripts/run_backtest.py --strategy breakout --years 2
    python scripts/run_backtest.py --strategy smc --equity 5000
    python scripts/run_backtest.py --strategy breakout --tag pre_phase_c

Outputs:
    data/backtests/{strategy}_{tag}_trades.parquet
    data/backtests/{strategy}_{tag}_equity.parquet
    data/backtests/{strategy}_{tag}_summary.json
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

from _backtest_engine import (
    BacktestEngine, SymbolSpecs, CostModel, BacktestParams,
)
from _strategies import (
    evaluate_breakout, evaluate_smc, evaluate_mean_reversion,
    evaluate_liquidity_sweep,
    BreakoutSignalParams, SMCSignalParams, MeanReversionParams,
    LiquiditySweepParams,
)
from _config_loader import load_config


# ============================== HELPERS =============================
def _sanitize_symbol(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def _load_history(symbol: str, history_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SymbolSpecs]:
    """Load M15 + H1 + H4 + specs from data/history/."""
    sym = _sanitize_symbol(symbol)
    paths = {
        "M15": history_dir / f"{sym}_M15.parquet",
        "H1":  history_dir / f"{sym}_H1.parquet",
        "H4":  history_dir / f"{sym}_H4.parquet",
        "specs": history_dir / f"{sym}_specs.json",
    }
    for tf, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"missing {tf}: {p}\n"
                f"Run: python scripts/fetch_mt5_history.py --symbol \"{symbol}\" --years 4"
            )
    df15 = pd.read_parquet(paths["M15"])
    df1h = pd.read_parquet(paths["H1"])
    df4h = pd.read_parquet(paths["H4"])
    specs = SymbolSpecs.from_json(paths["specs"])
    return df15, df1h, df4h, specs


def _trim_to_years(df: pd.DataFrame, years: float | None) -> pd.DataFrame:
    """Trim to the most recent N years if specified."""
    if years is None or len(df) == 0:
        return df
    cutoff = df.index[-1] - pd.Timedelta(days=int(years * 365))
    return df[df.index >= cutoff]


def _params_for(strategy_name: str, bot_cfg):
    """Build signal params for the backtest from the live bot's config.
    Ensures backtest uses the SAME params the live bot uses.
    For 'mean_reversion', returns aggressive defaults (no config wiring yet)."""
    if strategy_name == "mean_reversion":
        return MeanReversionParams(
            rsi_oversold=30.0, rsi_overbought=70.0,
            adx_max_for_entry=25.0,
            proximity_atr=0.3,
            min_rr=1.5,
            require_candle_confirmation=True,
        )
    if strategy_name == "liquidity_sweep":
        return LiquiditySweepParams()    # defaults (see dataclass)
    s = bot_cfg.strategy
    if strategy_name == "breakout":
        return BreakoutSignalParams(
            ema_fast=s.ema_fast, ema_slow=s.ema_slow,
            atr_period=s.atr_period, atr_min=s.atr_min,
            atr_pct_min=s.atr_pct_min, min_trend_strength=s.min_trend_strength,
            use_4h_trend_gate=s.use_4h_trend_gate,
            k_sl=s.k_sl, k_tp=s.k_tp,
        )
    else:
        return SMCSignalParams(
            htf_pivot=s.htf_pivot, ltf_pivot=s.ltf_pivot,
            min_impulse_bars=s.min_impulse_bars,
            poi_freshness_bars=s.poi_freshness_bars,
            min_poi_score=s.min_poi_score,
            sl_buffer_atr_frac=s.sl_buffer_atr_frac,
            require_ltf_choch=s.require_ltf_choch,
            min_rr=s.min_rr, atr_period=s.atr_period,
            max_structure_lookback_bars=s.max_structure_lookback_bars,
        )


def _run_one(strategy_name: str, df15, df1h, df4h, specs, args, bot_cfg) -> dict:
    """Run a single backtest, save artifacts, return summary dict."""
    print(f"\n=== Running backtest: {strategy_name} ===")
    print(f"  bars: 15m={len(df15):,}, 1h={len(df1h):,}, 4h={len(df4h):,}")
    print(f"  span: {df15.index[0]} -> {df15.index[-1]} "
          f"({(df15.index[-1] - df15.index[0]).days / 365.25:.2f} years)")

    params = BacktestParams(
        starting_equity=args.equity,
        risk_per_trade_pct=args.risk_pct,
        warmup_bars=args.warmup,
        k_sl=bot_cfg.strategy.k_sl if (bot_cfg and strategy_name == "breakout") else 1.5,
        k_tp=bot_cfg.strategy.k_tp if (bot_cfg and strategy_name == "breakout") else 2.5,
        poll_every_bars=args.poll_every,
    )
    cost = CostModel(
        spread_points=args.spread or specs.avg_spread_points,
        slippage_entry_pips_max=args.slip_entry,
        slippage_stop_pips_max=args.slip_stop,
        commission_per_lot_rt_usd=args.commission or specs.commission_per_lot_rt_usd,
        pessimistic_intrabar=True,
    )
    print(f"  costs: spread={cost.spread_points}pts, "
          f"slip_entry≤{cost.slippage_entry_pips_max}pips, "
          f"slip_stop≤{cost.slippage_stop_pips_max}pips, "
          f"commission=${cost.commission_per_lot_rt_usd}/lot RT")
    print(f"  risk: {params.risk_per_trade_pct*100:.2f}%/trade, start=${params.starting_equity:,.0f}")
    print(f"  poll: every {params.poll_every_bars} bar(s)")

    sig_params = _params_for(strategy_name, bot_cfg)
    if strategy_name == "breakout":
        strategy_fn = evaluate_breakout
    elif strategy_name == "smc":
        strategy_fn = evaluate_smc
    elif strategy_name == "mean_reversion":
        strategy_fn = evaluate_mean_reversion
    elif strategy_name == "liquidity_sweep":
        strategy_fn = evaluate_liquidity_sweep
    else:
        raise ValueError(f"unknown strategy: {strategy_name}")

    engine = BacktestEngine(specs, cost=cost, params=params)
    t0 = time.time()
    # Only breakout needs df4h; others ignore it
    result = engine.run(df15, df1h, df4h if strategy_name == "breakout" else None,
                        strategy_fn, sig_params, strategy_name)
    elapsed = time.time() - t0
    summary = result.summary()
    print(f"  elapsed: {elapsed:.1f}s")

    # Save artifacts
    out_dir = HERE / "data" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trades_path = out_dir / f"{strategy_name}_{tag}_trades.parquet"
    equity_path = out_dir / f"{strategy_name}_{tag}_equity.parquet"
    summary_path = out_dir / f"{strategy_name}_{tag}_summary.json"

    result.trades.to_parquet(trades_path)
    result.equity.to_parquet(equity_path)
    full_summary = {
        "strategy": strategy_name,
        "tag": tag,
        "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed, 1),
        "params": {
            "starting_equity": params.starting_equity,
            "risk_per_trade_pct": params.risk_per_trade_pct,
            "warmup_bars": params.warmup_bars,
            "k_sl": params.k_sl, "k_tp": params.k_tp,
            "poll_every_bars": params.poll_every_bars,
        },
        "costs": {
            "spread_points": cost.spread_points,
            "slip_entry_max": cost.slippage_entry_pips_max,
            "slip_stop_max": cost.slippage_stop_pips_max,
            "commission_per_lot_rt_usd": cost.commission_per_lot_rt_usd,
            "pessimistic_intrabar": cost.pessimistic_intrabar,
        },
        "signal_params": asdict(sig_params),
        "data": {
            "symbol": specs.symbol,
            "first_bar_ts": str(df15.index[0]),
            "last_bar_ts": str(df15.index[-1]),
            "bars_15m": len(df15),
            "bars_1h": len(df1h),
            "bars_4h": len(df4h),
        },
        "metrics": summary,
    }
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2, default=str)

    print(f"\n  -> trades:  {trades_path}")
    print(f"  -> equity:  {equity_path}")
    print(f"  -> summary: {summary_path}")
    print(f"\n=== Summary ({strategy_name}) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return full_summary


# ============================== MAIN ================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",
                   choices=["breakout", "smc", "mean_reversion", "liquidity_sweep", "both", "all"],
                   default="both",
                   help="Strategies to backtest. 'all' = breakout + smc + mean_reversion + liquidity_sweep.")
    p.add_argument("--symbol", default="GOLD.i#")
    p.add_argument("--history-dir", default=str(HERE / "data" / "history"))
    p.add_argument("--years", type=float, default=None,
                   help="Trim to most recent N years (default: use all available)")
    p.add_argument("--equity", type=float, default=10000.0,
                   help="Starting equity in USD")
    p.add_argument("--risk-pct", type=float, default=0.01,
                   help="Risk per trade as fraction of equity")
    p.add_argument("--warmup", type=int, default=250,
                   help="Bars to skip at start before evaluating")
    p.add_argument("--spread", type=int, default=None,
                   help="Override spread in points (default: use symbol specs)")
    p.add_argument("--slip-entry", type=float, default=2.0)
    p.add_argument("--slip-stop", type=float, default=3.0)
    p.add_argument("--commission", type=float, default=None,
                   help="Override commission $/lot RT (default: from specs)")
    p.add_argument("--poll-every", type=int, default=1,
                   help="Evaluate strategy every N bars (1=every bar, 4=every hour on 15m). "
                        "Higher = faster backtest, slightly fewer trades.")
    p.add_argument("--tag", default=None,
                   help="Suffix for output filenames (default: timestamp)")
    args = p.parse_args()

    # Load history
    df15, df1h, df4h, specs = _load_history(args.symbol, Path(args.history_dir))
    df15 = _trim_to_years(df15, args.years)
    df1h = _trim_to_years(df1h, args.years)
    df4h = _trim_to_years(df4h, args.years)

    # Load bot configs only for strategies that have config sections
    cfg_breakout = load_config("breakout") if args.strategy in ("breakout", "both", "all") else None
    cfg_smc = load_config("smc") if args.strategy in ("smc", "both", "all") else None

    summaries = []
    if args.strategy in ("breakout", "both", "all"):
        summaries.append(_run_one("breakout", df15, df1h, df4h, specs, args, cfg_breakout))
    if args.strategy in ("smc", "both", "all"):
        summaries.append(_run_one("smc", df15, df1h, df4h, specs, args, cfg_smc))
    if args.strategy in ("mean_reversion", "all"):
        # MR uses hardcoded defaults (no config wiring yet)
        summaries.append(_run_one("mean_reversion", df15, df1h, df4h, specs, args, None))
    if args.strategy in ("liquidity_sweep", "all"):
        # LS uses hardcoded defaults (no config wiring yet)
        summaries.append(_run_one("liquidity_sweep", df15, df1h, df4h, specs, args, None))

    print("\n\n========== OVERALL ==========")
    for s in summaries:
        m = s["metrics"]
        if m.get("trades", 0) == 0:
            print(f"  {s['strategy']:>8}: NO TRADES")
            continue
        print(f"  {s['strategy']:>8}: {m['trades']:>4} trades  "
              f"WR={m['win_rate_pct']:>5.1f}%  "
              f"PF={m['profit_factor']:>5}  "
              f"PnL=${m['net_pnl_usd']:>+9.2f} ({m['net_pnl_pct']:+5.1f}%)  "
              f"DD={m['max_dd_pct']:>5.2f}%  "
              f"Sharpe={m['sharpe_annualized']:>5.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
