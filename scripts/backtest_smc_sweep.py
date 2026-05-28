"""
SMC parameter sweep.

Tests 8 SMCParams configurations on the same 60d gold history. Goal: find
the best config that hits >= 20 trades (statistically meaningful) with
expectancy >= breakout_trend's baseline 0.254R, ideally with lower drawdown.

Run:
    python scripts/backtest_smc_sweep.py
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from app.core.events import Severity                                # noqa: E402
from app.strategy.base import MarketState                           # noqa: E402
from app.strategy.smc import SMCParams, SMCStrategy                 # noqa: E402

warnings.filterwarnings("ignore")

SYMBOL = "GC=F"
LOOKBACK_15M_DAYS = 60
LOOKBACK_1H_DAYS = 730
INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.02
K_SL_FALLBACK = 1.5
K_TP_FALLBACK = 2.5
WARMUP_BARS = 220


@dataclass
class SweepResult:
    name: str
    params: SMCParams
    n_trades: int
    wins: int
    losses: int
    win_rate: float
    total_return_pct: float
    expectancy_R: float
    avg_win_R: float
    avg_loss_R: float
    max_dd_pct: float
    final_equity: float


def fetch():
    print(f"Downloading {SYMBOL} 15m + 1h...")
    df15 = yf.download(SYMBOL, interval="15m", period=f"{LOOKBACK_15M_DAYS}d",
                       auto_adjust=True, progress=False)
    df1h = yf.download(SYMBOL, interval="1h", period=f"{LOOKBACK_1H_DAYS}d",
                       auto_adjust=True, progress=False)
    for df in (df15, df1h):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    return df15.dropna(), df1h.dropna()


def _resolve_sl_tp(sig, side, price, atr_val):
    extras = getattr(sig, "extras", None) or {}
    if "sl_suggested" in extras and "tp_suggested" in extras:
        return float(extras["sl_suggested"]), float(extras["tp_suggested"])
    if side == "BUY":
        return price - K_SL_FALLBACK * atr_val, price + K_TP_FALLBACK * atr_val
    return price + K_SL_FALLBACK * atr_val, price - K_TP_FALLBACK * atr_val


def run_one(params: SMCParams, df15, df1h, name: str) -> SweepResult:
    strat = SMCStrategy(params)
    equity = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    max_dd_pct = 0.0
    open_pos = None
    trades = []

    h_idx = 0
    for i in range(WARMUP_BARS, len(df15) - 1):
        bar = df15.iloc[i]; bar_next = df15.iloc[i + 1]
        ts = df15.index[i]
        while h_idx + 1 < len(df1h) and df1h.index[h_idx + 1] <= ts:
            h_idx += 1
        if h_idx < 60:
            continue

        if open_pos is not None:
            nb = bar_next
            high = float(nb["High"]); low = float(nb["Low"])
            ex = None; reason = None
            if open_pos["side"] == "BUY":
                if low <= open_pos["sl"]: ex, reason = open_pos["sl"], "SL"
                elif high >= open_pos["tp"]: ex, reason = open_pos["tp"], "TP"
            else:
                if high >= open_pos["sl"]: ex, reason = open_pos["sl"], "SL"
                elif low <= open_pos["tp"]: ex, reason = open_pos["tp"], "TP"
            if ex is not None:
                pnl = ((ex - open_pos["entry"]) if open_pos["side"] == "BUY"
                       else (open_pos["entry"] - ex)) * open_pos["qty"]
                stop_dist = abs(open_pos["entry"] - open_pos["sl"])
                r = pnl / (stop_dist * open_pos["qty"]) if stop_dist > 0 else 0
                equity += pnl
                peak = max(peak, equity)
                max_dd_pct = max(max_dd_pct, (peak - equity) / peak * 100)
                trades.append({"pnl": pnl, "r": r, "exit_reason": reason})
                open_pos = None

        if open_pos is None:
            ms = MarketState(symbol="GOLD",
                             bars_entry=df15.iloc[:i + 1],
                             bars_trend=df1h.iloc[:h_idx + 1])
            sig = strat.evaluate(ms)
            if sig is None or sig.severity not in (Severity.BUY_READY, Severity.SELL_READY):
                continue
            price = float(bar["Close"])
            atr_val = sig.atr if sig.atr > 0 else 1.0
            sl, tp = _resolve_sl_tp(sig, sig.side, price, atr_val)
            stop_dist = abs(price - sl)
            if stop_dist <= 0:
                continue
            qty = (equity * RISK_PER_TRADE) / stop_dist
            open_pos = {"side": sig.side, "entry": price, "sl": sl, "tp": tp, "qty": qty}

    n = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = n - wins
    return SweepResult(
        name=name, params=params, n_trades=n,
        wins=wins, losses=losses,
        win_rate=(wins / n if n else 0.0),
        total_return_pct=(equity / INITIAL_EQUITY - 1) * 100,
        expectancy_R=(sum(t["r"] for t in trades) / n if n else 0.0),
        avg_win_R=(sum(t["r"] for t in trades if t["pnl"] > 0) /
                   max(1, sum(1 for t in trades if t["pnl"] > 0))),
        avg_loss_R=(sum(t["r"] for t in trades if t["pnl"] <= 0) /
                    max(1, sum(1 for t in trades if t["pnl"] <= 0))),
        max_dd_pct=max_dd_pct,
        final_equity=equity,
    )


def main():
    df15, df1h = fetch()
    print(f"Got {len(df15)} 15m bars and {len(df1h)} 1h bars\n")

    # 8 configs from MOST RELAXED to MOST STRICT
    configs = [
        ("relaxed-A", SMCParams(min_poi_score=1, require_ltf_choch=False,
                                 min_rr=1.0, min_impulse_bars=2,
                                 poi_freshness_bars_htf=60)),
        ("relaxed-B", SMCParams(min_poi_score=1, require_ltf_choch=True,
                                 min_rr=1.0, min_impulse_bars=2,
                                 poi_freshness_bars_htf=60)),
        ("balanced-A", SMCParams(min_poi_score=2, require_ltf_choch=False,
                                  min_rr=1.2, min_impulse_bars=2,
                                  poi_freshness_bars_htf=50)),
        ("balanced-B", SMCParams(min_poi_score=2, require_ltf_choch=True,
                                  min_rr=1.2, min_impulse_bars=2,
                                  poi_freshness_bars_htf=50)),
        ("balanced-C", SMCParams(min_poi_score=2, require_ltf_choch=False,
                                  min_rr=1.5, min_impulse_bars=3,
                                  poi_freshness_bars_htf=40)),
        ("balanced-D", SMCParams(min_poi_score=3, require_ltf_choch=False,
                                  min_rr=1.2, min_impulse_bars=2,
                                  poi_freshness_bars_htf=50)),
        ("strict-A",   SMCParams(min_poi_score=3, require_ltf_choch=False,
                                  min_rr=1.5, min_impulse_bars=3,
                                  poi_freshness_bars_htf=30)),
        ("strict-B",   SMCParams(min_poi_score=3, require_ltf_choch=True,
                                  min_rr=1.5, min_impulse_bars=3,
                                  poi_freshness_bars_htf=30)),
    ]

    results: list[SweepResult] = []
    for name, p in configs:
        print(f"  running {name}...", flush=True)
        results.append(run_one(p, df15, df1h, name))

    # Print summary table sorted by expectancy * trades (score) descending
    def score(r: SweepResult) -> float:
        # Penalize if too few trades (<10) by halving score
        s = r.expectancy_R * r.n_trades
        if r.n_trades < 10:
            s *= 0.5
        return s

    results.sort(key=score, reverse=True)

    print("\n" + "=" * 110)
    print(f"  {'NAME':<12} {'TRADES':<8} {'WR':<7} {'EXP/R':<8} {'AVG_W':<7} {'AVG_L':<7} {'RET%':<8} {'DD%':<7} {'EQUITY':<11} {'KEY PARAMS'}")
    print("-" * 110)
    for r in results:
        kp = (f"score>={r.params.min_poi_score} "
              f"choch={r.params.require_ltf_choch} "
              f"impulse={r.params.min_impulse_bars} "
              f"rr>={r.params.min_rr}")
        print(f"  {r.name:<12} {r.n_trades:<8} "
              f"{r.win_rate*100:>5.1f}%  {r.expectancy_R:+.3f}R  "
              f"{r.avg_win_R:+.2f}R  {r.avg_loss_R:+.2f}R  "
              f"{r.total_return_pct:+.2f}%  {r.max_dd_pct:>5.2f}%  "
              f"${r.final_equity:>9,.0f}  {kp}")
    print("=" * 110)

    # Print recommendation
    best = results[0]
    print("\n*** RECOMMENDATION ***")
    print(f"  Best SMC config: {best.name}")
    print(f"    {best.params}")
    print(f"  Trades: {best.n_trades}   Expectancy: {best.expectancy_R:+.3f}R   "
          f"Max DD: {best.max_dd_pct:.2f}%   Return: {best.total_return_pct:+.2f}%")
    print()
    print("  Comparison to breakout_trend baseline (from backtest_compare.py):")
    print(f"    breakout_trend  -> 151 trades, expectancy +0.254R, DD 19.55%, return +103.78%")
    print(f"    best SMC        -> {best.n_trades} trades, expectancy {best.expectancy_R:+.3f}R, "
          f"DD {best.max_dd_pct:.2f}%, return {best.total_return_pct:+.2f}%")


if __name__ == "__main__":
    main()
