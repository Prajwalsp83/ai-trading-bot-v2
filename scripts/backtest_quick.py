"""
Quick backtest — first real run of the breakout-trend strategy.

What it does:
  1. Pulls last ~60 days of gold (GC=F) at 15m + 1h from yfinance.
  2. Walks forward bar-by-bar (no lookahead) calling strategy.evaluate().
  3. On BUY_READY/SELL_READY -> opens a simulated trade with:
       SL = 1.5 * ATR away, TP = 3.0 * ATR away (RR 1:2)
       size = (equity * risk_per_trade) / stop_distance
  4. Each subsequent bar: check SL / TP hits.
  5. At end: prints trades, win rate, total return, max drawdown, expectancy.
  6. Saves equity curve to data/equity.csv and trades to data/trades.csv.

This is the SAME strategy code the live bot will call. Backtest == live, finally.

Run:
    cd ~/Documents/ai-trading-bot/v2
    source .venv/bin/activate
    python scripts/backtest_quick.py
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import yfinance as yf

# Make the app package importable
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from app.core.events import Severity                              # noqa: E402
from app.strategy.base import MarketState                         # noqa: E402
from app.strategy.breakout_trend import (                         # noqa: E402
    BreakoutTrendParams,
    BreakoutTrendStrategy,
)

warnings.filterwarnings("ignore")

# ----- config (mirrors v2 config.yaml; hardcoded for quick run) -----
SYMBOL = "GC=F"
LOOKBACK_15M_DAYS = 60        # yfinance 15m limit
LOOKBACK_1H_DAYS = 730        # plenty for trend EMAs

INITIAL_EQUITY = 100_000.0    # currency-agnostic
RISK_PER_TRADE = 0.02         # 2%
K_SL_ATR = 1.5
K_TP_ATR = 3.0                # RR 1:2

WARMUP_BARS = 220             # need at least ema200 + a few; 15m bars

DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class Trade:
    open_ts: pd.Timestamp
    close_ts: pd.Timestamp
    side: str
    entry: float
    exit: float
    sl: float
    tp: float
    qty: float
    pnl: float
    r_realised: float
    duration_bars: int
    exit_reason: str


def fetch() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Downloading {SYMBOL} 15m ({LOOKBACK_15M_DAYS}d) and 1h ({LOOKBACK_1H_DAYS}d)…")
    df15 = yf.download(SYMBOL, interval="15m", period=f"{LOOKBACK_15M_DAYS}d",
                       auto_adjust=True, progress=False)
    df1h = yf.download(SYMBOL, interval="1h", period=f"{LOOKBACK_1H_DAYS}d",
                       auto_adjust=True, progress=False)
    # Flatten multi-index columns if yfinance returns them
    for df in (df15, df1h):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    df15 = df15.dropna()
    df1h = df1h.dropna()
    print(f"  15m bars: {len(df15)}   1h bars: {len(df1h)}")
    return df15, df1h


def run_backtest():
    df15, df1h = fetch()
    if len(df15) < WARMUP_BARS + 50:
        print(f"Not enough 15m bars ({len(df15)}); need >= {WARMUP_BARS + 50}")
        return

    strategy = BreakoutTrendStrategy(BreakoutTrendParams(
        ema_fast=50, ema_slow=200, atr_period=14, atr_min=5.0,
    ))

    equity = INITIAL_EQUITY
    peak_equity = equity
    max_dd = 0.0
    open_pos = None       # currently held position
    trades: list[Trade] = []
    equity_curve: list[tuple[pd.Timestamp, float]] = []

    # Align 1h bars to each 15m bar's "available-as-of" timestamp.
    print("Walking forward bar by bar…")
    for i in range(WARMUP_BARS, len(df15)):
        slice_15 = df15.iloc[: i + 1]
        cutoff = slice_15.index[-1]
        slice_1h = df1h.loc[: cutoff]
        if len(slice_1h) < 210:   # need ema200 + breathing room
            continue

        bar = slice_15.iloc[-1]
        ts = slice_15.index[-1]

        # ----- manage open position (SL/TP within the bar) -----
        if open_pos is not None:
            sl = open_pos["sl"]
            tp = open_pos["tp"]
            side = open_pos["side"]
            high = float(bar["High"])
            low = float(bar["Low"])
            exit_reason = None
            exit_price = None
            # Conservative tie-break: assume SL hits first within a bar.
            if side == "BUY":
                if low <= sl:
                    exit_price, exit_reason = sl, "SL"
                elif high >= tp:
                    exit_price, exit_reason = tp, "TP"
            else:  # SELL
                if high >= sl:
                    exit_price, exit_reason = sl, "SL"
                elif low <= tp:
                    exit_price, exit_reason = tp, "TP"

            if exit_reason is not None:
                pnl = (exit_price - open_pos["entry"]) * open_pos["qty"] \
                       if side == "BUY" else \
                       (open_pos["entry"] - exit_price) * open_pos["qty"]
                r_planned = abs(open_pos["entry"] - open_pos["sl"])
                r_realised = (pnl / (r_planned * open_pos["qty"])) if r_planned > 0 else 0
                equity += pnl
                trades.append(Trade(
                    open_ts=open_pos["ts"], close_ts=ts, side=side,
                    entry=open_pos["entry"], exit=exit_price,
                    sl=open_pos["sl"], tp=open_pos["tp"], qty=open_pos["qty"],
                    pnl=pnl, r_realised=r_realised,
                    duration_bars=i - open_pos["bar_idx"],
                    exit_reason=exit_reason,
                ))
                open_pos = None

        equity_curve.append((ts, equity))
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity
        max_dd = max(max_dd, dd)

        # ----- look for new entry (only if flat) -----
        if open_pos is None:
            sig = strategy.evaluate(MarketState(
                symbol=SYMBOL, bars_entry=slice_15, bars_trend=slice_1h,
            ))
            if sig and sig.severity in (Severity.BUY_READY, Severity.SELL_READY):
                entry = float(bar["Close"])
                atr_val = sig.atr
                if sig.side == "BUY":
                    sl = entry - K_SL_ATR * atr_val
                    tp = entry + K_TP_ATR * atr_val
                else:
                    sl = entry + K_SL_ATR * atr_val
                    tp = entry - K_TP_ATR * atr_val
                stop_dist = abs(entry - sl)
                if stop_dist <= 0:
                    continue
                qty = (equity * RISK_PER_TRADE) / stop_dist
                open_pos = {
                    "ts": ts, "bar_idx": i, "side": sig.side,
                    "entry": entry, "sl": sl, "tp": tp, "qty": qty,
                }

    # ---- report ----
    print_report(trades, equity_curve, max_dd)
    save_outputs(trades, equity_curve)


def print_report(trades: list[Trade], equity_curve: list, max_dd: float):
    n = len(trades)
    print("\n" + "=" * 56)
    print(" BACKTEST RESULTS — breakout_trend on GC=F")
    print("=" * 56)
    if n == 0:
        print(" No trades taken. Strategy may need looser ATR threshold,")
        print(" or the period was too quiet. Try lowering atr_min.")
        return
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    win_rate = len(wins) / n * 100
    avg_r = sum(t.r_realised for t in trades) / n
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
    profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) \
                    if losses and sum(t.pnl for t in losses) != 0 else float("inf")

    final_equity = equity_curve[-1][1] if equity_curve else INITIAL_EQUITY
    ret_pct = (final_equity / INITIAL_EQUITY - 1) * 100

    print(f" Trades              : {n}")
    print(f" Wins / Losses       : {len(wins)} / {len(losses)}")
    print(f" Win rate            : {win_rate:.1f}%")
    print(f" Avg R (per trade)   : {avg_r:+.2f}")
    print(f" Avg win / avg loss  : {avg_win:+,.2f} / {avg_loss:+,.2f}")
    print(f" Profit factor       : {profit_factor:.2f}")
    print(f" Total P&L           : {total_pnl:+,.2f}")
    print(f" Starting equity     : {INITIAL_EQUITY:,.2f}")
    print(f" Ending equity       : {final_equity:,.2f}  ({ret_pct:+.2f}%)")
    print(f" Max drawdown        : {max_dd * 100:.2f}%")
    print("=" * 56)

    # Verdict
    print("\n VERDICT:")
    issues = []
    if n < 10:
        issues.append("- Too few trades to be statistically meaningful.")
    if win_rate < 35:
        issues.append("- Win rate < 35%: hits common for trend-following, OK if avg R > 0.5.")
    if avg_r < 0:
        issues.append("- Negative expectancy: strategy loses money on average.")
    if max_dd > 0.20:
        issues.append("- Drawdown > 20%: dangerous for a small account.")
    if not issues:
        print("  Strategy looks promising — proceed to parameter sweep + walk-forward.")
    else:
        for ln in issues:
            print(f"  {ln}")


def save_outputs(trades: list[Trade], equity_curve: list):
    if trades:
        pd.DataFrame([asdict(t) for t in trades]).to_csv(
            DATA_DIR / "trades.csv", index=False)
        print(f"\n  Saved trades       -> {DATA_DIR / 'trades.csv'}")
    if equity_curve:
        pd.DataFrame(equity_curve, columns=["ts", "equity"]).to_csv(
            DATA_DIR / "equity.csv", index=False)
        print(f"  Saved equity curve -> {DATA_DIR / 'equity.csv'}")


if __name__ == "__main__":
    run_backtest()
