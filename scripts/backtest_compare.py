"""
Backtest comparison: SMC strategy vs breakout_trend baseline.

Runs both strategies bar-by-bar on the same historical 15m+1h gold data,
simulates trades, and prints a side-by-side metrics table.

Goal: decide whether the new SMC strategy is at least as good as the
existing breakout_trend on out-of-sample-ish data, BEFORE deploying live.

Run:
    cd ~/Documents/ai-trading-bot/v2
    source .venv/bin/activate
    python scripts/backtest_compare.py
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from app.core.events import Severity                                # noqa: E402
from app.strategy.base import MarketState                           # noqa: E402
from app.strategy.breakout_trend import (                           # noqa: E402
    BreakoutTrendParams, BreakoutTrendStrategy,
)
from app.strategy.smc import SMCParams, SMCStrategy                 # noqa: E402

warnings.filterwarnings("ignore")


# --- shared config ---
SYMBOL = "GC=F"
LOOKBACK_15M_DAYS = 60
LOOKBACK_1H_DAYS = 730
INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.02
K_SL_FALLBACK = 1.5  # used if strategy doesn't supply SL
K_TP_FALLBACK = 2.5
WARMUP_BARS = 220


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
    r: float
    exit_reason: str


@dataclass
class BacktestResult:
    label: str
    trades: list[Trade] = field(default_factory=list)
    final_equity: float = 0.0
    peak_equity: float = 0.0
    max_dd_pct: float = 0.0

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / INITIAL_EQUITY - 1.0) * 100

    @property
    def expectancy_R(self) -> float:
        return (sum(t.r for t in self.trades) / self.n) if self.n else 0.0

    @property
    def avg_win_R(self) -> float:
        ws = [t.r for t in self.trades if t.pnl > 0]
        return sum(ws) / len(ws) if ws else 0.0

    @property
    def avg_loss_R(self) -> float:
        ls = [t.r for t in self.trades if t.pnl <= 0]
        return sum(ls) / len(ls) if ls else 0.0


def fetch() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Downloading {SYMBOL} 15m + 1h...")
    df15 = yf.download(SYMBOL, interval="15m", period=f"{LOOKBACK_15M_DAYS}d",
                       auto_adjust=True, progress=False)
    df1h = yf.download(SYMBOL, interval="1h", period=f"{LOOKBACK_1H_DAYS}d",
                       auto_adjust=True, progress=False)
    for df in (df15, df1h):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    return df15.dropna(), df1h.dropna()


def _resolve_sl_tp(sig, side: str, price: float, atr_val: float) -> tuple[float, float]:
    """Use strategy-supplied SL/TP if present (SMC), else fall back to ATR multiples."""
    extras = getattr(sig, "extras", None) or {}
    if "sl_suggested" in extras and "tp_suggested" in extras:
        return float(extras["sl_suggested"]), float(extras["tp_suggested"])
    if side == "BUY":
        return price - K_SL_FALLBACK * atr_val, price + K_TP_FALLBACK * atr_val
    return price + K_SL_FALLBACK * atr_val, price - K_TP_FALLBACK * atr_val


def run_backtest(label: str, strategy, df15: pd.DataFrame, df1h: pd.DataFrame) -> BacktestResult:
    """Walk forward bar-by-bar. Apply signals using the strategy's own SL/TP if provided."""
    print(f"\n=== Backtest: {label} ===")
    result = BacktestResult(label=label, final_equity=INITIAL_EQUITY, peak_equity=INITIAL_EQUITY)
    equity = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    open_pos: dict | None = None

    # Pre-slice 1h to align: for each 15m bar at time T, use 1h bars <= T
    df1h_index = df1h.index
    h_idx = 0

    for i in range(WARMUP_BARS, len(df15) - 1):
        bar = df15.iloc[i]
        bar_next = df15.iloc[i + 1]
        ts = df15.index[i]

        # advance 1h slice up to current 15m timestamp
        while h_idx + 1 < len(df1h) and df1h_index[h_idx + 1] <= ts:
            h_idx += 1
        if h_idx < 60:
            continue
        df1h_slice = df1h.iloc[:h_idx + 1]
        df15_slice = df15.iloc[:i + 1]

        # 1) check open position SL/TP on the next bar (avoid lookahead within same bar)
        if open_pos is not None:
            nb = bar_next
            high = float(nb["High"]); low = float(nb["Low"])
            exit_price = None; reason = None
            if open_pos["side"] == "BUY":
                if low <= open_pos["sl"]:
                    exit_price, reason = open_pos["sl"], "SL"
                elif high >= open_pos["tp"]:
                    exit_price, reason = open_pos["tp"], "TP"
            else:
                if high >= open_pos["sl"]:
                    exit_price, reason = open_pos["sl"], "SL"
                elif low <= open_pos["tp"]:
                    exit_price, reason = open_pos["tp"], "TP"

            if exit_price is not None:
                pnl = ((exit_price - open_pos["entry"]) if open_pos["side"] == "BUY"
                       else (open_pos["entry"] - exit_price)) * open_pos["qty"]
                stop_dist = abs(open_pos["entry"] - open_pos["sl"])
                r = pnl / (stop_dist * open_pos["qty"]) if stop_dist > 0 else 0
                equity += pnl
                peak = max(peak, equity)
                result.trades.append(Trade(
                    open_ts=open_pos["open_ts"], close_ts=df15.index[i + 1],
                    side=open_pos["side"], entry=open_pos["entry"], exit=exit_price,
                    sl=open_pos["sl"], tp=open_pos["tp"], qty=open_pos["qty"],
                    pnl=pnl, r=r, exit_reason=reason,
                ))
                # track drawdown
                dd = (peak - equity) / peak * 100
                result.max_dd_pct = max(result.max_dd_pct, dd)
                open_pos = None

        # 2) if flat, evaluate strategy on the now-closed bar
        if open_pos is None:
            ms = MarketState(symbol="GOLD", bars_entry=df15_slice, bars_trend=df1h_slice)
            sig = strategy.evaluate(ms)
            if sig is None:
                continue
            if sig.severity not in (Severity.BUY_READY, Severity.SELL_READY):
                continue

            side = sig.side
            price = float(bar["Close"])
            atr_val = sig.atr if sig.atr > 0 else 1.0
            sl, tp = _resolve_sl_tp(sig, side, price, atr_val)
            stop_dist = abs(price - sl)
            if stop_dist <= 0:
                continue
            qty = (equity * RISK_PER_TRADE) / stop_dist
            open_pos = {
                "side": side, "entry": price, "sl": sl, "tp": tp, "qty": qty,
                "open_ts": ts,
            }

    result.final_equity = equity
    result.peak_equity = peak
    return result


def print_comparison(a: BacktestResult, b: BacktestResult):
    rows = [
        ("Trades",            f"{a.n}", f"{b.n}"),
        ("Wins / Losses",     f"{a.wins} / {a.losses}", f"{b.wins} / {b.losses}"),
        ("Win rate",          f"{a.win_rate*100:.1f}%", f"{b.win_rate*100:.1f}%"),
        ("Total return",      f"{a.total_return_pct:+.2f}%", f"{b.total_return_pct:+.2f}%"),
        ("Expectancy / trade", f"{a.expectancy_R:+.3f}R", f"{b.expectancy_R:+.3f}R"),
        ("Avg win",            f"{a.avg_win_R:+.2f}R", f"{b.avg_win_R:+.2f}R"),
        ("Avg loss",           f"{a.avg_loss_R:+.2f}R", f"{b.avg_loss_R:+.2f}R"),
        ("Max drawdown",       f"{a.max_dd_pct:.2f}%", f"{b.max_dd_pct:.2f}%"),
        ("Final equity",       f"${a.final_equity:,.0f}", f"${b.final_equity:,.0f}"),
    ]
    w_metric = max(len(r[0]) for r in rows)
    w_a = max(len(r[1]) for r in rows + [("", a.label, "")])
    w_b = max(len(r[2]) for r in rows + [("", "", b.label)])
    print("")
    print("=" * (w_metric + w_a + w_b + 8))
    print(f"  {'METRIC'.ljust(w_metric)}  {a.label.ljust(w_a)}  {b.label.ljust(w_b)}")
    print("-" * (w_metric + w_a + w_b + 8))
    for m, va, vb in rows:
        print(f"  {m.ljust(w_metric)}  {va.ljust(w_a)}  {vb.ljust(w_b)}")
    print("=" * (w_metric + w_a + w_b + 8))


def main():
    df15, df1h = fetch()
    print(f"Got {len(df15)} 15m bars from {df15.index[0]} -> {df15.index[-1]}")
    print(f"Got {len(df1h)} 1h bars")

    breakout = BreakoutTrendStrategy(BreakoutTrendParams(
        ema_fast=50, ema_slow=200, atr_period=14,
        atr_min=10.0, atr_pct_min=0.5, min_trend_strength=0.0,
        use_higher_tf_gate=False,  # disable 4H gate to keep comparison apples-to-apples
    ))
    smc = SMCStrategy(SMCParams(
        htf_pivot=2, ltf_pivot=2,
        min_impulse_bars=3,
        poi_freshness_bars_htf=30,
        min_poi_score=3,
        require_ltf_choch=True,
        min_rr=1.5,
    ))

    res_breakout = run_backtest("breakout_trend", breakout, df15, df1h)
    res_smc = run_backtest("smc", smc, df15, df1h)

    print_comparison(res_breakout, res_smc)

    # Save trades for inspection
    out_dir = HERE / "data"
    out_dir.mkdir(exist_ok=True)
    for r in (res_breakout, res_smc):
        if r.trades:
            df = pd.DataFrame([t.__dict__ for t in r.trades])
            df.to_csv(out_dir / f"compare_{r.label}_trades.csv", index=False)
            print(f"  wrote {len(df)} {r.label} trades -> data/compare_{r.label}_trades.csv")


if __name__ == "__main__":
    main()
