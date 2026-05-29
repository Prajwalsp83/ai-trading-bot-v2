"""
Phase B.3 — Realistic backtest engine for the live strategies.

Bar-by-bar simulation with:
  - Spread modeling (BUY pays ask, SELL pays bid; exit reversed)
  - Slippage (random 0-2 pips on entry, 1-3 pips on stop)
  - Commission ($7/lot round-trip = $3.50 per side, configurable)
  - First-touch SL/TP resolution within a bar
  - Optional intra-bar conservative rule (if both SL+TP hit same bar, assume SL)
  - Account equity tracking, max drawdown, peak equity

Public API:
    engine = BacktestEngine(specs, params)
    result = engine.run(df15, df1h, df4h, strategy_fn, signal_params,
                         starting_equity=1000.0, risk_per_trade_pct=0.01)
    result.trades  -> pd.DataFrame   (one row per closed trade)
    result.equity  -> pd.DataFrame   (one row per bar with equity, dd, pos)
    result.summary() -> dict         (basic metrics)

Strategy_fn signature: (df15_slice, df1h_slice, df4h_slice_or_None,
                        signal_params) -> dict | None
                       Must return signal dict with: severity, side, price, atr.
                       For SMC also: sl_suggested, tp_suggested.
                       Engine reads atr + applies k_sl/k_tp for breakout.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd


# ============================== PARAMS ==============================
@dataclass
class SymbolSpecs:
    """Broker contract specs — load from data/history/{symbol}_specs.json."""
    symbol: str
    contract_size: float          # oz per lot
    volume_min: float
    volume_step: float
    point: float                  # min price increment
    digits: int
    avg_spread_points: int        # snapshot spread in points (1 point = 1 * point)
    commission_per_lot_rt_usd: float

    @classmethod
    def from_json(cls, path: Path | str) -> "SymbolSpecs":
        with open(path) as f:
            d = json.load(f)
        return cls(
            symbol=d["symbol"],
            contract_size=float(d["contract_size"]),
            volume_min=float(d["volume_min"]),
            volume_step=float(d["volume_step"]),
            point=float(d["point"]),
            digits=int(d["digits"]),
            avg_spread_points=int(d.get("current_spread_points", 25)),
            commission_per_lot_rt_usd=float(d.get("assumed_commission_per_lot_rt_usd", 7.0)),
        )


@dataclass
class CostModel:
    """How to simulate transaction costs. Defaults reasonable for XM Gold."""
    spread_points: int = 25              # override symbol specs if set
    slippage_entry_pips_max: float = 2.0
    slippage_stop_pips_max: float = 3.0
    commission_per_lot_rt_usd: float = 7.0
    pessimistic_intrabar: bool = True    # if SL+TP same bar, count SL (conservative)


@dataclass
class BacktestParams:
    starting_equity: float = 1000.0
    risk_per_trade_pct: float = 0.01
    warmup_bars: int = 250                # don't evaluate until this many 15m bars
    k_sl: float = 1.5                     # ATR multiplier for SL (breakout only)
    k_tp: float = 2.5                     # ATR multiplier for TP (breakout only)
    poll_every_bars: int = 1              # 1 = check every bar; 4 = check every 4 bars (faster)

    # === Window-cap to keep evaluator O(1) per bar ===
    # The strategy uses EMA-200 + ATR-100 percentile, both stabilize well within
    # ~500 bars. Passing the full history per bar makes backtest O(n^2).
    # We pass only the last N bars to each call.
    eval_window_15m: int = 500
    eval_window_1h: int = 300
    eval_window_4h: int = 200


# ============================== RESULT ==============================
@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    params: BacktestParams
    cost: CostModel
    specs: SymbolSpecs
    strategy_name: str

    def summary(self) -> dict:
        """Compact metrics dict for quick assessment."""
        t = self.trades
        e = self.equity
        if len(t) == 0:
            return {"trades": 0, "note": "no trades fired"}

        wins = t[t["pnl_usd"] > 0]
        losses = t[t["pnl_usd"] <= 0]
        n = len(t); nw = len(wins); nl = len(losses)
        win_rate = nw / n if n else 0
        avg_win = wins["pnl_usd"].mean() if nw else 0
        avg_loss = losses["pnl_usd"].mean() if nl else 0
        gross_win = wins["pnl_usd"].sum() if nw else 0
        gross_loss = abs(losses["pnl_usd"].sum()) if nl else 0
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

        # Longest losing streak
        streak = 0; max_streak = 0
        for pnl in t["pnl_usd"]:
            if pnl <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        # Equity-curve metrics
        eq = e["equity"]
        net_pnl = eq.iloc[-1] - self.params.starting_equity if len(eq) else 0
        max_dd_pct = e["dd_pct"].max() * 100 if len(e) else 0
        # Sharpe — simple proxy from daily returns
        daily = e.set_index("ts")["equity"].resample("1D").last().pct_change().dropna()
        sharpe = (daily.mean() / daily.std() * (252 ** 0.5)) if (len(daily) > 1 and daily.std() > 0) else 0

        return {
            "trades": n,
            "wins": nw, "losses": nl,
            "win_rate_pct": round(win_rate * 100, 2),
            "net_pnl_usd": round(net_pnl, 2),
            "net_pnl_pct": round(net_pnl / self.params.starting_equity * 100, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "avg_r": round(t["r_realised"].mean(), 3),
            "max_dd_pct": round(max_dd_pct, 2),
            "sharpe_annualized": round(sharpe, 2),
            "longest_losing_streak": max_streak,
            "first_trade_ts": str(t["open_time"].iloc[0]),
            "last_trade_ts": str(t["close_time"].iloc[-1]),
        }


# ============================== ENGINE ==============================
class BacktestEngine:
    """Bar-by-bar walk. One open position at a time. No partial fills."""

    def __init__(self, specs: SymbolSpecs, cost: CostModel | None = None,
                 params: BacktestParams | None = None, seed: int = 42):
        self.specs = specs
        self.cost = cost or CostModel()
        # Use snapshot spread if not overridden in CostModel
        if self.cost.spread_points == 25 and specs.avg_spread_points != 25:
            self.cost.spread_points = specs.avg_spread_points
        self.params = params or BacktestParams()
        self.rng = random.Random(seed)

    # ---- cost helpers ----
    def _spread_price(self) -> float:
        return self.cost.spread_points * self.specs.point

    def _slip(self, max_pips: float) -> float:
        """Random slippage in price units (0 to max_pips * point)."""
        return self.rng.uniform(0, max_pips) * self.specs.point

    def _commission(self, lots: float) -> float:
        """Per-side commission. Round-trip = 2 * this."""
        return lots * self.cost.commission_per_lot_rt_usd / 2.0

    def _round_lot(self, lots: float) -> float:
        step = self.specs.volume_step
        rounded = round(round(lots / step) * step, 8)
        if rounded < self.specs.volume_min:
            return 0.0
        return rounded

    def _size_lots(self, equity: float, risk_pct: float, stop_dist: float) -> float:
        if stop_dist <= 0:
            return 0.0
        risk_usd = equity * risk_pct
        oz = risk_usd / stop_dist
        return self._round_lot(oz / self.specs.contract_size)

    # ---- main loop ----
    def run(self, df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame | None,
            strategy_fn: Callable, signal_params, strategy_name: str = "unknown") -> BacktestResult:
        """Walk df15 from warmup_bars onwards. Call strategy_fn on every Nth bar.
        On BUY_READY/SELL_READY, open position. Walk subsequent bars for SL/TP."""
        trades = []
        equity_rows = []
        equity = self.params.starting_equity
        peak_equity = equity
        open_pos = None     # dict with entry side/price/sl/tp/lots/open_time/atr/risk_pct
        trade_id = 0

        # Pre-align: build searchsorted indices for 1h/4h alignment per 15m bar
        # (faster than .loc[<= ts] in tight loops)
        h1_ts = df1h.index.values
        h4_ts = df4h.index.values if df4h is not None else None

        n = len(df15)
        import time as _t
        t_start = _t.time()
        report_every = max(1000, n // 20)
        for i in range(self.params.warmup_bars, n):
            if i % report_every == 0 and i > self.params.warmup_bars:
                elapsed = _t.time() - t_start
                pct = (i - self.params.warmup_bars) / (n - self.params.warmup_bars) * 100
                rate = (i - self.params.warmup_bars) / max(elapsed, 0.001)
                eta = (n - i) / max(rate, 0.001)
                print(f"  [progress] bar {i:>7,}/{n:,} ({pct:>5.1f}%) "
                      f"trades={len(trades):>4} eq=${equity:>9.2f} "
                      f"rate={rate:>5.0f}b/s ETA={eta:>4.0f}s", flush=True)
            ts = df15.index[i]
            bar = df15.iloc[i]
            bar_high = float(bar["High"])
            bar_low = float(bar["Low"])
            bar_open = float(bar["Open"])
            bar_close = float(bar["Close"])

            # Drawdown tracking
            peak_equity = max(peak_equity, equity)
            dd_pct = max(0.0, (peak_equity - equity) / peak_equity) if peak_equity > 0 else 0.0
            equity_rows.append({"ts": ts, "equity": equity, "peak_equity": peak_equity,
                                 "dd_pct": dd_pct, "open_position": open_pos is not None})

            # ===== Manage open position: check SL/TP this bar =====
            if open_pos is not None:
                hit_sl, hit_tp = self._check_intrabar(bar_high, bar_low, open_pos)
                exit_price = None
                exit_reason = None
                if hit_sl and hit_tp:
                    # Both hit same bar — pessimistic = assume SL
                    if self.cost.pessimistic_intrabar:
                        exit_price = open_pos["sl"]
                        exit_reason = "SL"
                    else:
                        exit_price = open_pos["tp"]
                        exit_reason = "TP"
                elif hit_sl:
                    exit_price = open_pos["sl"] + (self._slip(self.cost.slippage_stop_pips_max)
                                                    * (-1 if open_pos["side"] == "BUY" else 1))
                    exit_reason = "SL"
                elif hit_tp:
                    exit_price = open_pos["tp"]
                    exit_reason = "TP"

                if exit_price is not None:
                    # Compute P&L
                    direction = 1 if open_pos["side"] == "BUY" else -1
                    pnl_price = (exit_price - open_pos["entry"]) * direction
                    pnl_usd = (pnl_price * open_pos["lots"] * self.specs.contract_size
                                - 2 * self._commission(open_pos["lots"]))
                    stop_dist = abs(open_pos["entry"] - open_pos["sl"])
                    r_realised = (pnl_usd / (stop_dist * open_pos["lots"] * self.specs.contract_size)
                                    if stop_dist > 0 else 0.0)
                    duration = int((ts - open_pos["open_time"]).total_seconds() // 60)

                    trades.append({
                        "trade_id": open_pos["trade_id"],
                        "side": open_pos["side"],
                        "open_time": open_pos["open_time"],
                        "close_time": ts,
                        "entry": open_pos["entry"],
                        "exit": exit_price,
                        "sl": open_pos["sl"],
                        "tp": open_pos["tp"],
                        "lots": open_pos["lots"],
                        "pnl_usd": pnl_usd,
                        "r_realised": r_realised,
                        "duration_min": duration,
                        "exit_reason": exit_reason,
                        "atr_at_entry": open_pos["atr"],
                        "risk_pct_used": open_pos["risk_pct"],
                    })
                    equity += pnl_usd
                    open_pos = None

            # ===== Look for new entries (only if flat + on poll cadence) =====
            if open_pos is None and (i % self.params.poll_every_bars == 0):
                # Window-cap each timeframe so strategy is O(1) per call.
                # EMA-200 + ATR-percentile-100 fully stabilize within these windows.
                lo15 = max(0, i + 1 - self.params.eval_window_15m)
                df15_slice = df15.iloc[lo15:i + 1]
                # 1h/4h: searchsorted finds positions, then we take the most recent N
                h1_pos = int(np.searchsorted(h1_ts, ts.to_numpy(), side="right"))
                lo1h = max(0, h1_pos - self.params.eval_window_1h)
                df1h_slice = df1h.iloc[lo1h:h1_pos]
                df4h_slice = None
                if df4h is not None:
                    h4_pos = int(np.searchsorted(h4_ts, ts.to_numpy(), side="right"))
                    lo4h = max(0, h4_pos - self.params.eval_window_4h)
                    df4h_slice = df4h.iloc[lo4h:h4_pos]

                sig = self._call_strategy(strategy_fn, df15_slice, df1h_slice, df4h_slice,
                                           signal_params, strategy_name)
                if sig is None:
                    continue
                sev = sig.get("severity")
                if sev not in ("BUY_READY", "SELL_READY"):
                    continue

                # Determine entry/SL/TP
                side = sig["side"]
                # ENTRY at NEXT bar's open (more realistic than current close)
                if i + 1 >= n:
                    continue
                next_bar = df15.iloc[i + 1]
                next_open = float(next_bar["Open"])
                spread = self._spread_price()
                slip = self._slip(self.cost.slippage_entry_pips_max)
                if side == "BUY":
                    entry = next_open + spread / 2 + slip   # pay ask + slippage
                else:
                    entry = next_open - spread / 2 - slip   # sell at bid - slippage

                # SL/TP: SMC provides its own, breakout uses k_sl/k_tp from params
                if "sl_suggested" in sig and "tp_suggested" in sig:
                    sl = float(sig["sl_suggested"])
                    tp = float(sig["tp_suggested"])
                else:
                    atr_val = float(sig.get("atr", 0))
                    if atr_val <= 0:
                        continue
                    if side == "BUY":
                        sl = entry - self.params.k_sl * atr_val
                        tp = entry + self.params.k_tp * atr_val
                    else:
                        sl = entry + self.params.k_sl * atr_val
                        tp = entry - self.params.k_tp * atr_val

                stop_dist = abs(entry - sl)
                if stop_dist <= 0:
                    continue
                lots = self._size_lots(equity, self.params.risk_per_trade_pct, stop_dist)
                if lots <= 0:
                    continue

                trade_id += 1
                open_pos = {
                    "trade_id": trade_id,
                    "side": side, "entry": entry, "sl": sl, "tp": tp,
                    "lots": lots, "open_time": df15.index[i + 1],   # entry timestamp = next bar
                    "atr": float(sig.get("atr", 0)),
                    "risk_pct": self.params.risk_per_trade_pct,
                }

        # Build result
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["trade_id","side","open_time","close_time","entry","exit","sl","tp",
                     "lots","pnl_usd","r_realised","duration_min","exit_reason",
                     "atr_at_entry","risk_pct_used"])
        equity_df = pd.DataFrame(equity_rows)
        return BacktestResult(trades=trades_df, equity=equity_df,
                              params=self.params, cost=self.cost,
                              specs=self.specs, strategy_name=strategy_name)

    def _check_intrabar(self, bar_high: float, bar_low: float, pos: dict) -> tuple[bool, bool]:
        """Did the bar's [low,high] range touch SL and/or TP?"""
        if pos["side"] == "BUY":
            hit_sl = bar_low <= pos["sl"]
            hit_tp = bar_high >= pos["tp"]
        else:
            hit_sl = bar_high >= pos["sl"]
            hit_tp = bar_low <= pos["tp"]
        return hit_sl, hit_tp

    def _call_strategy(self, fn, df15, df1h, df4h, params, strategy_name):
        """Strategy signature varies — breakout takes df4h, others take just df15+df1h."""
        try:
            if strategy_name in ("smc", "mean_reversion"):
                return fn(df15, df1h, params)
            else:
                return fn(df15, df1h, df4h, params)
        except Exception as e:
            print(f"[backtest] strategy error at {df15.index[-1]}: {type(e).__name__}: {e}")
            return None
