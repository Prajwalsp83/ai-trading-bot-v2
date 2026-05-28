"""
Live paper trading loop. Runs forever (or until Ctrl+C).

What it does (every poll_seconds, default 60s):
  1. Fetch latest 15m + 1h gold bars from yfinance.
  2. If a new 15m bar closed:
       a. Check open position (if any) against the new bar's H/L for SL/TP hit.
       b. If still flat, ask strategy.evaluate() for a fresh signal.
       c. If BUY_READY/SELL_READY: open a simulated position with ATR stops.
  3. Otherwise: print a heartbeat and wait.

  4. Every entry/exit:
       - persists state to .paper_state.json (survives restart)
       - appends to data/paper_trades.csv
       - fires Telegram alert with severity-tagged message

Same `strategy.evaluate()` call as the backtest. By design — proves out the
live/backtest match. Currently uses yfinance/GC=F; switches to Upstox/GOLDM
after KYC clears (just swap the data fetcher).

Run in a dedicated terminal (it loops forever):
    cd ~/Documents/ai-trading-bot/v2
    source .venv/bin/activate
    python scripts/paper_live.py

Stop with Ctrl+C. State is saved so you can restart and continue.
"""
from __future__ import annotations

import csv
import json
import os
import signal
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml
import yfinance as yf
from dotenv import load_dotenv

# IST is UTC+5:30; we use a fixed offset so we don't depend on tz database/dst.
IST = timezone(timedelta(hours=5, minutes=30))

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from app.core.events import Severity                              # noqa: E402
from app.strategy.base import MarketState                         # noqa: E402
from app.strategy.breakout_trend import (                         # noqa: E402
    BreakoutTrendParams,
    BreakoutTrendStrategy,
)

warnings.filterwarnings("ignore")
load_dotenv(HERE / ".env")

# ---- config ----
with open(HERE / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PAPER_DATA_SYMBOL = "GC=F"                    # yfinance gold ticker
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")

POLL_SECONDS = CFG["poll_seconds"]
INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = CFG["risk"]["risk_per_trade_pct"]
K_SL = CFG["stops"]["k_sl"]
K_TP = CFG["stops"]["k_tp"]

# ---- P1: consecutive-loss cooldown ----
COOLDOWN_AFTER_N_LOSSES = CFG["risk"].get("cooldown_after_consecutive_losses", 2)
COOLDOWN_MINUTES_LOSSES = CFG["risk"].get("cooldown_minutes_after_losses", 240)

# ---- P2: drawdown-scaled risk ----
DD_SCALE_THRESHOLD = CFG["risk"].get("dd_scale_threshold_pct", 0.03)
DD_SCALE_FACTOR = CFG["risk"].get("dd_scale_factor", 0.5)

# ---- P6: reentry block ----
REENTRY_BLOCK_MIN = CFG["risk"].get("reentry_block_minutes", 120)

# ---- P3: daily summary timing (IST) ----
SUMMARY_HOUR_IST = CFG.get("reporting", {}).get("daily_summary_hour_ist", 23)
SUMMARY_MIN_IST = CFG.get("reporting", {}).get("daily_summary_minute_ist", 55)

# ---- P4: 4H trend gate ----
USE_4H_GATE = CFG["strategy"].get("use_4h_trend_gate", False)

STRATEGY = BreakoutTrendStrategy(BreakoutTrendParams(
    ema_fast=CFG["strategy"]["ema_fast"],
    ema_slow=CFG["strategy"]["ema_slow"],
    atr_period=CFG["strategy"]["atr_period"],
    atr_min=CFG["strategy"]["atr_min"],
    atr_pct_min=CFG["strategy"]["atr_pct_min"],
    min_trend_strength=CFG["strategy"]["min_trend_strength"],
    use_higher_tf_gate=USE_4H_GATE,
))

STATE_FILE = HERE / ".paper_state.json"
TRADES_CSV = HERE / "data" / "paper_trades.csv"
TRADES_CSV.parent.mkdir(exist_ok=True)

TRADE_COLS = [
    "trade_id", "open_time", "close_time", "side", "entry", "exit",
    "qty", "sl", "tp", "pnl", "r_realised", "duration_minutes",
    "atr_at_entry", "exit_reason",
]


# ============== state ==============
def load_state() -> dict:
    defaults = {
        "equity": INITIAL_EQUITY,
        "peak_equity": INITIAL_EQUITY,
        "open_position": None,
        "last_bar_ts": None,
        "next_trade_id": 1,
        "trades_today": 0,
        "pnl_today": 0.0,
        "today": None,
        # P1: consecutive-loss cooldown
        "consecutive_losses": 0,
        "cooldown_until_iso": None,
        # P6: reentry block
        "last_exit_iso": None,
        # P3: daily summary bookkeeping
        "last_summary_date_ist": None,
        "counters_today": {
            "entries": 0,
            "watches": 0,
            "breakouts": 0,
            "rejections": {},   # {reason: count}
        },
    }
    if STATE_FILE.exists():
        loaded = json.loads(STATE_FILE.read_text())
        # Forward-compat: backfill any missing keys from defaults
        for k, v in defaults.items():
            if k not in loaded:
                loaded[k] = v
        # Migrate counters_today subkeys
        if isinstance(loaded.get("counters_today"), dict):
            for sk, sv in defaults["counters_today"].items():
                if sk not in loaded["counters_today"]:
                    loaded["counters_today"][sk] = sv
        return loaded
    return defaults


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def ensure_journal_header():
    if not TRADES_CSV.exists():
        with TRADES_CSV.open("w", newline="") as f:
            csv.writer(f).writerow(TRADE_COLS)


def append_trade(record: dict):
    with TRADES_CSV.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADE_COLS).writerow(record)


# ============== telegram ==============
def tg_send(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log(f"telegram error: {e}")


def severity_tag(sev: Severity) -> str:
    return {
        Severity.WATCHLIST: "[WATCH]",
        Severity.BREAKOUT_WATCH: "[BREAKOUT WATCH]",
        Severity.BUY_READY: "[BUY READY]",
        Severity.SELL_READY: "[SELL READY]",
        Severity.ENTRY_CONFIRMED: "[ENTRY]",
        Severity.EXIT_ALERT: "[EXIT]",
        Severity.RISK_ALERT: "[RISK]",
    }.get(sev, "[INFO]")


# ============== data ==============
def _resample_4h(df1h: pd.DataFrame) -> pd.DataFrame:
    """yfinance doesn't expose 4h; we synthesize it from the 1h series."""
    if df1h.empty:
        return df1h
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df1h.columns:
        agg["Volume"] = "sum"
    return df1h.resample("4h").agg(agg).dropna()


def fetch_bars() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    try:
        df15 = yf.download(PAPER_DATA_SYMBOL, interval="15m", period="60d",
                           auto_adjust=True, progress=False)
        df1h = yf.download(PAPER_DATA_SYMBOL, interval="1h", period="730d",
                           auto_adjust=True, progress=False)
        for df in (df15, df1h):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        df15 = df15.dropna()
        df1h = df1h.dropna()
        df4h = _resample_4h(df1h)
        return df15, df1h, df4h
    except Exception as e:
        log(f"data fetch error: {e}")
        return None


# ============== logging ==============
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============== gates (P1 cooldown, P2 DD-scaling, P6 reentry block) ==============
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def can_open_new_trade(state: dict) -> tuple[bool, str]:
    """Return (allowed, reason_if_blocked). Checks cooldowns + reentry block."""
    now = _now_utc()
    cu = _parse_iso(state.get("cooldown_until_iso"))
    if cu and now < cu:
        mins = int((cu - now).total_seconds() // 60)
        return False, f"loss_cooldown ({mins}m left, after {state.get('consecutive_losses', 0)} losses)"
    le = _parse_iso(state.get("last_exit_iso"))
    if le and (now - le) < timedelta(minutes=REENTRY_BLOCK_MIN):
        mins = int((timedelta(minutes=REENTRY_BLOCK_MIN) - (now - le)).total_seconds() // 60)
        return False, f"reentry_block ({mins}m left)"
    return True, ""


def effective_risk_pct(state: dict) -> tuple[float, bool]:
    """P2: scale risk down when equity is in drawdown from peak."""
    peak = state.get("peak_equity") or INITIAL_EQUITY
    eq = state["equity"]
    if peak <= 0:
        return RISK_PER_TRADE, False
    dd = (peak - eq) / peak
    if dd >= DD_SCALE_THRESHOLD:
        return RISK_PER_TRADE * DD_SCALE_FACTOR, True
    return RISK_PER_TRADE, False


# ============== trade lifecycle ==============
def open_position(state: dict, signal_side: str, bar, atr_val: float) -> None:
    # P1 + P6: cooldown / reentry gates
    allowed, why = can_open_new_trade(state)
    if not allowed:
        log(f"entry blocked: {why}")
        # one-time alert per block reason+bar
        return

    entry = float(bar["Close"])
    if signal_side == "BUY":
        sl = entry - K_SL * atr_val
        tp = entry + K_TP * atr_val
    else:
        sl = entry + K_SL * atr_val
        tp = entry - K_TP * atr_val
    stop_dist = abs(entry - sl)
    if stop_dist <= 0:
        return

    # P2: drawdown-scaled risk
    risk_pct, scaled = effective_risk_pct(state)
    qty = (state["equity"] * risk_pct) / stop_dist

    pos = {
        "trade_id": state["next_trade_id"],
        "side": signal_side,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "qty": qty,
        "open_time": _now_utc().isoformat(timespec="seconds"),
        "atr_at_entry": atr_val,
        "risk_pct_used": risk_pct,
    }
    state["open_position"] = pos
    state["next_trade_id"] += 1
    state["counters_today"]["entries"] = state["counters_today"].get("entries", 0) + 1

    risk_note = f" (risk halved: DD)" if scaled else ""
    log(f"OPEN {signal_side}  entry={entry:.2f}  sl={sl:.2f}  tp={tp:.2f}  qty={qty:.2f}  risk={risk_pct*100:.1f}%{risk_note}")
    tg_send(
        f"<b>[ENTRY]</b> {signal_side} GOLD\n"
        f"Entry: {entry:.2f}\nSL: {sl:.2f}  TP: {tp:.2f}\n"
        f"Qty: {qty:.2f}  ATR: {atr_val:.2f}\n"
        f"Risk: {risk_pct*100:.1f}%{risk_note}\n"
        f"Equity: {state['equity']:,.0f}"
    )


def try_close_on_bar(state: dict, bar) -> None:
    """Check the just-closed bar for SL/TP hits. Conservative: SL wins ties."""
    pos = state["open_position"]
    if not pos:
        return
    high = float(bar["High"])
    low = float(bar["Low"])
    exit_price = None
    exit_reason = None

    if pos["side"] == "BUY":
        if low <= pos["sl"]:
            exit_price, exit_reason = pos["sl"], "SL"
        elif high >= pos["tp"]:
            exit_price, exit_reason = pos["tp"], "TP"
    else:
        if high >= pos["sl"]:
            exit_price, exit_reason = pos["sl"], "SL"
        elif low <= pos["tp"]:
            exit_price, exit_reason = pos["tp"], "TP"

    if exit_price is None:
        return

    pnl = ((exit_price - pos["entry"]) if pos["side"] == "BUY"
            else (pos["entry"] - exit_price)) * pos["qty"]
    stop_dist = abs(pos["entry"] - pos["sl"])
    r = pnl / (stop_dist * pos["qty"]) if stop_dist > 0 else 0
    state["equity"] += pnl
    state["peak_equity"] = max(state["peak_equity"], state["equity"])
    state["pnl_today"] += pnl
    state["trades_today"] += 1

    open_time_iso = pos["open_time"]
    close_time = datetime.now(timezone.utc)
    duration_min = int(
        (close_time - datetime.fromisoformat(open_time_iso)).total_seconds() // 60
    )

    record = {
        "trade_id": pos["trade_id"],
        "open_time": open_time_iso,
        "close_time": close_time.isoformat(timespec="seconds"),
        "side": pos["side"],
        "entry": round(pos["entry"], 4),
        "exit": round(exit_price, 4),
        "qty": round(pos["qty"], 4),
        "sl": round(pos["sl"], 4),
        "tp": round(pos["tp"], 4),
        "pnl": round(pnl, 2),
        "r_realised": round(r, 3),
        "duration_minutes": duration_min,
        "atr_at_entry": round(pos["atr_at_entry"], 3),
        "exit_reason": exit_reason,
    }
    append_trade(record)
    state["open_position"] = None
    state["last_exit_iso"] = close_time.isoformat(timespec="seconds")

    # P1: consecutive-loss tracker + cooldown trigger
    if pnl > 0:
        state["consecutive_losses"] = 0
        state["cooldown_until_iso"] = None  # winning resets any cooldown
        cooldown_note = ""
    else:
        state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
        if state["consecutive_losses"] >= COOLDOWN_AFTER_N_LOSSES:
            cu = close_time + timedelta(minutes=COOLDOWN_MINUTES_LOSSES)
            state["cooldown_until_iso"] = cu.isoformat(timespec="seconds")
            cooldown_note = (
                f"\nCooldown: paused {COOLDOWN_MINUTES_LOSSES//60}h "
                f"({state['consecutive_losses']} losses in a row)"
            )
        else:
            cooldown_note = ""

    emoji = "[WIN]" if pnl > 0 else "[LOSS]"
    log(f"CLOSE {pos['side']}  reason={exit_reason}  pnl={pnl:+.2f}  R={r:+.2f}  "
        f"equity={state['equity']:,.0f}  losses_in_row={state['consecutive_losses']}")
    tg_send(
        f"<b>[EXIT]</b> {emoji} {pos['side']} closed by {exit_reason}\n"
        f"P&amp;L: {pnl:+.2f} ({r:+.2f}R)\n"
        f"Equity: {state['equity']:,.0f}\n"
        f"Today: {state['trades_today']} trades, P&amp;L {state['pnl_today']:+.2f}"
        f"{cooldown_note}"
    )


# ============== signal handling ==============
LAST_ALERT_BAR: dict[Severity, str] = {}      # dedup per (severity, bar_ts)


def maybe_alert_for_signal(sig, bar_ts_str: str):
    """Send Telegram alerts for non-execution severities (watchlist/breakout watch)."""
    if sig.severity in (Severity.WATCHLIST, Severity.BREAKOUT_WATCH):
        key = f"{sig.severity.name}_{bar_ts_str}"
        if LAST_ALERT_BAR.get(sig.severity) == bar_ts_str:
            return
        LAST_ALERT_BAR[sig.severity] = bar_ts_str
        tg_send(
            f"<b>{severity_tag(sig.severity)}</b> {sig.side} GOLD\n"
            f"{sig.reason}\nPrice: {sig.price:.2f}  ATR: {sig.atr:.2f}"
        )


# ============== daily reset ==============
def reset_daily_if_needed(state: dict):
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("today") != today:
        state["today"] = today
        state["pnl_today"] = 0.0
        state["trades_today"] = 0
        # Daily telemetry counters reset at UTC midnight too
        state["counters_today"] = {
            "entries": 0,
            "watches": 0,
            "breakouts": 0,
            "rejections": {},
        }


# ============== P3: daily summary at 23:55 IST ==============
def maybe_send_daily_summary(state: dict):
    """Send a one-shot daily Telegram recap once we cross the configured IST time."""
    now_ist = datetime.now(IST)
    today_ist = now_ist.date().isoformat()
    if state.get("last_summary_date_ist") == today_ist:
        return
    if not (now_ist.hour > SUMMARY_HOUR_IST
            or (now_ist.hour == SUMMARY_HOUR_IST and now_ist.minute >= SUMMARY_MIN_IST)):
        return

    c = state.get("counters_today", {})
    rej = c.get("rejections", {}) or {}
    top_rej = ""
    if rej:
        top = Counter(rej).most_common(1)[0]
        top_rej = f"{top[0]} ({top[1]}x)"
    else:
        top_rej = "(none)"

    peak = state.get("peak_equity") or INITIAL_EQUITY
    eq = state["equity"]
    dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0.0

    pos = state.get("open_position")
    open_str = (f"{pos['side']} @ {pos['entry']:.2f} (sl {pos['sl']:.2f} tp {pos['tp']:.2f})"
                if pos else "flat")
    cu = _parse_iso(state.get("cooldown_until_iso"))
    cooldown_str = ""
    if cu and cu > _now_utc():
        mins_left = int((cu - _now_utc()).total_seconds() // 60)
        cooldown_str = f"\nCooldown: {mins_left}m left ({state.get('consecutive_losses',0)} losses)"

    tg_send(
        f"<b>[DAILY SUMMARY]</b> {today_ist}\n"
        f"Entries: {c.get('entries',0)}\n"
        f"Breakout watches: {c.get('breakouts',0)}\n"
        f"Trend watches: {c.get('watches',0)}\n"
        f"Top rejection: {top_rej}\n"
        f"Equity: {eq:,.0f}  |  DD: -{dd_pct:.2f}%\n"
        f"Position: {open_str}{cooldown_str}"
    )
    state["last_summary_date_ist"] = today_ist


# ============== main loop ==============
RUNNING = True


def handle_sigint(signum, frame):
    global RUNNING
    RUNNING = False
    print("\nStopping after current cycle… (Ctrl+C again to force)")


def main() -> int:
    if not TG_TOKEN or not TG_CHAT:
        log("WARNING: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in .env — alerts disabled.")

    signal.signal(signal.SIGINT, handle_sigint)
    ensure_journal_header()
    state = load_state()
    log(f"Paper trading started. equity={state['equity']:,.0f}  "
        f"open_position={'yes' if state['open_position'] else 'no'}")
    tg_send(f"<b>[BOT START]</b>\nEquity: {state['equity']:,.0f}  "
            f"Open: {'yes' if state['open_position'] else 'no'}")

    while RUNNING:
        try:
            reset_daily_if_needed(state)
            bars = fetch_bars()
            if bars is None:
                time.sleep(POLL_SECONDS)
                continue
            df15, df1h, df4h = bars
            if len(df15) < 220 or len(df1h) < 220:
                log("not enough history yet, waiting…")
                time.sleep(POLL_SECONDS)
                continue

            latest_ts = str(df15.index[-1])
            new_bar = state["last_bar_ts"] != latest_ts

            if new_bar:
                # 1. Check SL/TP on the bar that just closed.
                try_close_on_bar(state, df15.iloc[-1])

                # 2. If flat, evaluate strategy.
                if state["open_position"] is None:
                    sig = STRATEGY.evaluate(MarketState(
                        symbol="GOLD",
                        bars_entry=df15, bars_trend=df1h,
                        bars_higher=df4h if USE_4H_GATE else None,
                    ))
                    if sig is not None:
                        log(f"signal: {sig.severity.name} {sig.side}  "
                            f"price={sig.price:.2f} reason={sig.reason}")
                        # Count rejections / watches / breakouts for the daily summary.
                        ct = state["counters_today"]
                        if sig.extras and sig.extras.get("skipped"):
                            reason = sig.extras.get("rejection_reason", "unknown")
                            rj = ct.setdefault("rejections", {})
                            rj[reason] = rj.get(reason, 0) + 1
                        elif sig.severity == Severity.WATCHLIST:
                            ct["watches"] = ct.get("watches", 0) + 1
                            maybe_alert_for_signal(sig, latest_ts)
                        elif sig.severity == Severity.BREAKOUT_WATCH:
                            ct["breakouts"] = ct.get("breakouts", 0) + 1
                            maybe_alert_for_signal(sig, latest_ts)
                        elif sig.severity in (Severity.BUY_READY, Severity.SELL_READY):
                            open_position(state, sig.side, df15.iloc[-1], sig.atr)
                else:
                    pos = state["open_position"]
                    log(f"holding {pos['side']} @ {pos['entry']:.2f}  "
                        f"sl={pos['sl']:.2f} tp={pos['tp']:.2f}  qty={pos['qty']:.2f}")

                state["last_bar_ts"] = latest_ts
                save_state(state)
            else:
                # Heartbeat — no new bar yet.
                last_close = float(df15.iloc[-1]["Close"])
                pos = state["open_position"]
                pos_str = (f"{pos['side']} @ {pos['entry']:.2f}"
                           if pos else "flat")
                log(f"heartbeat  price={last_close:.2f}  {pos_str}  "
                    f"equity={state['equity']:,.0f}  today={state['pnl_today']:+.2f}")

            # P3: independent of new-bar — check every cycle for the IST summary trigger.
            maybe_send_daily_summary(state)
            save_state(state)

        except Exception as e:
            log(f"loop error: {e}")
            tg_send(f"<b>[RISK]</b> loop error: {e}")

        time.sleep(POLL_SECONDS)

    log("Stopped. State saved.")
    tg_send(f"<b>[BOT STOP]</b>\nEquity: {state['equity']:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
