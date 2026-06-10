"""
MT5 live trading bot for XM demo gold (GOLD.i#).

Mirrors paper_live.py but uses MetaTrader5 for data + execution.

Strategy logic is INLINED for VPS deployment so this file is the only
Python file you need on the VPS (besides a .env with Telegram creds).
Once we set up git on the VPS we'll refactor to import from v2/app/.

Run:
    python C:\\bot\\mt5_live.py

Stop with Ctrl+C. State is saved to .mt5_state.json so you can restart
and continue. Open positions are tracked by MT5 itself — we just query.
"""
from __future__ import annotations

import csv
import json
import os
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import requests
from dotenv import load_dotenv


# HERE = project root. Bot lives at <ROOT>/scripts/mt5_live.py, so parent.parent
# is the project root. Works for both Mac dev (v2/scripts/...) and VPS
# (C:\ai-trading-bot\scripts\...). State files, data/, logs/ all sit at root.
SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
load_dotenv(HERE / ".env")
sys.path.insert(0, str(SCRIPT_DIR))

# Shared pro-trader infrastructure (sessions, calendar, news, kelly, dd, regime)
from _bot_common import (  # noqa: E402
    is_tradeable_session, current_session,
    GateConfig, composite_pre_signal_gate, directional_news_gate,
    classify_regime, RegimeParams,
    compute_effective_risk, KellyParams, DEFAULT_DD_TIERS,
    init_mt5_headless, check_mt5_alive_or_reconnect, reset_mt5_failure_counter,
)

# Postgres journal (degrades gracefully if DATABASE_URL not set / DB unreachable)
from _journal import record_trade as pg_record_trade, record_signal as pg_record_signal  # noqa: E402
from _journal import snapshot_equity as pg_snapshot_equity, log_event as pg_log_event  # noqa: E402

# ML meta-labeler (shadow mode default — logs decisions, doesn't act on them)
from _meta_scorer import score_signal_live  # noqa: E402

BOT_NAME = "breakout"


# ============================== CONFIG ===============================
# All knobs loaded from config.yaml + .env via _config_loader.
# Compat aliases below keep the rest of the bot unchanged — the names
# `RISK_PER_TRADE_PCT`, `MAGIC` etc still work but now resolve from config.
from _config_loader import load_config  # noqa: E402

CFG = load_config("breakout")

# --- MT5 / symbol ---
SYMBOL = CFG.mt5.symbol
MAGIC = CFG.strategy.magic
POLL_SECONDS = CFG.mt5.poll_seconds

# --- risk + capital control ---
RISK_PER_TRADE_PCT = CFG.risk.risk_per_trade_pct
DAILY_LOSS_CAP_PCT = CFG.risk.daily_loss_cap_pct
MAX_DD_PCT = CFG.risk.max_drawdown_pct
COOLDOWN_AFTER_N_LOSSES = CFG.risk.cooldown_after_consecutive_losses
COOLDOWN_MINUTES_LOSSES = CFG.risk.cooldown_minutes_after_losses
REENTRY_BLOCK_MIN = CFG.risk.reentry_block_minutes
USE_KELLY = CFG.risk.kelly.enabled
USE_REGIME_WEIGHT = CFG.regime.enabled

# --- strategy params ---
EMA_FAST = CFG.strategy.ema_fast
EMA_SLOW = CFG.strategy.ema_slow
ATR_PERIOD = CFG.strategy.atr_period
ATR_MIN = CFG.strategy.atr_min
ATR_PCT_MIN = CFG.strategy.atr_pct_min
MIN_TREND_STRENGTH = CFG.strategy.min_trend_strength
K_SL = CFG.strategy.k_sl
K_TP = CFG.strategy.k_tp
USE_4H_GATE = CFG.strategy.use_4h_trend_gate

# --- telegram (still from env directly for module-level helpers) ---
TG_TOKEN = CFG.telegram.bot_token
TG_CHAT = CFG.telegram.chat_id

# --- daily summary timing (IST) ---
IST = timezone(timedelta(hours=5, minutes=30))
SUMMARY_HOUR_IST = CFG.reporting.daily_summary_hour_ist
SUMMARY_MIN_IST = CFG.reporting.daily_summary_minute_ist

# --- files ---
STATE_FILE = HERE / ".mt5_state.json"
TRADES_CSV = HERE / CFG.journal_csv.lstrip("./")
TRADES_CSV.parent.mkdir(exist_ok=True)
TRADE_COLS = [
    "trade_id", "open_time", "close_time", "side", "entry", "exit",
    "lots", "sl", "tp", "pnl_usd", "r_realised", "duration_minutes",
    "atr_at_entry", "exit_reason", "ticket",
]

# Calendar + news cache paths
CALENDAR_PATH = HERE / CFG.calendar.path.lstrip("./")
NEWS_CACHE = HERE / CFG.news.cache_path.lstrip("./")


# ============================== HELPERS ==============================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def ensure_journal_header():
    if not TRADES_CSV.exists():
        with TRADES_CSV.open("w", newline="") as f:
            csv.writer(f).writerow(TRADE_COLS)


def append_trade(record: dict):
    # CSV (kept as local backup, always succeeds)
    with TRADES_CSV.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADE_COLS).writerow(record)
    # Postgres (best-effort; logs warning on failure, never raises)
    try:
        pg_record_trade(BOT_NAME, MAGIC, record)
    except Exception as e:
        log(f"pg_record_trade error (non-fatal): {e}")


# ============================== STATE ================================
def load_state() -> dict:
    defaults = {
        "peak_equity": None,        # set on first cycle from MT5
        "open_ticket": None,        # MT5 position ticket if we have an open trade
        "open_meta": None,          # our metadata for the open position
        "last_bar_ts": None,
        "next_trade_id": 1,
        "trades_today": 0,
        "pnl_today_usd": 0.0,
        "today_utc": None,
        # P1 cooldown
        "consecutive_losses": 0,
        "cooldown_until_iso": None,
        # P6 reentry block
        "last_exit_iso": None,
        # P3 summary bookkeeping
        "last_summary_date_ist": None,
        "counters_today": {
            "entries": 0,
            "watches": 0,
            "breakouts": 0,
            "rejections": {},
        },
    }
    if STATE_FILE.exists():
        loaded = json.loads(STATE_FILE.read_text())
        for k, v in defaults.items():
            if k not in loaded:
                loaded[k] = v
        if isinstance(loaded.get("counters_today"), dict):
            for sk, sv in defaults["counters_today"].items():
                if sk not in loaded["counters_today"]:
                    loaded["counters_today"][sk] = sv
        return loaded
    return defaults


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ============================ INDICATORS =============================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    # Wilder's smoothing
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def atr_percentile(atr_series: pd.Series, lookback: int = 100) -> pd.Series:
    return atr_series.rolling(lookback).rank(pct=True)


# ============================ STRATEGY ===============================
def evaluate(df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame | None):
    """Inlined breakout-trend strategy. Returns a dict with side/severity/etc, or None."""
    if len(df15) < EMA_SLOW + 2 or len(df1h) < EMA_SLOW + 2:
        return None

    ema_f_e = ema(df15["Close"], EMA_FAST)
    ema_s_e = ema(df15["Close"], EMA_SLOW)
    atr_e = atr(df15["High"], df15["Low"], df15["Close"], ATR_PERIOD)

    ema_f_t = ema(df1h["Close"], EMA_FAST)
    ema_s_t = ema(df1h["Close"], EMA_SLOW)

    last = df15.iloc[-1]
    prev = df15.iloc[-2]
    price = float(last["Close"])
    atr_val = float(atr_e.iloc[-1])

    if pd.isna(atr_val) or any(pd.isna(s.iloc[-1]) for s in
                               (ema_f_e, ema_s_e, ema_f_t, ema_s_t)):
        return None

    # Vol/trend filters
    rejection = None
    if ATR_PCT_MIN > 0:
        ap = atr_percentile(atr_e, lookback=100).iloc[-1]
        if pd.isna(ap) or ap < ATR_PCT_MIN:
            rejection = "atr_pct_too_low"
    if rejection is None and MIN_TREND_STRENGTH > 0:
        ts_strength = abs(ema_f_e.iloc[-1] - ema_s_e.iloc[-1]) / ema_s_e.iloc[-1]
        if ts_strength < MIN_TREND_STRENGTH:
            rejection = "trend_strength_too_low"

    # 4H gate
    h4_up = h4_dn = True
    if USE_4H_GATE and df4h is not None and len(df4h) >= EMA_SLOW + 2:
        ema_f_h = ema(df4h["Close"], EMA_FAST)
        ema_s_h = ema(df4h["Close"], EMA_SLOW)
        if not (pd.isna(ema_f_h.iloc[-1]) or pd.isna(ema_s_h.iloc[-1])):
            h4_up = bool(ema_f_h.iloc[-1] > ema_s_h.iloc[-1])
            h4_dn = bool(ema_f_h.iloc[-1] < ema_s_h.iloc[-1])

    if rejection is not None:
        return {"severity": "SKIPPED", "side": None, "price": price, "atr": atr_val,
                "reason": f"skipped: {rejection}", "rejection_reason": rejection}

    long_cond = {
        "15m_stack_up": bool(ema_f_e.iloc[-1] > ema_s_e.iloc[-1]),
        "1h_stack_up":  bool(ema_f_t.iloc[-1] > ema_s_t.iloc[-1]),
        "breakout_up":  bool(last["High"] > prev["High"]),
        "atr_ok":       bool(atr_val >= ATR_MIN),
        "4h_stack_up":  h4_up,
    }
    short_cond = {
        "15m_stack_dn": bool(ema_f_e.iloc[-1] < ema_s_e.iloc[-1]),
        "1h_stack_dn":  bool(ema_f_t.iloc[-1] < ema_s_t.iloc[-1]),
        "breakout_dn":  bool(last["Low"] < prev["Low"]),
        "atr_ok":       bool(atr_val >= ATR_MIN),
        "4h_stack_dn":  h4_dn,
    }
    long_n = sum(long_cond.values())
    short_n = sum(short_cond.values())
    n_total = len(long_cond)

    if long_n == n_total:
        return {"severity": "BUY_READY", "side": "BUY", "price": price, "atr": atr_val,
                "reason": f"All {n_total} long conditions met", "conditions": long_cond}
    if short_n == n_total:
        return {"severity": "SELL_READY", "side": "SELL", "price": price, "atr": atr_val,
                "reason": f"All {n_total} short conditions met", "conditions": short_cond}
    if long_n == n_total - 1:
        return {"severity": "BREAKOUT_WATCH", "side": "BUY", "price": price, "atr": atr_val,
                "reason": f"{long_n} of {n_total} long conditions met", "conditions": long_cond}
    if short_n == n_total - 1:
        return {"severity": "BREAKOUT_WATCH", "side": "SELL", "price": price, "atr": atr_val,
                "reason": f"{short_n} of {n_total} short conditions met", "conditions": short_cond}
    if long_cond["1h_stack_up"]:
        return {"severity": "WATCHLIST", "side": "BUY", "price": price, "atr": atr_val,
                "reason": "1H trend up, awaiting 15m alignment"}
    if short_cond["1h_stack_dn"]:
        return {"severity": "WATCHLIST", "side": "SELL", "price": price, "atr": atr_val,
                "reason": "1H trend down, awaiting 15m alignment"}
    return None


# ============================== DATA =================================
def _rates_to_df(rates) -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close",
                            "tick_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_bars():
    """Returns (df15, df1h, df4h) DataFrames pulled from MT5."""
    try:
        # start_pos=1 drops bar 0 (the in-progress candle, which repaints
        # until it closes). The backtest only ever sees completed bars, so
        # the live bot must too -- otherwise it evaluates a forming bar that
        # can reverse before close (look-ahead bias vs the backtest). The
        # last_bar_ts dedupe still yields exactly one evaluation per closed bar.
        r15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 1, 500)
        r1h = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 1, 500)
        r4h = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 1, 500)
        return _rates_to_df(r15), _rates_to_df(r1h), _rates_to_df(r4h)
    except Exception as e:
        log(f"data fetch error: {e}")
        return None, None, None


# =========================== ACCOUNT/EQUITY ==========================
def get_equity() -> float:
    info = mt5.account_info()
    return float(info.equity) if info else 0.0


def get_balance() -> float:
    info = mt5.account_info()
    return float(info.balance) if info else 0.0


# ============================== GATES ================================
def _build_gate_config() -> GateConfig:
    return GateConfig(
        use_session_filter=True,
        use_calendar_filter=True,
        use_news_filter=True,
        news_block_threshold=0.35,
        calendar_path=CALENDAR_PATH,
        av_api_key=os.getenv("ALPHA_VANTAGE_KEY"),
        av_cache_path=NEWS_CACHE,
    )


def can_open_new_trade(state: dict, side: str | None = None, gate_cfg: GateConfig | None = None):
    """Composite gate. Returns (allowed, reason, news_summary_or_None).

    Order of checks (cheapest + most-impactful first):
      1. Max-DD kill-switch (HARD halt — bot won't trade until peak resets)
      2. Daily loss cap (HARD halt for the day)
      3. Cooldown after consecutive losses
      4. Reentry block after any exit
      5. Composite pre-signal gates (sessions / calendar)
      6. Directional news gate (sentiment vs trade direction)
    """
    now = _now_utc()

    # === 1. MAX DRAWDOWN KILL-SWITCH (HARD) ===
    # Equity has fallen >= MAX_DD_PCT below peak. Bot refuses new trades.
    # Requires MANUAL reset of state["peak_equity"] to recover.
    equity = get_equity() if 'get_equity' in globals() else None
    peak = state.get("peak_equity")
    if equity and peak and peak > 0:
        dd = (peak - equity) / peak
        if dd >= MAX_DD_PCT:
            return False, (f"MAX_DD_KILL_SWITCH: drawdown {dd*100:.2f}% "
                           f">= cap {MAX_DD_PCT*100:.0f}% (peak ${peak:,.2f}, eq ${equity:,.2f}). "
                           f"Manual reset of .mt5_state.json required."), None

    # === 2. DAILY LOSS CAP (HARD for the day) ===
    pnl_today = state.get("pnl_today_usd", 0.0)
    if equity and pnl_today < 0:
        loss_pct_today = abs(pnl_today) / equity
        if loss_pct_today >= DAILY_LOSS_CAP_PCT:
            return False, (f"DAILY_LOSS_CAP: today ${pnl_today:+.2f} "
                           f"= {loss_pct_today*100:.2f}% >= cap "
                           f"{DAILY_LOSS_CAP_PCT*100:.1f}%. Resets at UTC midnight."), None

    # === 3. Cooldown after consecutive losses ===
    cu = _parse_iso(state.get("cooldown_until_iso"))
    if cu and now < cu:
        mins = int((cu - now).total_seconds() // 60)
        return False, f"loss_cooldown ({mins}m left, {state.get('consecutive_losses',0)} losses)", None

    # === 4. Reentry block after any exit ===
    le = _parse_iso(state.get("last_exit_iso"))
    if le and (now - le) < timedelta(minutes=REENTRY_BLOCK_MIN):
        mins = int((timedelta(minutes=REENTRY_BLOCK_MIN) - (now - le)).total_seconds() // 60)
        return False, f"reentry_block ({mins}m left)", None

    # === 5+6. Composite gates (sessions, calendar, news) ===
    if gate_cfg is not None:
        ok, why = composite_pre_signal_gate(gate_cfg)
        if not ok:
            return False, why, None
        if side is not None:
            ok, why, news = directional_news_gate(side, gate_cfg)
            if not ok:
                return False, why, news
            return True, why, news
    return True, "", None


def effective_risk_pct(state: dict, equity: float) -> tuple[float, bool]:
    """Legacy P2-only DD-scaling. Kept for backwards compat; new code path
    uses compute_effective_risk() in _bot_common (DD-tier + Kelly + regime)."""
    peak = state.get("peak_equity") or equity
    if peak <= 0:
        return RISK_PER_TRADE_PCT, False
    dd = (peak - equity) / peak
    if dd >= DD_SCALE_THRESHOLD:
        return RISK_PER_TRADE_PCT * DD_SCALE_FACTOR, True
    return RISK_PER_TRADE_PCT, False


# ============================ ORDERING ===============================
def pick_filling_mode(sym) -> int:
    """Detect supported filling mode (FOK vs IOC) for this symbol."""
    fm = sym.filling_mode
    if fm & 2:  # SYMBOL_FILLING_IOC
        return mt5.ORDER_FILLING_IOC
    if fm & 1:  # SYMBOL_FILLING_FOK
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_IOC


def round_to_step(value: float, step: float) -> float:
    return round(round(value / step) * step, 8)


def size_in_lots(equity: float, risk_pct: float, stop_distance_usd: float,
                 contract_size: float, lot_step: float, lot_min: float, lot_max: float) -> float:
    """Convert risk to lots.
    risk_usd = equity * risk_pct
    oz_to_trade = risk_usd / stop_distance_usd_per_oz
    lots = oz / contract_size (since 1 lot = contract_size oz)
    """
    if stop_distance_usd <= 0:
        return 0.0
    risk_usd = equity * risk_pct
    oz = risk_usd / stop_distance_usd
    lots = oz / contract_size
    lots = round_to_step(lots, lot_step)
    if lots < lot_min:
        return 0.0
    if lots > lot_max:
        lots = lot_max
    return lots


def open_market(state: dict, side: str, atr_val: float, regime_snap=None) -> bool:
    """Send a market order for SIDE with SL/TP based on ATR. Returns True on success."""
    sym = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    if sym is None or tick is None:
        log("open: symbol_info/tick missing"); return False

    if side == "BUY":
        entry_px = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
        sl = entry_px - K_SL * atr_val
        tp = entry_px + K_TP * atr_val
    else:
        entry_px = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
        sl = entry_px + K_SL * atr_val
        tp = entry_px - K_TP * atr_val

    stop_distance = abs(entry_px - sl)
    if stop_distance <= 0:
        log("open: zero stop distance, abort"); return False

    equity = get_equity()
    if equity <= 0:
        log("open: account equity 0, abort"); return False

    # Pro upgrade: composite risk = base * dd_tier * kelly * regime_weight
    decision = compute_effective_risk(
        base_risk_pct=RISK_PER_TRADE_PCT,
        equity=equity,
        peak_equity=state.get("peak_equity"),
        journal_path=TRADES_CSV if USE_KELLY else None,
        regime_snapshot=regime_snap,
        strategy_name="breakout",
        kelly_params=KellyParams(),
        dd_tiers=DEFAULT_DD_TIERS,
        use_kelly=USE_KELLY,
        use_regime=USE_REGIME_WEIGHT,
    )
    if decision.halted:
        log(f"open: HALTED by risk layer: {decision.explanation}")
        tg_send(f"<b>[RISK HALT]</b>\n{decision.explanation}")
        return False

    risk_pct = decision.risk_pct
    scaled = decision.dd_mult < 1.0
    lots = size_in_lots(
        equity=equity, risk_pct=risk_pct, stop_distance_usd=stop_distance,
        contract_size=sym.trade_contract_size, lot_step=sym.volume_step,
        lot_min=sym.volume_min, lot_max=sym.volume_max,
    )
    if lots <= 0:
        log(f"open: computed lots <= min ({lots}); skip"); return False

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lots,
        "type": order_type,
        "price": entry_px,
        "sl": round(sl, sym.digits),
        "tp": round(tp, sym.digits),
        "deviation": 20,
        "magic": MAGIC,
        "comment": "PSP_AI_Trader",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": pick_filling_mode(sym),
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        rc = getattr(result, "retcode", None)
        cm = getattr(result, "comment", "")
        log(f"open: order_send rejected (retcode={rc} comment='{cm}')")
        tg_send(f"<b>[RISK]</b> Order rejected — retcode={rc} '{cm}'")
        return False

    # success — persist our meta
    state["open_ticket"] = int(result.order)
    state["open_meta"] = {
        "trade_id": state["next_trade_id"],
        "side": side,
        "entry": entry_px,
        "sl": round(sl, sym.digits),
        "tp": round(tp, sym.digits),
        "lots": lots,
        "open_time_iso": _now_utc().isoformat(timespec="seconds"),
        "atr_at_entry": atr_val,
        "risk_pct_used": risk_pct,
        "magic": MAGIC,
        "deal_ticket": int(result.deal),
        "position_ticket": int(result.order),
    }
    state["next_trade_id"] += 1
    state["counters_today"]["entries"] = state["counters_today"].get("entries", 0) + 1

    log(f"OPEN {side}  entry={entry_px:.2f}  sl={sl:.2f}  tp={tp:.2f}  "
        f"lots={lots}  {decision.explanation}  ticket={state['open_ticket']}")
    tg_send(
        f"<b>[ENTRY]</b> {side} GOLD (MT5 demo)\n"
        f"Entry: {entry_px:.2f}\nSL: {sl:.2f}  TP: {tp:.2f}\n"
        f"Lots: {lots}  ATR: {atr_val:.2f}\n"
        f"Risk: {risk_pct*100:.3f}% (base {RISK_PER_TRADE_PCT*100:.1f}% "
        f"x dd {decision.dd_mult:.2f} x kelly {decision.kelly_mult:.2f} "
        f"x regime {decision.regime_mult:.2f})\n"
        f"Regime: {regime_snap.regime.value if regime_snap else 'n/a'}\n"
        f"Equity: ${equity:,.2f}"
    )
    return True


def find_position_by_ticket(ticket: int):
    """Return the MT5 position dict for ticket, or None if not open anymore."""
    if ticket is None:
        return None
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return None
    return positions[0]


def reconcile_closed_position(state: dict):
    """When our tracked position is no longer open, look up the close in history and journal it."""
    meta = state.get("open_meta") or {}
    pos_ticket = meta.get("position_ticket")
    if not pos_ticket:
        return

    # Find closing deal in history (last 7 days is plenty for an intraday bot)
    deals = mt5.history_deals_get(
        _now_utc() - timedelta(days=7), _now_utc() + timedelta(minutes=5),
        position=pos_ticket,
    )
    if not deals:
        log(f"reconcile: no deals found for position {pos_ticket} yet; will retry")
        return

    # Closing deal: highest time, entry IN/OUT we treat as the exit fill
    closing = max(deals, key=lambda d: d.time)
    exit_price = float(closing.price)
    pnl_usd = float(closing.profit) + float(getattr(closing, "swap", 0.0)) + float(getattr(closing, "commission", 0.0))

    entry = meta["entry"]
    sl = meta["sl"]
    tp = meta["tp"]
    side = meta["side"]
    lots = meta["lots"]
    stop_distance = abs(entry - sl)

    # Reason: SL/TP based on price proximity
    if side == "BUY":
        exit_reason = "TP" if exit_price >= (tp - stop_distance * 0.05) else \
                      "SL" if exit_price <= (sl + stop_distance * 0.05) else "OTHER"
    else:
        exit_reason = "TP" if exit_price <= (tp + stop_distance * 0.05) else \
                      "SL" if exit_price >= (sl - stop_distance * 0.05) else "OTHER"

    r_realised = (pnl_usd / (stop_distance * lots * mt5.symbol_info(SYMBOL).trade_contract_size)) \
        if stop_distance > 0 else 0.0

    close_time = datetime.fromtimestamp(closing.time, tz=timezone.utc)
    open_time = datetime.fromisoformat(meta["open_time_iso"])
    dur_min = int((close_time - open_time).total_seconds() // 60)

    append_trade({
        "trade_id": meta["trade_id"],
        "open_time": meta["open_time_iso"],
        "close_time": close_time.isoformat(timespec="seconds"),
        "side": side,
        "entry": round(entry, 4),
        "exit": round(exit_price, 4),
        "lots": lots,
        "sl": round(sl, 4),
        "tp": round(tp, 4),
        "pnl_usd": round(pnl_usd, 2),
        "r_realised": round(r_realised, 3),
        "duration_minutes": dur_min,
        "atr_at_entry": round(meta["atr_at_entry"], 3),
        "exit_reason": exit_reason,
        "ticket": pos_ticket,
    })

    # Update state
    equity = get_equity()
    state["peak_equity"] = max(state.get("peak_equity") or equity, equity)
    state["pnl_today_usd"] += pnl_usd
    state["trades_today"] += 1
    state["last_exit_iso"] = close_time.isoformat(timespec="seconds")
    state["open_ticket"] = None
    state["open_meta"] = None

    if pnl_usd > 0:
        state["consecutive_losses"] = 0
        state["cooldown_until_iso"] = None
        cooldown_note = ""
    else:
        state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
        if state["consecutive_losses"] >= COOLDOWN_AFTER_N_LOSSES:
            cu = close_time + timedelta(minutes=COOLDOWN_MINUTES_LOSSES)
            state["cooldown_until_iso"] = cu.isoformat(timespec="seconds")
            cooldown_note = (f"\nCooldown: paused {COOLDOWN_MINUTES_LOSSES//60}h "
                             f"({state['consecutive_losses']} losses in a row)")
        else:
            cooldown_note = ""

    emoji = "[WIN]" if pnl_usd > 0 else "[LOSS]"
    log(f"CLOSE {side}  reason={exit_reason}  pnl=${pnl_usd:+.2f}  R={r_realised:+.2f}  "
        f"equity=${equity:,.2f}  losses_in_row={state['consecutive_losses']}")
    tg_send(
        f"<b>[EXIT]</b> {emoji} {side} closed by {exit_reason}\n"
        f"P&amp;L: ${pnl_usd:+.2f} ({r_realised:+.2f}R)\n"
        f"Equity: ${equity:,.2f}\n"
        f"Today: {state['trades_today']} trades, P&amp;L ${state['pnl_today_usd']:+.2f}"
        f"{cooldown_note}"
    )


# =========================== DAILY RESET =============================
def reset_daily_if_needed(state: dict):
    today = _now_utc().date().isoformat()
    if state.get("today_utc") != today:
        state["today_utc"] = today
        state["pnl_today_usd"] = 0.0
        state["trades_today"] = 0
        state["counters_today"] = {
            "entries": 0, "watches": 0, "breakouts": 0, "rejections": {},
        }


# =========================== DAILY SUMMARY ===========================
def maybe_send_daily_summary(state: dict):
    now_ist = datetime.now(IST)
    today_ist = now_ist.date().isoformat()
    if state.get("last_summary_date_ist") == today_ist:
        return
    if not (now_ist.hour > SUMMARY_HOUR_IST
            or (now_ist.hour == SUMMARY_HOUR_IST and now_ist.minute >= SUMMARY_MIN_IST)):
        return

    c = state.get("counters_today", {})
    rej = c.get("rejections", {}) or {}
    top_rej = "(none)"
    if rej:
        top = Counter(rej).most_common(1)[0]
        top_rej = f"{top[0]} ({top[1]}x)"

    equity = get_equity()
    peak = state.get("peak_equity") or equity
    dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0

    meta = state.get("open_meta") or {}
    if meta:
        open_str = f"{meta['side']} @ {meta['entry']:.2f} (sl {meta['sl']:.2f} tp {meta['tp']:.2f})"
    else:
        open_str = "flat"

    cu = _parse_iso(state.get("cooldown_until_iso"))
    cooldown_str = ""
    if cu and cu > _now_utc():
        mins_left = int((cu - _now_utc()).total_seconds() // 60)
        cooldown_str = f"\nCooldown: {mins_left}m left ({state.get('consecutive_losses',0)} losses)"

    tg_send(
        f"<b>[DAILY SUMMARY]</b> {today_ist} (MT5/XM)\n"
        f"Entries: {c.get('entries',0)}\n"
        f"Breakout watches: {c.get('breakouts',0)}\n"
        f"Trend watches: {c.get('watches',0)}\n"
        f"Top rejection: {top_rej}\n"
        f"Equity: ${equity:,.2f}  |  DD: -{dd_pct:.2f}%\n"
        f"Position: {open_str}{cooldown_str}"
    )
    state["last_summary_date_ist"] = today_ist


# ============================== MAIN =================================
RUNNING = True


def handle_sigint(signum, frame):
    global RUNNING
    RUNNING = False
    print("\nStopping after current cycle… (Ctrl+C again to force)")


def run_test_trade(side: str) -> int:
    """One-shot: open ONE trade through the bot's full pipeline, then exit.
    Validates the auto-trade code path without waiting for a real signal.
    Uses magic 20260522 (same as the bot), so this becomes the bot's open
    position — when you restart the bot normally, it'll manage this trade."""
    if not init_mt5_headless():
        log(f"mt5.initialize() failed: {mt5.last_error()}")
        return 1
    info = mt5.account_info()
    if info is None:
        log("no account_info (is MT5 logged in?)"); mt5.shutdown(); return 1
    if not mt5.symbol_select(SYMBOL, True):
        log(f"symbol_select({SYMBOL}) failed"); mt5.shutdown(); return 1

    ensure_journal_header()
    state = load_state()
    if state.get("peak_equity") is None:
        state["peak_equity"] = get_equity()

    # Compute current ATR(14) from the 15m series
    df15, df1h, df4h = fetch_bars()
    if df15.empty or len(df15) < 30:
        log("not enough bars to compute ATR"); mt5.shutdown(); return 1
    atr_val = float(atr(df15["High"], df15["Low"], df15["Close"], ATR_PERIOD).iloc[-1])
    # Sanity check: latest bar close vs live tick. If they diverge by more than
    # 2*ATR, the bars are likely stale — abort rather than place a bad trade.
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None or tick.ask == 0:
        log("no live tick available"); mt5.shutdown(); return 1
    last_close = float(df15["Close"].iloc[-1])
    live_price = tick.ask if side == "BUY" else tick.bid
    if abs(last_close - live_price) > 2 * atr_val:
        log(f"STALE BARS: bar_close={last_close:.2f} live={live_price:.2f} "
            f"diff={abs(last_close-live_price):.2f} > 2*ATR ({2*atr_val:.2f}). Aborting test.")
        mt5.shutdown(); return 1
    log(f"TEST MODE: forcing {side} entry through bot pipeline. "
        f"live={live_price:.2f}  ATR={atr_val:.2f}")
    tg_send(f"<b>[TEST MODE]</b> Forcing {side} GOLD via bot (magic {MAGIC})")

    regime_snap = classify_regime(df1h) if not df1h.empty else None
    ok = open_market(state, side, atr_val, regime_snap)
    save_state(state)
    mt5.shutdown()
    if ok:
        log("Test trade placed. Now run 'python mt5_live.py' normally to manage it.")
    return 0 if ok else 1


def main() -> int:
    # CLI test mode: --test-buy / --test-sell -> fire one trade through bot, exit
    if len(sys.argv) > 1 and sys.argv[1] in ("--test-buy", "--test-sell"):
        side = "BUY" if sys.argv[1] == "--test-buy" else "SELL"
        return run_test_trade(side)

    if not TG_TOKEN or not TG_CHAT:
        log("WARNING: TELEGRAM creds missing — alerts disabled.")

    if not init_mt5_headless():
        log(f"mt5.initialize() failed: {mt5.last_error()}")
        return 1

    info = mt5.account_info()
    if info is None:
        log("mt5.account_info() returned None (is MT5 logged in?)")
        mt5.shutdown(); return 1

    # Make sure symbol is selected
    if not mt5.symbol_select(SYMBOL, True):
        log(f"symbol_select({SYMBOL}) failed: {mt5.last_error()}")
        mt5.shutdown(); return 1

    signal.signal(signal.SIGINT, handle_sigint)
    ensure_journal_header()
    state = load_state()

    # seed peak_equity on first run
    eq0 = get_equity()
    if state["peak_equity"] is None:
        state["peak_equity"] = eq0

    log(f"MT5 paper bot started.  account={info.login}  server={info.server}  "
        f"equity=${eq0:,.2f}  symbol={SYMBOL}")
    try:
        CFG.print_summary()
    except Exception as e:
        log(f"print_summary error (non-fatal): {e}")
    tg_send(
        f"<b>[BOT START — MT5/XM]</b>\n"
        f"Account: {info.login} ({info.server})\n"
        f"Equity: ${eq0:,.2f}\n"
        f"Symbol: {SYMBOL}\n"
        f"Open: {'yes (ticket '+str(state['open_ticket'])+')' if state.get('open_ticket') else 'no'}"
    )
    pg_log_event(BOT_NAME, "bot_start", {
        "account": info.login, "server": info.server,
        "equity": eq0, "symbol": SYMBOL,
        "open_ticket": state.get("open_ticket"),
        "magic": MAGIC,
    })
    pg_snapshot_equity(str(info.login), eq0, get_balance(),
                       peak_equity=state.get("peak_equity"),
                       open_positions=1 if state.get("open_ticket") else 0)

    gate_cfg = _build_gate_config()
    if not os.getenv("ALPHA_VANTAGE_KEY"):
        log("WARNING: ALPHA_VANTAGE_KEY missing — news filter disabled.")

    while RUNNING:
        try:
            reset_daily_if_needed(state)
            df15, df1h, df4h = fetch_bars()
            if df15 is None or df15.empty or df1h is None or df1h.empty:
                check_mt5_alive_or_reconnect(state)  # watchdog: reconnect or sys.exit for NSSM
                time.sleep(POLL_SECONDS); continue
            reset_mt5_failure_counter(state)
            if len(df15) < EMA_SLOW + 5 or len(df1h) < EMA_SLOW + 5:
                log("not enough history yet, waiting…")
                time.sleep(POLL_SECONDS); continue

            latest_ts = str(df15.index[-1])
            new_bar = state["last_bar_ts"] != latest_ts

            # If we had an open position, check whether it's still there
            if state.get("open_ticket"):
                pos = find_position_by_ticket(state["open_ticket"])
                if pos is None:
                    # Position closed — broker hit SL or TP. Reconcile.
                    reconcile_closed_position(state)
                    save_state(state)

            if new_bar:
                if not state.get("open_ticket"):
                    # Classify regime each new-bar cycle (1H ADX-based)
                    regime_snap = classify_regime(df1h) if not df1h.empty else None
                    sig = evaluate(df15, df1h, df4h if USE_4H_GATE else None)
                    if sig is not None:
                        sev = sig["severity"]
                        log(f"signal: {sev} {sig.get('side')}  price={sig['price']:.2f}  {sig['reason']}")
                        # Postgres signal log (best-effort)
                        try:
                            pg_record_signal(BOT_NAME, sig,
                                             regime=regime_snap.regime.value if regime_snap else None)
                        except Exception as e:
                            log(f"pg_record_signal error (non-fatal): {e}")
                        ct = state["counters_today"]
                        if sev == "SKIPPED":
                            r = sig.get("rejection_reason", "unknown")
                            ct.setdefault("rejections", {})[r] = ct["rejections"].get(r, 0) + 1
                        elif sev == "WATCHLIST":
                            ct["watches"] = ct.get("watches", 0) + 1
                            # Telegram WATCH alerts suppressed when breakout is
                            # disabled (regime weights all 0). Counter still
                            # increments for the daily summary so we can see
                            # the strategy is alive. Re-enable by un-commenting
                            # the tg_send when breakout regime weights go >0.
                            # tg_send(f"<b>[WATCH]</b> {sig['side']} GOLD\n"
                            #         f"{sig['reason']}\nPrice: {sig['price']:.2f}  ATR: {sig['atr']:.2f}")
                        elif sev == "BREAKOUT_WATCH":
                            ct["breakouts"] = ct.get("breakouts", 0) + 1
                            # Same suppression as WATCHLIST — see comment above.
                            # tg_send(f"<b>[BREAKOUT WATCH]</b> {sig['side']} GOLD\n"
                            #         f"{sig['reason']}\nPrice: {sig['price']:.2f}  ATR: {sig['atr']:.2f}")
                        elif sev in ("BUY_READY", "SELL_READY"):
                            # === ML META-LABELER (shadow mode default) ===
                            # Score the signal. Log decision. In shadow mode,
                            # bot trades regardless. Once shadow-mode results
                            # validate the model, set ML_SHADOW_MODE=false in
                            # .env and ml_result.would_trade will gate trades.
                            ml_result = score_signal_live(
                                df15, df1h, sig["side"], BOT_NAME,
                                rr_target=K_TP / K_SL,
                            )
                            log(f"ml: {ml_result.note}  shadow={ml_result.shadow_mode}")
                            # Log to Postgres signals.extras for analysis
                            try:
                                pg_record_signal(
                                    BOT_NAME, sig,
                                    regime=regime_snap.regime.value if regime_snap else None,
                                    extras=ml_result.to_dict(),
                                )
                            except Exception:
                                pass

                            # Gate checks (sessions, calendar, news, cooldown)
                            allowed, why, news = can_open_new_trade(state, sig["side"], gate_cfg)
                            if not allowed:
                                log(f"entry blocked: {why}")
                                tg_send(f"<b>[BLOCKED]</b> {sig['side']} signal blocked: {why}")
                                ct.setdefault("rejections", {})[why.split(' ')[0]] = (
                                    ct["rejections"].get(why.split(' ')[0], 0) + 1)
                                continue
                            # ML veto (only outside shadow mode)
                            if not ml_result.shadow_mode and not ml_result.would_trade:
                                log(f"ML veto: {ml_result.note}")
                                tg_send(f"<b>[ML VETO]</b> {sig['side']} skipped — {ml_result.note}")
                                ct.setdefault("rejections", {})["ml_veto"] = (
                                    ct["rejections"].get("ml_veto", 0) + 1)
                                continue
                            if why:
                                log(f"gate ok: {why}")
                            open_market(state, sig["side"], sig["atr"], regime_snap)
                else:
                    meta = state["open_meta"]
                    log(f"holding {meta['side']} @ {meta['entry']:.2f}  "
                        f"sl={meta['sl']:.2f} tp={meta['tp']:.2f}  lots={meta['lots']}")

                state["last_bar_ts"] = latest_ts
                save_state(state)
            else:
                # Heartbeat
                tick = mt5.symbol_info_tick(SYMBOL)
                px = tick.bid if tick else 0
                equity = get_equity()
                # update peak as we go
                if state.get("peak_equity") is None or equity > state["peak_equity"]:
                    state["peak_equity"] = equity
                meta = state.get("open_meta") or {}
                pos_str = (f"{meta.get('side')} @ {meta.get('entry'):.2f}" if meta else "flat")
                sess_ok, sess_name = is_tradeable_session(_now_utc())
                sess_str = sess_name if sess_ok else f"NO_SESSION({sess_name})"
                log(f"heartbeat  price={px:.2f}  {pos_str}  equity=${equity:,.2f}  "
                    f"today=${state['pnl_today_usd']:+.2f}  {sess_str}")
                # Periodic equity snapshot (every ~5 min based on heartbeat counter)
                state["_hb_count"] = state.get("_hb_count", 0) + 1
                if state["_hb_count"] >= 5:
                    state["_hb_count"] = 0
                    try:
                        info_now = mt5.account_info()
                        if info_now:
                            pg_snapshot_equity(str(info_now.login), info_now.equity,
                                               info_now.balance,
                                               peak_equity=state.get("peak_equity"),
                                               open_positions=1 if state.get("open_ticket") else 0)
                    except Exception:
                        pass

            maybe_send_daily_summary(state)
            save_state(state)

        except Exception as e:
            log(f"loop error: {e}")
            tg_send(f"<b>[RISK]</b> loop error: {e}")

        time.sleep(POLL_SECONDS)

    log("Stopped. State saved.")
    final_eq = get_equity()
    tg_send(f"<b>[BOT STOP — MT5/XM]</b>\nEquity: ${final_eq:,.2f}")
    pg_log_event(BOT_NAME, "bot_stop", {"equity": final_eq})
    mt5.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
