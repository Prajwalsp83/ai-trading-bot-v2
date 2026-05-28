"""
MT5 live trading bot — SMC (Smart Money Concepts) variant for XM demo gold.

Runs IN PARALLEL with mt5_live.py (breakout-trend). Each uses its own MAGIC
number so positions and accounting stay separate.

Strategy:
  1. HTF (1H) market structure -> bias (up/down).
  2. HTF unmitigated POIs (OB+FVG overlaps preferred, scored).
  3. Wait for price to enter a high-scoring POI in bias direction.
  4. LTF (15m) CHoCH or BOS in bias direction -> confirmation.
  5. Entry market; SL beyond POI + ATR buffer; TP at next opposite-side liquidity.
  6. RR >= 1.5 required, else skip.

Shared gates (from _bot_common.py): session filter, calendar block, news
sentiment, drawdown tiers, fractional Kelly, ADX regime weight.

Run:
    python C:\\bot\\mt5_smc.py             # live loop
    python C:\\bot\\mt5_smc.py --test-buy  # one-shot trial through pipeline

Files written:
    .mt5_smc_state.json
    data/mt5_smc_trades.csv
"""
from __future__ import annotations

import csv
import json
import os
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

# See comment in mt5_live.py — HERE = project root, SCRIPT_DIR = scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
load_dotenv(HERE / ".env")
sys.path.insert(0, str(SCRIPT_DIR))

from _bot_common import (  # noqa: E402
    IST, tg_send,
    SessionConfig, current_session, is_tradeable_session, minutes_until_next_session,
    load_calendar, calendar_block, next_calendar_event,
    fetch_gold_sentiment,
    GateConfig, composite_pre_signal_gate, directional_news_gate,
    classify_regime, RegimeParams,
    compute_effective_risk, KellyParams, DEFAULT_DD_TIERS,
    init_mt5_headless, check_mt5_alive_or_reconnect, reset_mt5_failure_counter,
)

# Postgres journal (degrades gracefully if DATABASE_URL not set / DB unreachable)
from _journal import record_trade as pg_record_trade, record_signal as pg_record_signal  # noqa: E402
from _journal import snapshot_equity as pg_snapshot_equity, log_event as pg_log_event  # noqa: E402

# ML meta-labeler (shadow mode default)
from _meta_scorer import score_signal_live  # noqa: E402

BOT_NAME = "smc"


# ============================== CONFIG ===============================
SYMBOL = "GOLD.i#"
MAGIC = 20260601                 # distinct from breakout bot (20260522)
POLL_SECONDS = 60

# --- risk + capital control ---
RISK_PER_TRADE_PCT = 0.02
DAILY_LOSS_CAP_PCT = 0.03
MAX_DD_PCT = 0.15
COOLDOWN_AFTER_N_LOSSES = 2
COOLDOWN_MINUTES_LOSSES = 240
REENTRY_BLOCK_MIN = 120

USE_KELLY = True
USE_REGIME_WEIGHT = True

# --- SMC params ---
# 2026-05-28: LOOSENED for trade frequency — ran 1 week with 0 entries.
# Tighten back if win rate craters below 40%.
HTF_PIVOT = 2                    # 1H swing sensitivity
LTF_PIVOT = 2                    # 15m swing sensitivity
MIN_IMPULSE_BARS = 3
POI_FRESHNESS_BARS = 60          # was 30 — POIs stay valid ~10 days on H1
MIN_POI_SCORE = 2                # was 3 — accept OB+FVG overlap alone (no bonus required)
SL_BUFFER_ATR_FRAC = 0.25
REQUIRE_LTF_CHOCH = False        # was True — enter on POI mitigation alone (no 15m struct)
MIN_RR = 1.5
ATR_PERIOD = 14
MAX_STRUCTURE_LOOKBACK_BARS = 300

# --- daily summary timing (IST) ---
SUMMARY_HOUR_IST = 23
SUMMARY_MIN_IST = 55

# --- files ---
STATE_FILE = HERE / ".mt5_smc_state.json"
TRADES_CSV = HERE / "data" / "mt5_smc_trades.csv"
TRADES_CSV.parent.mkdir(exist_ok=True)
TRADE_COLS = [
    "trade_id", "open_time", "close_time", "side", "entry", "exit",
    "lots", "sl", "tp", "pnl_usd", "r_realised", "duration_minutes",
    "atr_at_entry", "exit_reason", "ticket", "poi_score", "rr_at_entry",
    "regime", "news_bias", "news_score",
]

CALENDAR_PATH = HERE / "data" / "economic_calendar.json"
NEWS_CACHE = HERE / "data" / ".av_news_cache.json"


# ============================== HELPERS ==============================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SMC] {msg}", flush=True)


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
    # Postgres (best-effort)
    try:
        pg_record_trade(BOT_NAME, MAGIC, record)
    except Exception as e:
        log(f"pg_record_trade error (non-fatal): {e}")


# ============================== STATE ================================
def load_state() -> dict:
    defaults = {
        "peak_equity": None,
        "open_ticket": None,
        "open_meta": None,
        "last_bar_ts": None,
        "next_trade_id": 1,
        "trades_today": 0,
        "pnl_today_usd": 0.0,
        "today_utc": None,
        "consecutive_losses": 0,
        "cooldown_until_iso": None,
        "last_exit_iso": None,
        "last_summary_date_ist": None,
        "counters_today": {
            "entries": 0,
            "watches": 0,
            "poi_approach": 0,
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
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ========================== SMC: STRUCTURE ===========================
Direction = Literal["up", "down", "none"]


@dataclass
class Swing:
    idx: int
    ts: pd.Timestamp
    price: float
    kind: Literal["high", "low"]


@dataclass
class StructureEvent:
    idx: int
    ts: pd.Timestamp
    kind: Literal["BOS", "CHoCH"]
    side: Direction
    broken_swing_price: float
    close: float


@dataclass
class StructureSnapshot:
    swings: list[Swing]
    events: list[StructureEvent]
    current_bias: Direction
    dealing_range_high: float | None
    dealing_range_low: float | None


def find_swings(df: pd.DataFrame, pivot: int = 2) -> list[Swing]:
    if len(df) < 2 * pivot + 1:
        return []
    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index
    swings: list[Swing] = []
    for i in range(pivot, len(df) - pivot):
        wh = highs[i - pivot:i + pivot + 1]
        wl = lows[i - pivot:i + pivot + 1]
        if highs[i] == wh.max() and wh.argmax() == pivot:
            swings.append(Swing(i, idx[i], float(highs[i]), "high"))
        if lows[i] == wl.min() and wl.argmin() == pivot:
            swings.append(Swing(i, idx[i], float(lows[i]), "low"))
    swings.sort(key=lambda s: s.idx)
    return swings


def find_structure_events(df: pd.DataFrame, swings: list[Swing]) -> list[StructureEvent]:
    events: list[StructureEvent] = []
    if not swings:
        return events
    closes = df["Close"].values
    idx = df.index
    bias: Direction = "none"
    unbroken_highs: list[Swing] = []
    unbroken_lows: list[Swing] = []
    next_swing = 0
    for i in range(len(df)):
        while next_swing < len(swings) and swings[next_swing].idx <= i:
            s = swings[next_swing]
            (unbroken_highs if s.kind == "high" else unbroken_lows).append(s)
            next_swing += 1
        c = closes[i]
        broken_h = None
        for sh in reversed(unbroken_highs):
            if sh.idx >= i: continue
            if c > sh.price:
                broken_h = sh; break
        if broken_h is not None:
            kind = "BOS" if bias == "up" else "CHoCH"
            events.append(StructureEvent(i, idx[i], kind, "up", broken_h.price, float(c)))
            bias = "up"
            unbroken_highs = [sh for sh in unbroken_highs if sh.idx > broken_h.idx]
        broken_l = None
        for sl in reversed(unbroken_lows):
            if sl.idx >= i: continue
            if c < sl.price:
                broken_l = sl; break
        if broken_l is not None:
            kind = "BOS" if bias == "down" else "CHoCH"
            events.append(StructureEvent(i, idx[i], kind, "down", broken_l.price, float(c)))
            bias = "down"
            unbroken_lows = [sl for sl in unbroken_lows if sl.idx > broken_l.idx]
    return events


def analyse_structure(df: pd.DataFrame, pivot: int = 2) -> StructureSnapshot:
    swings = find_swings(df, pivot=pivot)
    events = find_structure_events(df, swings)
    bias: Direction = events[-1].side if events else "none"
    LOOKBACK = 8
    recent = swings[-LOOKBACK:]
    dr_h = max((s.price for s in recent if s.kind == "high"), default=None)
    dr_l = min((s.price for s in recent if s.kind == "low"), default=None)
    return StructureSnapshot(swings, events, bias, dr_h, dr_l)


def equilibrium(snap: StructureSnapshot) -> float | None:
    if snap.dealing_range_high is None or snap.dealing_range_low is None:
        return None
    return (snap.dealing_range_high + snap.dealing_range_low) / 2.0


# ============================== SMC: FVG =============================
@dataclass
class FVG:
    side: Literal["bull", "bear"]
    top: float
    bottom: float
    created_idx: int
    mitigated: bool = False

    @property
    def height(self) -> float: return self.top - self.bottom

    @property
    def mid(self) -> float: return (self.top + self.bottom) / 2.0


def find_fvgs(df: pd.DataFrame, max_age_bars: int | None = None) -> list[FVG]:
    if len(df) < 3:
        return []
    highs = df["High"].values
    lows = df["Low"].values
    fvgs: list[FVG] = []
    for i in range(2, len(df)):
        if highs[i - 2] < lows[i]:
            fvgs.append(FVG("bull", float(lows[i]), float(highs[i - 2]), i))
        if lows[i - 2] > highs[i]:
            fvgs.append(FVG("bear", float(lows[i - 2]), float(highs[i]), i))
    for fvg in fvgs:
        for j in range(fvg.created_idx + 1, len(df)):
            if lows[j] <= fvg.top and highs[j] >= fvg.bottom:
                fvg.mitigated = True; break
    if max_age_bars is not None:
        cutoff = len(df) - max_age_bars
        fvgs = [f for f in fvgs if f.created_idx >= cutoff]
    return fvgs


# ============================ SMC: OB ================================
@dataclass
class OrderBlock:
    side: Literal["bull", "bear"]
    top: float
    bottom: float
    created_idx: int
    impulse_idx: int
    mitigated: bool = False

    @property
    def height(self) -> float: return self.top - self.bottom

    @property
    def mid(self) -> float: return (self.top + self.bottom) / 2.0


def find_order_blocks(df: pd.DataFrame, events: list[StructureEvent],
                      min_impulse_bars: int = 3) -> list[OrderBlock]:
    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    obs: list[OrderBlock] = []
    for ev in events:
        bos_idx = ev.idx
        if ev.side == "up":
            ob_idx = None
            for j in range(bos_idx - 1, max(-1, bos_idx - 30), -1):
                if closes[j] < opens[j]:
                    ob_idx = j; break
            if ob_idx is None: continue
            bullish = sum(1 for k in range(ob_idx + 1, bos_idx + 1) if closes[k] > opens[k])
            if bullish < min_impulse_bars: continue
            obs.append(OrderBlock("bull", float(highs[ob_idx]), float(lows[ob_idx]),
                                  ob_idx, bos_idx))
        elif ev.side == "down":
            ob_idx = None
            for j in range(bos_idx - 1, max(-1, bos_idx - 30), -1):
                if closes[j] > opens[j]:
                    ob_idx = j; break
            if ob_idx is None: continue
            bearish = sum(1 for k in range(ob_idx + 1, bos_idx + 1) if closes[k] < opens[k])
            if bearish < min_impulse_bars: continue
            obs.append(OrderBlock("bear", float(highs[ob_idx]), float(lows[ob_idx]),
                                  ob_idx, bos_idx))
    for ob in obs:
        for j in range(ob.impulse_idx + 1, len(df)):
            if lows[j] <= ob.top and highs[j] >= ob.bottom:
                ob.mitigated = True; break
    return obs


# ============================ SMC: POI ===============================
@dataclass
class POI:
    side: Literal["bull", "bear"]
    top: float
    bottom: float
    score: int
    reasons: list[str] = field(default_factory=list)
    created_idx: int = 0

    @property
    def height(self) -> float: return self.top - self.bottom

    @property
    def mid(self) -> float: return (self.top + self.bottom) / 2.0

    def contains(self, p: float) -> bool: return self.bottom <= p <= self.top


def _overlap(t1, b1, t2, b2):
    t, b = min(t1, t2), max(b1, b2)
    return (t, b) if t > b else None


def build_pois(snap: StructureSnapshot, obs: list[OrderBlock], fvgs: list[FVG],
               current_idx: int, atr_val: float,
               freshness_bars: int = 30, min_zone_atr_frac: float = 0.3) -> list[POI]:
    pois: list[POI] = []
    eq = equilibrium(snap)
    # OB+FVG confluences
    for ob in obs:
        if ob.mitigated: continue
        for fvg in fvgs:
            if fvg.mitigated or fvg.side != ob.side: continue
            ov = _overlap(ob.top, ob.bottom, fvg.top, fvg.bottom)
            if ov is None: continue
            top, bot = ov
            score = 2
            reasons = ["OB+FVG"]
            mid = (top + bot) / 2
            if ob.side == "bull" and eq is not None and mid <= eq:
                score += 1; reasons.append("discount")
            if ob.side == "bear" and eq is not None and mid >= eq:
                score += 1; reasons.append("premium")
            if (current_idx - max(ob.created_idx, fvg.created_idx)) <= freshness_bars:
                score += 1; reasons.append("fresh")
            if atr_val > 0 and (top - bot) >= min_zone_atr_frac * atr_val:
                score += 1; reasons.append("width_ok")
            pois.append(POI(ob.side, top, bot, score, reasons,
                            max(ob.created_idx, fvg.created_idx)))
    # standalone OBs
    used_obs = set(id(p) for p in pois)
    for ob in obs:
        if ob.mitigated: continue
        score = 1
        reasons = ["OB_only"]
        if ob.side == "bull" and eq is not None and ob.mid <= eq:
            score += 1; reasons.append("discount")
        if ob.side == "bear" and eq is not None and ob.mid >= eq:
            score += 1; reasons.append("premium")
        if (current_idx - ob.created_idx) <= freshness_bars:
            score += 1; reasons.append("fresh")
        if atr_val > 0 and ob.height >= min_zone_atr_frac * atr_val:
            score += 1; reasons.append("width_ok")
        pois.append(POI(ob.side, ob.top, ob.bottom, score, reasons, ob.created_idx))
    pois.sort(key=lambda p: (p.score, p.created_idx), reverse=True)
    return pois


# =========================== SMC: STRATEGY ===========================
def evaluate_smc(df15: pd.DataFrame, df1h: pd.DataFrame):
    """Returns a signal dict or None. Severity values match breakout bot for shared UX."""
    be = df15.iloc[-MAX_STRUCTURE_LOOKBACK_BARS:] if len(df15) > MAX_STRUCTURE_LOOKBACK_BARS else df15
    bt = df1h.iloc[-MAX_STRUCTURE_LOOKBACK_BARS:] if len(df1h) > MAX_STRUCTURE_LOOKBACK_BARS else df1h
    if len(be) < 60 or len(bt) < 60:
        return None

    htf = analyse_structure(bt, pivot=HTF_PIVOT)
    if htf.current_bias == "none":
        return None

    htf_obs = find_order_blocks(bt, htf.events, min_impulse_bars=MIN_IMPULSE_BARS)
    htf_fvgs = find_fvgs(bt, max_age_bars=200)
    atr_htf = atr(bt["High"], bt["Low"], bt["Close"], ATR_PERIOD)
    atr_htf_val = float(atr_htf.iloc[-1]) if not pd.isna(atr_htf.iloc[-1]) else 0.0

    pois = build_pois(htf, htf_obs, htf_fvgs, len(bt) - 1, atr_htf_val,
                      freshness_bars=POI_FRESHNESS_BARS)
    side_str = "bull" if htf.current_bias == "up" else "bear"
    directional = [p for p in pois if p.side == side_str]
    if not directional:
        return {"severity": "WATCHLIST", "side": "BUY" if side_str == "bull" else "SELL",
                "price": float(be.iloc[-1]["Close"]), "atr": 0.0,
                "reason": f"HTF {htf.current_bias} but no directional POIs"}
    good = [p for p in directional if p.score >= MIN_POI_SCORE]
    if not good:
        return {"severity": "WATCHLIST", "side": "BUY" if side_str == "bull" else "SELL",
                "price": float(be.iloc[-1]["Close"]), "atr": 0.0,
                "reason": f"POIs exist but max_score<{MIN_POI_SCORE}"}

    last15 = be.iloc[-1]
    price = float(last15["Close"])
    matches = [p for p in good if p.contains(price)]
    atr_ltf = atr(be["High"], be["Low"], be["Close"], ATR_PERIOD)
    atr_ltf_val = float(atr_ltf.iloc[-1]) if not pd.isna(atr_ltf.iloc[-1]) else 0.0

    if not matches:
        nearest = min(good, key=lambda p: abs(price - p.mid))
        if atr_ltf_val > 0 and abs(price - nearest.mid) <= 1.5 * atr_ltf_val:
            return {"severity": "BREAKOUT_WATCH",
                    "side": "BUY" if side_str == "bull" else "SELL",
                    "price": price, "atr": atr_ltf_val,
                    "reason": f"approaching POI score={nearest.score} ({','.join(nearest.reasons)})"}
        return {"severity": "WATCHLIST", "side": "BUY" if side_str == "bull" else "SELL",
                "price": price, "atr": atr_ltf_val,
                "reason": f"HTF {htf.current_bias}: waiting for POI mitigation"}

    active = max(matches, key=lambda p: p.score)

    # LTF confirmation
    if REQUIRE_LTF_CHOCH:
        ltf = analyse_structure(be, pivot=LTF_PIVOT)
        recent_events = [e for e in ltf.events if e.idx >= len(be) - 10]
        confirmed = (recent_events and recent_events[-1].side == htf.current_bias
                     and recent_events[-1].kind in ("CHoCH", "BOS"))
        if not confirmed:
            return {"severity": "BREAKOUT_WATCH",
                    "side": "BUY" if side_str == "bull" else "SELL",
                    "price": price, "atr": atr_ltf_val,
                    "reason": f"in POI score={active.score}, awaiting 15m CHoCH/BOS"}

    # Compute SL / TP
    buf = SL_BUFFER_ATR_FRAC * atr_ltf_val
    if htf.current_bias == "up":
        sl = active.bottom - buf
        future_highs = [s for s in htf.swings if s.kind == "high" and s.price > price]
        tp = future_highs[0].price if future_highs else price + 2.5 * atr_ltf_val
        side = "BUY"
    else:
        sl = active.top + buf
        future_lows = [s for s in htf.swings if s.kind == "low" and s.price < price]
        tp = future_lows[0].price if future_lows else price - 2.5 * atr_ltf_val
        side = "SELL"

    risk = abs(price - sl)
    reward = abs(tp - price)
    if risk <= 0 or (reward / risk) < MIN_RR:
        return {"severity": "SKIPPED", "side": side, "price": price, "atr": atr_ltf_val,
                "reason": f"rr_too_low ({reward/max(risk,1e-9):.2f})",
                "rejection_reason": "rr_too_low"}

    return {
        "severity": "BUY_READY" if side == "BUY" else "SELL_READY",
        "side": side, "price": price, "atr": atr_ltf_val,
        "reason": (f"POI mitigated + 15m CHoCH in {htf.current_bias} bias, "
                   f"score={active.score} ({','.join(active.reasons)})"),
        "sl_suggested": sl, "tp_suggested": tp,
        "rr": reward / risk, "poi_score": active.score, "htf_bias": htf.current_bias,
    }


# ============================== DATA =================================
def _rates_to_df(rates) -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_bars():
    try:
        r15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 500)
        r1h = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 500)
        return _rates_to_df(r15), _rates_to_df(r1h)
    except Exception as e:
        log(f"data fetch error: {e}")
        return None, None


def get_equity() -> float:
    info = mt5.account_info()
    return float(info.equity) if info else 0.0


# ============================ GATES ==================================
def can_open_new_trade(state: dict, side: str, gate_cfg: GateConfig):
    """Composite gate. Returns (allowed, reason, news_summary_or_None)."""
    now = _now_utc()
    cu = _parse_iso(state.get("cooldown_until_iso"))
    if cu and now < cu:
        mins = int((cu - now).total_seconds() // 60)
        return False, f"loss_cooldown ({mins}m left)", None
    le = _parse_iso(state.get("last_exit_iso"))
    if le and (now - le) < timedelta(minutes=REENTRY_BLOCK_MIN):
        mins = int((timedelta(minutes=REENTRY_BLOCK_MIN) - (now - le)).total_seconds() // 60)
        return False, f"reentry_block ({mins}m left)", None
    ok, why = composite_pre_signal_gate(gate_cfg)
    if not ok:
        return False, why, None
    ok, why, news = directional_news_gate(side, gate_cfg)
    if not ok:
        return False, why, news
    return True, why, news


# ============================ ORDERING ===============================
def pick_filling_mode(sym) -> int:
    fm = sym.filling_mode
    if fm & 2: return mt5.ORDER_FILLING_IOC
    if fm & 1: return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_IOC


def round_to_step(value: float, step: float) -> float:
    return round(round(value / step) * step, 8)


def size_in_lots(equity, risk_pct, stop_dist, contract_size, lot_step, lot_min, lot_max):
    if stop_dist <= 0: return 0.0
    risk_usd = equity * risk_pct
    oz = risk_usd / stop_dist
    lots = round_to_step(oz / contract_size, lot_step)
    if lots < lot_min: return 0.0
    if lots > lot_max: lots = lot_max
    return lots


def open_market(state: dict, side: str, signal_dict: dict, regime_snap,
                news_summary) -> bool:
    sym = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    if sym is None or tick is None:
        log("open: symbol_info/tick missing"); return False

    entry_px = tick.ask if side == "BUY" else tick.bid
    sl = signal_dict["sl_suggested"]
    tp = signal_dict["tp_suggested"]
    stop_dist = abs(entry_px - sl)
    if stop_dist <= 0:
        log("open: zero stop dist"); return False

    equity = get_equity()
    if equity <= 0:
        log("open: equity 0"); return False

    # === Composite risk: base * dd_tier * kelly * regime ===
    decision = compute_effective_risk(
        base_risk_pct=RISK_PER_TRADE_PCT,
        equity=equity,
        peak_equity=state.get("peak_equity"),
        journal_path=TRADES_CSV if USE_KELLY else None,
        regime_snapshot=regime_snap,
        strategy_name="smc",
        kelly_params=KellyParams(),
        dd_tiers=DEFAULT_DD_TIERS,
        use_kelly=USE_KELLY,
        use_regime=USE_REGIME_WEIGHT,
    )
    if decision.halted:
        log(f"open: HALTED by risk layer: {decision.explanation}")
        tg_send(f"<b>[SMC RISK HALT]</b>\n{decision.explanation}")
        return False

    lots = size_in_lots(equity, decision.risk_pct, stop_dist,
                        sym.trade_contract_size, sym.volume_step,
                        sym.volume_min, sym.volume_max)
    if lots <= 0:
        log(f"open: lots<=min ({lots}) at risk {decision.risk_pct*100:.3f}%"); return False

    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL,
        "volume": lots, "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": entry_px,
        "sl": round(sl, sym.digits), "tp": round(tp, sym.digits),
        "deviation": 20, "magic": MAGIC, "comment": "PSP_SMC",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": pick_filling_mode(sym),
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        rc = getattr(result, "retcode", None)
        cm = getattr(result, "comment", "")
        log(f"open: order rejected retcode={rc} '{cm}'")
        tg_send(f"<b>[SMC RISK]</b> Order rejected — retcode={rc} '{cm}'")
        return False

    state["open_ticket"] = int(result.order)
    state["open_meta"] = {
        "trade_id": state["next_trade_id"],
        "side": side, "entry": entry_px,
        "sl": round(sl, sym.digits), "tp": round(tp, sym.digits),
        "lots": lots, "open_time_iso": _now_utc().isoformat(timespec="seconds"),
        "atr_at_entry": signal_dict["atr"],
        "risk_pct_used": decision.risk_pct,
        "poi_score": signal_dict.get("poi_score", 0),
        "rr_at_entry": signal_dict.get("rr", 0.0),
        "regime": regime_snap.regime.value if regime_snap else "unknown",
        "news_bias": news_summary.bias if news_summary else "none",
        "news_score": news_summary.score if news_summary else 0.0,
        "magic": MAGIC, "deal_ticket": int(result.deal),
        "position_ticket": int(result.order),
    }
    state["next_trade_id"] += 1
    state["counters_today"]["entries"] = state["counters_today"].get("entries", 0) + 1

    log(f"OPEN {side}  entry={entry_px:.2f}  sl={sl:.2f}  tp={tp:.2f}  lots={lots}  "
        f"{decision.explanation}")
    tg_send(
        f"<b>[SMC ENTRY]</b> {side} GOLD\n"
        f"Entry: {entry_px:.2f}\nSL: {sl:.2f}  TP: {tp:.2f}\n"
        f"Lots: {lots}  RR: {signal_dict.get('rr', 0):.2f}\n"
        f"POI score: {signal_dict.get('poi_score', 0)} | Regime: "
        f"{regime_snap.regime.value if regime_snap else 'unknown'}\n"
        f"News: {news_summary.bias if news_summary else 'n/a'} "
        f"({news_summary.score if news_summary else 0:+.2f})\n"
        f"Risk: {decision.risk_pct*100:.3f}% (base {RISK_PER_TRADE_PCT*100:.1f}% "
        f"x dd {decision.dd_mult:.2f} x kelly {decision.kelly_mult:.2f} x regime "
        f"{decision.regime_mult:.2f})\n"
        f"Equity: ${equity:,.2f}"
    )
    return True


def find_position_by_ticket(ticket: int):
    if ticket is None: return None
    positions = mt5.positions_get(ticket=ticket)
    return positions[0] if positions else None


def reconcile_closed_position(state: dict):
    meta = state.get("open_meta") or {}
    pos_ticket = meta.get("position_ticket")
    if not pos_ticket: return

    deals = mt5.history_deals_get(_now_utc() - timedelta(days=7),
                                  _now_utc() + timedelta(minutes=5),
                                  position=pos_ticket)
    if not deals:
        log(f"reconcile: no deals yet for {pos_ticket}; will retry"); return

    closing = max(deals, key=lambda d: d.time)
    exit_price = float(closing.price)
    pnl_usd = (float(closing.profit) + float(getattr(closing, "swap", 0.0)) +
               float(getattr(closing, "commission", 0.0)))

    entry, sl, tp, side, lots = meta["entry"], meta["sl"], meta["tp"], meta["side"], meta["lots"]
    stop_dist = abs(entry - sl)
    if side == "BUY":
        exit_reason = ("TP" if exit_price >= (tp - stop_dist * 0.05)
                       else "SL" if exit_price <= (sl + stop_dist * 0.05) else "OTHER")
    else:
        exit_reason = ("TP" if exit_price <= (tp + stop_dist * 0.05)
                       else "SL" if exit_price >= (sl - stop_dist * 0.05) else "OTHER")
    r_realised = (pnl_usd / (stop_dist * lots * mt5.symbol_info(SYMBOL).trade_contract_size)
                  if stop_dist > 0 else 0.0)
    close_time = datetime.fromtimestamp(closing.time, tz=timezone.utc)
    open_time = datetime.fromisoformat(meta["open_time_iso"])
    dur_min = int((close_time - open_time).total_seconds() // 60)

    append_trade({
        "trade_id": meta["trade_id"],
        "open_time": meta["open_time_iso"],
        "close_time": close_time.isoformat(timespec="seconds"),
        "side": side, "entry": round(entry, 4), "exit": round(exit_price, 4),
        "lots": lots, "sl": round(sl, 4), "tp": round(tp, 4),
        "pnl_usd": round(pnl_usd, 2), "r_realised": round(r_realised, 3),
        "duration_minutes": dur_min, "atr_at_entry": round(meta["atr_at_entry"], 3),
        "exit_reason": exit_reason, "ticket": pos_ticket,
        "poi_score": meta.get("poi_score", 0),
        "rr_at_entry": round(meta.get("rr_at_entry", 0.0), 2),
        "regime": meta.get("regime", "unknown"),
        "news_bias": meta.get("news_bias", "none"),
        "news_score": round(meta.get("news_score", 0.0), 3),
    })

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
    log(f"CLOSE {side}  reason={exit_reason}  pnl=${pnl_usd:+.2f}  R={r_realised:+.2f}")
    tg_send(
        f"<b>[SMC EXIT]</b> {emoji} {side} closed by {exit_reason}\n"
        f"P&amp;L: ${pnl_usd:+.2f} ({r_realised:+.2f}R)\n"
        f"Equity: ${equity:,.2f}\n"
        f"Today: {state['trades_today']} trades, P&amp;L ${state['pnl_today_usd']:+.2f}"
        f"{cooldown_note}"
    )


# =========================== DAILY HOUSEKEEPING =======================
def reset_daily_if_needed(state: dict):
    today = _now_utc().date().isoformat()
    if state.get("today_utc") != today:
        state["today_utc"] = today
        state["pnl_today_usd"] = 0.0
        state["trades_today"] = 0
        state["counters_today"] = {
            "entries": 0, "watches": 0, "poi_approach": 0, "rejections": {},
        }


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
    top_rej = "(none)" if not rej else f"{Counter(rej).most_common(1)[0][0]} ({Counter(rej).most_common(1)[0][1]}x)"
    equity = get_equity()
    peak = state.get("peak_equity") or equity
    dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0
    meta = state.get("open_meta") or {}
    open_str = (f"{meta['side']} @ {meta['entry']:.2f} (sl {meta['sl']:.2f} tp {meta['tp']:.2f})"
                if meta else "flat")
    cu = _parse_iso(state.get("cooldown_until_iso"))
    cooldown_str = ""
    if cu and cu > _now_utc():
        mins_left = int((cu - _now_utc()).total_seconds() // 60)
        cooldown_str = f"\nCooldown: {mins_left}m left ({state.get('consecutive_losses',0)} losses)"
    tg_send(
        f"<b>[SMC DAILY SUMMARY]</b> {today_ist}\n"
        f"Entries: {c.get('entries',0)}\n"
        f"POI approaches: {c.get('poi_approach',0)}\n"
        f"Watches: {c.get('watches',0)}\n"
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
    print("\nStopping after current cycle…")


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


def run_test_trade(side: str) -> int:
    if not init_mt5_headless():
        log(f"mt5.initialize() failed: {mt5.last_error()}"); return 1
    if mt5.account_info() is None:
        log("no account_info"); mt5.shutdown(); return 1
    if not mt5.symbol_select(SYMBOL, True):
        log(f"symbol_select({SYMBOL}) failed"); mt5.shutdown(); return 1
    ensure_journal_header()
    state = load_state()
    if state.get("peak_equity") is None:
        state["peak_equity"] = get_equity()
    df15, df1h = fetch_bars()
    if df15.empty or len(df15) < 30:
        log("not enough bars"); mt5.shutdown(); return 1
    atr_val = float(atr(df15["High"], df15["Low"], df15["Close"], ATR_PERIOD).iloc[-1])
    # Use LIVE tick price for SL/TP, not stale bar close — historical bar data
    # can be 1-2 bars behind tick especially right after MT5 init.
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None or tick.ask == 0:
        log("no tick available"); mt5.shutdown(); return 1
    price = tick.ask if side == "BUY" else tick.bid
    sig = {"severity": f"{side}_READY", "side": side, "price": price, "atr": atr_val,
           "sl_suggested": price - 1.5 * atr_val if side == "BUY" else price + 1.5 * atr_val,
           "tp_suggested": price + 2.5 * atr_val if side == "BUY" else price - 2.5 * atr_val,
           "rr": 1.67, "poi_score": 0, "reason": "TEST MODE — forced", "htf_bias": "test"}
    log(f"TEST MODE: forcing {side} through SMC pipeline. live_price={price:.2f} atr={atr_val:.2f}")
    tg_send(f"<b>[SMC TEST MODE]</b> Forcing {side} GOLD (magic {MAGIC})")
    regime_snap = classify_regime(df1h) if not df1h.empty else None
    ok = open_market(state, side, sig, regime_snap, None)
    save_state(state)
    mt5.shutdown()
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("--test-buy", "--test-sell"):
        return run_test_trade("BUY" if sys.argv[1] == "--test-buy" else "SELL")

    if not os.getenv("TELEGRAM_BOT_TOKEN") or not os.getenv("TELEGRAM_CHAT_ID"):
        log("WARNING: TELEGRAM creds missing — alerts disabled.")
    if not os.getenv("ALPHA_VANTAGE_KEY"):
        log("WARNING: ALPHA_VANTAGE_KEY missing — news filter disabled.")

    if not init_mt5_headless():
        log(f"mt5.initialize() failed: {mt5.last_error()}"); return 1
    info = mt5.account_info()
    if info is None:
        log("mt5.account_info() None (logged in?)"); mt5.shutdown(); return 1
    if not mt5.symbol_select(SYMBOL, True):
        log(f"symbol_select({SYMBOL}) failed"); mt5.shutdown(); return 1

    signal.signal(signal.SIGINT, handle_sigint)
    ensure_journal_header()
    state = load_state()
    eq0 = get_equity()
    if state["peak_equity"] is None:
        state["peak_equity"] = eq0

    log(f"MT5 SMC bot started.  account={info.login}  server={info.server}  "
        f"equity=${eq0:,.2f}  symbol={SYMBOL}  magic={MAGIC}")
    tg_send(
        f"<b>[SMC BOT START — MT5/XM]</b>\n"
        f"Account: {info.login}\nEquity: ${eq0:,.2f}\n"
        f"Symbol: {SYMBOL} (magic {MAGIC})\n"
        f"Open: {'yes (ticket '+str(state['open_ticket'])+')' if state.get('open_ticket') else 'no'}"
    )
    pg_log_event(BOT_NAME, "bot_start", {
        "account": info.login, "server": info.server,
        "equity": eq0, "symbol": SYMBOL,
        "open_ticket": state.get("open_ticket"),
        "magic": MAGIC,
    })
    pg_snapshot_equity(str(info.login), eq0, info.balance,
                       peak_equity=state.get("peak_equity"),
                       open_positions=1 if state.get("open_ticket") else 0)

    gate_cfg = _build_gate_config()

    while RUNNING:
        try:
            reset_daily_if_needed(state)
            df15, df1h = fetch_bars()
            if df15 is None or df15.empty or df1h is None or df1h.empty:
                check_mt5_alive_or_reconnect(state)  # watchdog: reconnect or sys.exit for NSSM
                time.sleep(POLL_SECONDS); continue
            reset_mt5_failure_counter(state)
            if len(df15) < 60 or len(df1h) < 60:
                log("not enough history yet, waiting…")
                time.sleep(POLL_SECONDS); continue

            latest_ts = str(df15.index[-1])
            new_bar = state["last_bar_ts"] != latest_ts

            # If we had an open position, check whether it's still there
            if state.get("open_ticket"):
                pos = find_position_by_ticket(state["open_ticket"])
                if pos is None:
                    reconcile_closed_position(state)
                    save_state(state)

            if new_bar:
                if not state.get("open_ticket"):
                    # Classify regime for this cycle
                    regime_snap = classify_regime(df1h)
                    sig = evaluate_smc(df15, df1h)
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
                        elif sev == "BREAKOUT_WATCH":
                            ct["poi_approach"] = ct.get("poi_approach", 0) + 1
                            tg_send(f"<b>[SMC WATCH]</b> {sig['side']} GOLD\n"
                                    f"{sig['reason']}\n"
                                    f"Price: {sig['price']:.2f}  ATR: {sig['atr']:.2f}")
                        elif sev in ("BUY_READY", "SELL_READY"):
                            # === ML META-LABELER (shadow mode default) ===
                            ml_result = score_signal_live(
                                df15, df1h, sig["side"], BOT_NAME,
                                rr_target=sig.get("rr", 1.5),
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

                            allowed, why, news = can_open_new_trade(state, sig["side"], gate_cfg)
                            if not allowed:
                                log(f"entry blocked: {why}")
                                tg_send(f"<b>[SMC BLOCKED]</b> {sig['side']} signal blocked: {why}")
                                ct.setdefault("rejections", {})[why.split(" ")[0]] = (
                                    ct["rejections"].get(why.split(" ")[0], 0) + 1)
                                continue
                            # ML veto (only outside shadow mode)
                            if not ml_result.shadow_mode and not ml_result.would_trade:
                                log(f"ML veto: {ml_result.note}")
                                tg_send(f"<b>[SMC ML VETO]</b> {sig['side']} skipped — {ml_result.note}")
                                ct.setdefault("rejections", {})["ml_veto"] = (
                                    ct["rejections"].get("ml_veto", 0) + 1)
                                continue
                            open_market(state, sig["side"], sig, regime_snap, news)
                else:
                    meta = state["open_meta"]
                    log(f"holding {meta['side']} @ {meta['entry']:.2f}  "
                        f"sl={meta['sl']:.2f} tp={meta['tp']:.2f} lots={meta['lots']}")

                state["last_bar_ts"] = latest_ts
                save_state(state)
            else:
                # heartbeat
                tick = mt5.symbol_info_tick(SYMBOL)
                px = tick.bid if tick else 0
                equity = get_equity()
                if state.get("peak_equity") is None or equity > state["peak_equity"]:
                    state["peak_equity"] = equity
                meta = state.get("open_meta") or {}
                pos_str = f"{meta.get('side')} @ {meta.get('entry'):.2f}" if meta else "flat"
                sess_ok, sess_name = is_tradeable_session(_now_utc())
                sess_str = sess_name if sess_ok else f"NO_SESSION({sess_name})"
                log(f"heartbeat  price={px:.2f}  {pos_str}  equity=${equity:,.2f}  "
                    f"today=${state['pnl_today_usd']:+.2f}  {sess_str}")
                # Periodic equity snapshot (every ~5 min)
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
            tg_send(f"<b>[SMC RISK]</b> loop error: {e}")
        time.sleep(POLL_SECONDS)

    log("Stopped. State saved.")
    final_eq = get_equity()
    tg_send(f"<b>[SMC BOT STOP]</b>\nEquity: ${final_eq:,.2f}")
    pg_log_event(BOT_NAME, "bot_stop", {"equity": final_eq})
    mt5.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
