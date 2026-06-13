"""
Shared infrastructure for the live bots (mt5_live.py + mt5_smc.py).

Everything here is *self-contained* — no imports from app/ — so the file
can be dropped on the VPS via Notepad clipboard along with the two bots.

Contains:
  - Telegram helper
  - Trading session filter (London, NY overlap, NY afternoon — IST)
  - Economic calendar block (CPI / NFP / FOMC ± window)
  - News sentiment fetch + cache (Alpha Vantage)
  - 3-tier drawdown risk scaling
  - Fractional-Kelly sizing from CSV journal
  - ADX-based regime classifier (TREND / CHOP / TRANSITION / HIGH_VOL)
  - can_open_new_trade() composite gate

Each section is independently testable; check_news.py uses some of these.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import requests


IST = timezone(timedelta(hours=5, minutes=30))


# ============================== MT5 INIT =============================
def init_mt5_headless():
    """Initialize MT5 with optional headless-launch credentials from env vars.

    Environment variables (set in .env on the VPS):
      MT5_PATH     — full path to terminal64.exe (e.g. C:\\Program Files\\XM Global MT5\\terminal64.exe)
      MT5_LOGIN    — account number (integer as string)
      MT5_PASSWORD — account password
      MT5_SERVER   — broker server name (e.g. XMGlobal-MT5 2)

    Behavior:
      - If all 4 env vars are set: launches MT5 itself (true headless — survives RDP logout)
      - If any are missing: falls back to bare mt5.initialize() (connects to MT5 already
        running in the same Windows session — requires interactive RDP session to be alive)

    Returns: True on success, False on failure. Caller must check and handle.
    """
    import MetaTrader5 as mt5  # imported lazily so non-VPS scripts can use this module
    path = os.getenv("MT5_PATH")
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if path and login and password and server:
        return mt5.initialize(
            path=path,
            login=int(login),
            password=password,
            server=server,
            timeout=60000,
        )
    return mt5.initialize()


def check_mt5_alive_or_reconnect(state: dict, max_failures_before_exit: int = 5) -> bool:
    """Call this after a failed bar fetch. Detects mid-session MT5 death.

    Logic (escalating):
      1st empty fetch:      treat as transient, return True (try again next cycle)
      2nd-4th empty fetch:  log, attempt mt5.shutdown() + init_mt5_headless()
      >= max_failures:      sys.exit(1) so NSSM restarts the bot fresh

    Mutates state['mt5_consec_failures']. Caller should call
    reset_mt5_failure_counter(state) after every SUCCESSFUL fetch.

    Returns True normally; never returns False (it either reconnects or exits).
    """
    import sys
    import time as _time
    import MetaTrader5 as mt5

    state["mt5_consec_failures"] = state.get("mt5_consec_failures", 0) + 1
    n = state["mt5_consec_failures"]

    if n == 1:
        print(f"[watchdog] empty bar fetch (1/{max_failures_before_exit}) — transient, will retry", flush=True)
        return True

    if n >= max_failures_before_exit:
        msg = (f"MT5 unresponsive for {n} consecutive cycles "
               f"(~{n * 60}s) — exiting for NSSM restart")
        print(f"[watchdog] {msg}", flush=True)
        tg_send(f"<b>[WATCHDOG]</b> {msg}")
        try:
            mt5.shutdown()
        except Exception:
            pass
        sys.exit(1)

    # n in [2, max-1] — try to reconnect in-process
    print(f"[watchdog] empty bar fetch ({n}/{max_failures_before_exit}) — "
          f"attempting mt5.shutdown() + init_mt5_headless()...", flush=True)
    if n == 2:
        # Notify once on second failure so user knows recovery started
        tg_send(f"<b>[WATCHDOG]</b> MT5 unresponsive — attempting reconnect "
                f"({n}/{max_failures_before_exit})")
    try:
        mt5.shutdown()
    except Exception:
        pass
    _time.sleep(2)
    ok = init_mt5_headless()
    if ok:
        print(f"[watchdog] MT5 reconnected after {n} failures", flush=True)
        tg_send(f"<b>[WATCHDOG]</b> MT5 reconnected after {n} failed cycles")
        state["mt5_consec_failures"] = 0
    return True


def reset_mt5_failure_counter(state: dict) -> None:
    """Call after every successful bar fetch to reset the watchdog counter."""
    if state.get("mt5_consec_failures", 0) > 0:
        state["mt5_consec_failures"] = 0


# =============================== Telegram ============================
def tg_send(text: str, token: str | None = None, chat: str | None = None) -> None:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = chat or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


# ====================== Remote pause control =========================
# Flag file written by telegram_control.py (/pause, /resume). Bots check it
# at the top of can_open_new_trade: paused blocks NEW entries only -- open
# positions keep their server-side SL/TP and are still managed/reconciled.
CONTROL_PATH = Path(__file__).resolve().parent.parent / "data" / ".control.json"


def control_read(path: Path | None = None) -> dict:
    p = path or CONTROL_PATH
    try:
        with open(p) as f:
            return json.load(f) or {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def control_paused(path: Path | None = None) -> tuple[bool, str]:
    """Returns (paused, reason). Fail-open: unreadable flag = not paused."""
    c = control_read(path)
    if not c.get("paused"):
        return False, ""
    by = c.get("by", "telegram")
    at = c.get("at_utc", "?")
    return True, f"remote_pause (by {by} at {at}; send /resume to unpause)"


def control_set(paused: bool, by: str = "telegram", path: Path | None = None) -> dict:
    p = path or CONTROL_PATH
    c = {"paused": bool(paused), "by": by,
         "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(c, f)
    os.replace(tmp, p)
    return c


# ============================ Session filter =========================
@dataclass
class SessionWindow:
    name: str
    start_ist: time
    end_ist: time

    def contains(self, ts_ist: datetime) -> bool:
        t = ts_ist.time()
        if self.start_ist <= self.end_ist:
            return self.start_ist <= t <= self.end_ist
        return t >= self.start_ist or t <= self.end_ist  # wraps midnight


@dataclass
class SessionConfig:
    enabled: bool = True
    windows: list[SessionWindow] = field(default_factory=lambda: [
        SessionWindow(name="London",       start_ist=time(12, 30), end_ist=time(16, 30)),
        SessionWindow(name="NY_overlap",   start_ist=time(18, 0),  end_ist=time(21, 0)),
        SessionWindow(name="NY_afternoon", start_ist=time(21, 0),  end_ist=time(23, 30)),
    ])
    block_weekend: bool = True


def current_session(ts_utc: datetime, cfg: SessionConfig | None = None) -> str | None:
    cfg = cfg or SessionConfig()
    ts_ist = ts_utc.astimezone(IST)
    for w in cfg.windows:
        if w.contains(ts_ist):
            return w.name
    return None


def is_tradeable_session(ts_utc: datetime, cfg: SessionConfig | None = None) -> tuple[bool, str]:
    cfg = cfg or SessionConfig()
    if not cfg.enabled:
        return True, ""
    if cfg.block_weekend and ts_utc.weekday() >= 5:
        return False, "weekend_block"
    sess = current_session(ts_utc, cfg)
    if sess:
        return True, sess
    return False, "outside_session"


def minutes_until_next_session(ts_utc: datetime, cfg: SessionConfig | None = None) -> int | None:
    cfg = cfg or SessionConfig()
    ts_ist = ts_utc.astimezone(IST)
    candidates = []
    for w in cfg.windows:
        start_today = ts_ist.replace(hour=w.start_ist.hour, minute=w.start_ist.minute,
                                     second=0, microsecond=0)
        start_tomorrow = start_today + timedelta(days=1)
        for s in (start_today, start_tomorrow):
            if s > ts_ist:
                candidates.append(s)
    if not candidates:
        return None
    nxt = min(candidates)
    return int((nxt - ts_ist).total_seconds() // 60)


# ========================== Economic calendar ========================
@dataclass
class CalendarEvent:
    ts_utc: datetime
    name: str
    impact: Literal["high", "medium", "low"]
    currency: str


def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def load_calendar(path: Path) -> list[CalendarEvent]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text())
    except Exception:
        return []
    events: list[CalendarEvent] = []
    for r in rows:
        try:
            events.append(CalendarEvent(
                ts_utc=_parse_iso(r["ts_utc"]),
                name=r.get("name", "Unknown"),
                impact=r.get("impact", "low"),
                currency=r.get("currency", ""),
            ))
        except Exception:
            continue
    events.sort(key=lambda e: e.ts_utc)
    return events


def calendar_block(now_utc: datetime, events: list[CalendarEvent],
                   before_minutes: int = 30, after_minutes: int = 60,
                   impact_threshold: Literal["high", "medium"] = "high",
                   currencies: list[str] | None = None) -> tuple[bool, str]:
    rank = {"high": 3, "medium": 2, "low": 1}
    min_rank = rank[impact_threshold]
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    for ev in events:
        if rank.get(ev.impact, 1) < min_rank:
            continue
        if currencies and ev.currency not in currencies:
            continue
        start = ev.ts_utc - timedelta(minutes=before_minutes)
        end = ev.ts_utc + timedelta(minutes=after_minutes)
        if start <= now_utc <= end:
            mins = int(abs((now_utc - ev.ts_utc).total_seconds() // 60))
            when = "before" if now_utc < ev.ts_utc else "after"
            return True, f"calendar_block: {ev.name} ({ev.impact}) {mins}m {when}"
    return False, ""


def next_calendar_event(now_utc: datetime, events: list[CalendarEvent],
                        impact_threshold: Literal["high", "medium"] = "high") -> CalendarEvent | None:
    rank = {"high": 3, "medium": 2, "low": 1}
    min_rank = rank[impact_threshold]
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    upcoming = [e for e in events if e.ts_utc > now_utc and rank.get(e.impact, 1) >= min_rank]
    return upcoming[0] if upcoming else None


# ============================ News sentiment =========================
# Verified 2026-05-25: combining tickers behaves as AND on AV free tier and
# returns 0. GLD alone returns ~50 articles. Use GLD only (purest gold proxy).
# Skip GC/XAU/IAU/GOLD/GDX/NEM combos — only single-ticker queries are stable.
GOLD_TICKERS = ["GLD"]
GOLD_TOPICS: list[str] = []
BULL_CUTOFF = 0.15
BEAR_CUTOFF = -0.15


@dataclass
class SentimentSummary:
    n_articles: int
    score: float
    bias: Literal["bullish", "bearish", "neutral"]
    latest_ts_utc: datetime | None


def fetch_gold_sentiment(api_key: str | None, cache_path: Path,
                         cache_ttl_minutes: int = 25,
                         max_age_hours: int = 6) -> SentimentSummary:
    """Pull gold-related sentiment from Alpha Vantage, cached on disk.
    Returns a neutral summary if no api_key or fetch fails."""
    if not api_key:
        return SentimentSummary(0, 0.0, "neutral", None)

    payload = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            cached_at = datetime.fromisoformat(cached.get("_cached_at", ""))
            if datetime.now(timezone.utc) - cached_at <= timedelta(minutes=cache_ttl_minutes):
                payload = cached.get("payload")
        except Exception:
            pass

    if payload is None:
        try:
            r = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "NEWS_SENTIMENT",
                    "apikey": api_key,
                    "sort": "LATEST",
                    "limit": "50",
                    "tickers": ",".join(GOLD_TICKERS),
                    "topics": ",".join(GOLD_TOPICS),
                },
                timeout=15,
            )
            r.raise_for_status()
            payload = r.json()
            if "feed" in payload:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({
                    "_cached_at": datetime.now(timezone.utc).isoformat(),
                    "payload": payload,
                }, default=str))
        except Exception:
            return SentimentSummary(0, 0.0, "neutral", None)

    feed = payload.get("feed", []) or []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    fresh = []
    for item in feed:
        try:
            ts = datetime.strptime(item.get("time_published", ""), "%Y%m%dT%H%M%S")
            ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        relevance = 0.0
        score = float(item.get("overall_sentiment_score", 0.0) or 0.0)
        for ts_score in item.get("ticker_sentiment", []) or []:
            if ts_score.get("ticker") in GOLD_TICKERS:
                rel = float(ts_score.get("relevance_score", 0.0) or 0.0)
                if rel > relevance:
                    relevance = rel
                    score = float(ts_score.get("ticker_sentiment_score", score) or score)
        fresh.append((ts, score, max(relevance, 0.1)))

    if not fresh:
        return SentimentSummary(0, 0.0, "neutral", None)

    total_w = sum(w for _, _, w in fresh)
    weighted = sum(s * w for _, s, w in fresh) / total_w if total_w > 0 else 0.0
    bias = "bullish" if weighted > BULL_CUTOFF else "bearish" if weighted < BEAR_CUTOFF else "neutral"
    latest = max(t for t, _, _ in fresh)
    return SentimentSummary(len(fresh), weighted, bias, latest)


# =========================== Risk: DD tiers ==========================
@dataclass
class DDTier:
    threshold_pct: float
    multiplier: float


DEFAULT_DD_TIERS: list[DDTier] = [
    DDTier(threshold_pct=0.12, multiplier=0.0),
    DDTier(threshold_pct=0.07, multiplier=0.25),
    DDTier(threshold_pct=0.03, multiplier=0.50),
    DDTier(threshold_pct=0.0,  multiplier=1.00),
]


def dd_multiplier(equity: float, peak_equity: float,
                  tiers: list[DDTier] | None = None) -> tuple[float, str]:
    if peak_equity is None or peak_equity <= 0:
        return 1.0, "full"
    dd = max(0.0, (peak_equity - equity) / peak_equity)
    tiers = sorted(tiers or DEFAULT_DD_TIERS, key=lambda t: -t.threshold_pct)
    for t in tiers:
        if dd >= t.threshold_pct:
            name = (f"halt_dd>={t.threshold_pct*100:.0f}%" if t.multiplier == 0
                    else f"dd>={t.threshold_pct*100:.0f}%_x{t.multiplier:.2f}")
            return t.multiplier, name
    return 1.0, "full"


# ========================== Risk: Kelly ==============================
@dataclass
class KellyParams:
    lookback_trades: int = 30
    fraction: float = 0.25
    min_trades_required: int = 10
    max_multiplier: float = 2.0
    min_multiplier: float = 0.25
    default_multiplier: float = 1.0


def _read_recent_r(journal_path: Path, lookback: int) -> list[float]:
    if not journal_path.exists():
        return []
    try:
        with journal_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    out = []
    for row in rows[-lookback:]:
        try:
            out.append(float(row.get("r_realised", 0) or 0))
        except Exception:
            continue
    return out


def kelly_multiplier(journal_path: Path, params: KellyParams | None = None) -> tuple[float, str]:
    p = params or KellyParams()
    rs = _read_recent_r(journal_path, p.lookback_trades)
    n = len(rs)
    if n < p.min_trades_required:
        return p.default_multiplier, f"kelly:sample_too_small({n}/{p.min_trades_required})"

    wins = [r for r in rs if r > 0]
    losses = [-r for r in rs if r < 0]
    if not losses:
        return min(p.max_multiplier, 1.0 + p.fraction), f"kelly:no_losses_in_{n}_trades"
    if not wins:
        return p.min_multiplier, f"kelly:no_wins_in_{n}_trades"

    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    b = avg_win / avg_loss
    raw = (b * win_rate - (1 - win_rate)) / b
    fractional = raw * p.fraction
    mult = max(p.min_multiplier, min(p.max_multiplier, 1.0 + fractional * 5.0))
    note = (f"kelly:n={n} wr={win_rate*100:.0f}% b={b:.2f} "
            f"raw={raw:+.3f} frac={fractional:+.3f} -> x{mult:.2f}")
    return mult, note


# ============================ Regime (ADX) ===========================
class Regime(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CHOP = "chop"
    TRANSITION = "transition"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


@dataclass
class RegimeParams:
    adx_period: int = 14
    adx_trend_min: float = 25.0
    adx_chop_max: float = 20.0
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    atr_pct_lookback: int = 100
    high_vol_pct: float = 0.95
    weights: dict = field(default_factory=lambda: {
        "trend_up":   {"breakout": 1.0, "smc": 0.3},
        "trend_down": {"breakout": 1.0, "smc": 0.3},
        "chop":       {"breakout": 0.0, "smc": 1.0},
        "transition": {"breakout": 0.5, "smc": 0.5},
        "high_vol":   {"breakout": 0.5, "smc": 0.5},
        "unknown":    {"breakout": 0.0, "smc": 0.0},
    })


@dataclass
class RegimeSnapshot:
    regime: Regime
    adx: float
    note: str
    weight_breakout: float
    weight_smc: float


def _wilder_ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr_w = _wilder_ema(tr, period)
    plus_di = 100 * _wilder_ema(plus_dm, period) / atr_w.replace(0, pd.NA)
    minus_di = 100 * _wilder_ema(minus_dm, period) / atr_w.replace(0, pd.NA)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)) * 100
    return _wilder_ema(dx.fillna(0), period), plus_di.fillna(0), minus_di.fillna(0)


def classify_regime(bars_trend: pd.DataFrame,
                    params: RegimeParams | None = None) -> RegimeSnapshot:
    p = params or RegimeParams()
    if len(bars_trend) < max(p.ema_slow, p.adx_period) + 5:
        return _unknown_regime(p, "not_enough_history")

    high, low, close = bars_trend["High"], bars_trend["Low"], bars_trend["Close"]
    ema_f = close.ewm(span=p.ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=p.ema_slow, adjust=False).mean()
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr_s = _wilder_ema(tr, p.atr_period)
    atr_pct = atr_s.rolling(p.atr_pct_lookback).rank(pct=True)
    adx_s, di_p, di_m = _adx(high, low, close, p.adx_period)

    if pd.isna(adx_s.iloc[-1]) or pd.isna(ema_f.iloc[-1]) or pd.isna(ema_s.iloc[-1]):
        return _unknown_regime(p, "indicator_nan")

    adx_val = float(adx_s.iloc[-1])
    di_p_val = float(di_p.iloc[-1])
    di_m_val = float(di_m.iloc[-1])
    ef, es = float(ema_f.iloc[-1]), float(ema_s.iloc[-1])
    atr_rank = float(atr_pct.iloc[-1]) if not pd.isna(atr_pct.iloc[-1]) else 0.5

    if atr_rank >= p.high_vol_pct:
        regime, note = Regime.HIGH_VOL, f"atr_pct={atr_rank:.2f}>={p.high_vol_pct}"
    elif adx_val >= p.adx_trend_min and ef > es and di_p_val > di_m_val:
        regime, note = Regime.TREND_UP, f"adx={adx_val:.1f}+ema_up"
    elif adx_val >= p.adx_trend_min and ef < es and di_m_val > di_p_val:
        regime, note = Regime.TREND_DOWN, f"adx={adx_val:.1f}+ema_dn"
    elif adx_val <= p.adx_chop_max:
        regime, note = Regime.CHOP, f"adx={adx_val:.1f}<={p.adx_chop_max}"
    else:
        regime, note = Regime.TRANSITION, f"adx={adx_val:.1f}_mid"

    w = p.weights.get(regime.value, {"breakout": 0.5, "smc": 0.5})
    return RegimeSnapshot(
        regime=regime, adx=adx_val, note=note,
        weight_breakout=float(w.get("breakout", 0.0)),
        weight_smc=float(w.get("smc", 0.0)),
    )


def _unknown_regime(p: RegimeParams, why: str) -> RegimeSnapshot:
    w = p.weights.get("unknown", {"breakout": 0.0, "smc": 0.0})
    return RegimeSnapshot(
        regime=Regime.UNKNOWN, adx=0.0, note=why,
        weight_breakout=float(w.get("breakout", 0.0)),
        weight_smc=float(w.get("smc", 0.0)),
    )


# ======================= Composite gate (NEWS=optional) ==============
@dataclass
class GateConfig:
    use_session_filter: bool = True
    use_calendar_filter: bool = True
    use_news_filter: bool = True
    news_block_threshold: float = 0.35     # block when |sentiment| exceeds this against direction
    calendar_path: Path | None = None
    av_api_key: str | None = None
    av_cache_path: Path | None = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def composite_pre_signal_gate(gc: GateConfig) -> tuple[bool, str]:
    """Returns (allowed, reason). Run BEFORE strategy evaluation.
    Direction-agnostic — blocks sessions/calendar only."""
    now = _now_utc()
    if gc.use_session_filter:
        ok, why = is_tradeable_session(now)
        if not ok:
            return False, why
    if gc.use_calendar_filter and gc.calendar_path:
        events = load_calendar(gc.calendar_path)
        blocked, reason = calendar_block(now, events)
        if blocked:
            return False, reason
    return True, ""


def directional_news_gate(side: str, gc: GateConfig) -> tuple[bool, str, SentimentSummary | None]:
    """Returns (allowed, reason, summary). Use AFTER strategy fires to
    veto trades fighting strong contrary news. BUY blocked if sentiment
    strongly bearish; SELL blocked if strongly bullish. Neutral always passes."""
    if not gc.use_news_filter or not gc.av_api_key or not gc.av_cache_path:
        return True, "news_disabled", None
    s = fetch_gold_sentiment(gc.av_api_key, gc.av_cache_path)
    if s.n_articles == 0:
        return True, "no_news_data", s
    if side == "BUY" and s.score < -gc.news_block_threshold:
        return False, f"news_contra: BUY vs bearish news ({s.score:+.2f}, n={s.n_articles})", s
    if side == "SELL" and s.score > gc.news_block_threshold:
        return False, f"news_contra: SELL vs bullish news ({s.score:+.2f}, n={s.n_articles})", s
    return True, f"news_ok: {s.bias} ({s.score:+.2f}, n={s.n_articles})", s


# ====================== Combined risk computation ====================
@dataclass
class RiskDecision:
    risk_pct: float
    base_risk_pct: float
    dd_mult: float
    dd_tier: str
    kelly_mult: float
    kelly_note: str
    regime_mult: float
    regime_note: str
    halted: bool
    explanation: str


def compute_effective_risk(
    base_risk_pct: float,
    equity: float,
    peak_equity: float | None,
    journal_path: Path | None,
    regime_snapshot: RegimeSnapshot | None,
    strategy_name: str,              # "breakout" or "smc"
    kelly_params: KellyParams | None = None,
    dd_tiers: list[DDTier] | None = None,
    use_kelly: bool = True,
    use_regime: bool = True,
) -> RiskDecision:
    dd_m, dd_tier = dd_multiplier(equity, peak_equity or equity, dd_tiers)
    if use_kelly and journal_path is not None:
        k_m, k_note = kelly_multiplier(journal_path, kelly_params)
    else:
        k_m, k_note = 1.0, "kelly_disabled"
    if use_regime and regime_snapshot is not None:
        r_m = regime_snapshot.weight_breakout if strategy_name == "breakout" else regime_snapshot.weight_smc
        r_note = f"regime:{regime_snapshot.regime.value} x{r_m:.2f} ({regime_snapshot.note})"
    else:
        r_m, r_note = 1.0, "regime_disabled"

    eff = base_risk_pct * dd_m * k_m * r_m
    halted = (dd_m <= 0.0) or (r_m <= 0.0)
    expl = (f"base={base_risk_pct*100:.2f}% x dd={dd_m:.2f}({dd_tier}) "
            f"x kelly={k_m:.2f} x regime={r_m:.2f}({strategy_name}) "
            f"= {eff*100:.3f}%")
    return RiskDecision(
        risk_pct=eff, base_risk_pct=base_risk_pct,
        dd_mult=dd_m, dd_tier=dd_tier,
        kelly_mult=k_m, kelly_note=k_note,
        regime_mult=r_m, regime_note=r_note,
        halted=halted, explanation=expl,
    )
