"""
Trading session filter for gold.

Gold has well-known behaviour patterns by session:

  Asian session    (02:00 - 07:30 IST):  thin liquidity, range-bound, frequent
                                          false breakouts. Pros skip this.
  London open      (12:30 - 16:30 IST):  highest gold volume in the day.
                                          Real institutional flow. Trend forms.
  NY open / overlap (18:00 - 21:00 IST): London+NY overlap. Biggest moves.
                                          Best window for trend-following.
  NY afternoon     (21:00 - 23:30 IST):  follow-through and reversals.
                                          Still tradeable but more chop.
  Late NY / pre-Asian (23:30 - 02:00 IST): low liquidity, often whip.

This filter implements an ALLOW list (London open + NY overlap + NY afternoon
by default) and blocks the Asian session and the late-NY dead zone.

All times are IST (Asia/Kolkata, UTC+5:30). The trader is in India.

Stateless: pass in a UTC datetime, get back True/False.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class SessionWindow:
    name: str
    start_ist: time
    end_ist: time

    def contains(self, ts_ist: datetime) -> bool:
        t = ts_ist.time()
        # Handle wrap-around windows (e.g., 22:00 - 02:00)
        if self.start_ist <= self.end_ist:
            return self.start_ist <= t < self.end_ist
        return t >= self.start_ist or t < self.end_ist


@dataclass
class SessionConfig:
    """Which windows the bot is ALLOWED to enter new trades in."""
    enabled: bool = True
    windows: list[SessionWindow] = field(default_factory=lambda: [
        SessionWindow(name="London",       start_ist=time(12, 30), end_ist=time(16, 30)),
        SessionWindow(name="NY_overlap",   start_ist=time(18, 0),  end_ist=time(21, 0)),
        SessionWindow(name="NY_afternoon", start_ist=time(21, 0),  end_ist=time(23, 30)),
    ])
    block_weekend: bool = True   # gold spot pauses Fri 23:00 UTC - Sun 22:00 UTC


def to_ist(ts_utc: datetime) -> datetime:
    """Convert a UTC datetime to IST."""
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    return ts_utc.astimezone(IST)


def current_session(ts_utc: datetime, cfg: SessionConfig | None = None) -> str | None:
    """Return the name of the current session window if `ts_utc` is inside one,
    otherwise None. Useful for logging."""
    cfg = cfg or SessionConfig()
    ts_ist = to_ist(ts_utc)
    for w in cfg.windows:
        if w.contains(ts_ist):
            return w.name
    return None


def is_tradeable(ts_utc: datetime, cfg: SessionConfig | None = None) -> tuple[bool, str]:
    """Return (allowed, reason_if_blocked) for the given UTC timestamp.

    The current bar's timestamp must fall inside one of the configured
    `windows` AND it must not be a weekend (when enabled).
    """
    cfg = cfg or SessionConfig()
    if not cfg.enabled:
        return True, ""

    ts_ist = to_ist(ts_utc)

    # Weekend block (gold spot pauses)
    if cfg.block_weekend:
        # weekday(): Monday=0 ... Sunday=6
        # Gold spot: pause Friday 23:00 UTC (Sat 04:30 IST) to Sunday 22:00 UTC (Mon 03:30 IST)
        ts_utc_aware = ts_utc if ts_utc.tzinfo else ts_utc.replace(tzinfo=timezone.utc)
        wd = ts_utc_aware.weekday()
        if wd == 5:  # Saturday: full day
            return False, "weekend_closed"
        if wd == 6:  # Sunday: closed until 22:00 UTC
            if ts_utc_aware.hour < 22:
                return False, "weekend_closed"
        if wd == 4:  # Friday: closed after 23:00 UTC (rare in normal hours)
            if ts_utc_aware.hour >= 23:
                return False, "weekend_closed"

    # Session window check
    in_session = current_session(ts_utc, cfg)
    if in_session is None:
        return False, "outside_session"

    return True, ""


def minutes_until_next_session(ts_utc: datetime, cfg: SessionConfig | None = None) -> int | None:
    """How long until the next allowed session starts? None if currently inside one."""
    cfg = cfg or SessionConfig()
    ts_ist = to_ist(ts_utc)
    if current_session(ts_utc, cfg) is not None:
        return None
    # Check windows for next start time today, then tomorrow
    now_t = ts_ist.time()
    today_starts = sorted(w.start_ist for w in cfg.windows if w.start_ist > now_t)
    if today_starts:
        next_start = today_starts[0]
        delta = (datetime.combine(ts_ist.date(), next_start, tzinfo=IST) - ts_ist)
        return int(delta.total_seconds() // 60)
    # Tomorrow
    earliest = sorted(w.start_ist for w in cfg.windows)
    if earliest:
        next_dt = datetime.combine(ts_ist.date() + timedelta(days=1), earliest[0], tzinfo=IST)
        return int((next_dt - ts_ist).total_seconds() // 60)
    return None
