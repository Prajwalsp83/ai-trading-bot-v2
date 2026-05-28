"""
Economic calendar block.

Gold reacts violently to a small set of high-impact events. Trading WITHIN
the +-30 min window around these events is asking for stop-runs.

For v1 we keep a manually-maintained JSON file of upcoming events. This is
fine for the user since the calendar is well-known (FOMC, CPI, NFP dates
are public weeks ahead). For v2 we'll scrape ForexFactory.

Events JSON schema (data/economic_calendar.json):
  [
    {"ts_utc": "2026-06-04T18:00:00Z", "name": "US CPI",
     "impact": "high", "currency": "USD"},
    {"ts_utc": "2026-06-11T18:00:00Z", "name": "FOMC Statement",
     "impact": "high", "currency": "USD"},
    ...
  ]

Impact levels: "high" (block), "medium" (warn), "low" (ignore).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal


@dataclass
class CalendarEvent:
    ts_utc: datetime
    name: str
    impact: Literal["high", "medium", "low"]
    currency: str


def _parse_iso(s: str) -> datetime:
    # Accept "2026-06-04T18:00:00Z" or "...+00:00"
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


def block_window(now_utc: datetime, events: list[CalendarEvent],
                 before_minutes: int = 30, after_minutes: int = 60,
                 impact_threshold: Literal["high", "medium"] = "high",
                 currencies: list[str] | None = None) -> tuple[bool, str]:
    """Return (blocked, reason) if `now_utc` is inside the block window of any qualifying event."""
    impact_rank = {"high": 3, "medium": 2, "low": 1}
    min_rank = impact_rank[impact_threshold]
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    for ev in events:
        if impact_rank.get(ev.impact, 1) < min_rank:
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


def next_event(now_utc: datetime, events: list[CalendarEvent],
               impact_threshold: Literal["high", "medium"] = "high") -> CalendarEvent | None:
    impact_rank = {"high": 3, "medium": 2, "low": 1}
    min_rank = impact_rank[impact_threshold]
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    upcoming = [e for e in events
                if e.ts_utc > now_utc and impact_rank.get(e.impact, 1) >= min_rank]
    return upcoming[0] if upcoming else None
