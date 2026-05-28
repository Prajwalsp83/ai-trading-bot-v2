"""
Smoke test for the news + sessions + calendar modules.

Run:
    export ALPHA_VANTAGE_KEY=your_key
    python scripts/check_news.py

What it does:
  1. Pulls today's gold-related news from Alpha Vantage, prints aggregated sentiment.
  2. Shows the current trading session (London / NY overlap / NY afternoon / blocked).
  3. Shows whether we are currently blocked by an upcoming economic event.

You need a free Alpha Vantage API key: https://www.alphavantage.co/support/#api-key
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from app.news.alpha_vantage import AlphaVantageNews   # noqa: E402
from app.risk.sessions import (                       # noqa: E402
    current_session, is_tradeable, minutes_until_next_session,
)
from app.risk.calendar import (                       # noqa: E402
    load_calendar, block_window, next_event,
)


def main():
    now = datetime.now(timezone.utc)

    # ---- Sessions ----
    print("=== Session ===")
    sess = current_session(now)
    allowed, reason = is_tradeable(now)
    print(f"  now (UTC)        : {now.isoformat(timespec='seconds')}")
    print(f"  current session  : {sess or '(outside)'}")
    print(f"  tradeable        : {allowed}{'  (' + reason + ')' if not allowed else ''}")
    if not allowed:
        mins = minutes_until_next_session(now)
        if mins is not None:
            print(f"  next session in  : {mins} min")
    print()

    # ---- Calendar ----
    print("=== Economic calendar ===")
    cal_path = HERE / "data" / "economic_calendar.json"
    events = load_calendar(cal_path)
    print(f"  loaded {len(events)} events from {cal_path.name}")
    blocked, why = block_window(now, events)
    print(f"  blocked now      : {blocked}{'  (' + why + ')' if blocked else ''}")
    nxt = next_event(now, events)
    if nxt:
        mins = int((nxt.ts_utc - now).total_seconds() // 60)
        print(f"  next high-impact : {nxt.name} ({nxt.currency}) in {mins} min  @ {nxt.ts_utc}")
    else:
        print("  next high-impact : none in calendar")
    print()

    # ---- News ----
    key = os.environ.get("ALPHA_VANTAGE_KEY")
    if not key:
        print("=== News ===")
        print("  ALPHA_VANTAGE_KEY not set in env — skipping.")
        print("  Get a free key: https://www.alphavantage.co/support/#api-key")
        print("  Then: export ALPHA_VANTAGE_KEY=your_key && python scripts/check_news.py")
        return

    print("=== Gold news (Alpha Vantage) ===")
    cache_path = HERE / "data" / ".av_news_cache.json"
    news = AlphaVantageNews(api_key=key, cache_path=cache_path)
    # Wider window for smoke test — gold news is intermittent, weekends create
    # 60+ hr gaps. Live bots use 6h (the default) so stale weekend sentiment
    # doesn't influence Monday trades.
    s = news.latest_gold_sentiment(max_age_hours=72)
    print(f"  articles (72h)  : {s.n_articles}")
    print(f"  weighted score  : {s.score:+.3f}  (range -1..+1)")
    print(f"  bias            : {s.bias.upper()}")
    print(f"  bullish/bearish/neutral : {s.bullish_count}/{s.bearish_count}/{s.neutral_count}")
    print(f"  latest article  : {s.latest_ts_utc}")
    print()

    articles = news.fetch_gold_articles(limit=5)
    print("--- Latest 5 headlines ---")
    for a in articles[:5]:
        tag = ("BULL" if a.overall_score > 0.15 else
               "BEAR" if a.overall_score < -0.15 else "NEUT")
        print(f"  [{tag}] [{a.overall_score:+.2f} rel={a.relevance:.2f}]  {a.title}")
        print(f"         {a.source}  {a.time_published_utc.isoformat(timespec='minutes')}")


if __name__ == "__main__":
    main()
