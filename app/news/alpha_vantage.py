"""
Alpha Vantage NEWS_SENTIMENT fetcher.

Free tier: 25 requests/day, 5 requests/minute. We poll every ~30 min,
so daily limit is plenty (~48 polls).

Each article comes with:
  - title, summary, url, time_published
  - overall_sentiment_score (-1 bearish ... +1 bullish)
  - topic-level sentiment for tickers (GLD, GC, etc) if relevant

We cache results to a JSON file so repeated reads don't burn the quota.

Free key: https://www.alphavantage.co/support/#api-key

Usage:
    import os
    from app.news.alpha_vantage import AlphaVantageNews
    feed = AlphaVantageNews(api_key=os.environ["ALPHA_VANTAGE_KEY"])
    s = feed.latest_gold_sentiment()  # -> {"score": +0.18, "n": 12, "bias": "bullish", ...}
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import requests


@dataclass
class NewsArticle:
    title: str
    url: str
    source: str
    time_published_utc: datetime
    overall_score: float        # -1 (bearish) ... +1 (bullish)
    relevance: float            # 0 ... 1 for gold/XAU specifically
    topics: list[str] = field(default_factory=list)


@dataclass
class SentimentSummary:
    n_articles: int
    score: float                          # weighted-by-relevance average
    bias: Literal["bullish", "bearish", "neutral"]
    latest_ts_utc: datetime | None
    bullish_count: int
    bearish_count: int
    neutral_count: int


# NOTE: AV's NEWS_SENTIMENT only indexes equity tickers (not commodity codes).
# Verified 2026-05-25: GLD/GDX/NEM each return ~50 articles individually,
# but combining tickers behaves as AND and returns 0 — query a single ticker.
# GLD (SPDR Gold ETF) tracks physical gold most directly, so it's our anchor.
GOLD_TICKERS = ["GLD"]
# Topics filter narrows results further; leave empty for max recall.
GOLD_TOPICS: list[str] = []

# Thresholds Alpha Vantage uses internally:
#   bearish if score < -0.15
#   somewhat-bearish if score in [-0.35, -0.15]
#   neutral if [-0.15, +0.15]
#   somewhat-bullish if [+0.15, +0.35]
#   bullish if > +0.35
BULL_CUTOFF = 0.15
BEAR_CUTOFF = -0.15


class AlphaVantageNews:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, cache_path: Path | None = None,
                 cache_ttl_minutes: int = 25):
        self.api_key = api_key
        self.cache_path = cache_path
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)

    # ---- fetch ----
    def _fetch_raw(self, tickers: list[str] | None = None,
                   topics: list[str] | None = None,
                   limit: int = 50) -> dict:
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "sort": "LATEST",
            "limit": str(limit),
        }
        if tickers:
            params["tickers"] = ",".join(tickers)
        if topics:
            params["topics"] = ",".join(topics)
        r = requests.get(self.BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    # ---- cache helpers ----
    def _read_cache(self) -> dict | None:
        if not self.cache_path or not self.cache_path.exists():
            return None
        try:
            data = json.loads(self.cache_path.read_text())
        except Exception:
            return None
        cached_at = datetime.fromisoformat(data.get("_cached_at", ""))
        if datetime.now(timezone.utc) - cached_at > self.cache_ttl:
            return None
        return data.get("payload")

    def _write_cache(self, payload: dict) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps({
            "_cached_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }, default=str))

    # ---- public API ----
    def fetch_gold_articles(self, limit: int = 50) -> list[NewsArticle]:
        cached = self._read_cache()
        if cached is not None:
            payload = cached
        else:
            payload = self._fetch_raw(tickers=GOLD_TICKERS, topics=GOLD_TOPICS, limit=limit)
            # Detect rate limit / empty response
            if "feed" not in payload:
                return []
            self._write_cache(payload)

        feed = payload.get("feed", []) or []
        articles: list[NewsArticle] = []
        for item in feed:
            try:
                t_str = item.get("time_published", "")
                # AV format: YYYYMMDDTHHMMSS
                ts = datetime.strptime(t_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)

            # Pull the highest gold-related ticker relevance
            relevance = 0.0
            score = float(item.get("overall_sentiment_score", 0.0) or 0.0)
            for ts_score in item.get("ticker_sentiment", []) or []:
                if ts_score.get("ticker", "") in GOLD_TICKERS:
                    rel = float(ts_score.get("relevance_score", 0.0) or 0.0)
                    if rel > relevance:
                        relevance = rel
                        # Use the ticker-specific sentiment if more relevant
                        score = float(ts_score.get("ticker_sentiment_score", score) or score)

            articles.append(NewsArticle(
                title=item.get("title", ""),
                url=item.get("url", ""),
                source=item.get("source", ""),
                time_published_utc=ts,
                overall_score=score,
                relevance=relevance,
                topics=[t.get("topic", "") for t in item.get("topics", []) or []],
            ))

        articles.sort(key=lambda a: a.time_published_utc, reverse=True)
        return articles

    def latest_gold_sentiment(self, max_age_hours: int = 6) -> SentimentSummary:
        """Aggregate sentiment over articles within the last `max_age_hours`."""
        articles = self.fetch_gold_articles()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        fresh = [a for a in articles if a.time_published_utc >= cutoff]
        if not fresh:
            return SentimentSummary(n_articles=0, score=0.0, bias="neutral",
                                     latest_ts_utc=None,
                                     bullish_count=0, bearish_count=0, neutral_count=0)

        weights = [max(a.relevance, 0.1) for a in fresh]
        weighted = sum(a.overall_score * w for a, w in zip(fresh, weights))
        total_w = sum(weights)
        score = weighted / total_w if total_w > 0 else 0.0

        bullish = sum(1 for a in fresh if a.overall_score > BULL_CUTOFF)
        bearish = sum(1 for a in fresh if a.overall_score < BEAR_CUTOFF)
        neutral = len(fresh) - bullish - bearish

        bias: Literal["bullish", "bearish", "neutral"]
        if score > BULL_CUTOFF:
            bias = "bullish"
        elif score < BEAR_CUTOFF:
            bias = "bearish"
        else:
            bias = "neutral"

        return SentimentSummary(
            n_articles=len(fresh),
            score=score,
            bias=bias,
            latest_ts_utc=fresh[0].time_published_utc,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
        )
