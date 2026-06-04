"""
Ad-hoc reader for the `signals` table — inspect recent bot signal evaluations,
including the ML meta-labeler's shadow-mode FIRE/VETO decisions.

Connects with the same DB_* env vars as _journal.py (loaded from .env), so it
works anywhere the bot runs. Read-only: runs a single SELECT, prints a table.

Run on the VPS (flat layout):
    python scripts\\query_signals.py
    python scripts\\query_signals.py --bot smc --hours 48 --limit 80
    python scripts\\query_signals.py --severity ENTRY      # only fired/scored entries
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # env vars may already be set in the environment


def _connect():
    host = os.getenv("DB_HOST")
    if not host:
        print("DB_HOST not set (check .env). Cannot query.")
        sys.exit(1)
    return psycopg2.connect(
        host=host,
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        dbname=os.getenv("DB_NAME", "postgres"),
        connect_timeout=10,
    )


def _fmt(v, width):
    s = "" if v is None else str(v)
    if len(s) > width:
        s = s[: width - 1] + "."
    return s.ljust(width)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bot", default="smc", help="bot_name filter (default smc)")
    p.add_argument("--hours", type=int, default=24, help="look-back window")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--severity", default=None,
                   help="optional severity filter (e.g. ENTRY, WATCHLIST)")
    args = p.parse_args()

    sql = """
        SELECT ts, severity, side, price, reason, rejection_reason, regime,
               extras->>'probability' AS ml_prob,
               extras->>'threshold'   AS ml_thr,
               extras->>'would_trade' AS ml_fire,
               extras->>'shadow_mode' AS shadow
        FROM signals
        WHERE bot_name = %s
          AND ts > now() - (%s || ' hours')::interval
          AND (%s IS NULL OR severity = %s)
        ORDER BY ts DESC
        LIMIT %s
    """
    params = (args.bot, str(args.hours), args.severity, args.severity, args.limit)

    try:
        conn = _connect()
    except Exception as e:
        print(f"connect failed: {type(e).__name__}: {e}")
        return 1

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print(f"No signals for bot={args.bot} in last {args.hours}h"
              + (f" severity={args.severity}" if args.severity else ""))
        return 0

    hdr = (_fmt("ts (UTC)", 19) + _fmt("sev", 10) + _fmt("side", 5)
           + _fmt("price", 10) + _fmt("prob", 6) + _fmt("thr", 6)
           + _fmt("fire", 6) + _fmt("shdw", 6) + _fmt("regime", 11)
           + "reason / rejection")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ts = str(r["ts"])[:19]
        prob = r["ml_prob"]
        prob = f"{float(prob):.3f}" if prob not in (None, "") else ""
        thr = r["ml_thr"]
        thr = f"{float(thr):.2f}" if thr not in (None, "") else ""
        msg = r["rejection_reason"] or r["reason"] or ""
        print(_fmt(ts, 19) + _fmt(r["severity"], 10) + _fmt(r["side"], 5)
              + _fmt(r["price"], 10) + _fmt(prob, 6) + _fmt(thr, 6)
              + _fmt(r["ml_fire"], 6) + _fmt(r["shadow"], 6)
              + _fmt(r["regime"], 11) + msg)

    fired = sum(1 for r in rows if r["ml_fire"] == "true")
    scored = sum(1 for r in rows if r["ml_prob"] not in (None, ""))
    print("-" * len(hdr))
    print(f"{len(rows)} rows | {scored} ML-scored | {fired} would-FIRE "
          f"| {scored - fired} would-VETO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
