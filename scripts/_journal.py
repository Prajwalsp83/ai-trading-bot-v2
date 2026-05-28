"""
Postgres journal writer for the live bots.

Self-contained — drops on VPS alongside the bot files via Notepad clipboard.

Reads `DATABASE_URL` from environment (or .env). If unset or DB unreachable,
all writes become no-ops with a single warning per failure. Bots never crash
because the DB is down.

Functions:
    record_trade(bot_name, magic, record)        — write a closed trade
    record_signal(bot_name, sig)                 — write a signal evaluation
    snapshot_equity(account, equity, balance,    — write equity snapshot
                    peak_equity, open_positions)
    log_event(bot_name, kind, payload)           — write a bot/system event

All writes are synchronous but fast (single INSERT). Connection is pooled
via psycopg2.pool.SimpleConnectionPool — connections recycled.

Install:
    pip install psycopg2-binary
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

try:
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    from psycopg2.extras import Json
    _PSYCOPG_OK = True
except ImportError:
    _PSYCOPG_OK = False


# ---------- Pool ----------
_POOL: SimpleConnectionPool | None = None
_POOL_LOCK = threading.Lock()
_LAST_ERROR: str | None = None
_DB_DISABLED = False     # set True after persistent failures to silence warnings


def _print(msg: str) -> None:
    """Logger that always goes to stdout (so NSSM captures it)."""
    print(f"[journal] {msg}", flush=True)


def _get_pool() -> SimpleConnectionPool | None:
    """Lazy-init the connection pool. Returns None if DB is disabled/misconfigured."""
    global _POOL, _DB_DISABLED, _LAST_ERROR
    if _DB_DISABLED:
        return None
    if _POOL is not None:
        return _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            return _POOL
        if not _PSYCOPG_OK:
            _print("psycopg2 not installed — journal disabled. Run: pip install psycopg2-binary")
            _DB_DISABLED = True
            return None

        # Two ways to configure: individual DB_* vars (preferred — avoids URL
        # encoding headaches with special chars in passwords) or DATABASE_URL.
        host = os.getenv("DB_HOST")
        if host:
            kwargs = {
                "host": host,
                "port": int(os.getenv("DB_PORT", "5432")),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", ""),
                "dbname": os.getenv("DB_NAME", "postgres"),
                "connect_timeout": 10,
            }
            try:
                _POOL = SimpleConnectionPool(minconn=1, maxconn=3, **kwargs)
                _print(f"connected to Postgres ({host})")
            except Exception as e:
                _LAST_ERROR = str(e)
                _print(f"could not connect via DB_* params ({type(e).__name__}: {str(e)[:200]}). journal disabled.")
                _DB_DISABLED = True
                return None
            return _POOL

        # Fallback to DATABASE_URL
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            _print("neither DB_HOST nor DATABASE_URL set — journal disabled.")
            _DB_DISABLED = True
            return None
        try:
            _POOL = SimpleConnectionPool(minconn=1, maxconn=3, dsn=dsn)
            _print("connected to Postgres (via DATABASE_URL)")
        except Exception as e:
            _LAST_ERROR = str(e)
            _print(f"could not connect via DATABASE_URL ({type(e).__name__}: {str(e)[:200]}). journal disabled.")
            _DB_DISABLED = True
            return None
    return _POOL


def _execute(sql: str, params: tuple) -> bool:
    """Run a single INSERT. Returns True on success, False (and logs) on failure.
    Never raises — the bot must keep trading even if the DB is misbehaving."""
    pool = _get_pool()
    if pool is None:
        return False
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
        return True
    except Exception as e:
        _print(f"insert failed ({type(e).__name__}: {str(e)[:200]})")
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn:
            try:
                pool.putconn(conn)
            except Exception:
                pass


def _ts(value) -> datetime | None:
    """Coerce ISO string or datetime to timezone-aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            v = value.replace("Z", "+00:00") if value.endswith("Z") else value
            dt = datetime.fromisoformat(v)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


# ---------- Public API ----------
def record_trade(bot_name: str, magic: int, record: dict) -> bool:
    """Write a closed-trade record. `record` is the dict already written to CSV."""
    sql = """
        INSERT INTO trades (
            bot_name, magic, trade_id, symbol, side,
            open_time, close_time, entry, exit, lots, sl, tp,
            pnl_usd, r_realised, duration_minutes,
            atr_at_entry, exit_reason, ticket,
            poi_score, rr_at_entry, regime, news_bias, news_score,
            risk_pct_used
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s
        )
        ON CONFLICT (bot_name, magic, trade_id) DO NOTHING
    """
    params = (
        bot_name, magic, int(record.get("trade_id", 0)),
        record.get("symbol", "GOLD.i#"), record.get("side"),
        _ts(record.get("open_time")), _ts(record.get("close_time")),
        float(record.get("entry", 0)), float(record.get("exit", 0)),
        float(record.get("lots", 0)),
        float(record.get("sl", 0)), float(record.get("tp", 0)),
        float(record.get("pnl_usd", 0)), float(record.get("r_realised", 0)),
        int(record.get("duration_minutes", 0)),
        _float_or_none(record.get("atr_at_entry")),
        record.get("exit_reason", "UNKNOWN"),
        _int_or_none(record.get("ticket")),
        _int_or_none(record.get("poi_score")),
        _float_or_none(record.get("rr_at_entry")),
        record.get("regime"),
        record.get("news_bias"),
        _float_or_none(record.get("news_score")),
        _float_or_none(record.get("risk_pct_used")),
    )
    return _execute(sql, params)


def record_signal(bot_name: str, sig: dict, regime: str | None = None,
                   extras: dict | None = None) -> bool:
    """Write a signal evaluation. `sig` is the dict returned from evaluate()."""
    if not sig:
        return False
    sql = """
        INSERT INTO signals (
            bot_name, ts, symbol, severity, side, price, atr,
            reason, rejection_reason, regime, extras
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        bot_name,
        sig.get("ts") or datetime.now(timezone.utc),
        sig.get("symbol", "GOLD.i#"),
        sig.get("severity", "UNKNOWN"),
        sig.get("side"),
        _float_or_none(sig.get("price")),
        _float_or_none(sig.get("atr")),
        sig.get("reason"),
        sig.get("rejection_reason"),
        regime,
        Json(extras) if extras else None,
    )
    return _execute(sql, params)


def snapshot_equity(account: str, equity: float, balance: float,
                    peak_equity: float | None = None,
                    open_positions: int | None = None) -> bool:
    """Write an equity snapshot for plotting / DD tracking."""
    dd_pct = None
    if peak_equity and peak_equity > 0:
        dd_pct = max(0.0, (peak_equity - equity) / peak_equity)
    sql = """
        INSERT INTO equity_snapshots (
            ts, account, equity, balance, peak_equity, dd_pct, open_positions
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        datetime.now(timezone.utc),
        str(account), float(equity), float(balance),
        _float_or_none(peak_equity), _float_or_none(dd_pct),
        _int_or_none(open_positions),
    )
    return _execute(sql, params)


def log_event(bot_name: str | None, kind: str, payload: dict | None = None) -> bool:
    """Write a bot lifecycle / ops event."""
    sql = """
        INSERT INTO events (bot_name, ts, kind, payload)
        VALUES (%s, %s, %s, %s)
    """
    params = (
        bot_name,
        datetime.now(timezone.utc),
        kind,
        Json(payload) if payload else None,
    )
    return _execute(sql, params)


# ---------- helpers ----------
def _float_or_none(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _int_or_none(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None


# ---------- Smoke test ----------
if __name__ == "__main__":
    """Run directly to verify the connection:
        python _journal.py
    """
    from dotenv import load_dotenv
    load_dotenv()
    ok = log_event(None, "smoke_test", {"msg": "journal smoke test", "ts": datetime.now(timezone.utc).isoformat()})
    if ok:
        print("OK — journal write succeeded. Check `SELECT * FROM events WHERE kind='smoke_test';`")
    else:
        print("FAIL — see error above")
