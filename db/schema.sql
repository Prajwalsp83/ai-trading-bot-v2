-- =============================================================================
-- AI Trading Bot — Postgres schema (v1)
-- =============================================================================
-- 4 tables, shared by both bots (breakout + smc).
-- Each bot writes with its bot_name + magic so we can filter or aggregate.
--
-- Run once on a fresh database:
--   psql "$DATABASE_URL" -f db/schema.sql
--
-- Idempotent — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- TRADES: closed positions, one row per round-trip.
-- Mirrors the columns of mt5_trades.csv / mt5_smc_trades.csv plus DB metadata.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL PRIMARY KEY,
    bot_name        TEXT        NOT NULL,           -- 'breakout' | 'smc'
    magic           INTEGER     NOT NULL,           -- MT5 magic number
    trade_id        INTEGER     NOT NULL,           -- bot-local incrementing id
    symbol          TEXT        NOT NULL,
    side            TEXT        NOT NULL CHECK (side IN ('BUY', 'SELL')),
    open_time       TIMESTAMPTZ NOT NULL,
    close_time      TIMESTAMPTZ NOT NULL,
    entry           DOUBLE PRECISION NOT NULL,
    exit            DOUBLE PRECISION NOT NULL,
    lots            DOUBLE PRECISION NOT NULL,
    sl              DOUBLE PRECISION NOT NULL,
    tp              DOUBLE PRECISION NOT NULL,
    pnl_usd         DOUBLE PRECISION NOT NULL,
    r_realised      DOUBLE PRECISION NOT NULL,
    duration_minutes INTEGER    NOT NULL,
    atr_at_entry    DOUBLE PRECISION,
    exit_reason     TEXT        NOT NULL,           -- 'TP' | 'SL' | 'OTHER'
    ticket          BIGINT,                         -- MT5 position ticket

    -- SMC-only fields (nullable for breakout trades)
    poi_score       INTEGER,
    rr_at_entry     DOUBLE PRECISION,
    regime          TEXT,                           -- regime at entry
    news_bias       TEXT,                           -- 'bullish' | 'bearish' | 'neutral' | 'none'
    news_score      DOUBLE PRECISION,

    -- Composite-risk bookkeeping
    risk_pct_used   DOUBLE PRECISION,               -- final risk % after dd/kelly/regime

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (bot_name, magic, trade_id)              -- de-dupes if migration runs twice
);

CREATE INDEX IF NOT EXISTS idx_trades_bot_close   ON trades (bot_name, close_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_magic       ON trades (magic);
CREATE INDEX IF NOT EXISTS idx_trades_close_time  ON trades (close_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades (exit_reason);

COMMENT ON TABLE trades IS 'Closed trades from all bots. One row per round-trip.';
COMMENT ON COLUMN trades.r_realised IS 'P&L in R-multiples (P&L / initial risk).';
COMMENT ON COLUMN trades.risk_pct_used IS 'Effective risk after DD tier x Kelly x regime weight.';


-- ---------------------------------------------------------------------------
-- SIGNALS: every signal evaluation (firing, watch, skip).
-- Lets us analyse what the bot "saw" even when it didn't trade.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
    id              BIGSERIAL PRIMARY KEY,
    bot_name        TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    severity        TEXT        NOT NULL,           -- BUY_READY | SELL_READY | BREAKOUT_WATCH | WATCHLIST | SKIPPED
    side            TEXT,                           -- BUY | SELL | NULL
    price           DOUBLE PRECISION,
    atr             DOUBLE PRECISION,
    reason          TEXT,
    rejection_reason TEXT,                          -- for SKIPPED
    regime          TEXT,                           -- regime when evaluated
    extras          JSONB,                          -- bot-specific (e.g. SMC: poi_top/bottom, rr)

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_bot_ts      ON signals (bot_name, ts DESC);
CREATE INDEX IF NOT EXISTS idx_signals_severity    ON signals (severity);
CREATE INDEX IF NOT EXISTS idx_signals_rejection   ON signals (rejection_reason) WHERE rejection_reason IS NOT NULL;

COMMENT ON TABLE signals IS 'Every signal evaluation. High volume — informs gate tuning.';


-- ---------------------------------------------------------------------------
-- EQUITY_SNAPSHOTS: periodic equity capture for curve plotting + DD tracking.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL,
    account         TEXT        NOT NULL,           -- MT5 account number as string
    equity          DOUBLE PRECISION NOT NULL,
    balance         DOUBLE PRECISION NOT NULL,
    peak_equity     DOUBLE PRECISION,
    dd_pct          DOUBLE PRECISION,
    open_positions  INTEGER,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots (ts DESC);

COMMENT ON TABLE equity_snapshots IS 'Equity curve data. Bot writes every N minutes.';


-- ---------------------------------------------------------------------------
-- EVENTS: ops log — bot lifecycle, watchdog events, errors, daily summaries.
-- JSONB payload lets us store arbitrary structured data without schema changes.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    id              BIGSERIAL PRIMARY KEY,
    bot_name        TEXT,                           -- NULL for system-level events
    ts              TIMESTAMPTZ NOT NULL,
    kind            TEXT        NOT NULL,           -- bot_start, bot_stop, watchdog_reconnect, watchdog_exit, error, daily_summary, gate_block, ...
    payload         JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_bot_ts ON events (bot_name, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_kind   ON events (kind);
CREATE INDEX IF NOT EXISTS idx_events_payload ON events USING GIN (payload);

COMMENT ON TABLE events IS 'Ops log. Filterable by kind. JSONB allows free-form details.';


-- ---------------------------------------------------------------------------
-- Useful views for dashboards (Phase 2)
-- ---------------------------------------------------------------------------

-- Per-bot daily P&L
CREATE OR REPLACE VIEW v_daily_pnl AS
SELECT
    bot_name,
    DATE(close_time AT TIME ZONE 'UTC') AS day,
    COUNT(*) AS trades,
    SUM(pnl_usd) AS pnl_usd,
    SUM(r_realised) AS r_total,
    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN pnl_usd <= 0 THEN 1 ELSE 0 END) AS losses,
    AVG(pnl_usd) AS avg_pnl_usd,
    AVG(r_realised) AS avg_r
FROM trades
GROUP BY bot_name, DATE(close_time AT TIME ZONE 'UTC');

COMMENT ON VIEW v_daily_pnl IS 'Per-bot daily P&L roll-up. Use for equity curves.';

-- Most common signal rejections per bot
CREATE OR REPLACE VIEW v_rejection_counts AS
SELECT
    bot_name,
    rejection_reason,
    COUNT(*) AS n,
    MIN(ts) AS first_seen,
    MAX(ts) AS last_seen
FROM signals
WHERE rejection_reason IS NOT NULL
GROUP BY bot_name, rejection_reason
ORDER BY n DESC;

COMMENT ON VIEW v_rejection_counts IS 'Why is each bot rejecting trades? Drives strategy tuning.';
