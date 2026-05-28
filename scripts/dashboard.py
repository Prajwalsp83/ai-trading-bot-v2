"""
Phase 2 — Streamlit dashboard for the AI trading bot.

Reads from Postgres (Supabase). Shows bot status, equity curve, recent
trades, signal funnel, rejection breakdown, regime distribution.

Run locally on Mac:
    pip install streamlit plotly psycopg2-binary python-dotenv
    streamlit run scripts/dashboard.py

Open browser: http://localhost:8501

Reads DB credentials from .env in the same way the bots do (DB_HOST,
DB_PORT, DB_USER, DB_PASSWORD, DB_NAME).

Auto-refreshes every 60 seconds. Each panel is cached separately to keep
the page fast even with thousands of trades.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv


# ============================== CONFIG ==============================
HERE = Path(__file__).resolve().parent.parent
load_dotenv(HERE / ".env")
load_dotenv(HERE / "scripts" / ".env")     # if .env lives next to scripts

REFRESH_SECONDS = 60
CACHE_TTL = 30   # seconds — DB queries cached this long

st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================== DB ==================================
@st.cache_resource
def get_conn():
    import psycopg2
    host = os.getenv("DB_HOST")
    if host:
        return psycopg2.connect(
            host=host,
            port=int(os.getenv("DB_PORT", "5432")),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            dbname=os.getenv("DB_NAME", "postgres"),
            connect_timeout=10,
        )
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        st.error("No DB credentials in .env (DB_HOST or DATABASE_URL)")
        st.stop()
    return psycopg2.connect(dsn, connect_timeout=10)


def fetch_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        # Reconnect on stale connection
        try:
            conn.close()
        except Exception:
            pass
        get_conn.clear()
        conn = get_conn()
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data(ttl=CACHE_TTL)
def q_latest_equity() -> pd.DataFrame:
    return fetch_df("""
        SELECT account, equity, balance, peak_equity, dd_pct, open_positions, ts
        FROM equity_snapshots
        ORDER BY ts DESC LIMIT 1
    """)


@st.cache_data(ttl=CACHE_TTL)
def q_equity_curve(days: int = 30) -> pd.DataFrame:
    return fetch_df("""
        SELECT ts, equity, balance, dd_pct
        FROM equity_snapshots
        WHERE ts >= NOW() - %s::interval
        ORDER BY ts
    """, (f"{days} days",))


@st.cache_data(ttl=CACHE_TTL)
def q_bot_starts() -> pd.DataFrame:
    return fetch_df("""
        SELECT bot_name, ts, payload->>'equity' AS equity
        FROM events
        WHERE kind = 'bot_start' AND bot_name IS NOT NULL
        ORDER BY ts DESC
        LIMIT 20
    """)


@st.cache_data(ttl=CACHE_TTL)
def q_recent_trades(n: int = 50) -> pd.DataFrame:
    return fetch_df("""
        SELECT id, bot_name, magic, trade_id, side, symbol,
               open_time, close_time, entry, exit, lots, sl, tp,
               pnl_usd, r_realised, duration_minutes, exit_reason,
               regime, news_bias, poi_score
        FROM trades
        ORDER BY close_time DESC
        LIMIT %s
    """, (n,))


@st.cache_data(ttl=CACHE_TTL)
def q_daily_pnl() -> pd.DataFrame:
    return fetch_df("""
        SELECT bot_name, day, trades, pnl_usd, r_total, wins, losses
        FROM v_daily_pnl
        WHERE day >= CURRENT_DATE - INTERVAL '30 days'
        ORDER BY day, bot_name
    """)


@st.cache_data(ttl=CACHE_TTL)
def q_signal_funnel(hours: int = 24) -> pd.DataFrame:
    return fetch_df("""
        SELECT bot_name, severity, COUNT(*) AS n
        FROM signals
        WHERE ts >= NOW() - %s::interval
        GROUP BY bot_name, severity
        ORDER BY bot_name, severity
    """, (f"{hours} hours",))


@st.cache_data(ttl=CACHE_TTL)
def q_rejection_counts() -> pd.DataFrame:
    return fetch_df("""
        SELECT bot_name, rejection_reason, n, first_seen, last_seen
        FROM v_rejection_counts
        LIMIT 20
    """)


@st.cache_data(ttl=CACHE_TTL)
def q_regime_timeline(hours: int = 72) -> pd.DataFrame:
    return fetch_df("""
        SELECT bot_name, regime, COUNT(*) AS n
        FROM signals
        WHERE ts >= NOW() - %s::interval AND regime IS NOT NULL
        GROUP BY bot_name, regime
        ORDER BY bot_name, regime
    """, (f"{hours} hours",))


@st.cache_data(ttl=CACHE_TTL)
def q_trade_summary() -> pd.DataFrame:
    return fetch_df("""
        SELECT bot_name,
               COUNT(*) AS trades,
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN pnl_usd <= 0 THEN 1 ELSE 0 END) AS losses,
               ROUND(AVG(r_realised)::numeric, 3) AS avg_r,
               ROUND(SUM(pnl_usd)::numeric, 2) AS total_pnl,
               ROUND(AVG(duration_minutes)::numeric, 0) AS avg_duration_min
        FROM trades
        GROUP BY bot_name
        ORDER BY bot_name
    """)


# === Dashboard v2 queries ===

@st.cache_data(ttl=CACHE_TTL)
def q_hourly_heatmap(hours: int = 168) -> pd.DataFrame:
    """Signal counts per hour-of-day per severity for last N hours."""
    return fetch_df("""
        SELECT EXTRACT(hour FROM ts AT TIME ZONE 'UTC')::int AS hour_utc,
               severity,
               COUNT(*) AS n
        FROM signals
        WHERE ts >= NOW() - %s::interval
        GROUP BY hour_utc, severity
        ORDER BY hour_utc, severity
    """, (f"{hours} hours",))


@st.cache_data(ttl=CACHE_TTL)
def q_r_distribution() -> pd.DataFrame:
    """R-multiple per closed trade for histogram."""
    return fetch_df("""
        SELECT bot_name, r_realised, exit_reason, close_time
        FROM trades
        ORDER BY close_time DESC
        LIMIT 500
    """)


@st.cache_data(ttl=CACHE_TTL)
def q_session_activity(hours: int = 168) -> pd.DataFrame:
    """Signal + entry activity by session over the last N hours.
    Sessions: London (12:30-16:30 IST = 07:00-11:00 UTC),
              NY_overlap (18:00-21:00 IST = 12:30-15:30 UTC),
              NY_afternoon (21:00-23:30 IST = 15:30-18:00 UTC)."""
    return fetch_df("""
        SELECT bot_name,
               CASE
                   WHEN EXTRACT(hour FROM ts AT TIME ZONE 'UTC') BETWEEN 7 AND 10 THEN 'London'
                   WHEN EXTRACT(hour FROM ts AT TIME ZONE 'UTC') BETWEEN 12 AND 14 THEN 'NY_overlap'
                   WHEN EXTRACT(hour FROM ts AT TIME ZONE 'UTC') BETWEEN 15 AND 17 THEN 'NY_afternoon'
                   ELSE 'outside'
               END AS session,
               severity,
               COUNT(*) AS n
        FROM signals
        WHERE ts >= NOW() - %s::interval
        GROUP BY bot_name, session, severity
        ORDER BY session, severity
    """, (f"{hours} hours",))


@st.cache_data(ttl=CACHE_TTL)
def q_blocked_entries(n: int = 20) -> pd.DataFrame:
    """Last N signals that were ready-to-fire but didn't make it to a trade
    (severity ended up as SKIPPED or there's an explicit rejection_reason)."""
    return fetch_df("""
        SELECT bot_name, ts, severity, side, price, atr, reason, rejection_reason, regime
        FROM signals
        WHERE (severity = 'SKIPPED' OR rejection_reason IS NOT NULL)
              OR (severity IN ('BUY_READY', 'SELL_READY')
                  AND ts > (SELECT COALESCE(MAX(open_time), '1970-01-01') FROM trades))
        ORDER BY ts DESC
        LIMIT %s
    """, (n,))


@st.cache_data(ttl=CACHE_TTL)
def q_open_position() -> pd.DataFrame:
    """If there are open positions (snapshots show open_positions > 0), show details."""
    return fetch_df("""
        SELECT account, equity, balance, peak_equity, dd_pct, open_positions, ts
        FROM equity_snapshots
        WHERE open_positions > 0
        ORDER BY ts DESC
        LIMIT 1
    """)


@st.cache_data(ttl=CACHE_TTL)
def q_recent_loss_streak() -> pd.DataFrame:
    """For each bot: current loss streak (consecutive losses ending most recently)."""
    return fetch_df("""
        WITH recent AS (
            SELECT bot_name, close_time, pnl_usd,
                   ROW_NUMBER() OVER (PARTITION BY bot_name ORDER BY close_time DESC) AS rn
            FROM trades
        )
        SELECT bot_name,
               COUNT(*) FILTER (
                   WHERE pnl_usd <= 0
                   AND rn <= (SELECT MIN(rn) FROM recent r2 WHERE r2.bot_name = recent.bot_name AND r2.pnl_usd > 0)
               ) AS streak_losses,
               MAX(close_time) AS last_trade_at
        FROM recent
        GROUP BY bot_name
    """)


@st.cache_data(ttl=CACHE_TTL)
def q_cooldown_events(hours: int = 168) -> pd.DataFrame:
    """Recent gate-block events from logs (watchdog, cooldown, regime halt)."""
    return fetch_df("""
        SELECT bot_name, ts, kind, payload
        FROM events
        WHERE kind IN ('watchdog_reconnect', 'watchdog_exit', 'gate_block')
              AND ts >= NOW() - %s::interval
        ORDER BY ts DESC
        LIMIT 20
    """, (f"{hours} hours",))


# ============================== UI ==================================
def metric_card(col, label: str, value, delta=None, help=None):
    with col:
        st.metric(label, value, delta=delta, help=help)


# --- Auto-refresh ---
st_autorefresh_count = st.empty()
st.markdown(
    f"<script>setTimeout(function(){{window.location.reload();}}, {REFRESH_SECONDS*1000});</script>",
    unsafe_allow_html=True,
)

# --- Header ---
st.title("📈 AI Gold Trading Bot")
last_eq = q_latest_equity()
if not last_eq.empty:
    e = last_eq.iloc[0]
    age = datetime.now(timezone.utc) - pd.to_datetime(e["ts"]).to_pydatetime()
    age_str = f"{int(age.total_seconds())}s ago" if age.total_seconds() < 120 else f"{int(age.total_seconds()/60)}m ago"
    st.caption(f"Account `{e['account']}` · last update {age_str} · refresh every {REFRESH_SECONDS}s")
else:
    st.warning("No equity snapshots yet — bots may not have written to DB.")

# --- Open position card (only shown if a position is open) ---
open_pos = q_open_position()
if not open_pos.empty:
    e = open_pos.iloc[0]
    age = datetime.now(timezone.utc) - pd.to_datetime(e["ts"]).to_pydatetime()
    st.warning(
        f"🔴 **OPEN POSITION** · Account `{e['account']}` · "
        f"Equity ${e['equity']:,.2f} · Drawdown {(float(e['dd_pct'] or 0)*100):.2f}% · "
        f"snapshot {int(age.total_seconds()/60)}m ago"
    )

# --- KPIs ---
st.subheader("Live state")
c1, c2, c3, c4, c5 = st.columns(5)
if not last_eq.empty:
    e = last_eq.iloc[0]
    metric_card(c1, "Equity", f"${e['equity']:,.2f}")
    metric_card(c2, "Balance", f"${e['balance']:,.2f}")
    metric_card(c3, "Peak Equity",
                f"${(e['peak_equity'] or e['equity']):,.2f}")
    dd = float(e["dd_pct"] or 0) * 100
    metric_card(c4, "Drawdown",
                f"{dd:.2f}%",
                delta=f"{-dd:.2f}%" if dd > 0 else "0.00%",
                help="From peak equity")
    metric_card(c5, "Open positions", int(e["open_positions"] or 0))
else:
    for c, lbl in zip([c1, c2, c3, c4, c5],
                      ["Equity", "Balance", "Peak", "Drawdown", "Open"]):
        metric_card(c, lbl, "—")

# --- Risk meter row: cooldown / streak / recent watchdog events ---
streak_df = q_recent_loss_streak()
cooldown_df = q_cooldown_events(168)
if not streak_df.empty or not cooldown_df.empty:
    st.subheader("Risk state")
    rcols = st.columns(4)
    # Streak per bot
    streak_dict = {r["bot_name"]: int(r["streak_losses"] or 0)
                   for _, r in streak_df.iterrows()}
    metric_card(rcols[0], "Breakout loss streak",
                streak_dict.get("breakout", 0),
                help="Consecutive losses ending most recently. After 2, bot pauses 4h.")
    metric_card(rcols[1], "SMC loss streak",
                streak_dict.get("smc", 0),
                help="Same as above for SMC bot.")
    # Watchdog events
    wd_count = (cooldown_df["kind"] == "watchdog_reconnect").sum() if not cooldown_df.empty else 0
    wd_exit = (cooldown_df["kind"] == "watchdog_exit").sum() if not cooldown_df.empty else 0
    metric_card(rcols[2], "Watchdog reconnects (7d)", int(wd_count),
                help="Times bot detected MT5 was dead and recovered.")
    metric_card(rcols[3], "Watchdog hard exits (7d)", int(wd_exit),
                delta=None if wd_exit == 0 else f"⚠️ +{wd_exit}",
                help="Times NSSM had to fully restart the bot. >2/week = investigate.")

# --- Per-bot summary ---
st.subheader("Per-bot summary")
summary = q_trade_summary()
if summary.empty:
    st.info("No closed trades yet. Loosened params just deployed — first entries usually within 6-24h of next London open.")
else:
    cols = st.columns(len(summary))
    for col, (_, r) in zip(cols, summary.iterrows()):
        with col:
            wr = (r["wins"] / r["trades"] * 100) if r["trades"] > 0 else 0
            st.markdown(f"**{r['bot_name']}**")
            st.metric("Trades", int(r["trades"]),
                      delta=f"{int(r['wins'])}W / {int(r['losses'])}L")
            st.metric("Win rate", f"{wr:.1f}%")
            st.metric("Avg R", f"{r['avg_r']:+.3f}",
                      delta=f"${r['total_pnl']:+,.2f}")

# --- Equity curve ---
st.subheader("Equity curve")
eq = q_equity_curve(30)
if eq.empty:
    st.info("No equity snapshots in the last 30 days.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["ts"], y=eq["equity"], mode="lines",
                              name="Equity", line=dict(color="#22c55e", width=2)))
    fig.add_trace(go.Scatter(x=eq["ts"], y=eq["balance"], mode="lines",
                              name="Balance", line=dict(color="#3b82f6", width=1, dash="dot")))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                       legend=dict(orientation="h", y=1.1),
                       xaxis_title=None, yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)

# --- Hourly signal heatmap ---
st.subheader("When does the bot find setups? (last 7 days, by hour UTC)")
heatmap = q_hourly_heatmap(168)
if heatmap.empty:
    st.info("Not enough signals to draw a heatmap yet.")
else:
    pivot = heatmap.pivot_table(index="severity", columns="hour_utc", values="n",
                                 fill_value=0, aggfunc="sum")
    # Order severities for readability
    severity_order = ["WATCHLIST", "BREAKOUT_WATCH", "SKIPPED", "BUY_READY", "SELL_READY"]
    pivot = pivot.reindex([s for s in severity_order if s in pivot.index])
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="Viridis", showscale=True,
        text=pivot.values, texttemplate="%{text}", textfont={"size": 10},
    ))
    fig.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0),
                       xaxis_title="Hour (UTC)", yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Reference: 07:00–11:00 UTC = London · 12:30–15:30 UTC = NY overlap · 15:30–18:00 UTC = NY afternoon")

# --- Daily P&L bars ---
st.subheader("Daily P&L (last 30 days)")
daily = q_daily_pnl()
if daily.empty:
    st.info("No closed trades to roll up by day yet.")
else:
    daily["day"] = pd.to_datetime(daily["day"])
    fig = px.bar(daily, x="day", y="pnl_usd", color="bot_name",
                  barmode="group", height=300,
                  color_discrete_map={"breakout": "#3b82f6", "smc": "#a855f7"})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None,
                       yaxis_title="P&L USD",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

# --- Recent trades ---
st.subheader("Recent trades")
trades = q_recent_trades(50)
if trades.empty:
    st.info("No trades closed yet.")
else:
    display = trades[[
        "bot_name", "side", "close_time", "entry", "exit",
        "pnl_usd", "r_realised", "duration_minutes", "exit_reason",
        "regime", "news_bias",
    ]].copy()
    display.columns = ["Bot", "Side", "Closed", "Entry", "Exit",
                        "P&L $", "R", "Duration m", "Reason", "Regime", "News"]
    st.dataframe(display, use_container_width=True, hide_index=True, height=350)

# --- Two-column: signal funnel + rejection breakdown ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Signal funnel (last 24h)")
    funnel = q_signal_funnel(24)
    if funnel.empty:
        st.info("No signals evaluated in last 24h.")
    else:
        fig = px.bar(funnel, x="severity", y="n", color="bot_name",
                      barmode="group", height=280,
                      color_discrete_map={"breakout": "#3b82f6", "smc": "#a855f7"})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title=None, yaxis_title=None,
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Top rejections (all-time)")
    rej = q_rejection_counts()
    if rej.empty:
        st.info("No rejections logged yet.")
    else:
        fig = px.bar(rej.head(10), x="n", y="rejection_reason", color="bot_name",
                      orientation="h", height=280,
                      color_discrete_map={"breakout": "#3b82f6", "smc": "#a855f7"})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Count", yaxis_title=None,
                          legend=dict(orientation="h", y=1.15))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

# --- Regime distribution ---
st.subheader("Regime distribution (last 72h)")
regime = q_regime_timeline(72)
if regime.empty:
    st.info("No regime data in last 72h.")
else:
    fig = px.bar(regime, x="regime", y="n", color="bot_name",
                  barmode="group", height=260,
                  color_discrete_map={"breakout": "#3b82f6", "smc": "#a855f7"})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                       xaxis_title=None, yaxis_title="Bars classified",
                       legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)

# --- R-multiple histogram + Session activity (side-by-side) ---
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("R-multiple distribution")
    r_dist = q_r_distribution()
    if r_dist.empty:
        st.info("No closed trades yet — R-multiples will populate as bots trade.")
    else:
        fig = px.histogram(r_dist, x="r_realised", color="bot_name", nbins=30, height=280,
                            color_discrete_map={"breakout": "#3b82f6", "smc": "#a855f7"})
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.add_vline(x=1.67, line_dash="dot", line_color="#22c55e", opacity=0.5,
                       annotation_text="TP=+1.67R", annotation_position="top")
        fig.add_vline(x=-1.0, line_dash="dot", line_color="#ef4444", opacity=0.5,
                       annotation_text="SL=-1R", annotation_position="top")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="R realised", yaxis_title="Trades",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Activity by session (last 7d)")
    sess = q_session_activity(168)
    if sess.empty:
        st.info("No signals yet.")
    else:
        sess_agg = sess.groupby(["session", "bot_name"], as_index=False)["n"].sum()
        # Order sessions intuitively
        order = ["London", "NY_overlap", "NY_afternoon", "outside"]
        sess_agg["session"] = pd.Categorical(sess_agg["session"], categories=order, ordered=True)
        sess_agg = sess_agg.sort_values("session")
        fig = px.bar(sess_agg, x="session", y="n", color="bot_name",
                      barmode="group", height=280,
                      color_discrete_map={"breakout": "#3b82f6", "smc": "#a855f7"})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title=None, yaxis_title="Signals",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

# --- Recent blocked entries — what the bot WANTED to do but couldn't ---
st.subheader("Recent blocked / skipped entries")
blocked = q_blocked_entries(20)
if blocked.empty:
    st.info("No blocked entries — bots haven't tried to fire yet.")
else:
    display = blocked[["ts", "bot_name", "severity", "side", "price",
                        "reason", "rejection_reason", "regime"]].copy()
    display["ts"] = pd.to_datetime(display["ts"]).dt.strftime("%Y-%m-%d %H:%M")
    display.columns = ["Time UTC", "Bot", "Severity", "Side", "Price",
                        "Reason", "Rejection", "Regime"]
    st.dataframe(display, use_container_width=True, hide_index=True, height=350)
    st.caption("These are the moments the bot saw a setup but didn't enter. Look for patterns — "
                "if rejection is consistently `atr_pct_too_low`, vol is just too dead. "
                "If `regime_router`, you're in chop and breakout is intentionally halted.")

# --- Recent bot starts (collapsed) ---
with st.expander("Bot lifecycle events"):
    starts = q_bot_starts()
    if starts.empty:
        st.info("No bot_start events yet.")
    else:
        starts["ts"] = pd.to_datetime(starts["ts"])
        st.dataframe(starts, use_container_width=True, hide_index=True)

st.caption("Auto-refreshes every 60 seconds. Cache TTL: 30 seconds.")
