# AI Quant Trading Bot — v2 Architecture

**Status:** Design proposal + skeleton (no functional code yet)
**Stage:** Plan-and-review. Nothing in `v2/` runs until you approve the shape.
**Live system:** `bot.py` is untouched and keeps running. v2 is built alongside it; we cut over only when each layer is proven.

---

## 1. Goals

1. Full automation: live data → signal → risk-checked order → broker → tracked position → exit → journal.
2. A real UI (Streamlit) showing live state, equity, journal, risk, and a kill switch.
3. Audible / Telegram / desktop alarms with severity levels.
4. SL / TP / trailing managed by the system, not by the broker only.
5. Risk management as a first-class blocker — the strategy proposes, risk disposes.
6. Backtest engine that runs the **same** strategy code as live (no more drift).
7. Staged rollout: paper → semi-auto (you confirm in Telegram) → full-auto.

## 2. Non-goals (for v2 first cut)

- No real money trading until paper + semi-auto pass.
- No multi-broker abstraction yet (MT5 first; structure leaves room).
- No deep-learning models in v2.0 — interface is reserved, but first release ships rule-based + classical ML feature scoring.
- No cloud deploy yet — runs locally; VPS/Docker is a follow-up milestone.

---

## 3. The core problem we are fixing

`bot.py` today has two different decision functions sharing the same indicators:

| Path | What decides a trade |
|---|---|
| Live signal | EMA50/200 alignment + breakout of prior candle + ATR filter |
| "Backtest" loop | RandomForest `predict()` over (ema50, ema200, atr) |

So your live bot's win-rate is **not** what the equity curve shows. v2 fixes this by making the strategy a single object that both the live runner and the backtester call.

---

## 4. Component map

```
                       ┌────────────────────────┐
                       │    Streamlit UI        │  ← you see + control
                       │ live | backtest | risk │
                       └───────────┬────────────┘
                                   │ reads state, sends commands
                                   ▼
┌──────────┐   ┌──────────┐   ┌───────────┐   ┌─────────┐   ┌──────────┐
│  Data    │──▶│Indicators│──▶│ Strategy  │──▶│  Risk   │──▶│Execution │──▶ broker
│ provider │   │  (pure)  │   │ (signal)  │   │ (gate)  │   │ (orders) │
└──────────┘   └──────────┘   └─────┬─────┘   └────┬────┘   └────┬─────┘
                                    │              │             │
                                    └──────────────┴─────────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │  Event bus      │
                                          └────┬───────┬────┘
                                               ▼       ▼
                                         ┌────────┐ ┌────────┐
                                         │ Alerts │ │Journal │
                                         └────────┘ └────────┘
```

**Key idea:** the strategy emits structured `Signal` objects. Risk decides if the signal becomes an `Order`. Execution turns orders into fills. Every step publishes events; alerts and journal are pure subscribers — they cannot block the trade pipeline.

---

## 5. Module-by-module spec

### 5.1 `core/`
- `config.py` — loads `config.yaml`, validates, exposes typed config object.
- `events.py` — dataclasses: `Signal`, `Order`, `Fill`, `Position`, `Alert`. Plus a tiny in-process pub/sub.
- `state.py` — single source of truth: balance, open positions, daily P&L, kill flag. Thread-safe.
- `kill_switch.py` — file-based + UI button. When tripped: cancel pending, do not open new, leave open positions to managed exits (or flatten — configurable).

### 5.2 `data/`
- `base.MarketDataProvider` — interface: `get_ohlcv(symbol, timeframe, lookback)`, `subscribe_live(symbol, callback)`.
- `yfinance_provider.py` — current source. Polls every N seconds, caches last bar.
- `upstox_provider.py` — live source. Websocket market feed for ticks, REST historical candles. Handles MCX front-month contract auto-roll.
- `oanda_provider.py`, `mt5_provider.py` — deferred (India build).

### 5.3 `indicators/`
Pure functions, no state. Each takes a DataFrame, returns a Series.
- `trend.py` — EMA, EMA slope, MA stack alignment.
- `volatility.py` — ATR, ATR percentile, expansion flag.
- `structure.py` — swing highs/lows, distance-to-prior-high (used for early-warning).

### 5.4 `strategy/`
- `base.Strategy` — interface:
  - `evaluate(market_state) -> Signal | None`
  - `name`, `params`, `required_indicators`
  - Used identically by live runner and backtest engine.
- `breakout_trend.py` — formalizes the current rules (15m + 1H EMA stack + breakout + ATR filter). Adds early-warning emission: when 3 of 4 conditions are true, emit `Signal(severity=BREAKOUT_WATCH)` instead of nothing.
- `regime.py` — placeholder for trend / chop / volatility-regime classifier (Priority 2 / 3).

### 5.5 `risk/`
The gate. A `Signal` only becomes an `Order` if every check passes.
- `sizing.py` — position size = `(equity * risk_per_trade) / (stop_distance_in_price)`. Stop distance comes from ATR.
- `limits.py` — daily loss cap, max drawdown cap, max concurrent positions, cooldown after N losses, no-trade windows (news, session open).
- `stops.py` — initial SL = entry ± k·ATR; trailing rule (move to BE after 1R, trail by m·ATR after 2R); time-based exit (close if no movement after N bars).

### 5.6 `execution/`
- `base.Broker` — interface: `place(order)`, `modify(order_id, sl=, tp=)`, `cancel(order_id)`, `positions()`, `account()`.
- `paper.py` — simulated fills with configurable spread + slippage; backtest and forward-paper share this.
- `upstox.py` — live broker. SEBI/Upstox doesn't support OCO on MCX, so `OrderManager` manages SL/TP as paired follow-up orders. Enforces `max_account_equity_inr` cap before every order. Daily OAuth re-login.
- `oanda.py`, `mt5.py` — deferred placeholders.
- `order_manager.py` — owns the OCO logic (SL + TP as bracket), handles partial fills, retries, and emergency flatten.

### 5.7 `alerts/`
Severity ladder (matches what you specified):

| Level | When | Channels |
|---|---|---|
| INFO | Heartbeats, status | log only |
| WATCHLIST | Trend forming on higher TF | Telegram |
| BREAKOUT_WATCH | 3 of 4 entry conditions met | Telegram |
| BUY_READY / SELL_READY | All conditions met, awaiting bar close | Telegram |
| ENTRY_CONFIRMED | Order filled | Telegram + sound |
| EXIT_ALERT | SL/TP/trail hit | Telegram + sound |
| RISK_ALERT | Daily loss approaching, kill switch, broker error | Telegram + sound + desktop |

- `router.py` — receives `Alert`, fans out to channels based on level.
- `telegram.py` — formatted messages (markdown), de-duplicated per (symbol, level, bar) like today.
- `sound.py` — local `.wav` playback for ENTRY/EXIT/RISK only.
- `desktop.py` — OS notification for RISK only.

### 5.8 `logging/`
- `trade_journal.py` — expanded CSV per your list:
  `trade_id, open_time, close_time, symbol, side, entry, exit, qty, sl, tp, rr_planned, rr_realised, pnl, pnl_pct, max_adverse_excursion, max_favorable_excursion, duration_bars, atr_at_entry, regime, exit_reason`
- `event_log.py` — append-only JSONL of every event for replay/debug.

### 5.9 `backtest/`
- `engine.py` — drives the same `Strategy` and `Risk` objects bar-by-bar over historical data via the **paper** broker. Same code path as live = no drift.
- `metrics.py` — Sharpe, Sortino, max DD, expectancy, profit factor, hit rate, avg R, trade duration distribution.
- `report.py` — writes summary + per-trade CSV; renders charts the dashboard reads.

### 5.10 `ai/`
- `features.py` — feature engineering (returns, vol percentiles, EMA distances, regime tags).
- `regime_classifier.py` — trains a classifier offline; output feeds `strategy/regime.py`.
- `signal_scorer.py` — given a rule-based signal, score 0–1 confidence; risk uses it to size up/down.

### 5.11 `dashboard/`
Streamlit. One process, four pages.

- `app.py` — sidebar: kill switch button, mode (paper/semi/auto), symbol picker.
- `pages/live.py` — current price, indicators, last signal, open positions, today's P&L, recent alerts feed.
- `pages/backtest.py` — pick date range + params, run, see equity + DD + metrics.
- `pages/journal.py` — trade journal table with filters, win/loss histograms, R distribution.
- `pages/risk.py` — current exposure, daily loss vs cap, drawdown vs cap, cooldown status.

The dashboard reads from `state.py` and the journal CSV; it does not call data providers directly. The kill switch button writes to `kill_switch.py`'s flag file.

---

## 6. Data flow — one full live tick

1. `main.py` loop wakes, asks data provider for latest 15m + 1H bars.
2. Indicators recompute (cached if no new bar).
3. Strategy `.evaluate()` returns either `None`, a `WATCHLIST`/`BREAKOUT_WATCH` signal, or a `BUY_READY`/`SELL_READY` signal.
4. Watchlist signals → alert router only. No order.
5. Ready signals → risk gate:
   - Within daily loss cap?
   - Below max-positions?
   - Not in cooldown?
   - Position size > 0?
   - If any fails → `RISK_ALERT`, no order.
6. Risk emits an `Order` with size, SL, TP.
7. Execution places it (paper or MT5), receives `Fill`, opens `Position`.
8. Each subsequent tick, `stops.py` updates trailing SL via `order_manager.modify(...)`.
9. On SL/TP hit → `Fill` close → journal writes a complete row → exit alert fires.
10. Every step publishes an event; dashboard re-reads on next refresh.

## 7. Data flow — backtest

Identical to above except: data provider = historical CSV iterator, broker = paper, no alerts to Telegram (logged only), runs as fast as the loop allows. Because the same `Strategy` and `Risk` objects are used, the backtest equity curve is the live behaviour.

---

## 8. Auto-trading staging plan (Mac + OANDA, small-account fast path)

You chose: skip semi-auto, go to live auto on a small account. Staging is
compressed but the gates remain — they exist so a broken strategy can't drain
the account.

| Stage | What happens | Exit criteria |
|---|---|---|
| 0 — Backtest | Strategy + risk run on history | Sharpe > 1, max DD < 15%, expectancy > 0R, hit-rate > 40% over 6 months of XAU_USD 15m data |
| 1 — Forward paper (OANDA practice) | Same code, live OANDA prices, practice account, full alerts | 1 week minimum, live equity curve tracks backtest within ±20%, zero execution errors |
| 2 — Live auto, small account | OANDA live, real money, kill switch armed, hard equity cap = `max_account_equity_usd` in config | ongoing; any cap breach trips kill switch and pauses trading |

**Hard guardrails enforced in code regardless of stage:**
- 2% risk per trade (cap; sizer never exceeds)
- 3% daily loss cap → no new trades for the rest of the day
- 15% peak-drawdown cap → kill switch trips, all trading halts pending review
- `max_account_equity_usd` cap → if account grows past it, new orders refused. Raising the cap requires re-running stage 1 at the new size.

Stages are flipped via `config.yaml` (`execution.broker` and `execution.mode`) — no code change.

---

## 9. Configuration (single source)

`config.yaml` controls everything user-tunable: symbol list, timeframes, EMA periods, ATR thresholds, risk-per-trade %, daily loss cap, max DD, broker (`paper` / `mt5`), execution mode (`backtest` / `paper` / `semi` / `auto`), Telegram token, alert sound paths, dashboard port. No magic numbers in code.

---

## 10. Decisions locked + remaining open questions

**Locked in:**
1. Platform → macOS
2. Broker → **Upstox v2 API** (OANDA wouldn't accept India). Symbol: **MCX GOLDM** (gold mini, 100g lot, fully SEBI-regulated).
3. Risk per trade → 2% of equity
4. Rollout → Backtest → 1wk paper → live auto on small account, hard equity cap (₹50,000 default)
5. Semi-auto stage → skipped
6. Auth → OAuth 2.0 (daily re-login required by SEBI; bot alerts before expiry)

**Still open — answer when convenient, doesn't block first build:**
1. **OANDA account region:** US, EU, or other? (Determines API base URL and available leverage/margin rules.)
2. **Small-account starting size:** what number goes into `max_account_equity_usd`? Default placeholder is $1500.
3. **UI host:** local Streamlit only, or reachable from your phone over LAN / Tailscale?
4. **Sound alarm:** any preferred .wav for entry/exit/risk, or ship a default set?
5. **Journal storage:** stay with CSV, or SQLite now (cleaner dashboard later)?
6. **Backtest data source:** keep yfinance for history, or download clean Dukascopy/HistData once and store locally? (yfinance gold data can have gaps.)

---

## 11. What lives where (folder map)

```
v2/
├── ARCHITECTURE.md         ← this file
├── README.md               ← run instructions (skeleton)
├── requirements.txt
├── config.yaml             ← user-editable settings
└── app/
    ├── main.py             ← orchestrator entry
    ├── core/               ← config, events, state, kill switch
    ├── data/               ← market data providers
    ├── indicators/         ← pure indicator functions
    ├── strategy/           ← signal logic (shared by live + backtest)
    ├── risk/               ← sizing, limits, stops
    ├── execution/          ← brokers + order manager
    ├── alerts/             ← severity router + channels
    ├── logging/            ← trade journal + event log
    ├── backtest/           ← engine + metrics + report
    ├── ai/                 ← features, regime, signal scoring
    └── dashboard/          ← Streamlit UI
```

Every file in `app/` is currently a skeleton: class signatures, docstrings, and `# TODO` markers. No business logic yet. Approve the shape, answer the seven questions in §10, and we start filling.
