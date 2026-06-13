# CLAUDE.md — AI Trading Bot Project Context

**Owner:** Praj (DevOps + full-stack founder, solo operator)
**Mac project root:** `~/Documents/ai-trading-bot/` (with active dev tree in `v2/`)
**Mac git repo:** `~/Documents/ai-trading-bot/v2/` — this is where `.git` lives. The `v1` `bot.py` at the parent is legacy, untracked, don't touch.
**VPS repo:** `C:\ai-trading-bot\` — FLAT layout, NO `v2\` subdir. The GitHub repo is rooted at what the Mac calls `v2/`, so on the VPS `scripts/`, `config.yaml`, `data/` etc. are directly at the repo root.
**VPS host:** `163.223.86.49`
**VPS NSSM binary:** `C:\ai-trading-bot\nssm.exe` (in the repo dir itself — not on PATH; always invoke as `.\nssm.exe`)
**VPS service logs:** `C:\ai-trading-bot\logs\` — files named `smc.out.log`, `smc.err.log`, `breakout.out.log`, etc.
**Broker:** XM Ultra Low (MT5 demo, account 168184358)
**Instrument:** XAUUSD via symbol `GOLD.i#` (contract 100 oz/lot, 0.01 min, 0.01 step — KNOWN ISSUE, see "Sharp edges" below)

> This document is the handoff from a prior Claude session that built most of this system. It's hand-written, not auto-generated — read it once at session start so you don't repeat decisions we already made and threw away.

---

## TL;DR — Current State (as of 2026-06-03)

**What's running on the VPS right now:**
- `psp_bot_smc` — SMC (Smart Money Concepts) strategy, magic 20260601, params from `aggressive_all` sweep. Just got a regime override: now trades in chop too (chop weight 0.0 → 1.0).
- `psp_bot_breakout` — running but **disabled** via regime weights (all 0.0). Kept alive to log signals for future ML/research.
- `psp_dashboard` — Streamlit on port 8501, password-gated, sidebar nav for Live Bot / Backtest Reports.

**What's built but NOT running:**
- `psp_bot_dca` — DCA gold buyer (mt5_dca.py). Fully plumbed (config, loader, runbook). User pivoted away from DCA after building it — service is NOT installed. See `v2/RUNBOOK_DCA.md` if it gets revived.
- `psp_bot_telegram` — Telegram control center (`scripts/telegram_control.py`, built 2026-06-13). Phone control: `/status /pnl /positions /pause /resume /risk /logs /restart` + daily summary + DD alerts. `/pause` writes `data/.control.json`, which both bots check at the top of `can_open_new_trade()` (blocks NEW entries only; open positions keep SL/TP). Install via `DEPLOY_TELEGRAM.md` (NSSM service, not yet registered on the VPS). Auth = `TELEGRAM_CHAT_ID` only. Reuses existing telegram .env creds.

**What we've learned that matters more than any code:**
1. Of 4 strategies backtested over 4.24 years, **only SMC has a real edge**, and even SMC is currently in its worst-ever 262-day drawdown.
2. The ML meta-labeler on a 1,350-sample combined dataset returns val AUC **0.5752** (honest; reproduced 2026-06-04). The overfit incumbent scored a higher AUC only because it was overfit. There is no easy ML rescue. NOTE: the model `.pkl` is **gitignored and per-machine** — the VPS and Mac each hold their own file (VPS incumbent was val_auc 0.749; Mac's was 0.391). Changing the model means retraining on the box that runs the bot, not editing a committed file.
3. Buy-and-hold gold (Sharpe ~1.16) beat 3 of 4 strategies. We considered pivoting to DCA, built it, then user reversed: wants algo trading despite the data.

---

## Architecture (one paragraph)

A single canonical `v2/config.yaml` drives both bots. Bots are independent Python long-running processes (one per strategy), managed as Windows services via NSSM. Each bot reads MT5 ticks/bars, runs gates (session window, economic calendar, news sentiment, ADX regime classifier, drawdown tiers, Kelly sizing), evaluates its strategy, opens market orders with SL+TP (except DCA which is no-SL accumulation). Closed trades land in a CSV journal AND Supabase Postgres. A Streamlit dashboard reads the journal + equity snapshots for monitoring. The whole thing is git-deployed: edit on Mac → push to GitHub → `git pull` on VPS → `nssm restart`.

```
Mac (dev) ──git push──> GitHub ──git pull──> Windows VPS
                                                    │
                            ┌───────────────────────┼───────────────────────┐
                            ▼                       ▼                       ▼
                    psp_bot_smc           psp_bot_breakout            psp_dashboard
                    (mt5_smc.py)          (mt5_live.py)               (dashboard.py)
                            │                       │                       │
                            └───────────┬───────────┘                       │
                                        │                                   │
                                        ▼                                   ▼
                                  Supabase Postgres ◀──────── reads ──────── │
                                  (trades, signals, equity_snapshots, events)
```

---

## Project layout (only the files that matter)

```
v2/
├── config.yaml                     # SINGLE SOURCE OF TRUTH — bots read at startup
├── ARCHITECTURE.md                 # older arch doc, mostly still accurate
├── DEPLOY.md                       # VPS install steps
├── README.md                       # public-facing
├── RUNBOOK_DCA.md                  # if/when we re-enable DCA
├── requirements.txt
├── data/
│   ├── history/                    # OHLC parquets pulled via fetch_mt5_history.py
│   │   ├── GOLD_i_M15.parquet      # 4.24yr × 15min
│   │   ├── GOLD_i_H1.parquet       # 6yr × 1h
│   │   └── GOLD_i_H4.parquet       # 6yr × 4h
│   ├── backtests/                  # *_trades.parquet, *_equity.parquet per strategy
│   ├── ml_dataset_combined.parquet # 1,350 labeled samples across 4 strategies (GITIGNORED; built on the VPS, not on the Mac)
│   ├── economic_calendar.json      # high-impact USD events
│   ├── .av_news_cache.json         # Alpha Vantage news cache (gitignored)
│   ├── .dca_state.json             # DCA idempotency state (gitignored)
│   ├── mt5_trades.csv              # breakout closed-trade journal
│   ├── mt5_smc_trades.csv          # smc closed-trade journal
│   └── mt5_dca_trades.csv          # dca buy log (when active)
├── models/                         # ENTIRE DIR IS GITIGNORED — per-machine, NOT shipped via git
│   ├── meta_labeler.pkl            # live model, set by train_meta_v2.py (honest model = val_auc 0.5752)
│   ├── meta_labeler.meta.json      # training metadata (features, chosen_threshold, val_auc)
│   └── meta_labeler.prev.*         # backup of prior model, written on each swap
└── scripts/
    ├── _config_loader.py           # typed config dataclasses + load_config()
    ├── _bot_common.py              # MT5 init, sessions, calendar, news, regime, Kelly, DD tiers
    ├── _strategies.py              # standalone evaluators used by live AND backtest
    ├── _backtest_engine.py         # bar-by-bar sim with realistic costs
    ├── _journal.py                 # Postgres writer (no-op if DB down)
    ├── _meta_scorer.py             # ML model loader + scoring wrapper
    ├── mt5_live.py                 # breakout bot (currently disabled via regime weights)
    ├── mt5_smc.py                  # SMC bot (live + active)
    ├── mt5_dca.py                  # DCA bot (built but NOT installed as service)
    ├── dashboard.py                # Streamlit; renders tearsheets via st.components.v1.html
    ├── fetch_mt5_history.py        # pulls M15/H1/H4 from MT5 via copy_rates_from_pos
    ├── run_backtest.py             # CLI: --strategy breakout|smc|mean_reversion|liquidity_sweep|all
    ├── sweep_smc.py                # 5 SMC variants — 'aggressive_all' won
    ├── walk_forward.py             # post-process trade logs into 6-month windows, R-based
    ├── generate_tearsheet.py       # QuantStats HTML reports per backtest
    ├── build_combined_ml_dataset.py# build training data from backtest trade logs
    ├── train_meta_v2.py            # retrain ML, chronological walk-forward CV
    └── check_news.py               # ad-hoc sentiment check
```

---

## What each bot actually does (in plain English)

### SMC bot (`mt5_smc.py`) — CURRENTLY THE ONLY ACTIVE TRADER

Reads M15 + H1 bars. Detects swing pivots, order blocks (OBs), fair value gaps (FVGs), then waits for price to retrace into a point-of-interest (POI). On confirmation, opens a market trade with SL behind the POI and TP at the next swing. Uses `aggressive_all` params from the parameter sweep:
- `htf_pivot: 1` — most sensitive swing detection
- `min_impulse_bars: 2`
- `poi_freshness_bars: 120` (~20 days on H1)
- `min_poi_score: 1` — accept standalone OB or FVG
- `min_rr: 1.0` — accept 1:1 RR setups

Risk: 1% per trade, max 1 concurrent position. Hard gates: session window (London / NY overlap / NY afternoon, IST), economic calendar block (±30/60 min around high-impact USD events), news sentiment block (block BUY if AV score < -0.35, etc), regime weights (see below).

### Breakout bot (`mt5_live.py`) — DISABLED, KEPT FOR LOGGING

Walk-forward over 4 years showed 2/9 positive 6-month windows, PF 1.04 — statistical noise. Disabled by setting `regime.weights.*.breakout = 0.0` everywhere. The signal logic still runs and writes to Postgres `signals` table for research, but the risk layer kills every trade attempt.

### DCA bot (`mt5_dca.py`) — BUILT, SHELVED

Schedule-driven (default: every Monday 12:30 IST = London open). Buys fixed USD notional at market, no SL/TP. Hard caps: `max_lot_per_buy`, `max_total_lots`, `max_buys_per_day`. State file `.dca_state.json` ensures idempotency across restarts. **Math problem we hit:** at standard gold contract (100 oz/lot) and $3,400/oz, the broker minimum 0.01 lot represents $3,400 of notional — so $50/wk floors to zero lots. User chose to abandon DCA rather than raise the buy amount.

---

## The regime classifier — read this before touching `config.yaml`

`_bot_common.py:classify_regime()` returns one of: `trend_up`, `trend_down`, `chop`, `transition`, `high_vol`, `unknown`. ADX-based with EMA50/EMA200 direction. The `regime.weights` table in `config.yaml` maps each regime to a per-strategy size multiplier (0.0 = halt that strategy in that regime, 1.0 = full size, 0.5 = half size).

**Current weights (after Phase H.5 override):**

```
              breakout    smc
trend_up      0.0         1.0
trend_down    0.0         1.0
chop          0.0         1.0   ← H.5 override 2026-06-03, was 0.0
transition    0.0         0.5
high_vol      0.0         0.5
unknown       0.0         0.0
```

The chop=1.0 was a deliberate user override against walk-forward evidence (SMC historically loses in chop). The rationale: the user wanted more trading activity and accepted the drawdown risk. Risk net (daily 3% cap, 15% max DD, DD tiers, 4hr cooldown after 2 consecutive SLs) is the safety layer.

If chop bleeds badly: revert that single line and we're back to the data-supported config.

---

## The risk system — non-negotiable safety layer

All defined in `config.yaml:risk` and enforced in `_bot_common.py`:

| Gate | Setting | What it does |
|---|---|---|
| Per-trade risk | `risk_per_trade_pct: 0.01` | 1% of equity max per trade |
| Daily loss cap | `daily_loss_cap_pct: 0.03` | Halt new entries at -3% on the day |
| Max DD kill | `max_drawdown_pct: 0.15` | HARD halt all trading at -15% from peak; requires manual reset of peak_equity |
| DD tiers | `dd_tiers` | At ≥12% DD multiply risk by 0.0 (halt), ≥7% × 0.25, ≥3% × 0.50, else × 1.0 |
| Cooldown | `cooldown_after_consecutive_losses: 2`, `cooldown_minutes_after_losses: 240` | After 2 SL in a row, pause 4hr |
| Re-entry block | `reentry_block_minutes: 120` | After any close (win or loss), 2hr before new entry |
| Kelly sizing | `kelly.fraction: 0.25` quarter-Kelly | Rolling 30-trade Kelly, capped at [0.25x, 2.0x] |
| Max concurrent | `max_concurrent_positions: 1` | One position per bot at a time |

**These cascade multiplicatively.** Final risk = base × dd_mult × kelly_mult × regime_mult. If any is 0, the trade is halted.

> **Don't loosen these without a backtest in front of you.** They are the reason the account hasn't blown up despite multiple bad-strategy iterations.

---

## Strategies tested — full results (so you don't relitigate)

| Strategy | Trades | PnL | PF | Sharpe | Max DD | Verdict |
|---|---|---|---|---|---|---|
| Breakout (4.24yr) | 322 | +5.5% | 1.04 | low | — | **Dead** (walk-forward 2/9; 2026-06-11 re-run w/ harsher costs: PF 1.08 — still dead) |
| SMC `aggressive_all` (4.24yr) | 185 | +381% | 1.89 | 0.89 | 13.57% | **Live** (best edge; in 262-day DD) |
| SMC **OOS walk-forward** (2026-06-11, 12mo train/6mo test) | — | +71.9R | **1.98 OOS** | — | 11.8R | **DEPLOYABLE** — honest out-of-sample, 5/7 windows positive, WR 30%. The number that justifies keeping SMC live. |
| FVG scalp (2026-06-11, M15 intraday) | 875 | **-99.7%** | 0.71 | — | 99.7% | **Rejected pre-deployment** — 37% WR at fixed 1.5RR dies on costs; ~0.9 trades/day, not the 5-8 targeted |
| SMC baseline (4.24yr) | 31 | +23.8% | — | — | — | Too few trades |
| Mean Reversion aggressive | 419 | -70.16% | 0.58 | — | 76.94% | **Catastrophic** |
| Mean Reversion textbook | not retested | — | — | — | — | Code exists, untested |
| Liquidity Sweep (4yr) | 241 | -17.4% | 0.88 | -0.51 | — | **Negative edge** |
| Buy-and-hold gold | n/a | spot move | — | 1.16 | — | **Beat 3 of 4 strategies** |

ML training on combined 1,350-sample dataset: val AUC 0.5752, vs old VPS model 0.7491 (old was overfit). The auto-swap gate keeps the higher-AUC model, so the honest model could never promote — fixed 2026-06-04 with a `--force-replace` flag. On the new model, threshold 0.78 gives 75% WR on only 12 trades over 4yr; the default `--target-wr 0.40` instead picks threshold ~0.05 which keeps 255/270 signals (near no-op). Choose `--target-wr` deliberately. **The bot scores in shadow mode by default (`ML_SHADOW_MODE=true`), so the model logs verdicts but does NOT veto live trades until that env var is flipped to false.**

---

## Conventions to keep

### Credentials policy (strict)
- `.env` at `v2/.env` — gitignored. Holds: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `ALPHA_VANTAGE_KEY`, `DASHBOARD_PASSWORD`.
- **NEVER** put passwords/tokens in `config.yaml` (it's committed) or in chat.
- VPS RDP password: the user keeps it. Don't ask. Don't store it.
- Tokens have been rotated once already after a leak (Telegram bot token, GitHub PAT). Treat them as sensitive.

### Windows-cp1252 console gotcha
Print statements that hit Windows NSSM stdout MUST be ASCII-only — no unicode arrows (→), `≥`, `×`, etc. Use `->`, `>=`, `x`. There's a `try/except` around `print_summary()` in `_config_loader.py` specifically because of this. New code: stick to ASCII in any string that might end up in a Windows log.

### Backtest engine scaling
`_backtest_engine.py` uses eval window caps (M15=500 bars, H1=300, H4=200) to keep strategy calls O(window) instead of O(history). Without these, a 4-year backtest hangs at 37 b/s. Don't remove the caps.

### Strategy dispatch
`_call_strategy` in the engine dispatches by name. Breakout takes `df4h`, the others don't. Adding a new strategy means: function in `_strategies.py`, dispatch case in the engine, optional live runner.

### Test-trade stale-bar guard
There's a stale-bars guard in the bots: if `df15.iloc[-1]['Close']` diverges from `tick.ask` by >2×ATR, the bot aborts before opening. This is from a real bug where SL ended up above entry on a BUY because the test trade used stale bar prices.

---

## Common dev workflows

### Edit a parameter and ship it
```bash
# Mac (note: Mac has no `python` alias, use `python3`)
cd ~/Documents/ai-trading-bot/v2
# edit config.yaml
python3 scripts/_config_loader.py smc    # verify it parses (needs venv with pyyaml)
git add config.yaml
git commit -m "tweak: <one-liner>"
git push origin main
```

```powershell
# RDP to VPS (163.223.86.49). Repo lives at C:\ai-trading-bot\ (flat — NO v2 subdir).
cd C:\ai-trading-bot
git pull
.\nssm.exe restart psp_bot_smc       # nssm.exe is in the repo dir, NOT on PATH
Start-Sleep -Seconds 5
.\nssm.exe status psp_bot_smc        # should report SERVICE_RUNNING
Get-Content .\logs\smc.out.log -Wait -Tail 40   # log naming is <bot>.out.log / .err.log
```

Restart gotcha: if `nssm restart` reports `Unexpected status SERVICE_STOP_PENDING`, the stop took too long and the service might have stayed stopped. Always run `.\nssm.exe status` after restart and explicitly `.\nssm.exe start` if it's `SERVICE_STOPPED`.

### Run a backtest locally
```bash
cd ~/Documents/ai-trading-bot/v2
python scripts/run_backtest.py --strategy smc --years 4 --poll-every 4
# trade log lands at data/backtests/smc_<timestamp>_trades.parquet
python scripts/generate_tearsheet.py
```

### Pull fresh history from MT5
```bash
# Only runs on VPS (or any machine with MT5)
python scripts/fetch_mt5_history.py
# Outputs to data/history/
```

### Retrain the ML meta-labeler
```bash
python scripts/build_combined_ml_dataset.py
python scripts/train_meta_v2.py
# --no-replace      just report, don't touch the .pkl
# --force-replace   promote the new model even if its AUC is lower than the incumbent
#                   (needed to swap an honest lower-AUC model over an overfit one)
# --target-wr 0.60  raise to pick a higher (more selective) threshold; default 0.40
#                   picks ~0.05 which barely vetoes anything
# Run this on the box that runs the bot (the VPS) — the .pkl is gitignored.
```

### Restart everything on VPS
```powershell
cd C:\ai-trading-bot
.\nssm.exe restart psp_bot_smc
.\nssm.exe restart psp_bot_breakout
.\nssm.exe restart psp_dashboard
```

### Tail logs
```powershell
Get-Content C:\ai-trading-bot\logs\smc.out.log -Wait -Tail 40
Get-Content C:\ai-trading-bot\logs\smc.err.log -Wait -Tail 40    # errors only
Get-Content C:\ai-trading-bot\logs\breakout.out.log -Wait -Tail 40
```

---

## Postgres schema (Supabase)

Connection via `aws-1-ap-south-1.pooler.supabase.com:6543`. Tables:

- `trades` — closed-trade journal. Cols: bot_name, magic, trade_id, side, open_time, close_time, entry, exit, lots, sl, tp, pnl_usd, r_realised, exit_reason, regime, poi_score, etc.
- `signals` — every signal evaluation including rejections. Cols: bot_name, ts, severity, side, price, reason, rejection_reason, regime, extras (JSONB).
- `equity_snapshots` — for plotting/DD tracking. Cols: ts, account, equity, balance, peak_equity, dd_pct, open_positions.
- `events` — bot lifecycle + ops events. Cols: bot_name, ts, kind, payload (JSONB). DCA writes `dca_buy` events here.

The journal writer (`_journal.py`) is fail-safe: if Postgres is down, writes become no-ops and the bot never crashes.

---

## Dashboard

`scripts/dashboard.py` runs on VPS port 8501 as service `psp_dashboard`. Streamlit app with sidebar nav:
- **Live Bot** — current positions, equity curve, recent trades, regime, gate status
- **Backtest Reports** — auto-discovers `data/backtests/*_tearsheet.html`, renders inline via `st.components.v1.html()` + download button

Password gate via `DASHBOARD_PASSWORD` env var. Cache TTLs tiered: LIVE=30s, SLOW=180s, LIST=300s.

---

## Decisions log (recent → oldest)

| Date | Phase | Decision |
|---|---|---|
| 2026-06-13 | I.4 | **ML veto is now LIVE on the VPS** (`ML_SHADOW_MODE=false` in VPS `.env`, SMC restarted, banner confirms `shadow=False`). User did this after SMC took 3 live losers and wanted "fewer losses" -- pushed back that 30% WR + a 3-loss run is normal variance (backtest had a 13-loss streak) and every non-SMC strategy tested loses money, so the honest lever is the ML filter, not a strategy swap. Live model targets 60% WR via threshold ~0.74, so it vetoes the large majority of signals -> expect FAR fewer trades (opposite of the earlier 5-8/day wish). Edge is marginal (val_auc 0.58) and live shadow data was near-zero, so monitor; if too quiet, retrain at a lower `--target-wr` to keep more trades. Tradeoff is explicit and user-owned. |
| 2026-06-13 | I.3 | Phase C hardening. `tests/test_risk_layer.py`: 27 tests pinning `dd_multiplier`/`kelly_multiplier`/`compute_effective_risk` (the untested money-sizing core) -- tiers, Kelly bounds, multiplicative cascade, halt conditions. GitHub Actions CI (`.github/workflows/ci.yml`): compileall + pytest on push/PR, Python 3.11, installs `requirements-dev.txt` (minimal fast set: pandas/numpy/pyyaml/python-dateutil/requests/python-dotenv/pytest -- NOT streamlit/sklearn/MetaTrader5, all lazy/unneeded for tests; verified in a clean-room venv). CI tests only, deploy stays manual. Log rotation + service hygiene: `RUNBOOK_OPS.md` (NSSM AppRotateFiles/Bytes/Online 10MB for all services -- rotation is per-service and NOT retroactive, so existing services need it applied). 41 tests total, all green. |
| 2026-06-13 | I.2 | Built Telegram control center (`scripts/telegram_control.py`, service `psp_bot_telegram`). Phone commands + daily summary + DD alerts. Added `control_paused/control_set/CONTROL_PATH` to `_bot_common.py`; both bots now check the remote-pause flag first in `can_open_new_trade()`. Pause = block NEW entries only (open positions keep server-side SL/TP). Auth strictly by `TELEGRAM_CHAT_ID`. Single getUpdates poller (don't run a 2nd on the same token -> 409). Deploy: `DEPLOY_TELEGRAM.md`. Tests: `tests/test_control_flag.py` (flag round-trip, fail-open, PnL bucketing). NOT yet registered as an NSSM service on the VPS. |
| 2026-06-11 | I.1 | Full re-validation after review fixes (completed-bar fetch, SELL-exit spread, swap modeling, chop revert). SMC OOS walk-forward: **PF 1.98, +71.9R, 5/7 windows -> DEPLOYABLE**, stays live. fvg_scalp PF 0.71/-99.7% -> rejected pre-deployment. breakout/MR/LS still negative. Report: `data/backtests/review_fixes_report.md`. Swap on this XM demo is DISABLED (swap_mode=0) so $0/night is correct; re-check on real account. Gotchas: cp1252 crash applies to *redirected* stdout too (Start-Process logs) — suite forces PYTHONIOENCODING=utf-8 for child steps; urllib hits CERTIFICATE_VERIFY_FAILED on the VPS, requests (certifi) works — use requests for Telegram. |
| 2026-06-04 | H.10 | Added `--force-replace` to train_meta_v2 to promote the honest meta-labeler (val_auc 0.5752) over the overfit 0.749 incumbent (the AUC gate blocked it). Swap runs on the VPS. Default `--target-wr 0.40` -> threshold ~0.05 (near no-op); tune `--target-wr` for selectivity. Stays shadow mode (no live veto) until `ML_SHADOW_MODE=false`. Documented model `.pkl` is gitignored/per-machine. |
| 2026-06-04 | H.9 | Fixed latent pickle bug: meta-labeler wrapper was a function-local class (`Can't pickle local object`), surfaced the first time a swap actually ran. Moved to module-level `_meta_scorer.MetaModel` so the bot unpickles it without importing the trainer. |
| 2026-06-04 | H.8 | Symbol info confirmed GOLD.i# = 100 oz/lot, min 0.01. On $960 account this means 1% target risk is unreachable; actual per-trade risk = 1.5-3%. No micro-gold on this XM account. User emailed XM support to ask about alternatives. Bot continues running with oversized risk in the interim. |
| 2026-06-03 | H.5 | User override: regime.weights.chop.smc 0.0 → 1.0. Contradicts walk-forward; accepted for activity |
| 2026-06-03 | H.4 | Built DCA bot (`mt5_dca.py`) + `RUNBOOK_DCA.md`. User reversed before deployment |
| 2026-06-02 | G | ML retrain on 1,350 samples → val AUC 0.58. Old 0.75 model kept (NOT swapped) |
| 2026-05-30 | F | Added Liquidity Sweep strategy + backtest. -17.4% PnL, PF 0.88. Disabled |
| 2026-05-29 | E | Added Mean Reversion strategy. Aggressive variant -70% PnL. Disabled |
| 2026-05-29 | C | Walk-forward analysis: breakout dead (2/9), SMC flipped to aggressive_all params. Regime weights flipped: trends → SMC, chop → halt |
| 2026-05-28 | B | Backtest engine + realistic costs (spread 25pts, slippage 0-3 pips, $7/lot RT commission) |
| 2026-05-27 | A | Config-driven architecture. `config.yaml` is now SSOT. Risk per trade 2% → 1% |

---

## Known sharp edges

1. **The model `.pkl` is gitignored and per-machine.** `models/` is NOT in git, so the Mac and VPS each carry their own model file — editing/training on the Mac does NOT change what the live bot loads. Any model swap must run on the VPS. To put the honest val_auc=0.5752 model live, retrain on the VPS with `train_meta_v2.py --force-replace`, which bypasses the AUC-comparison gate that otherwise locks in an overfit higher-AUC incumbent. Two gotchas: (a) the default `--target-wr 0.40` picks threshold ~0.05 which keeps ~255/270 signals — near no-op; raise `--target-wr` for a selective model. (b) Scoring is in shadow mode (`ML_SHADOW_MODE=true`) so the model does not actually veto live trades until that env var is set false in the VPS `.env`.

2. **The contract size makes the 1% risk target unreachable.** GOLD.i# is 100 oz per 1.0 lot, broker min 0.01 lot. At $4,470/oz that's $4,470 of notional per 0.01 lot. With $960 equity and 1% risk target = $9.60, the required SL distance is ~$9.60/(0.01×100) = $9.60 — which is way smaller than any typical 1.5×ATR stop ($15-$75). In practice every SMC trade risks 1.5-3% of equity, not 1%. The bot does not know this — config says 1% and the math layers underneath don't refuse the trade. The safety net (daily 3% cap, 15% max DD) still bites, but per-trade risk is dishonest. Fixes: (a) open XM account tier with smaller gold contract, (b) raise risk_per_trade_pct to 0.02 to be honest, (c) wait until equity > $5K. As of 2026-06-04, user emailed XM support asking which account tier offers micro-gold; waiting on reply.

3. **SMC is in a 262-day drawdown** as of last walk-forward analysis. The decision to keep running it is a bet that the regime returns to its historical mean. If the user starts asking why the account is bleeding, the answer is in `walk_forward.py` output.

4. **The breakout bot still writes signals to Postgres** even though every trade is halted. Don't be confused by activity in the `signals` table — check `bot_name='breakout' AND rejection_reason LIKE 'regime%'` to count vetoed-by-regime entries.

5. **WATCH/pre-entry Telegram alerts are silenced.** Past version spammed the user. Both `mt5_live.py` and `mt5_smc.py` have `tg_send()` commented out for WATCH-severity signals; SMC pre-entry alerts require `poi_score >= 3`.

6. **Magic numbers are sacred.** Each strategy has its own (breakout: 20260522, smc: 20260601, dca: 20260603). They're how MT5 lets the bot find ITS positions among everything in the account. Never change a magic on a strategy with open positions — those positions become orphaned.

7. **Cron parser in `mt5_dca.py` is intentionally minimal.** Supports `*`, single integer, comma list — not ranges (`1-5`) or steps (`*/2`). Sufficient for DCA schedules; don't extend without adding tests.

---

## When the user says "why isn't the bot trading?"

This question has come up multiple times. Run through this checklist:

1. Is `psp_bot_smc` running? `cd C:\ai-trading-bot && .\nssm.exe status psp_bot_smc`
2. Is the current time in a trading session? London 12:30-16:30 IST, NY overlap 18:00-21:00, NY afternoon 21:00-23:30.
3. Is the news/calendar gate blocking? Tail the log for `calendar_block` or `news_contra`.
4. What's the current regime? Look for `[regime]` log lines.
5. Are there any open positions? `max_concurrent_positions=1` means one open SMC trade blocks new ones.
6. Is the DD tier or daily cap triggered? Look for `halted` in risk decision logs.
7. Is the SMC scoring finding setups but rejecting them? Query Postgres: `SELECT severity, reason, rejection_reason, regime FROM signals WHERE bot_name='smc' AND ts > now() - interval '24h' ORDER BY ts DESC LIMIT 50;`

Don't just say "I'll loosen the gates." Find which gate is actually blocking first.

---

## What I'd build next if asked

Ranked roughly by value:

1. **Daily Postgres heartbeat to Telegram** — a `cron`/scheduled task that posts "Daily summary: X trades, Y signals, Z R, current DD W%" every 23:55 IST. The user doesn't currently get a daily report; only ad-hoc alerts. The reporting hook in config exists (`reporting.daily_summary_hour_ist`) but isn't wired to anything.

2. **Unit tests for the risk layer** — `compute_effective_risk` is THE most important function and has no tests. A small pytest suite over `_bot_common.py` covering DD tiers, Kelly bounds, regime multipliers would prevent regressions. The user has been moving fast and changing config; tests would catch the next foot-gun.

3. **Backtest the chop override before next month's review** — we just enabled SMC in chop on user request. Run a backtest with the new regime weights and compare to baseline. If chop adds -X R over the 4yr period, the user has receipts when they want to revert.

4. **Wire dashboard to surface "next scheduled DCA buy"** — if DCA ever gets re-enabled. Currently the dashboard knows nothing about it.

5. **Replace `meta_labeler.pkl` with the honest model** — and update the comparison gate to use a lower replacement threshold OR add manual `--force-replace` flag. The overfit-old-model situation is dishonest.

---

## How to interact with the user

- They prefer concrete actions over discussion. If you can show them a diff or a result, do that before asking.
- They've been burned by overconfident strategy claims (looking at every YouTube AI-bot video). Match data-driven claims with data; don't overpromise.
- When they ask for "more trades," push back with data first. The honest answer is usually a gate is doing its job, not a bug.
- When they make a choice against the data (like the chop override), implement it, document it as an override with the contradiction explicit, and trust them to own the consequence.
- Keep risk gates sacred. The 1% per trade / 15% max DD / daily cap is what's keeping the account alive through multiple failed strategy generations.
- No emojis unless they use them first.
- Sources, file paths, commit messages — keep these specific. Vague answers cost time on a project this hairy.

---

## If something breaks at 3am

Order of operations:
1. Check VPS is up: ping `163.223.86.49`.
2. RDP in, `cd C:\ai-trading-bot` then `.\nssm.exe status psp_bot_smc` — if it's "SERVICE_PAUSED" or "STOPPED", `.\nssm.exe start psp_bot_smc`.
3. Tail `C:\ai-trading-bot\logs\smc.out.log` (stdout) and `smc.err.log` (errors) for the last error.
4. If MT5 is the issue: `.\nssm.exe restart psp_bot_smc` triggers a full reconnect (`init_mt5_headless` with creds from `.env`). Watch for the `SERVICE_STOP_PENDING` warning — if you see it, manually `.\nssm.exe start psp_bot_smc` to be safe.
5. If Postgres is the issue: the bot keeps trading; `_journal.py` no-ops on failure.
6. If you can't get it back in <15min: stop the service. Don't let a degraded bot trade.

The watchdog in `_bot_common.py:check_mt5_alive_or_reconnect` will `sys.exit(1)` after 5 consecutive empty fetches, which triggers NSSM auto-restart. So short MT5 outages self-heal.

---

That's everything that matters. Welcome to the project.
