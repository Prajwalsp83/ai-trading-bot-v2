# AI Quant Trading Bot — v2

Production multi-strategy gold trading bot. Runs on a Windows VPS via MT5 (XM broker), with optional Mac-side dashboard.

## What's running

**Two live bots in parallel** on a Windows VPS via MetaTrader 5:

| Bot | Strategy | Magic | Files |
|---|---|---|---|
| `psp_bot_breakout` | EMA-50/200 + ATR breakout + 1H trend | 20260522 | `scripts/mt5_live.py` |
| `psp_bot_smc` | Smart Money Concepts: HTF bias + POI mitigation | 20260601 | `scripts/mt5_smc.py` |

Both share `scripts/_bot_common.py` (sessions, calendar, news, risk, regime, watchdog).

## Stack

- **Execution**: MetaTrader 5 (XM GOLD.i#)
- **Risk**: 2% base × 3-tier DD scaling × fractional Kelly × ADX regime weight
- **Gates**: session filter (London/NY), economic calendar block (CPI/NFP/FOMC), Alpha Vantage news sentiment
- **Storage**: Postgres (Supabase) for trades/signals/equity/events, CSV as local backup
- **Auto-restart**: NSSM Windows service with crash recovery + headless MT5 launch + mid-session MT5 death watchdog
- **Alerts**: Telegram
- **Dashboard**: Streamlit reading from Postgres

## Local development (Mac)

```bash
cd v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure .env from .env.example (do NOT commit your .env)
cp .env.example .env  # then edit
```

## VPS deployment

See `DEPLOY.md` for the full git-based update workflow.

Quick reference once deployed:
```powershell
# On VPS
cd C:\ai-trading-bot
git pull
sc.exe stop psp_bot_breakout; sc.exe stop psp_bot_smc
sc.exe start psp_bot_breakout; sc.exe start psp_bot_smc
```

## Layout

```
v2/
├── scripts/             # Live bots + utilities (the runnable code)
│   ├── mt5_live.py      # Breakout bot
│   ├── mt5_smc.py       # SMC bot
│   ├── _bot_common.py   # Shared infrastructure
│   ├── _journal.py      # Postgres writer
│   ├── dashboard.py     # Streamlit dashboard
│   └── generate_ml_dataset.py / train_meta_labeler.py  # ML pipeline (paused)
├── app/                 # Canonical library code (used by backtests + dashboard skeleton)
├── db/
│   └── schema.sql       # Postgres schema (4 tables + 2 views)
├── data/                # Sample calendar + (gitignored) trade journals
├── models/              # (gitignored) trained ML models
├── config.yaml          # All tunable settings
└── requirements.txt
```

## What's NOT in this repo

The `.gitignore` excludes:
- `.env` (secrets — Telegram token, MT5 password, Supabase password, API keys)
- `.mt5_state.json`, `.mt5_smc_state.json` (bot runtime state)
- `data/mt5_trades.csv`, `data/mt5_smc_trades.csv` (live trade journals)
- `models/*.pkl` (trained models)
- `logs/` (NSSM rotation logs)

Re-create from scratch when cloning:
1. Copy `.env.example` → `.env`, fill in values
2. Apply schema to Postgres: `psql $DATABASE_URL -f db/schema.sql`

## Risk

This is a real-money-capable trading bot. The `.env` on the VPS holds production credentials. Trade on demo until you're confident. The system has multiple safety layers (DD halt, daily loss cap, cooldown, news block, regime halt) but no guarantee.
