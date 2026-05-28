# Deployment Guide

How to deploy code changes to the Windows VPS via git.

## Prerequisites

Once-only setup (done on first deploy):
1. Mac side: this git repo exists at `~/Documents/ai-trading-bot/v2/`
2. VPS side: git installed, repo cloned to `C:\ai-trading-bot\`
3. VPS `.env` lives at `C:\ai-trading-bot\.env` (gitignored, copied once manually)
4. NSSM services `psp_bot_breakout` and `psp_bot_smc` point to `C:\ai-trading-bot\scripts\mt5_*.py`

## Day-to-day workflow

### 1. Make changes on Mac

Edit files locally, test in sandbox / on Mac if relevant.

```bash
cd ~/Documents/ai-trading-bot/v2
# ... edit files ...

# Optional: smoke test
python3 -m py_compile scripts/mt5_live.py scripts/mt5_smc.py scripts/_bot_common.py

git status
git diff
git add -A
git commit -m "tune SMC POI score threshold to 3 — too many false signals"
git push
```

### 2. Pull on VPS via RDP/PowerShell

```powershell
cd C:\ai-trading-bot\v2
git pull
```

If the change affects bot files (anything in `scripts/`), restart services:

```powershell
sc.exe stop psp_bot_breakout
sc.exe stop psp_bot_smc
Start-Sleep 5
Stop-Process -Name python -Force -ErrorAction SilentlyContinue
Start-Sleep 2
sc.exe start psp_bot_breakout
sc.exe start psp_bot_smc
Start-Sleep 15
Get-Service psp_bot_*
Get-Content C:\ai-trading-bot\logs\breakout.out.log -Tail 5
```

Dashboard changes don't need a restart — just refresh the Streamlit browser tab.

### 3. Verify in Telegram

Within ~30 sec you should get two `[BOT START — MT5/XM]` messages.

## What's NEVER deployed via git

Stored only on VPS, never committed:
- `.env` (Telegram token, MT5 password, Supabase password, Alpha Vantage key)
- `.mt5_state.json`, `.mt5_smc_state.json` (bot state)
- `data/mt5_trades.csv`, `data/mt5_smc_trades.csv` (trade journals)
- `data/.av_news_cache.json` (cache)
- `models/*.pkl` (trained models)
- `logs/` (NSSM rotation)

If you need to reset state (rare):
```powershell
cd C:\ai-trading-bot\v2
del .mt5_state.json -ErrorAction SilentlyContinue
del .mt5_smc_state.json -ErrorAction SilentlyContinue
sc.exe stop psp_bot_breakout; sc.exe start psp_bot_breakout
```

## Schema changes

If you edit `db/schema.sql`, run it through Supabase SQL Editor manually after `git pull`:
1. Open Supabase web → SQL Editor
2. Paste the updated schema (idempotent — uses `IF NOT EXISTS`)
3. Run

## Rolling back

If a deploy breaks:

```powershell
# VPS
cd C:\ai-trading-bot\v2
git log --oneline -10
git checkout <previous-commit-hash>
sc.exe stop psp_bot_breakout; sc.exe start psp_bot_breakout
sc.exe stop psp_bot_smc; sc.exe start psp_bot_smc
```

Then fix on Mac and push a new commit. After the fix is live, return VPS to `main`:
```powershell
git checkout main
git pull
```

## First-time VPS clone

See `GIT_INITIAL_SETUP.md`.
