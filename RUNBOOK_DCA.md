# DCA Cutover Runbook — Phase H

**Date authored:** 2026-06-02
**Author:** Praj (with Claude)
**Goal:** Cut over from algorithmic SMC trading to scheduled Dollar-Cost-Averaging gold buyer.

---

## Why we're doing this

After four backtested algorithmic strategies (Breakout, SMC, Mean Reversion, Liquidity Sweep) all failed to convincingly beat buy-and-hold gold (Sharpe 1.16) over the 4.24-year MT5 history, and the meta-labeler trained on a 1,350-sample combined dataset returned val AUC 0.5752 (vs the old 0.7491 model that was overfit on a smaller dataset), the honest conclusion is: we don't have a demonstrated short-term edge on XAUUSD with the strategies we built. DCA is the mathematically optimal alternative for someone who has conviction in long-term gold but no proven ability to time it.

Direction: **buy $50 of gold every Monday at London open, set it and forget it.**

---

## What changes

| Thing | Before | After |
| --- | --- | --- |
| Active strategy | SMC `aggressive_all` | DCA weekly buy |
| Magic number | 20260601 (SMC) | 20260603 (DCA) |
| Service | `psp_bot_smc` running | `psp_bot_smc` stopped, `psp_bot_dca` running |
| Risk model | 1%/trade with SL/TP | Fixed $USD nominal, NO SL, NO TP |
| Position cap | 1 concurrent | `max_total_lots: 1.0` cumulative |
| Telegram | SMC entry/exit alerts | DCA accumulation alerts only |

Existing SMC open positions: **leave them alone** until they close naturally on their own SL/TP. The DCA bot uses a different magic so it cannot interfere with them.

---

## Step-by-step (run on Windows VPS)

### 1. Local: push the new code

On your Mac:

```bash
cd ~/Documents/ai-trading-bot/v2
git status                                  # confirm changes look right
git add config.yaml scripts/_config_loader.py scripts/mt5_dca.py RUNBOOK_DCA.md
git commit -m "Phase H: add DCA gold buyer (mt5_dca.py)"
git push origin main
```

### 2. VPS: pull + smoke-test the loader

RDP into `163.223.86.49`, open PowerShell:

```powershell
cd C:\bots\ai-trading-bot\v2
git pull
# Verify config parses for the new strategy
python scripts\_config_loader.py dca
```

You should see a clean print summary ending with a `[config] dca: enabled=true ...` line. If it fails: stop and fix before installing the service.

### 3. VPS: dry-run the bot for one tick

```powershell
python scripts\mt5_dca.py --dry-run --once
```

This will:
- Connect to MT5 (or skip if `--dry-run` and MT5 isn't initialized that path)
- Evaluate the cron window
- Print what it WOULD buy if today were Monday and the window were open
- Exit cleanly

If today is NOT Monday between 12:30–13:00 IST, you'll see `outside_window`. That is correct behavior — confirm there are no Python errors.

### 4. VPS: stop the SMC service

```powershell
nssm stop psp_bot_smc
nssm status psp_bot_smc     # should report SERVICE_STOPPED
```

(We don't `nssm remove` it — keep it installed so we can re-enable later if our gold thesis changes. Just stopped.)

### 5. VPS: install the DCA service

```powershell
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"
$script = "C:\bots\ai-trading-bot\v2\scripts\mt5_dca.py"
$cwd    = "C:\bots\ai-trading-bot\v2"

nssm install psp_bot_dca $python $script
nssm set psp_bot_dca AppDirectory $cwd
nssm set psp_bot_dca AppStdout    "$cwd\logs\dca_stdout.log"
nssm set psp_bot_dca AppStderr    "$cwd\logs\dca_stderr.log"
nssm set psp_bot_dca AppRotateFiles 1
nssm set psp_bot_dca AppRotateBytes 10485760     # 10 MB
nssm set psp_bot_dca AppEnvironmentExtra "PYTHONUNBUFFERED=1"
nssm set psp_bot_dca Start SERVICE_AUTO_START
nssm start psp_bot_dca
nssm status psp_bot_dca     # should report SERVICE_RUNNING
```

### 6. VPS: tail the log

```powershell
Get-Content "$cwd\logs\dca_stdout.log" -Wait -Tail 30
```

You should see:
- The `[config] dca: ...` summary
- `[dca] MT5 connected`
- `[dca] state: last_buy_at=None total_buys=0`
- `[dca] poll interval: 60s | dry_run=False | once=False`
- Periodic `[dca] outside window. next target IST: <next Monday 12:30>` heartbeats

### 7. First Monday verification

The next Monday between 12:30–13:00 IST, expect:

1. Telegram: `[DCA] BUY GOLD  Lots: 0.01  @ $XXXX.XX  Notional: ~$XX  Ticket: NNNNN`
2. Log: `[dca] BOUGHT 0.01 lot @ $XXXX.XX  ticket=NNNNN`
3. New row in `data/mt5_dca_trades.csv`
4. New row in Postgres `events` table with `kind='dca_buy'`
5. MT5 terminal shows a position with magic 20260603 and no SL/TP

If any of those are missing → see "Failure modes" below.

---

## Failure modes & fixes

**Bot never fires on Monday.** Check the log around 12:30 IST. If you see `outside_window`, the cron parser found no match. Run `python scripts\mt5_dca.py --dry-run --once` and inspect the response. Most likely cause: VPS clock is in UTC; IST conversion is computed inside the bot from UTC so this should be fine, but verify with `Get-Date` (UTC) and confirm conversion.

**Bot fires twice on Monday.** This can't happen if `state.last_buy_at_utc` is being written. Check `data\.dca_state.json` exists and has the timestamp. If the file is missing after a buy, that's a write-permission issue — fix the path/perms.

**`retcode=...` in log on order send.** Common XM retcodes:
- `10009 / 10018` = success / market closed
- `10006` = rejected (margin? wrong filling mode?). Bot tries IOC → FOK → RETURN automatically; if all three fail, check broker support.
- `10027` = autotrading disabled in terminal. Enable it in MT5: Tools → Options → Expert Advisors → Allow algorithmic trading.

**Lot size below minimum.** At current gold prices around $3400, `$50 / ($3400 × 100 oz) = 0.000147 lot`. Rounded down to 0.01 step = 0.01 lot (~$3400 notional). This is **much larger than $50** — the broker minimum forces you up. Decide:
- Accept the larger notional ($3400 per Monday) — but that's NOT $50/wk DCA, it's $3400/wk
- OR reduce frequency to monthly so the $50 → 0.01 lot ratio makes more sense
- OR keep $50 weekly knowing it'll skip most weeks (`action=lot_below_min`)

**Decision needed from you** — the current default will SKIP every week because the lot math floors to zero. Two practical options:

1. **Set `buy_amount_usd: 3400`** in `config.yaml` so each buy maps cleanly to 0.01 lot (~real-world $50/week is just too small for gold lots).
2. **Set `buy_amount_usd: 50` and `min_lot: 0.01`** as today — bot will skip until you bump it.

We can adjust this together once you see how it behaves.

---

## Rollback (if you want SMC back)

```powershell
nssm stop psp_bot_dca
nssm start psp_bot_smc
```

Existing DCA-accumulated gold positions stay open (HODL). They are accounted for by magic 20260603 and don't conflict with SMC.

---

## Next time we revisit

- After 8–12 weekly buys, plot accumulated lots vs market price → confirm we're catching dips and not just chasing.
- If/when we want a sell rule: change `sell_mode` from `"never"` to one of `profit_take_pct`, `trailing_pct`, `rebalance` — the loader validates the enum and the bot can be extended to handle them.
- The SMC bot can be re-enabled at any time; the algo code is preserved.
