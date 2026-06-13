# DEPLOY — Telegram Control Center (`psp_bot_telegram`)

A long-running service that lets you query and control the bots from your phone.
It reuses the existing `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` in `.env` — no
new credentials. Only messages from your `TELEGRAM_CHAT_ID` are honored; everyone
else is silently ignored.

It does **not** place trades. The strongest thing it can do is `/pause` (block
NEW entries — open positions keep their server-side SL/TP) or `/restart` a
service.

## What it can do

| Command | Effect |
|---|---|
| `/status` | service states, paused?, equity, drawdown, open-position count |
| `/pnl` | realized P&L today + this week, per bot (from MT5 deal history) |
| `/positions` | live open positions across all bots |
| `/pause` | sets `data/.control.json` → both bots refuse new entries |
| `/resume` | clears the pause |
| `/risk` | risk config + live drawdown tier multiplier |
| `/logs [smc\|breakout\|telegram]` | last 15 log lines |
| `/restart <smc\|breakout\|dashboard>` | `nssm restart` that service |
| `/help` | command list |

It also pushes a **daily summary** at `reporting.daily_summary_hour_ist` and a
**drawdown alert** when equity crosses a new 3% / 7% / 12% tier below peak.

## How `/pause` reaches the bots

`telegram_control.py` writes `data/.control.json`. Both `mt5_smc.py` and
`mt5_live.py` call `control_paused()` at the top of `can_open_new_trade()`, so a
pause takes effect on each bot's **next evaluation cycle** (≤ its poll interval).
No restart needed. The flag is fail-open: a missing/corrupt file = not paused.

## Install on the VPS (RDP to 163.223.86.49)

The bots already deploy via `git pull`; this service just needs registering once.

```powershell
cd C:\ai-trading-bot
git pull

# adjust if your python lives elsewhere — check with: where.exe python
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"
$script = "C:\ai-trading-bot\scripts\telegram_control.py"
$cwd    = "C:\ai-trading-bot"

.\nssm.exe install psp_bot_telegram $python $script
.\nssm.exe set psp_bot_telegram AppDirectory $cwd
.\nssm.exe set psp_bot_telegram AppStdout "$cwd\logs\telegram.out.log"
.\nssm.exe set psp_bot_telegram AppStderr "$cwd\logs\telegram.err.log"
.\nssm.exe set psp_bot_telegram AppRotateFiles 1
.\nssm.exe set psp_bot_telegram AppRotateBytes 10485760     # 10 MB
.\nssm.exe set psp_bot_telegram AppEnvironmentExtra "PYTHONUNBUFFERED=1" "PYTHONIOENCODING=utf-8"
.\nssm.exe set psp_bot_telegram Start SERVICE_AUTO_START
.\nssm.exe start psp_bot_telegram
.\nssm.exe status psp_bot_telegram     # want SERVICE_RUNNING
```

Then on your phone, message the bot `/help`. You should get the command list and
a "Control center online" greeting in the log.

```powershell
Get-Content C:\ai-trading-bot\logs\telegram.out.log -Wait -Tail 30
```

## Updating it later

It's plain `git pull` + restart, same as the bots:

```powershell
cd C:\ai-trading-bot
git pull
.\nssm.exe restart psp_bot_telegram
```

## Notes / gotchas

- **One Telegram bot token, one consumer.** `getUpdates` long-polling and the
  bots' fire-and-forget `tg_send` alerts share the token fine (sending and
  receiving are independent). But do NOT run a second `getUpdates` poller on the
  same token — Telegram returns a 409 conflict. There's only this one poller.
- **MT5 multi-connect.** This is a 3rd Python process attaching to the same MT5
  terminal (alongside smc + breakout). MT5 supports that; it's read-only here
  (account_info / positions_get / history_deals_get).
- **`/restart` can't restart `psp_bot_telegram`** (it would kill the process
  mid-reply). Restart it from PowerShell instead.
- **Stop/remove:** `.\nssm.exe stop psp_bot_telegram` / `.\nssm.exe remove
  psp_bot_telegram confirm`.
- ASCII-only prints (cp1252 console); `PYTHONIOENCODING=utf-8` is set in the
  service env as a belt-and-suspenders for any stray non-ASCII.
