# RUNBOOK — VPS Ops (log rotation, service hygiene)

Operational tasks for the Windows VPS (`163.223.86.49`, flat layout
`C:\ai-trading-bot`). These are run-once / occasional PowerShell tasks, not part
of the git-deploy loop.

## Log rotation — apply to ALL services

NSSM can rotate its own stdout/stderr logs, but rotation must be set per service
and is **not** retroactive — services installed before rotation was configured
keep appending to one unbounded file. The bots heartbeat once a minute, so
`smc.out.log` grows steadily and will eventually fill the disk.

This applies (or re-applies) rotation to every service: rotate when a log passes
**10 MB**, keep the rotated file. Safe to re-run — `nssm set` is idempotent.

```powershell
cd C:\ai-trading-bot

# All bot/service names. Add psp_bot_telegram once it's installed
# (see DEPLOY_TELEGRAM.md); add psp_bot_dca only if/when it's revived.
$services = @("psp_bot_smc", "psp_bot_breakout", "psp_dashboard")

foreach ($svc in $services) {
    # only touch services that actually exist
    $status = (& .\nssm.exe status $svc) 2>$null
    if (-not $status) { Write-Host "skip $svc (not installed)"; continue }

    .\nssm.exe set $svc AppRotateFiles 1
    .\nssm.exe set $svc AppRotateOnline 1          # rotate without restarting
    .\nssm.exe set $svc AppRotateBytes 10485760    # 10 MB
    Write-Host "rotation set on $svc"
}
```

`AppRotateFiles 1` enables rotation; `AppRotateBytes` is the threshold;
`AppRotateOnline 1` lets NSSM rotate a live log without bouncing the service.
NSSM checks the size when it next writes, so rotation kicks in on the next log
line after the file crosses 10 MB. No restart required for the setting to take —
but a restart forces an immediate rotation if you want a clean cut now.

### Verify what's set on a service

```powershell
.\nssm.exe get psp_bot_smc AppRotateFiles
.\nssm.exe get psp_bot_smc AppRotateBytes
.\nssm.exe get psp_bot_smc AppRotateOnline
```

### One-off: truncate a log that's already huge

Rotation only acts going forward. If a log is already multi-GB, rotate it now by
restarting the service (NSSM rotates on startup when enabled), or manually:

```powershell
# keep the last 2000 lines, drop the rest
$f = "C:\ai-trading-bot\logs\smc.out.log"
Get-Content $f -Tail 2000 | Set-Content "$f.trim"; Move-Item "$f.trim" $f -Force
.\nssm.exe restart psp_bot_smc    # reopen the file handle cleanly
```

## Disk check

```powershell
Get-ChildItem C:\ai-trading-bot\logs\ | Sort-Object Length -Descending |
    Select-Object Name, @{N="MB";E={[math]::Round($_.Length/1MB,1)}} -First 10
Get-PSDrive C | Select-Object Used, Free
```

## Service status sweep

```powershell
cd C:\ai-trading-bot
foreach ($svc in @("psp_bot_smc","psp_bot_breakout","psp_dashboard","psp_bot_telegram")) {
    "{0,-20} {1}" -f $svc, ((& .\nssm.exe status $svc) 2>$null)
}
```

## Restart gotcha (recurring)

`nssm restart` sometimes reports `SERVICE_STOP_PENDING` and leaves the service
**stopped** rather than restarting it (seen on `psp_bot_breakout`). Always check
status after a restart and explicitly start if needed:

```powershell
.\nssm.exe restart psp_bot_smc
Start-Sleep -Seconds 5
if ((& .\nssm.exe status psp_bot_smc) -ne "SERVICE_RUNNING") { .\nssm.exe start psp_bot_smc }
```
