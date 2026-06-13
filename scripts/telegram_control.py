"""
Telegram control center -- query + control the bots from your phone.

A long-running service (NSSM: psp_bot_telegram) that long-polls Telegram for
commands and answers from live MT5 + the shared control flag. It does NOT place
trades; the worst it can do is PAUSE new entries (open positions keep their
server-side SL/TP) or restart a service.

Commands (only the configured chat_id is honored -- everyone else is ignored):
  /status        services up? paused? equity, drawdown, session, regime
  /pnl           realized P&L today + this week, per bot (from MT5 deal history)
  /positions     open positions across all bots (live from MT5)
  /pause         block NEW entries on every bot (sets data/.control.json)
  /resume        clear the pause
  /risk          current risk config + live drawdown tier
  /logs [bot]    last lines of a bot's log (smc|breakout|telegram)
  /restart <bot> nssm restart a service (smc|breakout|dashboard)
  /help          this list

Also pushes, unprompted:
  - daily summary at reporting.daily_summary_hour_ist
  - drawdown alerts when equity DD crosses a new 3/7/12% tier

Run on the VPS as a service (see DEPLOY_TELEGRAM.md). Needs MetaTrader5 +
TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env.
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))

from _config_loader import load_config
from _bot_common import IST, control_paused, control_set, CONTROL_PATH

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # lets the module import on a dev box; main() refuses to run

STATE_PATH = HERE / "data" / ".telegram_control_state.json"
LOGS_DIR = HERE / "logs"
NSSM = HERE / "nssm.exe"

# nssm service names by short alias
SERVICES = {"smc": "psp_bot_smc", "breakout": "psp_bot_breakout",
            "dashboard": "psp_dashboard", "telegram": "psp_bot_telegram"}
LOG_FILES = {"smc": "smc.out.log", "breakout": "breakout.out.log",
             "telegram": "telegram.out.log"}
DD_ALERT_TIERS = [0.12, 0.07, 0.03]   # high -> low; alert on first crossed


# ============================ config / state ========================
def _cfg():
    """Load shared config once. SMC carries the risk + reporting blocks; both
    bots' magics are read so PnL can be attributed per strategy."""
    smc = load_config("smc")
    try:
        bo_magic = load_config("breakout").strategy.magic
    except Exception:
        bo_magic = None
    return smc, {smc.strategy.magic: "smc", bo_magic: "breakout"}


def _load_state() -> dict:
    try:
        return json.load(open(STATE_PATH))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(s: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    json.dump(s, open(tmp, "w"))
    tmp.replace(STATE_PATH)


# ============================ telegram io ===========================
class Bot:
    def __init__(self, token: str, chat_id: str):
        self.api = f"https://api.telegram.org/bot{token}"
        self.chat_id = str(chat_id)

    def send(self, text: str) -> None:
        try:
            requests.post(f"{self.api}/sendMessage",
                          data={"chat_id": self.chat_id, "text": text,
                                "parse_mode": "HTML"}, timeout=15)
        except Exception as e:
            print(f"[tg] send failed: {e}", flush=True)

    def poll(self, offset: int | None) -> list:
        try:
            r = requests.get(f"{self.api}/getUpdates",
                             params={"offset": offset, "timeout": 30},
                             timeout=40)
            return r.json().get("result", []) if r.ok else []
        except Exception as e:
            print(f"[tg] poll failed: {e}", flush=True)
            time.sleep(5)
            return []


# ============================ MT5 queries ===========================
def _equity_balance():
    info = mt5.account_info()
    if not info:
        return None, None
    return float(info.equity), float(info.balance)


def _positions(symbol: str | None = None):
    pos = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    return list(pos) if pos else []


def _deals_pnl(since: datetime, magic_to_bot: dict) -> dict:
    """Sum realized P&L (profit+swap+commission) of closing deals since `since`,
    bucketed by bot via magic. Returns {bot: {pnl, n}}."""
    deals = mt5.history_deals_get(since, datetime.now(timezone.utc) + timedelta(minutes=5))
    out: dict = {}
    for d in (deals or []):
        # entry==1 (DEAL_ENTRY_OUT) marks a position close; that deal carries
        # the realized profit. Skip deposits/balance ops (no position).
        if getattr(d, "entry", None) != 1:
            continue
        bot = magic_to_bot.get(getattr(d, "magic", None), "other")
        pnl = float(d.profit) + float(getattr(d, "swap", 0.0)) + float(getattr(d, "commission", 0.0))
        b = out.setdefault(bot, {"pnl": 0.0, "n": 0})
        b["pnl"] += pnl
        b["n"] += 1
    return out


def _service_status(service: str) -> str:
    if not NSSM.exists():
        return "?"
    try:
        r = subprocess.run([str(NSSM), "status", service],
                           capture_output=True, text=True, timeout=10)
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else "?"
    except Exception:
        return "?"


# ============================ commands ==============================
def _utc_midnight() -> datetime:
    n = datetime.now(timezone.utc)
    return n.replace(hour=0, minute=0, second=0, microsecond=0)


def _week_start() -> datetime:
    m = _utc_midnight()
    return m - timedelta(days=m.weekday())   # Monday 00:00 UTC


def cmd_status(cfg, magic_to_bot) -> str:
    eq, bal = _equity_balance()
    st = _load_state()
    peak = max(st.get("peak_equity", 0.0) or 0.0, eq or 0.0)
    dd = ((peak - eq) / peak * 100) if (peak and eq) else 0.0
    paused, why = control_paused()
    lines = ["<b>STATUS</b>"]
    for alias in ("smc", "breakout"):
        lines.append(f"  {alias}: {_service_status(SERVICES[alias])}")
    lines.append(f"  trading: {'PAUSED' if paused else 'ACTIVE'}"
                 + (f" ({html.escape(why)})" if paused else ""))
    if eq is not None:
        lines.append(f"  equity: ${eq:,.2f}  balance: ${bal:,.2f}")
        lines.append(f"  drawdown: {dd:.2f}% from peak ${peak:,.2f}")
    else:
        lines.append("  equity: MT5 not connected")
    lines.append(f"  open positions: {len(_positions())}")
    return "\n".join(lines)


def cmd_pnl(cfg, magic_to_bot) -> str:
    today = _deals_pnl(_utc_midnight(), magic_to_bot)
    week = _deals_pnl(_week_start(), magic_to_bot)

    def fmt(bucket):
        if not bucket:
            return "  none"
        rows = []
        tot = 0.0
        for bot, v in sorted(bucket.items()):
            rows.append(f"  {bot}: ${v['pnl']:+.2f} ({v['n']} trades)")
            tot += v["pnl"]
        rows.append(f"  <b>total: ${tot:+.2f}</b>")
        return "\n".join(rows)

    return ("<b>P&amp;L today (UTC)</b>\n" + fmt(today)
            + "\n<b>P&amp;L this week</b>\n" + fmt(week))


def cmd_positions(cfg, magic_to_bot) -> str:
    pos = _positions()
    if not pos:
        return "No open positions."
    lines = ["<b>OPEN POSITIONS</b>"]
    for p in pos:
        side = "BUY" if p.type == 0 else "SELL"
        bot = magic_to_bot.get(getattr(p, "magic", None), "?")
        lines.append(f"  {bot} {side} {p.volume} @ {p.price_open:.2f}  "
                     f"SL {p.sl:.2f} TP {p.tp:.2f}  P&amp;L ${p.profit:+.2f}")
    return "\n".join(lines)


def cmd_pause(cfg, magic_to_bot) -> str:
    control_set(True, by="telegram")
    return ("Trading PAUSED. No NEW entries on any bot. Open positions keep "
            "their SL/TP and are still managed. Send /resume to re-enable.")


def cmd_resume(cfg, magic_to_bot) -> str:
    control_set(False, by="telegram")
    return "Trading RESUMED. Bots may open new entries again (gates still apply)."


def cmd_risk(cfg, magic_to_bot) -> str:
    r = cfg.risk
    eq, _ = _equity_balance()
    st = _load_state()
    peak = max(st.get("peak_equity", 0.0) or 0.0, eq or 0.0)
    dd = ((peak - eq) / peak) if (peak and eq) else 0.0
    tier_mult = 1.0
    for t in sorted(getattr(r, "dd_tiers", []) or [], key=lambda x: -x.threshold_pct):
        if dd >= t.threshold_pct:
            tier_mult = t.multiplier
            break
    return ("<b>RISK</b>\n"
            f"  per-trade: {r.risk_per_trade_pct*100:.2f}%\n"
            f"  daily cap: {r.daily_loss_cap_pct*100:.1f}%\n"
            f"  max DD halt: {r.max_drawdown_pct*100:.0f}%\n"
            f"  max concurrent: {r.max_concurrent_positions}/bot\n"
            f"  live drawdown: {dd*100:.2f}%  -> risk x{tier_mult:.2f}")


def cmd_logs(cfg, magic_to_bot, arg: str) -> str:
    alias = (arg or "smc").strip().lower()
    fname = LOG_FILES.get(alias)
    if not fname:
        return f"Unknown log '{alias}'. Try: {', '.join(LOG_FILES)}"
    fp = LOGS_DIR / fname
    if not fp.exists():
        return f"No log file at {fp}"
    try:
        tail = fp.read_text(errors="replace").splitlines()[-15:]
    except Exception as e:
        return f"Could not read log: {e}"
    return f"<b>{alias} (last 15)</b>\n<pre>{html.escape(chr(10).join(tail))}</pre>"


def cmd_restart(cfg, magic_to_bot, arg: str) -> str:
    alias = (arg or "").strip().lower()
    if alias not in SERVICES or alias == "telegram":
        return ("Usage: /restart smc | breakout | dashboard "
                "(can't restart myself)")
    if not NSSM.exists():
        return f"nssm.exe not found at {NSSM}"
    try:
        subprocess.run([str(NSSM), "restart", SERVICES[alias]],
                       capture_output=True, text=True, timeout=30)
        time.sleep(4)
        return f"Restarted {SERVICES[alias]} -> {_service_status(SERVICES[alias])}"
    except Exception as e:
        return f"Restart failed: {e}"


HELP = (
    "<b>COMMANDS</b>\n"
    "/status - services, pause, equity, drawdown\n"
    "/pnl - realized P&amp;L today + this week\n"
    "/positions - open positions\n"
    "/pause - block new entries\n"
    "/resume - re-enable entries\n"
    "/risk - risk config + live DD tier\n"
    "/logs [smc|breakout|telegram]\n"
    "/restart &lt;smc|breakout|dashboard&gt;\n"
    "/help - this list"
)


def handle(text: str, cfg, magic_to_bot) -> str:
    parts = text.strip().split(maxsplit=1)
    cmd = parts[0].lower().lstrip("/").split("@")[0]   # tolerate /cmd@BotName
    arg = parts[1] if len(parts) > 1 else ""
    table = {
        "status": lambda: cmd_status(cfg, magic_to_bot),
        "pnl": lambda: cmd_pnl(cfg, magic_to_bot),
        "positions": lambda: cmd_positions(cfg, magic_to_bot),
        "pause": lambda: cmd_pause(cfg, magic_to_bot),
        "resume": lambda: cmd_resume(cfg, magic_to_bot),
        "risk": lambda: cmd_risk(cfg, magic_to_bot),
        "logs": lambda: cmd_logs(cfg, magic_to_bot, arg),
        "restart": lambda: cmd_restart(cfg, magic_to_bot, arg),
        "help": lambda: HELP,
        "start": lambda: HELP,
    }
    fn = table.get(cmd)
    if not fn:
        return f"Unknown command /{html.escape(cmd)}. Send /help."
    try:
        return fn()
    except Exception as e:
        return f"Command /{cmd} errored: {html.escape(str(e))}"


# ====================== unprompted notifications ====================
def maybe_daily_summary(bot: Bot, cfg, magic_to_bot, st: dict) -> None:
    now_ist = datetime.now(IST)
    want = (now_ist.hour == cfg.reporting.daily_summary_hour_ist
            and now_ist.minute >= cfg.reporting.daily_summary_minute_ist)
    today_key = now_ist.strftime("%Y-%m-%d")
    if want and st.get("last_summary_date") != today_key:
        bot.send("<b>DAILY SUMMARY</b>\n" + cmd_status(cfg, magic_to_bot)
                 + "\n" + cmd_pnl(cfg, magic_to_bot))
        st["last_summary_date"] = today_key
        _save_state(st)


def maybe_dd_alert(bot: Bot, st: dict) -> None:
    eq, _ = _equity_balance()
    if not eq:
        return
    peak = max(st.get("peak_equity", 0.0) or 0.0, eq)
    st["peak_equity"] = peak
    dd = (peak - eq) / peak if peak else 0.0
    crossed = next((t for t in DD_ALERT_TIERS if dd >= t), None)
    last = st.get("last_dd_alert_tier")
    if crossed and crossed != last:
        bot.send(f"DRAWDOWN ALERT: equity ${eq:,.2f} is {dd*100:.1f}% below "
                 f"peak ${peak:,.2f} (crossed {crossed*100:.0f}% tier).")
        st["last_dd_alert_tier"] = crossed
    elif crossed is None and last is not None:
        st["last_dd_alert_tier"] = None   # recovered above 3%
    _save_state(st)


# ============================== main ================================
def main() -> int:
    import os
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("FATAL: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in .env", flush=True)
        return 1
    if mt5 is None:
        print("FATAL: MetaTrader5 not installed (run on the VPS)", flush=True)
        return 1

    from _bot_common import init_mt5_headless
    if not init_mt5_headless():
        print("FATAL: MT5 init failed", flush=True)
        return 1

    cfg, magic_to_bot = _cfg()
    bot = Bot(token, chat_id)
    st = _load_state()
    offset = st.get("offset")
    print(f"[telegram] control center up. chat_id={chat_id} "
          f"magics={magic_to_bot} control_flag={CONTROL_PATH}", flush=True)
    bot.send("Control center online. Send /help for commands.")

    while True:
        for upd in bot.poll(offset):
            offset = upd["update_id"] + 1
            st["offset"] = offset
            _save_state(st)
            msg = upd.get("message") or upd.get("edited_message") or {}
            if str(msg.get("chat", {}).get("id")) != bot.chat_id:
                continue   # auth: ignore everyone but the owner
            text = msg.get("text", "")
            if not text.startswith("/"):
                continue
            print(f"[telegram] cmd: {text}", flush=True)
            bot.send(handle(text, cfg, magic_to_bot))

        # heartbeat-driven pushes (also runs between polls since poll blocks 30s)
        try:
            maybe_dd_alert(bot, st)
            maybe_daily_summary(bot, cfg, magic_to_bot, st)
        except Exception as e:
            print(f"[telegram] notify loop error: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
