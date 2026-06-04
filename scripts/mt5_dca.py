"""
Phase H — DCA (Dollar-Cost-Averaging) gold buyer.

Why this exists (the receipts):
  After four algorithmic strategies failed to beat buy-and-hold gold on the
  4-year MT5 backtest, and the meta-labeler couldn't separate signal from noise
  on a 1,350-sample combined dataset (val AUC 0.58), we accepted the math:
  for someone with a long-term gold thesis and no demonstrated short-term edge,
  scheduled accumulation IS the strategy.

How it works:
  Every Monday at London open (12:30 IST), buy a fixed USD notional of gold
  at market. NO stop loss. NO take profit. Pure accumulation. Hard caps:
    - max_lot_per_buy   — never accidentally buy too large a lot
    - max_total_lots    — refuse new buys once cumulative open lots exceed this
    - max_buys_per_day  — defense against a scheduler retry / restart glitch

State:
  data/.dca_state.json tracks last_buy_at_utc. On restart, the bot rechecks
  MT5 for open positions by magic + queries the journal CSV — defense in depth.

Service:
  Designed to run as Windows NSSM service `psp_bot_dca` alongside (or instead
  of) the SMC bot. See RUNBOOK.md for the cutover.

Run locally for a smoke test (won't actually trade unless MT5 is connected):
    python scripts/mt5_dca.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# project-local imports
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from _config_loader import load_config, ConfigError, DCAParams
from _bot_common import (
    IST,
    init_mt5_headless,
    check_mt5_alive_or_reconnect,
    reset_mt5_failure_counter,
    tg_send,
)
import _journal as J


BOT_NAME = "dca"
STATE_FILE = ROOT / "data" / ".dca_state.json"


# ============================== STATE ===============================
def _state_load() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _state_save(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except Exception as e:
        print(f"[dca] state save failed: {e}", flush=True)


# ============================== CRON ================================
# Tiny dependency-free cron matcher. Supports only:
#   '*'                       any value
#   single integer            exact match
#   comma-separated integers  any of
# Sufficient for DCA schedules. NOT a full cron implementation.
def _cron_field_matches(field: str, value: int) -> bool:
    field = field.strip()
    if field == "*":
        return True
    parts = [p.strip() for p in field.split(",") if p.strip()]
    try:
        ints = {int(p) for p in parts}
    except ValueError:
        return False
    return value in ints


def cron_fire_window(now_ist: datetime, cron: str,
                     window_minutes: int) -> tuple[bool, datetime | None]:
    """Given an IST timestamp and a cron expression, return (in_window, target_dt).

    target_dt is the most recent scheduled minute that COULD have fired
    (today's match at hour:minute), regardless of whether now is past it.

    in_window is True iff:
      - today matches the cron's day-of-month, month, day-of-week fields
      - now_ist is within [target, target + window_minutes]
    """
    try:
        m, h, dom, mon, dow = cron.split()
    except ValueError:
        return False, None

    # Build today's target time in IST
    try:
        target_min = int(m)
        target_hr = int(h)
    except ValueError:
        return False, None

    target = now_ist.replace(hour=target_hr, minute=target_min, second=0, microsecond=0)

    # Date-level cron checks
    if not _cron_field_matches(dom, now_ist.day):
        return False, target
    if not _cron_field_matches(mon, now_ist.month):
        return False, target
    # cron dow: 0 or 7 = Sunday, 1 = Monday, ... ISO weekday() returns Mon=0..Sun=6
    iso_dow = now_ist.weekday()              # Mon=0 ... Sun=6
    cron_dow = (iso_dow + 1) % 7             # convert to cron convention
    if not (_cron_field_matches(dow, cron_dow) or
            (cron_dow == 0 and _cron_field_matches(dow, 7))):
        return False, target

    if now_ist < target:
        return False, target

    if (now_ist - target).total_seconds() <= window_minutes * 60:
        return True, target

    return False, target


# ============================== LOT MATH ============================
def round_down_to_step(x: float, step: float) -> float:
    """Round down to the nearest `step`. e.g. (0.0147, 0.01) -> 0.01."""
    if step <= 0:
        return x
    return (int(x / step)) * step


def compute_buy_lots(price: float, contract_size: float,
                      amount_usd: float, params: DCAParams) -> tuple[float, str]:
    """Compute lot size for a target USD notional. Returns (lots, reason).

    `lots` may be 0 if the math floors below min_lot — caller should skip.
    """
    if price <= 0 or contract_size <= 0:
        return 0.0, f"bad inputs price={price} contract_size={contract_size}"

    raw_lots = amount_usd / (price * contract_size)
    # Cap and round
    capped = min(raw_lots, params.max_lot_per_buy)
    rounded = round_down_to_step(capped, params.lot_step)

    if rounded < params.min_lot:
        return 0.0, (f"computed {raw_lots:.4f} lot -> rounded {rounded:.4f} "
                     f"below min_lot={params.min_lot} (price ${price:.2f})")
    if rounded < raw_lots * 0.5 and rounded == params.lot_step:
        # User wanted more than 2x what the min step gave — note in reason
        return rounded, (f"want={raw_lots:.4f} cap={capped:.4f} -> {rounded:.4f} "
                         f"(rounded down to step; ~${rounded * price * contract_size:.0f})")
    return rounded, f"want={raw_lots:.4f} cap={capped:.4f} -> {rounded:.4f}"


# ============================== MT5 HELPERS =========================
def _open_positions_for_magic(symbol: str, magic: int) -> list:
    import MetaTrader5 as mt5
    try:
        positions = mt5.positions_get(symbol=symbol) or []
    except Exception:
        return []
    return [p for p in positions if getattr(p, "magic", None) == magic]


def total_lots_open(symbol: str, magic: int) -> float:
    return sum(float(p.volume) for p in _open_positions_for_magic(symbol, magic))


def _symbol_info(symbol: str):
    import MetaTrader5 as mt5
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        if not info.visible:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        return info
    except Exception:
        return None


def _tick(symbol: str):
    import MetaTrader5 as mt5
    try:
        return mt5.symbol_info_tick(symbol)
    except Exception:
        return None


def open_market_buy_no_sltp(symbol: str, lots: float, magic: int,
                             comment: str = "dca") -> tuple[bool, str, dict]:
    """Open a market BUY with NO stop loss or take profit.

    Returns (ok, reason, result_dict). result_dict includes ticket, entry, ts.
    """
    import MetaTrader5 as mt5
    tick = _tick(symbol)
    info = _symbol_info(symbol)
    if tick is None or info is None:
        return False, "no tick/info from MT5", {}

    # Use ASK for BUY
    price = float(tick.ask)
    if price <= 0:
        return False, f"invalid ask price {price}", {}

    # Some brokers require explicit deviation/filling mode
    filling = info.filling_mode
    # filling_mode is a bitmask; pick a supported one
    if filling & mt5.ORDER_FILLING_IOC:
        filling_type = mt5.ORDER_FILLING_IOC
    elif filling & mt5.ORDER_FILLING_FOK:
        filling_type = mt5.ORDER_FILLING_FOK
    else:
        filling_type = mt5.ORDER_FILLING_RETURN

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 20,             # 20 points of slippage tolerance
        "magic": magic,
        "comment": comment[:31],     # MT5 comment limit
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }
    result = mt5.order_send(req)
    if result is None:
        return False, "order_send returned None", {}
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"retcode={result.retcode} ({getattr(result, 'comment', '')})", {
            "retcode": result.retcode,
            "comment": getattr(result, "comment", ""),
        }
    return True, "ok", {
        "ticket": int(result.order),
        "entry": float(result.price or price),
        "lots": float(result.volume or lots),
        "ts_utc": datetime.now(timezone.utc),
    }


# ============================== JOURNAL =============================
DCA_CSV_FIELDS = [
    "trade_id", "ts_utc", "symbol", "side", "lots", "entry",
    "ticket", "amount_usd_target", "amount_usd_actual",
    "price_at_buy", "contract_size", "total_lots_after", "magic", "note",
]


def journal_csv_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    try:
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DCA_CSV_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in DCA_CSV_FIELDS})
    except Exception as e:
        print(f"[dca] csv append failed: {e}", flush=True)


# ============================== CORE TICK ===========================
def consider_buy(cfg, state: dict, dry_run: bool = False) -> dict:
    """One evaluation step. Returns dict describing what happened."""
    import MetaTrader5 as mt5

    s: DCAParams = cfg.strategy
    if not s.enabled:
        return {"action": "disabled"}

    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc.astimezone(IST)

    in_window, target = cron_fire_window(now_ist, s.schedule_cron_ist,
                                          s.fire_window_minutes)
    if not in_window:
        return {"action": "outside_window", "next_target_ist": target}

    # Idempotency: have we already fired for this target?
    last_buy_at = state.get("last_buy_at_utc")
    if last_buy_at:
        try:
            last_dt = datetime.fromisoformat(str(last_buy_at).replace("Z", "+00:00"))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            # target is IST naive timezone — convert to compare
            target_utc = target.astimezone(timezone.utc) if target.tzinfo else \
                          target.replace(tzinfo=IST).astimezone(timezone.utc)
            if last_dt >= target_utc:
                return {"action": "already_fired_for_target", "target": target}
        except Exception:
            pass

    # Defense in depth: count of buys today in journal CSV
    buys_today = _count_buys_today_csv(Path(cfg.journal_csv), now_utc)
    if buys_today >= s.max_buys_per_day:
        return {"action": "max_buys_per_day_reached", "buys_today": buys_today}

    # Pre-trade caps from live MT5
    open_lots = total_lots_open(cfg.mt5.symbol, s.magic)
    if open_lots >= s.max_total_lots:
        return {"action": "max_total_lots_reached", "open_lots": open_lots}

    # Compute lot size
    tick = _tick(cfg.mt5.symbol)
    info = _symbol_info(cfg.mt5.symbol)
    if tick is None or info is None:
        return {"action": "no_market_data"}

    contract_size = float(getattr(info, "trade_contract_size", 100.0) or 100.0)
    price = float(tick.ask)
    lots, why = compute_buy_lots(price, contract_size, s.buy_amount_usd, s)

    # Re-check against max_total_lots AFTER computing lots
    if lots <= 0:
        return {"action": "lot_below_min", "reason": why, "price": price}
    if open_lots + lots > s.max_total_lots:
        # Try to shrink to fit; if still below min_lot, skip
        room = s.max_total_lots - open_lots
        shrunk = round_down_to_step(room, s.lot_step)
        if shrunk < s.min_lot:
            return {"action": "cap_exceeded", "open_lots": open_lots,
                    "wanted": lots, "room": room}
        lots = shrunk
        why += f" (shrunk to fit total cap, lots={lots})"

    if dry_run:
        return {
            "action": "dry_run_would_buy", "lots": lots,
            "price": price, "reason": why, "open_lots_before": open_lots,
            "target_ist": target,
        }

    # FIRE
    ok, reason, result = open_market_buy_no_sltp(
        cfg.mt5.symbol, lots, s.magic,
        comment=f"dca_{now_ist.strftime('%Y%m%d')}",
    )
    if not ok:
        return {"action": "buy_failed", "reason": reason,
                "lots": lots, "price": price}

    # Persist state
    state["last_buy_at_utc"] = result["ts_utc"].isoformat()
    state["last_buy_lots"] = result["lots"]
    state["last_buy_ticket"] = result["ticket"]
    state["total_buys"] = int(state.get("total_buys", 0)) + 1
    _state_save(state)

    # CSV + Postgres
    trade_id = uuid.uuid4().int >> 64
    notional_actual = result["lots"] * price * contract_size
    row = {
        "trade_id": trade_id,
        "ts_utc": result["ts_utc"].isoformat(),
        "symbol": cfg.mt5.symbol,
        "side": "BUY",
        "lots": result["lots"],
        "entry": result["entry"],
        "ticket": result["ticket"],
        "amount_usd_target": s.buy_amount_usd,
        "amount_usd_actual": round(notional_actual, 2),
        "price_at_buy": price,
        "contract_size": contract_size,
        "total_lots_after": round(open_lots + result["lots"], 4),
        "magic": s.magic,
        "note": why,
    }
    journal_csv_append(Path(cfg.journal_csv), row)
    if cfg.postgres_enabled:
        try:
            J.log_event(BOT_NAME, "dca_buy", {
                "trade_id": trade_id,
                "ticket": result["ticket"],
                "lots": result["lots"],
                "entry": result["entry"],
                "notional_usd": round(notional_actual, 2),
                "target_usd": s.buy_amount_usd,
                "total_lots_after": round(open_lots + result["lots"], 4),
                "magic": s.magic,
                "reason": why,
            })
        except Exception as e:
            print(f"[dca] postgres event failed: {e}", flush=True)

    # Telegram
    if s.send_buy_telegram:
        tg_send(
            f"<b>[DCA] BUY GOLD</b>\n"
            f"Lots: <b>{result['lots']:.2f}</b>  @ <b>${result['entry']:.2f}</b>\n"
            f"Notional: ~${notional_actual:.0f} (target ${s.buy_amount_usd:.0f})\n"
            f"Ticket: <code>{result['ticket']}</code>\n"
            f"Total accumulated: <b>{open_lots + result['lots']:.2f} lot</b>\n"
            f"Target time IST: {target.strftime('%Y-%m-%d %H:%M')}\n"
            f"<i>{why}</i>"
        )

    return {"action": "bought", "lots": result["lots"], "ticket": result["ticket"],
            "entry": result["entry"], "notional": notional_actual}


def _count_buys_today_csv(path: Path, now_utc: datetime) -> int:
    if not path.exists():
        return 0
    today_str = now_utc.strftime("%Y-%m-%d")
    try:
        with path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return 0
    n = 0
    for r in rows:
        ts = r.get("ts_utc", "")
        if ts.startswith(today_str):
            n += 1
    return n


# ============================== MAIN LOOP ===========================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't actually send orders; print what would happen.")
    ap.add_argument("--once", action="store_true",
                    help="Evaluate once and exit (useful for testing / cron).")
    args = ap.parse_args()

    try:
        cfg = load_config("dca")
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2

    cfg.print_summary()
    s: DCAParams = cfg.strategy

    if not s.enabled:
        print("[dca] strategy.dca.enabled=false — exiting", flush=True)
        return 0

    # MT5 init
    if not args.dry_run:
        ok = init_mt5_headless()
        if not ok:
            print("[dca] MT5 init FAILED — exiting (NSSM will restart)", flush=True)
            return 1
        print("[dca] MT5 connected", flush=True)

    if cfg.postgres_enabled:
        try:
            J.log_event(BOT_NAME, "startup", {
                "magic": s.magic, "symbol": cfg.mt5.symbol,
                "cron_ist": s.schedule_cron_ist,
                "buy_usd": s.buy_amount_usd,
                "max_total_lots": s.max_total_lots,
                "dry_run": args.dry_run,
            })
        except Exception:
            pass

    state = _state_load()
    print(f"[dca] state: last_buy_at={state.get('last_buy_at_utc')} "
          f"total_buys={state.get('total_buys', 0)}", flush=True)

    poll = max(30, int(cfg.mt5.poll_seconds))
    print(f"[dca] poll interval: {poll}s  | dry_run={args.dry_run}  | "
          f"once={args.once}", flush=True)

    watchdog_state = {"mt5_consec_failures": 0}
    last_heartbeat = 0
    heartbeat_every = 30 * 60   # 30 min

    while True:
        try:
            result = consider_buy(cfg, state, dry_run=args.dry_run)
            action = result.get("action", "?")
            if action == "bought":
                print(f"[dca] BOUGHT {result['lots']:.2f} lot @ "
                      f"${result['entry']:.2f}  ticket={result['ticket']}", flush=True)
                reset_mt5_failure_counter(watchdog_state)
            elif action == "dry_run_would_buy":
                print(f"[dca] DRY RUN would buy {result['lots']:.2f} lot "
                      f"@ ${result['price']:.2f}  -- {result['reason']}", flush=True)
            elif action == "outside_window":
                t = result.get("next_target_ist")
                if t and (time.time() - last_heartbeat) > heartbeat_every:
                    print(f"[dca] outside window. next target IST: "
                          f"{t.strftime('%Y-%m-%d %H:%M')}", flush=True)
                    last_heartbeat = time.time()
            elif action == "already_fired_for_target":
                if (time.time() - last_heartbeat) > heartbeat_every:
                    print(f"[dca] already fired for target {result.get('target')}", flush=True)
                    last_heartbeat = time.time()
            elif action == "no_market_data":
                check_mt5_alive_or_reconnect(watchdog_state)
            elif action in ("max_buys_per_day_reached", "max_total_lots_reached",
                            "cap_exceeded", "lot_below_min"):
                if s.send_skip_telegram:
                    tg_send(f"<b>[DCA SKIP]</b> {action}: {result}")
                print(f"[dca] SKIP: {action}  {result}", flush=True)
            else:
                # disabled / other — log and exit if disabled
                if action == "disabled":
                    print("[dca] disabled at runtime — exiting", flush=True)
                    return 0
                # Otherwise harmless
                pass
        except KeyboardInterrupt:
            print("[dca] interrupted — exiting", flush=True)
            return 0
        except Exception as e:
            # Never crash the loop; NSSM restart is last resort
            print(f"[dca] tick error: {type(e).__name__}: {e}", flush=True)
            if cfg.postgres_enabled:
                try:
                    J.log_event(BOT_NAME, "tick_error", {"err": str(e)[:500]})
                except Exception:
                    pass

        if args.once:
            return 0

        time.sleep(poll)


if __name__ == "__main__":
    sys.exit(main())
