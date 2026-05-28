"""
One-shot trial trade on XM demo GOLD.i#. Validates the full execution path.

What this does:
  1. Connects to MT5 (terminal must be logged in)
  2. Reads current ATR(14) from the 15m series
  3. Computes SL = entry ± 1.5*ATR, TP = entry ± 2.5*ATR (same as bot)
  4. Sizes position at 0.5% risk of current equity (deliberately small)
  5. Sends ONE market order with SL/TP attached
  6. Fires Telegram alert
  7. Exits — does NOT loop

The trade has magic number 99999 (different from the live bot's 20260522),
so the live bot will not consider this its own position and will not try to
manage it. You can let it run to SL/TP, or close it manually in MT5.

Usage (on the VPS, in PowerShell):
    python C:\\bot\\trial_trade.py BUY
    python C:\\bot\\trial_trade.py SELL
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")


SYMBOL = "GOLD.i#"
MAGIC = 99999                  # NOT the bot's magic (20260522)
RISK_PCT = 0.005               # 0.5% of equity for the trial
K_SL = 1.5
K_TP = 2.5
ATR_PERIOD = 14

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")


def tg_send(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT:
        print("(telegram creds missing; skipping alert)")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"telegram error: {e}")


def compute_atr(rates) -> float:
    """Wilder's ATR(14) on a numpy structured array from copy_rates_from_pos."""
    df = pd.DataFrame(rates)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean().iloc[-1])


def pick_filling_mode(sym) -> int:
    fm = sym.filling_mode
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_IOC


def round_to_step(v: float, step: float) -> float:
    return round(round(v / step) * step, 8)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1].upper() not in ("BUY", "SELL"):
        print("Usage: python trial_trade.py BUY|SELL")
        return 2
    side = sys.argv[1].upper()

    if not mt5.initialize():
        print(f"FAIL: mt5.initialize() — {mt5.last_error()}")
        return 1

    info = mt5.account_info()
    if info is None:
        print("FAIL: no account_info (is MT5 logged in?)")
        mt5.shutdown(); return 1

    sym = mt5.symbol_info(SYMBOL)
    if sym is None or not mt5.symbol_select(SYMBOL, True):
        print(f"FAIL: cannot select {SYMBOL}")
        mt5.shutdown(); return 1

    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("FAIL: no tick"); mt5.shutdown(); return 1

    # ATR(14) from last 100 bars of M15
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) < ATR_PERIOD + 5:
        print("FAIL: not enough bars"); mt5.shutdown(); return 1
    atr_val = compute_atr(rates)

    # Entry, SL, TP
    if side == "BUY":
        entry = tick.ask
        sl = entry - K_SL * atr_val
        tp = entry + K_TP * atr_val
        otype = mt5.ORDER_TYPE_BUY
    else:
        entry = tick.bid
        sl = entry + K_SL * atr_val
        tp = entry - K_TP * atr_val
        otype = mt5.ORDER_TYPE_SELL

    stop_distance = abs(entry - sl)
    equity = float(info.equity)
    risk_usd = equity * RISK_PCT
    oz = risk_usd / stop_distance
    lots_raw = oz / sym.trade_contract_size
    lots = round_to_step(lots_raw, sym.volume_step)
    lots = max(lots, sym.volume_min)
    lots = min(lots, sym.volume_max)

    print(f"--- Trial trade plan ---")
    print(f"  side           : {side}")
    print(f"  symbol         : {SYMBOL}")
    print(f"  bid/ask        : {tick.bid:.2f} / {tick.ask:.2f}")
    print(f"  ATR(14)        : {atr_val:.2f}")
    print(f"  entry          : {entry:.2f}")
    print(f"  SL             : {sl:.2f}  (distance {stop_distance:.2f})")
    print(f"  TP             : {tp:.2f}")
    print(f"  equity         : ${equity:,.2f}")
    print(f"  risk           : ${risk_usd:.2f} ({RISK_PCT*100:.1f}%)")
    print(f"  lots           : {lots}  (= {lots * sym.trade_contract_size:.1f} oz)")
    print(f"  magic          : {MAGIC}  (bot uses 20260522 — won't touch this)")
    print()

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lots,
        "type": otype,
        "price": entry,
        "sl": round(sl, sym.digits),
        "tp": round(tp, sym.digits),
        "deviation": 20,
        "magic": MAGIC,
        "comment": "trial_trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": pick_filling_mode(sym),
    }
    print(f"Sending order…")
    result = mt5.order_send(request)
    if result is None:
        print(f"FAIL: order_send returned None — {mt5.last_error()}")
        mt5.shutdown(); return 1
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"FAIL: retcode={result.retcode} comment='{result.comment}'")
        # common: 10030 unsupported filling mode, 10018 market closed, 10006 rejected
        tg_send(f"<b>[TRIAL]</b> Order REJECTED\n"
                f"retcode={result.retcode}\ncomment={result.comment}")
        mt5.shutdown(); return 1

    fill_px = result.price
    deal = result.deal
    order = result.order

    print(f"FILLED  order={order}  deal={deal}  fill_price={fill_px:.2f}")
    print(f"\nCheck MT5 'Trade' tab — you should see this position with magic 99999.")
    print(f"It will close automatically on SL or TP, or you can right-click → Close.")

    tg_send(
        f"<b>[TRIAL]</b> {side} GOLD (XM demo)\n"
        f"Fill: {fill_px:.2f}\n"
        f"SL: {sl:.2f}  TP: {tp:.2f}\n"
        f"Lots: {lots}  ATR: {atr_val:.2f}\n"
        f"Risk: {RISK_PCT*100:.1f}%  Equity: ${equity:,.2f}\n"
        f"Magic: {MAGIC} (bot will ignore this)"
    )

    mt5.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
