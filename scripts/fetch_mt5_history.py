"""
Phase B.1 — Pull multi-year XAUUSD history from MT5 for backtesting.

MT5's copy_rates_from_pos() is limited to ~100k bars per request on most
brokers. We paginate backwards from "now" in chunks, deduplicate, and save
to parquet. Also extracts broker symbol specs (spread, contract size,
commission hints) — needed for realistic backtest cost modeling.

Run on VPS where MT5 is connected:
    python scripts/fetch_mt5_history.py --years 4
    python scripts/fetch_mt5_history.py --years 4 --symbol "GOLD.i#"
    python scripts/fetch_mt5_history.py --years 4 --timeframes M15 H1 H4

Outputs:
    data/history/{symbol}_M15.parquet
    data/history/{symbol}_H1.parquet
    data/history/{symbol}_H4.parquet
    data/history/{symbol}_specs.json     <- contract specs for backtest cost model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))


# ============================== MT5 INIT ============================
def init_mt5():
    import MetaTrader5 as mt5
    from dotenv import load_dotenv
    import os

    load_dotenv(HERE / ".env")
    path = os.getenv("MT5_PATH")
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if path and login:
        ok = mt5.initialize(
            path=path, login=int(login),
            password=password, server=server, timeout=60000,
        )
    else:
        ok = mt5.initialize()
    if not ok:
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")
    return mt5


# ============================== FETCH ===============================
TIMEFRAME_MAP = {
    "M1":  ("TIMEFRAME_M1",  60),
    "M5":  ("TIMEFRAME_M5",  300),
    "M15": ("TIMEFRAME_M15", 900),
    "M30": ("TIMEFRAME_M30", 1800),
    "H1":  ("TIMEFRAME_H1",  3600),
    "H4":  ("TIMEFRAME_H4",  14400),
    "D1":  ("TIMEFRAME_D1",  86400),
}

# Bars per request. Most brokers allow 100k-200k. Stay safe.
CHUNK_SIZE = 50000


def fetch_history(mt5, symbol: str, tf_name: str, years: int) -> pd.DataFrame:
    """Paginate backwards from now, until we have `years` of data or MT5 stops."""
    if tf_name not in TIMEFRAME_MAP:
        raise ValueError(f"unknown timeframe {tf_name}; supported: {list(TIMEFRAME_MAP.keys())}")
    tf_attr, secs_per_bar = TIMEFRAME_MAP[tf_name]
    tf_const = getattr(mt5, tf_attr)

    bars_per_year = (365 * 86400) // secs_per_bar
    target_bars = int(bars_per_year * years * 1.05)   # +5% buffer
    print(f"  [{tf_name}] target bars: ~{target_bars:,} ({years} years)")

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select({symbol}) failed")

    all_chunks = []
    offset = 0
    while offset < target_bars:
        n = min(CHUNK_SIZE, target_bars - offset)
        chunk = mt5.copy_rates_from_pos(symbol, tf_const, offset, n)
        if chunk is None or len(chunk) == 0:
            print(f"  [{tf_name}] MT5 returned no bars at offset {offset:,}; stopping pagination")
            break
        df = pd.DataFrame(chunk)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        all_chunks.append(df)
        offset += len(chunk)
        print(f"  [{tf_name}] +{len(chunk):,} bars (total: {offset:,})", flush=True)
        if len(chunk) < n:
            print(f"  [{tf_name}] MT5 returned fewer bars than requested; reached history limit")
            break
        time.sleep(0.2)   # be polite to broker

    if not all_chunks:
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df = df.set_index("time")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "tick_volume": "Volume", "spread": "Spread", "real_volume": "RealVolume",
    })
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Spread", "RealVolume"]
                 if c in df.columns]
    return df[keep_cols]


def fetch_specs(mt5, symbol: str) -> dict:
    """Snapshot contract specs needed for realistic backtest cost modeling."""
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select({symbol}) failed")
    sym = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if sym is None or tick is None:
        raise RuntimeError("symbol_info or tick is None")

    # Convert MT5 raw swap (swap_long/swap_short) to USD per 1.0 lot per night.
    # The unit depends on swap_mode (SYMBOL_SWAP_MODE):
    #   1 = POINTS            -> swap_points * point * contract_size (USD, since
    #                            XAUUSD profit currency is USD on a USD account)
    #   2..6 = currency/%-of  -> already a money amount per lot, use as-is
    #   0 = DISABLED / other  -> no swap
    # If the profit currency is not USD this needs an FX conversion we don't do
    # here; for GOLD.i# on this XM USD account the assumption holds.
    swap_mode = int(getattr(sym, "swap_mode", 0))
    point = float(sym.point)
    contract = float(sym.trade_contract_size)

    def _swap_to_usd_per_lot_night(raw: float) -> float:
        raw = float(raw)
        if swap_mode == 1:                       # POINTS
            return raw * point * contract
        if swap_mode in (2, 3, 4, 5, 6):         # money / interest per lot
            return raw
        return 0.0                               # disabled / unknown

    return {
        "symbol": symbol,
        "snapshot_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contract_size": float(sym.trade_contract_size),
        "volume_min": float(sym.volume_min),
        "volume_max": float(sym.volume_max),
        "volume_step": float(sym.volume_step),
        "digits": int(sym.digits),
        "point": float(sym.point),
        "current_bid": float(tick.bid),
        "current_ask": float(tick.ask),
        "current_spread_price": float(tick.ask - tick.bid),
        "current_spread_points": int(round((tick.ask - tick.bid) / sym.point)),
        "stops_level_points": int(sym.trade_stops_level),
        "freeze_level_points": int(sym.trade_freeze_level),
        "swap_mode": swap_mode,
        "swap_long": float(sym.swap_long),       # raw MT5 value (unit per swap_mode)
        "swap_short": float(sym.swap_short),
        "swap_long_usd_per_lot_night": _swap_to_usd_per_lot_night(sym.swap_long),
        "swap_short_usd_per_lot_night": _swap_to_usd_per_lot_night(sym.swap_short),
        # Commission usually NOT in symbol_info — broker-specific. Hardcode our XM
        # Ultra Low gold commission. Update if your account differs.
        "assumed_commission_per_lot_rt_usd": 7.0,
        "assumed_commission_note": "XM Ultra Low gold typical $7/lot round-trip",
    }


# ============================== MAIN ================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="GOLD.i#")
    p.add_argument("--years", type=int, default=4)
    p.add_argument("--timeframes", nargs="+",
                   default=["M15", "H1", "H4"],
                   help="space-separated, e.g. M15 H1 H4")
    p.add_argument("--out-dir", default=str(HERE / "data" / "history"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== MT5 history fetch ===")
    print(f"  symbol:     {args.symbol}")
    print(f"  years:      {args.years}")
    print(f"  timeframes: {args.timeframes}")
    print(f"  out dir:    {out_dir}")
    print()

    mt5 = init_mt5()
    try:
        info = mt5.account_info()
        if info:
            print(f"  account: {info.login} ({info.server}), equity ${info.equity:,.2f}")
        print()

        # Specs
        specs = fetch_specs(mt5, args.symbol)
        specs_path = out_dir / f"{_sanitize(args.symbol)}_specs.json"
        with open(specs_path, "w") as f:
            json.dump(specs, f, indent=2)
        print(f"  specs saved: {specs_path}")
        print(f"    contract_size={specs['contract_size']}, "
              f"min_lot={specs['volume_min']}, "
              f"spread now={specs['current_spread_points']} pts ({specs['current_spread_price']:.4f}), "
              f"commission=${specs['assumed_commission_per_lot_rt_usd']}/lot RT")
        print()

        # Bars
        for tf in args.timeframes:
            print(f"Fetching {tf}...")
            df = fetch_history(mt5, args.symbol, tf, args.years)
            if df.empty:
                print(f"  [{tf}] NO DATA -- skipping save")
                continue
            out_path = out_dir / f"{_sanitize(args.symbol)}_{tf}.parquet"
            df.to_parquet(out_path)
            actual_years = (df.index[-1] - df.index[0]).days / 365.25
            print(f"  [{tf}] saved {out_path}")
            print(f"  [{tf}] span: {df.index[0]} -> {df.index[-1]} ({actual_years:.2f} years, {len(df):,} bars)")
            print()
    finally:
        mt5.shutdown()

    print("=== DONE ===")
    return 0


def _sanitize(s: str) -> str:
    """Make a filename-safe version of a symbol (e.g. 'GOLD.i#' -> 'GOLD_i')."""
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


if __name__ == "__main__":
    sys.exit(main())
