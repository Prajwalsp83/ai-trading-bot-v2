"""
Upstox smoke test.

Proves end-to-end connectivity:
  - reads saved token
  - hits /v2/user/profile     (auth works)
  - hits /v2/user/get-funds-and-margin (account access works)
  - hits /v2/market/timings   (market data works)

If any step fails, the bot won't be able to trade. This is the gate before
we start writing strategy code.

Run:
    cd ~/Documents/ai-trading-bot/v2
    source .venv/bin/activate
    python scripts/upstox_smoke.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent.parent
TOKEN_FILE = HERE / ".upstox_token.json"
BASE = "https://api.upstox.com/v2"


def load_token() -> str:
    if not TOKEN_FILE.exists():
        print(f"ERROR: {TOKEN_FILE.name} not found. Run scripts/upstox_login.py first.",
              file=sys.stderr)
        sys.exit(1)
    return json.loads(TOKEN_FILE.read_text())["access_token"]


def api_get(path: str, token: str) -> dict:
    r = requests.get(
        f"{BASE}{path}",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        timeout=15,
    )
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} from {path}\n  {r.text}", file=sys.stderr)
        return {}
    return r.json()


def fmt_inr(amount: float) -> str:
    # 1,23,456.78 (Indian numbering)
    s = f"{amount:,.2f}"
    return f"₹{s}"


def main() -> int:
    print("=" * 50)
    print(" Upstox Smoke Test")
    print("=" * 50)

    token = load_token()
    print(f"\n[1/3] Profile check…")
    prof = api_get("/user/profile", token)
    if prof.get("status") != "success":
        print(f"  FAIL: {prof}", file=sys.stderr)
        return 2
    d = prof["data"]
    print(f"  ✓ Name      : {d.get('user_name')}")
    print(f"  ✓ User ID   : {d.get('user_id')}")
    print(f"  ✓ Broker    : {d.get('broker')}")
    print(f"  ✓ Exchanges : {', '.join(d.get('exchanges', []))}")
    has_mcx = "MCX" in d.get("exchanges", [])
    print(f"  ✓ MCX active: {'YES — ready for gold trading' if has_mcx else 'NO — activation pending'}")

    print(f"\n[2/3] Funds check…")
    funds = api_get("/user/get-funds-and-margin", token)
    if funds.get("status") != "success":
        print(f"  FAIL: {funds}", file=sys.stderr)
        return 3
    f = funds["data"]
    # Funds split by segment: equity, commodity
    for seg, data in f.items():
        avail = data.get("available_margin", 0)
        used  = data.get("used_margin", 0)
        print(f"  ✓ {seg:10s}: available {fmt_inr(avail)} | used {fmt_inr(used)}")

    print(f"\n[3/3] Market timings…")
    today = date.today().isoformat()
    timings = api_get(f"/market/timings/{today}", token)
    if timings.get("status") != "success":
        print(f"  FAIL: {timings}", file=sys.stderr)
        return 4
    for t in timings.get("data", []):
        ex = t.get("exchange")
        if ex in ("MCX", "NSE"):
            print(f"  ✓ {ex}: open {t.get('start_time')} → close {t.get('end_time')}")

    print("\n" + "=" * 50)
    print(" ALL CHECKS PASSED")
    print("=" * 50)
    if not has_mcx:
        print("\nNote: MCX not yet active. Strategy + paper backtest can still")
        print("proceed; live MCX orders will fail until activation completes.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
