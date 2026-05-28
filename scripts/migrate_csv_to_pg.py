"""
Backfill existing CSV journals into Postgres.

Reads both:
  data/mt5_trades.csv      (breakout bot, magic 20260522)
  data/mt5_smc_trades.csv  (SMC bot,      magic 20260601)

Writes into the `trades` table with `bot_name` populated.
Idempotent — the UNIQUE (bot_name, magic, trade_id) constraint dedupes re-runs.

Run once after schema.sql is applied:
    DATABASE_URL=postgresql://... python scripts/migrate_csv_to_pg.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Path resolution: works in both Mac v2 layout (scripts/migrate.py — parent.parent=v2)
# and VPS flat layout (C:\bot\migrate.py — parent=C:\bot). We probe both.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_CANDIDATES = [SCRIPT_DIR.parent, SCRIPT_DIR]   # v2 layout first, then flat

sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv  # noqa: E402
for root in ROOT_CANDIDATES:
    if (root / ".env").exists():
        load_dotenv(root / ".env")
        break

from _journal import record_trade  # noqa: E402


def _resolve_data_path(filename: str) -> Path | None:
    for root in ROOT_CANDIDATES:
        p = root / "data" / filename
        if p.exists():
            return p
    return None


JOURNAL_MAP = [
    {"path": _resolve_data_path("mt5_trades.csv"),     "bot_name": "breakout", "magic": 20260522},
    {"path": _resolve_data_path("mt5_smc_trades.csv"), "bot_name": "smc",      "magic": 20260601},
]


def migrate_one(path: Path | None, bot_name: str, magic: int) -> tuple[int, int]:
    if path is None or not path.exists():
        print(f"  skip — no CSV found for {bot_name} (looked in: " +
              ", ".join(str(r / 'data') for r in ROOT_CANDIDATES) + ")")
        return (0, 0)
    print(f"  reading {path}")

    written = 0
    skipped = 0
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            # ensure symbol field exists for journal writer (CSV may omit it)
            row.setdefault("symbol", "GOLD.i#")
            ok = record_trade(bot_name, magic, row)
            if ok:
                written += 1
            else:
                skipped += 1
    return (written, skipped)


def main() -> int:
    print("=== CSV -> Postgres backfill ===")
    total_w = total_s = 0
    for cfg in JOURNAL_MAP:
        print(f"\n{cfg['bot_name']:>10} ({cfg['path'].name}):")
        w, s = migrate_one(cfg["path"], cfg["bot_name"], cfg["magic"])
        print(f"           wrote {w}, skipped {s}")
        total_w += w
        total_s += s
    print(f"\nTOTAL: wrote {total_w}, skipped {total_s}")
    print("(skipped are usually rows already present from a previous run — UNIQUE constraint dedupes)")
    return 0 if total_w + total_s > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
