"""
Trade journal — one row per closed trade, expanded schema.

Columns:
  trade_id, open_time, close_time, symbol, side, entry, exit, qty,
  sl, tp, rr_planned, rr_realised, pnl, pnl_pct,
  max_adverse_excursion, max_favorable_excursion,
  duration_bars, atr_at_entry, regime, exit_reason

Storage: CSV by default. SQLite is a config swap (see open question §10.6).

Skeleton — no implementation yet.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class TradeRecord:
    trade_id: str
    open_time: datetime
    close_time: datetime
    symbol: str
    side: Literal["BUY", "SELL"]
    entry: float
    exit: float
    qty: float
    sl: float
    tp: float
    rr_planned: float
    rr_realised: float
    pnl: float
    pnl_pct: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    duration_bars: int
    atr_at_entry: float
    regime: str
    exit_reason: str


class TradeJournal:
    HEADERS = list(TradeRecord.__annotations__.keys())

    def __init__(self, path: Path) -> None:
        self.path = path

    def ensure_header(self) -> None:
        """TODO: create file with header row if missing."""
        raise NotImplementedError

    def append(self, record: TradeRecord) -> None:
        """TODO: append one row, fsync."""
        raise NotImplementedError

    def read_all(self):
        """TODO: return DataFrame for dashboard."""
        raise NotImplementedError
