"""
Backtest report writer. Saves trades CSV + summary JSON + chart data files
that the dashboard reads.

Skeleton — no implementation yet.
"""
from pathlib import Path

from .engine import BacktestResult


def write_report(result: BacktestResult, out_dir: Path) -> None:
    """TODO:
      - trades.csv
      - equity.csv
      - summary.json
      - drawdown.csv
    The dashboard's backtest page reads these.
    """
    raise NotImplementedError
