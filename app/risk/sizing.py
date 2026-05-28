"""
Position sizing. ATR-based, capped by risk-per-trade %.

Skeleton — no implementation yet.
"""
from ..core.config import RiskConfig


def position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk: RiskConfig,
    contract_size: float = 1.0,
) -> float:
    """Return quantity such that loss at stop == risk_per_trade_pct * equity.

    qty = (equity * risk_per_trade_pct) / (|entry - stop| * contract_size)

    TODO: implement, round to broker step, enforce min/max lot.
    """
    raise NotImplementedError
