"""
Paper broker. Simulated fills with configurable spread + slippage.
Used by both forward-paper trading and the backtest engine.

Skeleton — no implementation yet.
"""
from ..core.config import ExecutionConfig
from ..core.events import Fill, Order, Position
from .base import Broker


class PaperBroker(Broker):
    def __init__(self, exec_cfg: ExecutionConfig, starting_equity: float) -> None:
        self.cfg = exec_cfg
        self._equity = starting_equity
        self._positions: dict[str, Position] = {}
        self._fill_callbacks: list = []

    def place(self, order: Order) -> str:
        """TODO: apply spread + slippage, create Position, fire on_fill."""
        raise NotImplementedError

    def modify(self, order_id, sl=None, tp=None) -> None:
        """TODO: update SL/TP on stored position."""
        raise NotImplementedError

    def cancel(self, order_id) -> None:
        """TODO: paper has no pending state in v1."""
        raise NotImplementedError

    def positions(self) -> list[Position]:
        return list(self._positions.values())

    def account_equity(self) -> float:
        return self._equity

    def on_fill(self, callback) -> None:
        self._fill_callbacks.append(callback)

    def on_bar(self, symbol: str, bar) -> None:
        """Called by engine on every new bar — checks SL/TP for each position
        and fires close fills.

        TODO: standard bar-by-bar SL/TP logic, with conservative tie-breaking
        (assume SL hits before TP within the same bar)."""
        raise NotImplementedError
