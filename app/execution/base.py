"""
Broker interface. Paper, MT5, future brokers all implement this.

Skeleton — no implementation yet.
"""
from abc import ABC, abstractmethod

from ..core.events import Fill, Order, Position


class Broker(ABC):
    @abstractmethod
    def place(self, order: Order) -> str:
        """Return broker order_id. Fills arrive via on_fill callback."""

    @abstractmethod
    def modify(self, order_id: str, sl: float | None = None, tp: float | None = None) -> None:
        """Modify an open order or position's SL/TP."""

    @abstractmethod
    def cancel(self, order_id: str) -> None:
        """Cancel pending order."""

    @abstractmethod
    def positions(self) -> list[Position]:
        """All open positions."""

    @abstractmethod
    def account_equity(self) -> float:
        ...

    @abstractmethod
    def on_fill(self, callback) -> None:
        """Register fill callback. Broker calls it as fills happen."""
