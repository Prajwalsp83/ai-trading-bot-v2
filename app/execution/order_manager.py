"""
Owns order lifecycle: brackets (SL+TP), modifications, retries, kill-switch flatten.
Sits between the strategy/risk pipeline and the broker.

Skeleton — no implementation yet.
"""
from ..core.events import Fill, Order, Position
from ..core.kill_switch import KillSwitch
from .base import Broker


class OrderManager:
    def __init__(self, broker: Broker, kill: KillSwitch) -> None:
        self.broker = broker
        self.kill = kill

    def submit(self, order: Order) -> str | None:
        """TODO:
          - if kill.is_tripped(): return None + RISK_ALERT
          - place order, attach SL/TP bracket
          - retry on broker error (idempotent via signal_id)
          - return order_id
        """
        raise NotImplementedError

    def update_stops(self, position: Position, new_sl: float | None, new_tp: float | None) -> None:
        """TODO: broker.modify with retry."""
        raise NotImplementedError

    def flatten_all(self, reason: str) -> None:
        """TODO: cancel pending; close every open position at market."""
        raise NotImplementedError
