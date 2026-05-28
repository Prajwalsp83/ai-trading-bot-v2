"""
Event types and a tiny in-process pub/sub bus.

Every meaningful action publishes an event. Alerts and journal subscribe.
The trade pipeline never blocks on subscribers.

Skeleton — no implementation yet.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Literal


class Severity(Enum):
    INFO = "INFO"
    WATCHLIST = "WATCHLIST"
    BREAKOUT_WATCH = "BREAKOUT_WATCH"
    BUY_READY = "BUY_READY"
    SELL_READY = "SELL_READY"
    ENTRY_CONFIRMED = "ENTRY_CONFIRMED"
    EXIT_ALERT = "EXIT_ALERT"
    RISK_ALERT = "RISK_ALERT"


Side = Literal["BUY", "SELL"]
ExitReason = Literal["SL", "TP", "TRAIL", "TIME", "MANUAL", "KILL"]


@dataclass
class Signal:
    ts: datetime
    symbol: str
    side: Side
    severity: Severity
    price: float
    atr: float
    reason: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    ts: datetime
    symbol: str
    side: Side
    qty: float
    sl: float
    tp: float
    signal_id: str | None = None


@dataclass
class Fill:
    ts: datetime
    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    is_open: bool        # True for entry, False for close


@dataclass
class Position:
    position_id: str
    symbol: str
    side: Side
    qty: float
    entry_price: float
    entry_ts: datetime
    sl: float
    tp: float
    atr_at_entry: float


@dataclass
class Alert:
    ts: datetime
    severity: Severity
    title: str
    body: str
    payload: dict[str, Any] = field(default_factory=dict)


# ----- pub/sub -----

Subscriber = Callable[[Any], None]


class EventBus:
    """Thread-safe in-process bus. Subscribers should be fast and non-blocking."""

    def subscribe(self, event_type: type, callback: Subscriber) -> None:
        """TODO: register callback for given dataclass type."""
        raise NotImplementedError

    def publish(self, event: Any) -> None:
        """TODO: dispatch to all subscribers of type(event); swallow exceptions."""
        raise NotImplementedError
