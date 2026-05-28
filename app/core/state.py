"""
Shared application state. Single source of truth for runtime values.
The dashboard reads this; nothing else mutates it directly except the
execution + risk layers via the methods below.

Skeleton — no implementation yet.
"""
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock

from .events import Position


@dataclass
class AccountState:
    equity: float = 0.0
    balance: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_reset_at: datetime | None = None
    consecutive_losses: int = 0


@dataclass
class RuntimeState:
    account: AccountState = field(default_factory=AccountState)
    open_positions: dict[str, Position] = field(default_factory=dict)  # by position_id
    last_signal_ts: dict[str, datetime] = field(default_factory=dict)  # by symbol
    cooldown_until: datetime | None = None
    paused: bool = False


class StateStore:
    """Thread-safe accessor around RuntimeState."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._state = RuntimeState()

    def snapshot(self) -> RuntimeState:
        """TODO: return a deep copy under lock for the dashboard."""
        raise NotImplementedError

    def update_account(self, **kwargs) -> None:
        """TODO: mutate account fields under lock."""
        raise NotImplementedError

    def add_position(self, position: Position) -> None:
        """TODO."""
        raise NotImplementedError

    def remove_position(self, position_id: str) -> Position | None:
        """TODO."""
        raise NotImplementedError
