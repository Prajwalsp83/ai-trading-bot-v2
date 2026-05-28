"""
Risk limit gates. The Risk layer asks each gate; any False blocks the order.

Skeleton — no implementation yet.
"""
from dataclasses import dataclass

from ..core.config import RiskConfig
from ..core.state import StateStore


@dataclass
class GateResult:
    allowed: bool
    reason: str


class RiskGate:
    def __init__(self, state: StateStore, risk: RiskConfig) -> None:
        self.state = state
        self.risk = risk

    def check_daily_loss(self) -> GateResult:
        """TODO: today's pnl vs daily_loss_cap_pct * starting balance."""
        raise NotImplementedError

    def check_drawdown(self) -> GateResult:
        """TODO: equity vs peak equity vs max_drawdown_pct."""
        raise NotImplementedError

    def check_max_positions(self) -> GateResult:
        """TODO: count open positions vs max_concurrent_positions."""
        raise NotImplementedError

    def check_cooldown(self) -> GateResult:
        """TODO: are we still inside cooldown_minutes after a loss streak?"""
        raise NotImplementedError

    def check_all(self) -> GateResult:
        """TODO: run every gate, first failure wins, return aggregate result."""
        raise NotImplementedError
