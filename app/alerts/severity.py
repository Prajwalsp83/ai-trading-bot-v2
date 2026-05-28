"""
Severity helpers. The Severity enum itself lives in core/events.py
(so the strategy can produce Signals without importing alerts).

Skeleton — no implementation yet.
"""
from ..core.events import Severity


SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.INFO: "i",
    Severity.WATCHLIST: "*",
    Severity.BREAKOUT_WATCH: "!",
    Severity.BUY_READY: "BUY",
    Severity.SELL_READY: "SELL",
    Severity.ENTRY_CONFIRMED: "ENTRY",
    Severity.EXIT_ALERT: "EXIT",
    Severity.RISK_ALERT: "RISK",
}


def title_for(severity: Severity) -> str:
    """TODO: map severity to short human title for alert headers."""
    raise NotImplementedError
