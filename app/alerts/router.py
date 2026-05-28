"""
Alert router. Subscribes to the event bus; fans out to channels by severity.

Severity → channels mapping (configurable):

  INFO              -> log only
  WATCHLIST         -> telegram
  BREAKOUT_WATCH    -> telegram
  BUY_READY         -> telegram
  SELL_READY        -> telegram
  ENTRY_CONFIRMED   -> telegram + sound
  EXIT_ALERT        -> telegram + sound
  RISK_ALERT        -> telegram + sound + desktop

De-duplication: don't re-emit the same (symbol, severity, bar_ts) twice.

Skeleton — no implementation yet.
"""
from ..core.events import Alert, Severity


class AlertRouter:
    def __init__(self, channels: dict) -> None:
        """channels: {"telegram": TelegramChannel, "sound": SoundChannel, ...}"""
        self.channels = channels
        self._dedup_cache: set[tuple] = set()

    def handle(self, alert: Alert) -> None:
        """TODO: check dedup, look up channels for severity, dispatch (non-blocking)."""
        raise NotImplementedError

    def channels_for(self, severity: Severity) -> list[str]:
        """TODO: return channel names for this severity."""
        raise NotImplementedError
