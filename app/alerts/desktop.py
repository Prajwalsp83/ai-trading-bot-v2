"""
Desktop OS notifications. Used for RISK_ALERT only (don't spam the OS).

Skeleton — no implementation yet.
"""
from ..core.events import Alert


class DesktopChannel:
    def send(self, alert: Alert) -> None:
        """TODO: use 'plyer' or 'osascript' on macOS, 'notify-send' on Linux,
        'win10toast' on Windows. Detect platform once at init."""
        raise NotImplementedError
