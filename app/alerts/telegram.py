"""
Telegram channel. Markdown messages with severity-tagged headers.
Supports inline buttons for semi-auto confirm (Confirm / Skip).

Skeleton — no implementation yet.
"""
from ..core.events import Alert


class TelegramChannel:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send(self, alert: Alert) -> None:
        """TODO: format message with severity emoji + body + payload table; POST."""
        raise NotImplementedError

    def send_with_confirm(self, alert: Alert, confirm_callback_id: str) -> None:
        """TODO: send with inline keyboard [Confirm] [Skip] for semi-auto."""
        raise NotImplementedError

    def poll_updates(self) -> list:
        """TODO: long-poll getUpdates so we can react to user button presses."""
        raise NotImplementedError
