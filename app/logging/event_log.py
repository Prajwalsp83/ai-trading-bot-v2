"""
Append-only JSONL log of every event on the bus. Used for replay + debugging.

Skeleton — no implementation yet.
"""
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        """TODO: append {ts, event_type, payload} as JSON line."""
        raise NotImplementedError

    def replay(self, since=None):
        """TODO: yield (ts, event_type, payload) tuples — for debugging."""
        raise NotImplementedError
