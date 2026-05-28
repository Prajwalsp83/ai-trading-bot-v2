"""
Local sound alarm. Plays .wav for ENTRY_CONFIRMED / EXIT_ALERT / RISK_ALERT.

Skeleton — no implementation yet.
"""
from pathlib import Path

from ..core.events import Alert, Severity


class SoundChannel:
    def __init__(self, sound_dir: Path) -> None:
        self.sound_dir = sound_dir
        # TODO: lazy import simpleaudio or playsound; pick a backend per OS.

    def send(self, alert: Alert) -> None:
        """TODO: pick wav by severity, play asynchronously."""
        raise NotImplementedError

    def _file_for(self, severity: Severity) -> Path:
        """TODO: map severity -> filename (entry.wav, exit.wav, risk.wav)."""
        raise NotImplementedError
