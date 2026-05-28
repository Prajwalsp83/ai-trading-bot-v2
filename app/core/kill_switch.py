"""
Emergency stop. Trips via:
  1. UI button (writes flag file)
  2. Risk layer (daily loss / DD breach)
  3. Manual: `touch .kill`

Behaviour when tripped (configurable):
  - cancel all pending orders
  - block new orders
  - either flatten open positions, or let SL/TP run

Skeleton — no implementation yet.
"""
from enum import Enum
from pathlib import Path


class KillBehaviour(Enum):
    BLOCK_NEW_ONLY = "block_new_only"
    FLATTEN_ALL = "flatten_all"


class KillSwitch:
    def __init__(
        self,
        flag_file: Path,
        behaviour: KillBehaviour = KillBehaviour.BLOCK_NEW_ONLY,
    ) -> None:
        self.flag_file = flag_file
        self.behaviour = behaviour

    def is_tripped(self) -> bool:
        """TODO: return True if flag_file exists or in-memory flag set."""
        raise NotImplementedError

    def trip(self, reason: str) -> None:
        """TODO: create flag file, log reason, publish RISK_ALERT."""
        raise NotImplementedError

    def reset(self) -> None:
        """TODO: remove flag file. Called only by user via UI."""
        raise NotImplementedError
