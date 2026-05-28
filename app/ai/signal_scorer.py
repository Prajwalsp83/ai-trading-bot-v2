"""
Signal scorer. Given a rule-based Signal, output 0..1 confidence.
Risk uses this to scale position size up or down (with caps).

This is where AI augments the strategy without replacing it — the rules
generate candidates, the model rates them.

Skeleton — no implementation yet.
"""
import pandas as pd

from ..core.events import Signal


class SignalScorer:
    def train(self, features: pd.DataFrame, outcomes: pd.Series) -> None:
        """outcomes: realised R per historical signal. TODO."""
        raise NotImplementedError

    def score(self, signal: Signal, features: pd.Series) -> float:
        """TODO: return calibrated 0..1 probability of profitable outcome."""
        raise NotImplementedError
