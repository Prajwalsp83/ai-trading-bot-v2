"""
Regime classifier. Trained offline; loaded by strategy/regime.py at runtime.

Approach (v1):
  - label historical bars by future N-bar return + realized vol bucket
  - train RF / GBM to predict regime from current features
  - export trained model (joblib) to disk

Skeleton — no implementation yet.
"""
from pathlib import Path

import pandas as pd

from ..strategy.regime import Regime


class RegimeClassifier:
    def train(self, features: pd.DataFrame, labels: pd.Series, out_path: Path) -> None:
        """TODO."""
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """TODO."""
        raise NotImplementedError

    def predict(self, feature_row: pd.Series) -> Regime:
        """TODO."""
        raise NotImplementedError
