"""
Strategy-aligned backtest engine. Replays historical bars through the SAME
Strategy + Risk + Broker (paper) used live. No lookahead, no separate logic.

This is the fix for the v1 problem where backtest used ML predictions
while live used breakout rules.

Skeleton — no implementation yet.
"""
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ..execution.paper import PaperBroker
from ..risk.limits import RiskGate
from ..risk.sizing import position_size
from ..risk.stops import StopParams, initial_sl_tp, update_trailing
from ..strategy.base import MarketState, Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        risk_gate: RiskGate,
        broker: PaperBroker,
        stop_params: StopParams,
    ) -> None:
        self.strategy = strategy
        self.risk = risk_gate
        self.broker = broker
        self.stops = stop_params

    def run(
        self,
        bars_entry: pd.DataFrame,
        bars_trend: pd.DataFrame,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> BacktestResult:
        """TODO:
        for each entry-TF bar in [start, end]:
          1. slice bars_entry up to and including this bar (NO lookahead)
          2. align trend-TF bars to same point in time
          3. broker.on_bar(...) updates SL/TP for any open positions
          4. risk_gate.check_all() — if blocked, skip
          5. strategy.evaluate(state) -> Signal
          6. if Signal is BUY_READY/SELL_READY:
               sl, tp = initial_sl_tp(...)
               qty   = position_size(...)
               broker.place(Order(...))
          7. record equity, exposure, drawdown
        finally: build BacktestResult
        """
        raise NotImplementedError
