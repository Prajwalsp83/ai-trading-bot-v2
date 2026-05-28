"""
Backtest page.

Sections:
  - Form: symbol, date range, strategy params (ema_fast, ema_slow, atr_min,
          k_sl, k_tp, k_trail), starting equity, risk_per_trade
  - Run button -> calls BacktestEngine.run synchronously, shows progress
  - Equity curve chart
  - Drawdown chart
  - Metrics summary (Sharpe, max DD, expectancy, hit rate, ...)
  - Trade list

Skeleton — no implementation yet.
"""


def render() -> None:
    """TODO."""
    raise NotImplementedError
