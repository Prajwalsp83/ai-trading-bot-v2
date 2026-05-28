"""
Streamlit dashboard entry point.

Run with:  streamlit run app/dashboard/app.py

Sidebar:
  - Mode toggle: backtest / paper / semi / auto  (writes to runtime config)
  - Symbol picker
  - Kill switch button (red)
  - Last refresh time

Pages (auto-discovered under pages/):
  - live      — current state, last signal, open positions, alerts feed
  - backtest  — run + view metrics
  - journal   — trade history
  - risk      — limits, exposure, drawdown

Skeleton — no implementation yet.
"""
# import streamlit as st
# from ..core.config import load_config
# from ..core.state import StateStore


def main() -> None:
    """TODO:
      - load config
      - render sidebar (mode selector, kill switch, symbol)
      - delegate to selected page
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
