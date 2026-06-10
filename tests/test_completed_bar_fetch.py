"""
Phase 1 regression guard: the live bots must fetch only COMPLETED bars.

mt5.copy_rates_from_pos(symbol, tf, start_pos, count) with start_pos=0 returns
the in-progress (forming) candle as the most recent bar, which repaints until
close. The backtest only ever sees completed bars, so the live bots must call
with start_pos=1 to drop bar 0. This test fails if anyone reverts to 0.

Runs on any machine (no real MetaTrader5 needed): a fake MetaTrader5 module is
injected into sys.modules before importing the bots, and it records every
copy_rates_from_pos call.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

_RATE_DTYPE = np.dtype([
    ("time", "i8"), ("open", "f8"), ("high", "f8"),
    ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
])


def _make_rates(count):
    """A minimal but valid structured array, as MT5 returns."""
    n = max(count, 1)
    arr = np.zeros(n, dtype=_RATE_DTYPE)
    arr["time"] = np.arange(n, dtype="i8") * 900  # 15-min steps
    for f in ("open", "high", "low", "close"):
        arr[f] = 2000.0
    arr["high"] = 2001.0
    arr["low"] = 1999.0
    arr["tick_volume"] = 100
    return arr


class _FakeMT5(types.ModuleType):
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388

    def __init__(self):
        super().__init__("MetaTrader5")
        self.calls = []  # list of (symbol, tf, start_pos, count)

    def copy_rates_from_pos(self, symbol, tf, start_pos, count):
        self.calls.append((symbol, tf, start_pos, count))
        return _make_rates(count)


@pytest.fixture
def fake_mt5(monkeypatch):
    fake = _FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return fake


def _import_fresh(module_name, monkeypatch):
    # Force a clean import so the fake mt5 is the one bound in the module.
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    import importlib
    return importlib.import_module(module_name)


def test_smc_fetch_bars_drops_forming_bar(fake_mt5, monkeypatch):
    mod = _import_fresh("mt5_smc", monkeypatch)
    fake_mt5.calls.clear()
    df15, df1h = mod.fetch_bars()
    assert df15 is not None and df1h is not None
    assert len(fake_mt5.calls) == 2
    for symbol, tf, start_pos, count in fake_mt5.calls:
        assert start_pos == 1, (
            f"copy_rates_from_pos called with start_pos={start_pos}; must be 1 "
            f"to drop the forming bar (look-ahead guard)"
        )


def test_live_fetch_bars_drops_forming_bar(fake_mt5, monkeypatch):
    mod = _import_fresh("mt5_live", monkeypatch)
    fake_mt5.calls.clear()
    df15, df1h, df4h = mod.fetch_bars()
    assert df15 is not None and df1h is not None and df4h is not None
    assert len(fake_mt5.calls) == 3
    for symbol, tf, start_pos, count in fake_mt5.calls:
        assert start_pos == 1, (
            f"copy_rates_from_pos called with start_pos={start_pos}; must be 1 "
            f"to drop the forming bar (look-ahead guard)"
        )
