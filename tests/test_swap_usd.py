"""
Phase 2 unit tests for BacktestEngine._swap_usd (overnight swap modeling).

Anchor dates (all in Jan 2024, a known week):
    2024-01-01 = Monday
    2024-01-04 = Thursday
    2024-01-05 = Friday   (triple_swap_weekday default = 4)
    2024-01-06 = Saturday
    2024-01-08 = Monday
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _backtest_engine import BacktestEngine, CostModel, SymbolSpecs


def _engine(swap_long=-5.0, swap_short=-3.0, triple_weekday=4):
    specs = SymbolSpecs(
        symbol="GOLD.i#", contract_size=100.0, volume_min=0.01, volume_step=0.01,
        point=0.01, digits=2, avg_spread_points=25, commission_per_lot_rt_usd=7.0,
    )
    cost = CostModel(
        swap_long_usd_per_lot_night=swap_long,
        swap_short_usd_per_lot_night=swap_short,
        triple_swap_weekday=triple_weekday,
    )
    return BacktestEngine(specs, cost=cost)


def test_swap_one_night_long():
    # Mon 10:00 -> Tue 10:00 = roll into Tue (weekday 1, normal) = 1 night
    eng = _engine()
    got = eng._swap_usd("BUY", lots=1.0, open_time=datetime(2024, 1, 1, 10),
                        close_time=datetime(2024, 1, 2, 10))
    assert got == pytest.approx(-5.0)        # 1 night * -5.0 * 1.0 lot


def test_swap_intraday_is_zero():
    # Same calendar date, no rollover crossed
    eng = _engine()
    got = eng._swap_usd("BUY", lots=2.0, open_time=datetime(2024, 1, 1, 9),
                        close_time=datetime(2024, 1, 1, 23))
    assert got == 0.0


def test_swap_triple_on_friday():
    # Thu 10:00 -> Fri 10:00 = roll into Fri (weekday 4 = triple) = 3 nights
    eng = _engine()
    got = eng._swap_usd("BUY", lots=1.0, open_time=datetime(2024, 1, 4, 10),
                        close_time=datetime(2024, 1, 5, 10))
    assert got == pytest.approx(-15.0)       # 3 nights * -5.0 * 1.0 lot


def test_swap_weekend_skips_sat_sun():
    # Fri 10:00 -> Mon 10:00 = roll into Sat(skip)+Sun(skip)+Mon(1 night) = 1 night
    # (the triple was charged on the Friday rollover, which is before open_time)
    eng = _engine()
    got = eng._swap_usd("BUY", lots=1.0, open_time=datetime(2024, 1, 5, 10),
                        close_time=datetime(2024, 1, 8, 10))
    assert got == pytest.approx(-5.0)        # only Monday's rollover


def test_swap_short_side_uses_short_rate():
    # Mon -> Tue, SELL side uses swap_short (-3.0), scaled by lots
    eng = _engine()
    got = eng._swap_usd("SELL", lots=2.0, open_time=datetime(2024, 1, 1, 10),
                        close_time=datetime(2024, 1, 2, 10))
    assert got == pytest.approx(-6.0)        # 1 night * -3.0 * 2.0 lots


def test_swap_zero_rate_returns_zero():
    # No real swap rate configured -> no swap cost (backward compatible)
    eng = _engine(swap_long=0.0, swap_short=0.0)
    got = eng._swap_usd("BUY", lots=1.0, open_time=datetime(2024, 1, 1, 10),
                        close_time=datetime(2024, 1, 5, 10))
    assert got == 0.0
