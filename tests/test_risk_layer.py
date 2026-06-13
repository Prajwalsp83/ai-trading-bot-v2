"""
Unit tests for the risk layer in _bot_common.py -- the most important and
previously-untested code in the repo (CLAUDE.md flags this as #2 to build).

Covers:
  - dd_multiplier: tier selection, clamping, custom tiers, edge cases
  - kelly_multiplier: every branch (small sample, no-loss, no-win, normal) + bounds
  - compute_effective_risk: multiplicative cascade + halt conditions

These functions multiply together to size every live trade. A regression here
silently over- or under-risks real money, so the math is pinned exactly.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _bot_common import (
    DDTier, DEFAULT_DD_TIERS, dd_multiplier,
    KellyParams, kelly_multiplier,
    Regime, RegimeSnapshot, compute_effective_risk,
)


# ============================ dd_multiplier =========================
def test_dd_no_peak_is_full():
    assert dd_multiplier(1000, None) == (1.0, "full")
    assert dd_multiplier(1000, 0) == (1.0, "full")


def test_dd_zero_drawdown_is_full():
    # dd exactly 0 matches the 0% tier (mult 1.0); name reflects full sizing
    mult, name = dd_multiplier(1000, 1000)
    assert mult == 1.0 and "x1.00" in name


def test_dd_equity_above_peak_clamps_to_full():
    # equity > peak -> negative dd clamped to 0 -> full size, never >1
    mult, _ = dd_multiplier(1200, 1000)
    assert mult == 1.0


@pytest.mark.parametrize("dd_pct,expected_mult", [
    (0.00, 1.00),   # full
    (0.02, 1.00),   # below 3% tier
    (0.03, 0.50),   # exactly 3%
    (0.05, 0.50),   # between 3 and 7
    (0.07, 0.25),   # exactly 7%
    (0.10, 0.25),   # between 7 and 12
    (0.12, 0.00),   # exactly 12% -> halt
    (0.30, 0.00),   # deep -> halt
])
def test_dd_tiers_default(dd_pct, expected_mult):
    peak = 1000.0
    equity = peak * (1 - dd_pct)
    mult, _ = dd_multiplier(equity, peak)
    assert mult == expected_mult


def test_dd_halt_tier_name():
    mult, name = dd_multiplier(880, 1000)   # 12% dd
    assert mult == 0.0 and "halt" in name


def test_dd_custom_tiers_unsorted_input():
    # pass tiers out of order -> function must sort internally
    tiers = [DDTier(0.0, 1.0), DDTier(0.10, 0.0), DDTier(0.05, 0.5)]
    assert dd_multiplier(960, 1000, tiers)[0] == 1.0   # 4% dd -> below 5% tier
    assert dd_multiplier(940, 1000, tiers)[0] == 0.5   # 6% dd -> 5% tier
    assert dd_multiplier(890, 1000, tiers)[0] == 0.0   # 11% dd -> halt


# ============================ kelly_multiplier ======================
def _journal(tmp_path, r_values) -> Path:
    p = tmp_path / "journal.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["r_realised"])
        w.writeheader()
        for r in r_values:
            w.writerow({"r_realised": r})
    return p


def test_kelly_missing_journal_returns_default():
    mult, note = kelly_multiplier(Path("/no/such/file.csv"))
    assert mult == 1.0 and "sample_too_small" in note


def test_kelly_small_sample_returns_default(tmp_path):
    p = _journal(tmp_path, [1, -1, 1, -1, 1])   # 5 < min 10
    mult, note = kelly_multiplier(p)
    assert mult == 1.0 and "sample_too_small(5/10)" in note


def test_kelly_no_losses_caps_at_one_plus_fraction(tmp_path):
    p = _journal(tmp_path, [1.0] * 10)
    mult, note = kelly_multiplier(p)
    assert mult == pytest.approx(1.25)   # min(max=2.0, 1.0+0.25)
    assert "no_losses" in note


def test_kelly_no_wins_returns_min(tmp_path):
    p = _journal(tmp_path, [-1.0] * 10)
    mult, note = kelly_multiplier(p)
    assert mult == 0.25 and "no_wins" in note


def test_kelly_normal_mix_is_exact(tmp_path):
    # 6 wins +1R, 4 losses -1R: wr=0.6, b=1, raw=0.2, frac=0.05, 1+0.05*5=1.25
    p = _journal(tmp_path, [1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
    mult, _ = kelly_multiplier(p)
    assert mult == pytest.approx(1.25)


def test_kelly_clamps_at_max(tmp_path):
    # huge edge: 9 wins +10R, 1 loss -1R -> raw 0.89, 1+0.89*0.25*5=2.11 -> cap 2.0
    p = _journal(tmp_path, [10] * 9 + [-1])
    mult, _ = kelly_multiplier(p)
    assert mult == 2.0


def test_kelly_clamps_at_min(tmp_path):
    # mostly losers: 2 wins +1R, 8 losses -1R -> raw -0.6, 1-0.75=0.25 (== min)
    p = _journal(tmp_path, [1, 1] + [-1] * 8)
    mult, _ = kelly_multiplier(p)
    assert mult == 0.25


def test_kelly_respects_lookback(tmp_path):
    # 30 ancient losers then 10 recent winners; lookback=10 should see only wins
    p = _journal(tmp_path, [-1] * 30 + [1] * 10)
    mult, note = kelly_multiplier(p, KellyParams(lookback_trades=10))
    assert "no_losses" in note and mult == pytest.approx(1.25)


# ======================= compute_effective_risk =====================
def _regime(weight_smc=1.0, weight_breakout=1.0, regime=Regime.CHOP):
    return RegimeSnapshot(regime=regime, adx=22.0, note="test",
                          weight_breakout=weight_breakout, weight_smc=weight_smc)


def test_effective_risk_full_passthrough(tmp_path):
    d = compute_effective_risk(
        base_risk_pct=0.01, equity=1000, peak_equity=1000,
        journal_path=None, regime_snapshot=_regime(), strategy_name="smc",
        use_kelly=False, use_regime=True)
    assert d.risk_pct == pytest.approx(0.01)
    assert d.dd_mult == 1.0 and d.kelly_mult == 1.0 and d.regime_mult == 1.0
    assert d.halted is False


def test_effective_risk_cascades_multiplicatively(tmp_path):
    # dd 5% -> 0.50, kelly 1.25 (from 6/4 journal), regime smc 0.5
    p = _journal(tmp_path, [1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
    d = compute_effective_risk(
        base_risk_pct=0.02, equity=950, peak_equity=1000,
        journal_path=p, regime_snapshot=_regime(weight_smc=0.5),
        strategy_name="smc")
    # 0.02 * 0.50 * 1.25 * 0.50 = 0.00625
    assert d.dd_mult == 0.5 and d.kelly_mult == pytest.approx(1.25) and d.regime_mult == 0.5
    assert d.risk_pct == pytest.approx(0.00625)
    assert d.halted is False


def test_effective_risk_halted_by_drawdown(tmp_path):
    d = compute_effective_risk(
        base_risk_pct=0.01, equity=850, peak_equity=1000,   # 15% dd -> halt
        journal_path=None, regime_snapshot=_regime(), strategy_name="smc",
        use_kelly=False)
    assert d.dd_mult == 0.0 and d.risk_pct == 0.0 and d.halted is True


def test_effective_risk_halted_by_regime_zero_weight(tmp_path):
    d = compute_effective_risk(
        base_risk_pct=0.01, equity=1000, peak_equity=1000,
        journal_path=None, regime_snapshot=_regime(weight_smc=0.0),
        strategy_name="smc", use_kelly=False)
    assert d.regime_mult == 0.0 and d.halted is True and d.risk_pct == 0.0


def test_effective_risk_strategy_picks_correct_regime_weight(tmp_path):
    rs = _regime(weight_breakout=1.0, weight_smc=0.0)
    smc = compute_effective_risk(0.01, 1000, 1000, None, rs, "smc", use_kelly=False)
    bo = compute_effective_risk(0.01, 1000, 1000, None, rs, "breakout", use_kelly=False)
    assert smc.regime_mult == 0.0 and smc.halted is True
    assert bo.regime_mult == 1.0 and bo.halted is False


def test_effective_risk_disabled_toggles(tmp_path):
    p = _journal(tmp_path, [1] * 10)   # would give kelly 1.25 if enabled
    d = compute_effective_risk(
        0.01, 1000, 1000, p, _regime(weight_smc=0.0), "smc",
        use_kelly=False, use_regime=False)
    assert d.kelly_mult == 1.0 and d.regime_mult == 1.0   # both ignored
    assert d.risk_pct == pytest.approx(0.01)
