"""
Unit tests for the remote pause control flag (_bot_common.control_*) and the
telegram_control PnL bucketing. These are the two pieces that decide whether
the bots stop trading and what the owner sees on /pnl, so they get coverage.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import _bot_common as bc


# ============================ control flag ==========================
def test_control_paused_fresh_is_false(tmp_path):
    p = tmp_path / ".control.json"
    paused, why = bc.control_paused(p)
    assert paused is False and why == ""


def test_control_set_and_read_roundtrip(tmp_path):
    p = tmp_path / ".control.json"
    bc.control_set(True, by="unit", path=p)
    paused, why = bc.control_paused(p)
    assert paused is True
    assert "unit" in why and "/resume" in why
    bc.control_set(False, by="unit", path=p)
    assert bc.control_paused(p)[0] is False


def test_control_unreadable_fails_open(tmp_path):
    """A corrupt flag file must NOT halt trading (fail-open)."""
    p = tmp_path / ".control.json"
    p.write_text("{ this is not json")
    assert bc.control_paused(p) == (False, "")


def test_control_set_is_atomic_no_tmp_left(tmp_path):
    p = tmp_path / ".control.json"
    bc.control_set(True, by="unit", path=p)
    assert p.exists()
    assert not (tmp_path / ".control.json.tmp").exists()


# ====================== telegram PnL bucketing ======================
def _fake_deal(entry, profit, swap, comm, magic):
    return types.SimpleNamespace(entry=entry, profit=profit, swap=swap,
                                 commission=comm, magic=magic)


def _load_tc():
    import telegram_control as tc
    return tc


def test_deals_pnl_buckets_by_magic_and_skips_non_close():
    tc = _load_tc()
    magic_to_bot = {20260601: "smc", 20260522: "breakout"}
    deals = [
        _fake_deal(1, 12.0, 0.0, -0.70, 20260601),   # smc close, net +11.30
        _fake_deal(1, -8.0, 0.0, -0.70, 20260522),   # breakout close, net -8.70
        _fake_deal(0, 0.0, 0.0, 0.0, 20260601),      # entry leg -> ignored
        _fake_deal(1, 5.0, -0.50, -0.70, 999),       # unknown magic -> "other"
    ]
    tc.mt5 = types.SimpleNamespace(history_deals_get=lambda a, b: deals)
    out = tc._deals_pnl(datetime(2024, 1, 1, tzinfo=timezone.utc), magic_to_bot)
    assert out["smc"]["n"] == 1
    assert out["smc"]["pnl"] == pytest.approx(11.30)
    assert out["breakout"]["pnl"] == pytest.approx(-8.70)
    assert out["other"]["pnl"] == pytest.approx(3.80)   # 5.0 - 0.50 - 0.70
    assert "smc" in out and out["smc"]["n"] == 1


def test_deals_pnl_empty_history():
    tc = _load_tc()
    tc.mt5 = types.SimpleNamespace(history_deals_get=lambda a, b: None)
    assert tc._deals_pnl(datetime(2024, 1, 1, tzinfo=timezone.utc), {}) == {}
