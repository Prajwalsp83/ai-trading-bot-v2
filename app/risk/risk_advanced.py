"""
Advanced risk sizing — pro-trader playbook.

Two layers on top of the base risk_per_trade_pct:

1. **3-tier drawdown scaling.** As equity slips below peak, risk is cut in
   discrete steps, not a binary toggle. Mimics how desks reduce size
   automatically when their P&L curve breaks structure.
       0%  .. 3%  DD  ->  100% of base risk
       3%  .. 7%  DD  ->   50% of base risk
       7%  .. 12% DD  ->   25% of base risk
      12%+ DD         ->    0%  (block trading; let max_drawdown_pct halt)

2. **Kelly-fractional sizing.** From the last N closed trades we compute
   win-rate p and avg-win/avg-loss ratio b. Kelly = (b*p - q) / b, where
   q = 1 - p. We then take a *fraction* of Kelly (typically 0.25-0.5) to
   protect against estimation error. The fractional Kelly is multiplied into
   the base risk_per_trade_pct, so it can scale risk UP in good streaks and
   DOWN in bad ones — entirely independently of DD scaling.

Both layers compose multiplicatively:

    effective_risk = base * dd_multiplier * kelly_multiplier

with sane floors/ceilings so a hot streak can't blow up the account
and a cold streak still leaves room to recover.

Trade history is read from a CSV journal — same format mt5_live.py writes.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# =========================== Drawdown tiers ==========================

@dataclass
class DDTier:
    threshold_pct: float    # equity is at least this much below peak
    multiplier: float       # apply this fraction of base risk


# Order matters: tiers are checked top-down, the deepest matching wins.
# Tune these in config.yaml -> risk.dd_tiers if your equity curve is bumpier.
DEFAULT_DD_TIERS: list[DDTier] = [
    DDTier(threshold_pct=0.12, multiplier=0.0),    # 12%+ DD: stop entirely
    DDTier(threshold_pct=0.07, multiplier=0.25),   # 7-12%: 25% size
    DDTier(threshold_pct=0.03, multiplier=0.50),   # 3-7%: 50% size
    DDTier(threshold_pct=0.0,  multiplier=1.00),   # 0-3%: full size
]


def dd_multiplier(equity: float, peak_equity: float,
                  tiers: list[DDTier] | None = None) -> tuple[float, str]:
    """Return (multiplier, tier_name)."""
    if peak_equity is None or peak_equity <= 0:
        return 1.0, "full"
    dd = max(0.0, (peak_equity - equity) / peak_equity)
    tiers = sorted(tiers or DEFAULT_DD_TIERS, key=lambda t: -t.threshold_pct)
    for t in tiers:
        if dd >= t.threshold_pct:
            name = (f"halt_dd>={t.threshold_pct*100:.0f}%" if t.multiplier == 0
                    else f"dd>={t.threshold_pct*100:.0f}%_x{t.multiplier:.2f}")
            return t.multiplier, name
    return 1.0, "full"


# ========================== Kelly sizing =============================

@dataclass
class KellyParams:
    """Tunable knobs for Kelly-fractional sizing."""
    lookback_trades: int = 30             # smaller = more reactive, noisier
    fraction: float = 0.25                # quarter-Kelly: standard safety margin
    min_trades_required: int = 10         # don't size up until we have a sample
    max_multiplier: float = 2.0           # never more than 2x base on hot streak
    min_multiplier: float = 0.25          # never less than 25% on cold streak
    default_multiplier: float = 1.0       # when sample too small, no scaling


@dataclass
class KellySnapshot:
    n: int
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    b_ratio: float
    raw_kelly: float
    fractional_kelly: float
    multiplier: float                     # what we actually apply
    note: str                             # human-readable explanation


def _read_recent_r(journal_path: Path, lookback: int) -> list[float]:
    """Read the last `lookback` trades' realised R from a CSV journal."""
    if not journal_path.exists():
        return []
    try:
        with journal_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    r_values: list[float] = []
    for row in rows[-lookback:]:
        try:
            r_values.append(float(row.get("r_realised", 0) or 0))
        except Exception:
            continue
    return r_values


def kelly_from_r(r_values: Iterable[float], params: KellyParams | None = None) -> KellySnapshot:
    """Compute fractional-Kelly multiplier from a list of realised R values."""
    p = params or KellyParams()
    rs = list(r_values)
    n = len(rs)

    if n < p.min_trades_required:
        return KellySnapshot(
            n=n, win_rate=0.0, avg_win_r=0.0, avg_loss_r=0.0, b_ratio=0.0,
            raw_kelly=0.0, fractional_kelly=0.0,
            multiplier=p.default_multiplier,
            note=f"sample_too_small ({n}/{p.min_trades_required}); using default {p.default_multiplier:.2f}x",
        )

    wins = [r for r in rs if r > 0]
    losses = [-r for r in rs if r < 0]  # positive magnitudes
    if not losses:
        # all wins? -> bullish, but stay capped (we have no loss data to size from)
        return KellySnapshot(
            n=n, win_rate=1.0, avg_win_r=sum(wins) / max(len(wins), 1),
            avg_loss_r=0.0, b_ratio=float("inf"),
            raw_kelly=1.0, fractional_kelly=p.fraction,
            multiplier=min(p.max_multiplier, 1.0 + p.fraction),
            note=f"no_losses_in_sample ({n} trades); capped x{p.max_multiplier:.2f}",
        )
    if not wins:
        return KellySnapshot(
            n=n, win_rate=0.0, avg_win_r=0.0,
            avg_loss_r=sum(losses) / len(losses), b_ratio=0.0,
            raw_kelly=-1.0, fractional_kelly=p.min_multiplier,
            multiplier=p.min_multiplier,
            note=f"no_wins_in_sample ({n} trades); floored x{p.min_multiplier:.2f}",
        )

    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    b = avg_win / avg_loss

    # Kelly fraction: f* = (b*p - q) / b, q = 1-p
    raw = (b * win_rate - (1 - win_rate)) / b
    fractional = raw * p.fraction

    # Map fractional Kelly (typically in [-0.05 .. +0.10]) to a multiplier
    # centred at 1.0. We treat 0 as "neutral, use base risk".
    mult = 1.0 + fractional * 5.0  # 0.05 fractional -> 1.25x ; -0.05 -> 0.75x
    mult = max(p.min_multiplier, min(p.max_multiplier, mult))

    note = (f"n={n} wr={win_rate*100:.0f}% avg_win={avg_win:.2f}R "
            f"avg_loss={avg_loss:.2f}R b={b:.2f} "
            f"raw_kelly={raw:+.3f} frac_kelly={fractional:+.3f} -> x{mult:.2f}")
    return KellySnapshot(
        n=n, win_rate=win_rate, avg_win_r=avg_win, avg_loss_r=avg_loss,
        b_ratio=b, raw_kelly=raw, fractional_kelly=fractional,
        multiplier=mult, note=note,
    )


def kelly_from_journal(journal_path: Path,
                       params: KellyParams | None = None) -> KellySnapshot:
    p = params or KellyParams()
    rs = _read_recent_r(journal_path, p.lookback_trades)
    return kelly_from_r(rs, p)


# ======================== Combined sizing ============================

@dataclass
class RiskDecision:
    risk_pct: float                # final multiplier-applied risk %
    base_risk_pct: float           # config base
    dd_mult: float
    dd_tier: str
    kelly_mult: float
    kelly_note: str
    halted: bool                   # True if DD tier said no trading
    explanation: str


def compute_effective_risk(
    base_risk_pct: float,
    equity: float,
    peak_equity: float | None,
    journal_path: Path | None = None,
    kelly_params: KellyParams | None = None,
    dd_tiers: list[DDTier] | None = None,
    use_kelly: bool = True,
) -> RiskDecision:
    """Compose DD-tier + fractional-Kelly multipliers onto base risk."""
    dd_mult, dd_tier = dd_multiplier(equity, peak_equity or equity, dd_tiers)
    if use_kelly and journal_path is not None:
        kelly = kelly_from_journal(journal_path, kelly_params)
        kelly_mult = kelly.multiplier
        kelly_note = kelly.note
    else:
        kelly_mult = 1.0
        kelly_note = "kelly_disabled" if not use_kelly else "no_journal"

    eff = base_risk_pct * dd_mult * kelly_mult
    halted = dd_mult <= 0.0

    expl = (f"base={base_risk_pct*100:.2f}% "
            f"x dd_mult={dd_mult:.2f} ({dd_tier}) "
            f"x kelly={kelly_mult:.2f} "
            f"= {eff*100:.3f}%")
    return RiskDecision(
        risk_pct=eff, base_risk_pct=base_risk_pct,
        dd_mult=dd_mult, dd_tier=dd_tier,
        kelly_mult=kelly_mult, kelly_note=kelly_note,
        halted=halted, explanation=expl,
    )
