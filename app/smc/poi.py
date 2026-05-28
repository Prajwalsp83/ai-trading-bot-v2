"""
Point of Interest (POI) scoring.

A POI is a high-probability reaction zone. We score by *confluence*:

  +2 if an Order Block overlaps a Fair Value Gap
  +1 if zone sits in discount/premium relative to current dealing range
  +1 if zone is fresh (within last N bars)
  +1 if zone is wide enough to allow a sensible stop (height >= some_fraction_of_atr)

Output: list of POI candidates ranked by score, with their zones.
Strategy then waits for price to enter the top POI in its bias direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from ..indicators.structure import StructureSnapshot, equilibrium
from .fvg import FVG
from .order_block import OrderBlock


@dataclass
class POI:
    side: Literal["bull", "bear"]
    top: float
    bottom: float
    score: int
    reasons: list[str] = field(default_factory=list)
    ob: OrderBlock | None = None
    fvg: FVG | None = None
    created_idx: int = 0

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


def _overlap(top1: float, bot1: float, top2: float, bot2: float) -> tuple[float, float] | None:
    """Return overlapping range or None."""
    top = min(top1, top2)
    bot = max(bot1, bot2)
    if top > bot:
        return (top, bot)
    return None


def build_pois(snap: StructureSnapshot,
               obs: list[OrderBlock], fvgs: list[FVG],
               current_idx: int, atr_val: float,
               freshness_bars: int = 30,
               min_zone_atr_frac: float = 0.3) -> list[POI]:
    """Combine OBs and FVGs into ranked POIs.

    Returns POIs sorted by score (descending).
    """
    pois: list[POI] = []
    eq = equilibrium(snap)

    # Pass A — OB+FVG confluences
    for ob in obs:
        if ob.mitigated:
            continue
        for fvg in fvgs:
            if fvg.mitigated or fvg.side != ob.side:
                continue
            ov = _overlap(ob.top, ob.bottom, fvg.top, fvg.bottom)
            if ov is None:
                continue
            top, bot = ov
            score = 2
            reasons = ["OB+FVG overlap"]
            # discount/premium bonus
            if ob.side == "bull" and eq is not None and (top + bot) / 2 <= eq:
                score += 1; reasons.append("in_discount")
            if ob.side == "bear" and eq is not None and (top + bot) / 2 >= eq:
                score += 1; reasons.append("in_premium")
            # freshness
            if (current_idx - max(ob.created_idx, fvg.created_idx)) <= freshness_bars:
                score += 1; reasons.append("fresh")
            # zone height vs ATR
            if atr_val > 0 and (top - bot) >= min_zone_atr_frac * atr_val:
                score += 1; reasons.append("zone_wide_enough")
            pois.append(POI(side=ob.side, top=top, bottom=bot, score=score,
                            reasons=reasons, ob=ob, fvg=fvg,
                            created_idx=max(ob.created_idx, fvg.created_idx)))

    # Pass B — standalone fresh unmitigated OBs (lower base score)
    for ob in obs:
        if ob.mitigated:
            continue
        if any(p.ob is ob for p in pois):
            continue
        score = 1
        reasons = ["OB_only"]
        if ob.side == "bull" and eq is not None and ob.mid <= eq:
            score += 1; reasons.append("in_discount")
        if ob.side == "bear" and eq is not None and ob.mid >= eq:
            score += 1; reasons.append("in_premium")
        if (current_idx - ob.created_idx) <= freshness_bars:
            score += 1; reasons.append("fresh")
        if atr_val > 0 and ob.height >= min_zone_atr_frac * atr_val:
            score += 1; reasons.append("zone_wide_enough")
        pois.append(POI(side=ob.side, top=ob.top, bottom=ob.bottom, score=score,
                        reasons=reasons, ob=ob, created_idx=ob.created_idx))

    # Pass C — standalone fresh unmitigated FVGs (lowest base)
    for fvg in fvgs:
        if fvg.mitigated:
            continue
        if any(p.fvg is fvg for p in pois):
            continue
        score = 1
        reasons = ["FVG_only"]
        if fvg.side == "bull" and eq is not None and fvg.mid <= eq:
            score += 1; reasons.append("in_discount")
        if fvg.side == "bear" and eq is not None and fvg.mid >= eq:
            score += 1; reasons.append("in_premium")
        if (current_idx - fvg.created_idx) <= freshness_bars:
            score += 1; reasons.append("fresh")
        if atr_val > 0 and fvg.height >= min_zone_atr_frac * atr_val:
            score += 1; reasons.append("zone_wide_enough")
        pois.append(POI(side=fvg.side, top=fvg.top, bottom=fvg.bottom, score=score,
                        reasons=reasons, fvg=fvg, created_idx=fvg.created_idx))

    pois.sort(key=lambda p: (p.score, p.created_idx), reverse=True)
    return pois


def pois_in_direction(pois: list[POI], side: Literal["bull", "bear"]) -> list[POI]:
    return [p for p in pois if p.side == side]


def best_poi_containing_price(pois: list[POI], price: float,
                               side: Literal["bull", "bear"]) -> POI | None:
    """Return the highest-scoring POI in `side` direction that currently contains `price`."""
    in_dir = pois_in_direction(pois, side)
    matches = [p for p in in_dir if p.contains(price)]
    if not matches:
        return None
    matches.sort(key=lambda p: p.score, reverse=True)
    return matches[0]
