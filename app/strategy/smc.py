"""
Smart Money Concepts (SMC) strategy.

Decision flow:

  1. HTF (1H) market structure -> bias (up / down / none).
     We trade WITH the bias only.

  2. HTF dealing range -> equilibrium (50% level).
     Long candidates must be in discount (below equilibrium).
     Short candidates must be in premium (above equilibrium).

  3. HTF unmitigated POIs (OB+FVG, OB-only, or FVG-only).
     We pick the highest-scored POI in the bias direction.

  4. LTF (15m) trigger:
       - price has entered the chosen HTF POI (mitigation), AND
       - 15m structure has just printed a CHoCH in the bias direction
         (= price reaction confirmed at the POI).
     Then we fire.

  5. Stop loss = beyond the POI's far edge by a small ATR buffer.
     Take profit = next opposite-side HTF liquidity (last unbroken swing).

Severity ladder reused from breakout_trend so the rest of the bot doesn't change.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from ..core.events import Severity, Signal
from ..indicators.structure import (
    analyse_structure, equilibrium, is_discount, is_premium,
)
from ..indicators.volatility import atr
from ..smc.fvg import find_fvgs, open_fvgs
from ..smc.order_block import find_order_blocks, open_order_blocks
from ..smc.poi import POI, build_pois, best_poi_containing_price, pois_in_direction
from .base import MarketState, Strategy


@dataclass
class SMCParams:
    htf_pivot: int = 2                # swing detection sensitivity on 1H
    ltf_pivot: int = 2                # ditto on 15m
    min_impulse_bars: int = 3         # for OB detection
    poi_freshness_bars_htf: int = 30  # HTF POIs older than this are stale
    min_poi_score: int = 3            # require at least this score to consider a POI
    atr_period: int = 14
    sl_buffer_atr_frac: float = 0.25  # SL = POI edge ± buffer * ATR
    require_ltf_choch: bool = True    # must see 15m CHoCH in bias direction
    min_rr: float = 1.5               # refuse trades where TP/SL ratio is worse than this
    # Rolling lookback — only scan the last N bars for structure/POIs. Keeps
    # evaluate() O(1) regardless of how much history we have. 300 bars on 1H
    # = ~12 days of structure, plenty for SMC. 300 on 15m = ~3 days.
    max_structure_lookback_bars: int = 300


class SMCStrategy(Strategy):
    name = "smc"
    required_indicators = ["structure", "fvg", "ob", "poi", "atr"]

    def __init__(self, params: SMCParams | None = None) -> None:
        self.params = params or SMCParams()

    def evaluate(self, state: MarketState) -> Signal | None:
        p = self.params
        # Cap scan window — see SMCParams.max_structure_lookback_bars.
        be_full = state.bars_entry
        bt_full = state.bars_trend
        be = be_full.iloc[-p.max_structure_lookback_bars:] if len(be_full) > p.max_structure_lookback_bars else be_full
        bt = bt_full.iloc[-p.max_structure_lookback_bars:] if len(bt_full) > p.max_structure_lookback_bars else bt_full

        if len(be) < 60 or len(bt) < 60:
            return None

        # === 1. HTF bias from 1H market structure ===
        htf = analyse_structure(bt, pivot=p.htf_pivot)
        if htf.current_bias == "none":
            return None

        # === 2. HTF POIs (OB+FVG confluences on 1H) ===
        htf_obs = find_order_blocks(bt, htf.events, min_impulse_bars=p.min_impulse_bars)
        htf_fvgs = find_fvgs(bt, max_age_bars=200)
        atr_htf = atr(bt["High"], bt["Low"], bt["Close"], p.atr_period)
        atr_htf_val = float(atr_htf.iloc[-1]) if not pd.isna(atr_htf.iloc[-1]) else 0.0

        pois = build_pois(
            snap=htf, obs=htf_obs, fvgs=htf_fvgs,
            current_idx=len(bt) - 1, atr_val=atr_htf_val,
            freshness_bars=p.poi_freshness_bars_htf,
        )

        side_str = "bull" if htf.current_bias == "up" else "bear"
        directional_pois = pois_in_direction(pois, side=side_str)
        if not directional_pois:
            return _watch_signal(state, be, htf, "no_directional_pois")

        # filter by min score
        good_pois = [poi for poi in directional_pois if poi.score >= p.min_poi_score]
        if not good_pois:
            return _watch_signal(state, be, htf, "no_high_score_poi")

        # === 3. Is current price inside any of these POIs? ===
        last15 = be.iloc[-1]
        price = float(last15["Close"])
        active_poi = best_poi_containing_price(good_pois, price, side=side_str)
        if active_poi is None:
            # Price hasn't reached a POI yet. Send breakout-watch if a POI is close.
            nearest = min(good_pois, key=lambda po: abs(price - po.mid))
            distance = abs(price - nearest.mid)
            atr_ltf = atr(be["High"], be["Low"], be["Close"], p.atr_period)
            atr_ltf_val = float(atr_ltf.iloc[-1]) if not pd.isna(atr_ltf.iloc[-1]) else 0.0
            if atr_ltf_val > 0 and distance <= 1.5 * atr_ltf_val:
                return Signal(
                    ts=_ts(be.index[-1]), symbol=state.symbol,
                    side="BUY" if side_str == "bull" else "SELL",
                    severity=Severity.BREAKOUT_WATCH,
                    price=price, atr=atr_ltf_val,
                    reason=f"approaching POI score={nearest.score} ({','.join(nearest.reasons)})",
                    extras={"poi_top": nearest.top, "poi_bottom": nearest.bottom},
                )
            return _watch_signal(state, be, htf, "waiting_for_poi_mitigation")

        # === 4. LTF (15m) confirmation: CHoCH in bias direction ===
        ltf = analyse_structure(be, pivot=p.ltf_pivot)
        atr_ltf = atr(be["High"], be["Low"], be["Close"], p.atr_period)
        atr_ltf_val = float(atr_ltf.iloc[-1])
        if pd.isna(atr_ltf_val):
            return None

        ltf_confirm = False
        if not p.require_ltf_choch:
            ltf_confirm = True
        else:
            # Look at the last few LTF events; require the latest event to match bias
            recent_events = [e for e in ltf.events if e.idx >= len(be) - 10]
            if recent_events:
                latest = recent_events[-1]
                if latest.side == htf.current_bias and latest.kind in ("CHoCH", "BOS"):
                    ltf_confirm = True

        if not ltf_confirm:
            return Signal(
                ts=_ts(be.index[-1]), symbol=state.symbol,
                side="BUY" if side_str == "bull" else "SELL",
                severity=Severity.BREAKOUT_WATCH,
                price=price, atr=atr_ltf_val,
                reason=f"in POI score={active_poi.score}, awaiting 15m CHoCH",
                extras={"poi_top": active_poi.top, "poi_bottom": active_poi.bottom},
            )

        # === 5. Compute SL and TP. Reject if RR too poor. ===
        buffer = p.sl_buffer_atr_frac * atr_ltf_val
        if htf.current_bias == "up":
            sl = active_poi.bottom - buffer
            # TP = next opposite-side liquidity (most recent unbroken swing high above price)
            future_highs = [s for s in htf.swings if s.kind == "high" and s.price > price]
            if future_highs:
                tp = future_highs[0].price  # nearest above
            else:
                tp = price + 2.5 * atr_ltf_val
            entry = price
            side = "BUY"
        else:
            sl = active_poi.top + buffer
            future_lows = [s for s in htf.swings if s.kind == "low" and s.price < price]
            if future_lows:
                tp = future_lows[0].price
            else:
                tp = price - 2.5 * atr_ltf_val
            entry = price
            side = "SELL"

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or (reward / risk) < p.min_rr:
            return Signal(
                ts=_ts(be.index[-1]), symbol=state.symbol, side="NONE",
                severity=Severity.WATCHLIST,
                price=price, atr=atr_ltf_val,
                reason=f"skipped: rr_too_low ({reward/max(risk,1e-9):.2f})",
                extras={"rejection_reason": "rr_too_low", "skipped": True},
            )

        # READY to fire
        severity = Severity.BUY_READY if side == "BUY" else Severity.SELL_READY
        return Signal(
            ts=_ts(be.index[-1]), symbol=state.symbol, side=side,
            severity=severity,
            price=price, atr=atr_ltf_val,
            reason=(f"POI mitigated + 15m {('BOS/CHoCH' if p.require_ltf_choch else 'confirm')} "
                    f"in {htf.current_bias} bias, score={active_poi.score} "
                    f"({','.join(active_poi.reasons)})"),
            extras={
                "poi_top": active_poi.top,
                "poi_bottom": active_poi.bottom,
                "sl_suggested": sl,
                "tp_suggested": tp,
                "rr": reward / risk,
                "htf_bias": htf.current_bias,
            },
        )


def _watch_signal(state: MarketState, be: pd.DataFrame, htf, reason: str) -> Signal:
    side_str = "BUY" if htf.current_bias == "up" else "SELL" if htf.current_bias == "down" else "NONE"
    price = float(be.iloc[-1]["Close"])
    return Signal(
        ts=_ts(be.index[-1]), symbol=state.symbol, side=side_str,
        severity=Severity.WATCHLIST,
        price=price, atr=0.0,
        reason=f"HTF bias {htf.current_bias}: {reason}",
    )


def _ts(idx_value) -> datetime:
    if isinstance(idx_value, pd.Timestamp):
        ts = idx_value.to_pydatetime()
    else:
        ts = idx_value
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts
