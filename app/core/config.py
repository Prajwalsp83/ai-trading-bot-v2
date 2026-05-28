"""
Loads and validates config.yaml. Single source of truth for all tunables.

Skeleton — no implementation yet.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float        # e.g. 0.01 == 1% of equity per trade
    daily_loss_cap_pct: float        # halt new trades if breached
    max_drawdown_pct: float          # halt all trades if breached
    max_concurrent_positions: int
    cooldown_losses: int             # N consecutive losses before pause
    cooldown_minutes: int


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    timeframe_entry: str             # e.g. "15m"
    timeframe_trend: str             # e.g. "1h"
    ema_fast: int
    ema_slow: int
    atr_period: int
    atr_min: float                   # min ATR to take a signal


@dataclass(frozen=True)
class ExecutionConfig:
    broker: Literal["paper", "mt5"]
    mode: Literal["backtest", "paper", "semi", "auto"]
    spread_pips: float
    slippage_pips: float
    commission_per_lot: float


@dataclass(frozen=True)
class AppConfig:
    symbols: list[str]
    poll_seconds: int
    telegram: TelegramConfig
    risk: RiskConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    journal_path: Path
    event_log_path: Path
    sound_dir: Path
    dashboard_port: int


def load_config(path: Path | str = "config.yaml") -> AppConfig:
    """Load YAML, validate, return typed AppConfig.

    TODO: read YAML, coerce types, raise on missing keys, expand env vars
    for secrets (TELEGRAM_BOT_TOKEN).
    """
    raise NotImplementedError
