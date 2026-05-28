"""
Phase A — canonical config loader.

Reads config.yaml from project root + .env for secrets. Validates required
keys. Returns a typed `BotConfig` that both bots consume at startup.

Design goals:
  - Bots have ZERO hardcoded constants. All knobs in config.yaml.
  - Missing config keys = log warning + use sensible default, never crash.
  - Sensitive values (passwords, tokens) ONLY in .env, never in config.yaml.
  - Single startup log line shows every effective param.

Use from bot scripts:
    from _config_loader import load_config
    CFG = load_config(strategy_name="breakout")
    CFG.print_summary()
    print(f"Trading with risk={CFG.risk.risk_per_trade_pct*100:.2f}%")

If config.yaml is missing or malformed, raises ConfigError with a clear msg.
We CRASH on bad config rather than trading with bad params.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import yaml


# ============================== PATHS ===============================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_PATH = ROOT / "config.yaml"


# ============================== ERRORS ==============================
class ConfigError(Exception):
    """Raised on missing required keys or malformed config."""


# ============================== DATACLASSES =========================
@dataclass
class MT5Config:
    symbol: str
    poll_seconds: int


@dataclass
class TelegramConfig:
    bot_token: str | None
    chat_id: str | None


@dataclass
class DDTier:
    threshold_pct: float
    multiplier: float


@dataclass
class KellyConfig:
    enabled: bool
    lookback_trades: int
    fraction: float
    min_trades_required: int
    max_multiplier: float
    min_multiplier: float
    default_multiplier: float


@dataclass
class RiskConfig:
    risk_per_trade_pct: float
    max_concurrent_positions: int
    daily_loss_cap_pct: float
    max_drawdown_pct: float
    cooldown_after_consecutive_losses: int
    cooldown_minutes_after_losses: int
    reentry_block_minutes: int
    dd_tiers: list[DDTier]
    kelly: KellyConfig


@dataclass
class SessionWindowConfig:
    name: str
    start_ist: str   # "HH:MM"
    end_ist: str


@dataclass
class SessionsConfig:
    enabled: bool
    block_weekend: bool
    windows: list[SessionWindowConfig]


@dataclass
class CalendarConfig:
    enabled: bool
    path: str
    before_minutes: int
    after_minutes: int
    impact_threshold: str
    currencies: list[str]


@dataclass
class NewsConfig:
    enabled: bool
    cache_path: str
    cache_ttl_minutes: int
    max_age_hours: int
    block_threshold: float
    api_key: str | None    # from env


@dataclass
class RegimeConfig:
    enabled: bool
    adx_period: int
    adx_trend_min: float
    adx_chop_max: float
    high_vol_pct: float
    weights: dict   # {regime_name: {strategy_name: weight}}


@dataclass
class BreakoutParams:
    magic: int
    ema_fast: int
    ema_slow: int
    atr_period: int
    atr_min: float
    atr_pct_min: float
    min_trend_strength: float
    use_4h_trend_gate: bool
    k_sl: float
    k_tp: float


@dataclass
class SMCParams:
    magic: int
    htf_pivot: int
    ltf_pivot: int
    min_impulse_bars: int
    poi_freshness_bars: int
    min_poi_score: int
    sl_buffer_atr_frac: float
    require_ltf_choch: bool
    min_rr: float
    atr_period: int
    max_structure_lookback_bars: int


@dataclass
class MLConfig:
    enabled: bool
    shadow_mode: bool
    model_path: str
    meta_path: str


@dataclass
class ReportingConfig:
    daily_summary_hour_ist: int
    daily_summary_minute_ist: int


@dataclass
class MT5Credentials:
    path: str | None
    login: int | None
    password: str | None
    server: str | None


@dataclass
class DBCredentials:
    host: str | None
    port: int
    user: str | None
    password: str | None
    dbname: str


@dataclass
class BotConfig:
    """Top-level config consumed by each bot. `strategy` is one of the per-bot params."""
    strategy_name: Literal["breakout", "smc"]
    mt5: MT5Config
    mt5_creds: MT5Credentials
    telegram: TelegramConfig
    postgres_enabled: bool
    db_creds: DBCredentials
    risk: RiskConfig
    sessions: SessionsConfig
    calendar: CalendarConfig
    news: NewsConfig
    regime: RegimeConfig
    ml: MLConfig
    reporting: ReportingConfig
    strategy: BreakoutParams | SMCParams
    journal_csv: str

    def print_summary(self) -> None:
        """Single line per concern so bot startup log shows everything."""
        print(f"[config] strategy={self.strategy_name} "
              f"magic={self.strategy.magic} symbol={self.mt5.symbol}", flush=True)
        print(f"[config] risk: {self.risk.risk_per_trade_pct*100:.2f}%/trade "
              f"daily_cap={self.risk.daily_loss_cap_pct*100:.1f}% "
              f"max_dd={self.risk.max_drawdown_pct*100:.0f}% "
              f"cooldown={self.risk.cooldown_after_consecutive_losses}L→{self.risk.cooldown_minutes_after_losses}m "
              f"reentry={self.risk.reentry_block_minutes}m", flush=True)
        print(f"[config] dd_tiers: " + " | ".join(
            f"≥{t.threshold_pct*100:.0f}%→×{t.multiplier:.2f}" for t in self.risk.dd_tiers), flush=True)
        print(f"[config] kelly: enabled={self.risk.kelly.enabled} "
              f"lookback={self.risk.kelly.lookback_trades} "
              f"fraction={self.risk.kelly.fraction} "
              f"cap=[×{self.risk.kelly.min_multiplier},×{self.risk.kelly.max_multiplier}]", flush=True)
        print(f"[config] gates: sessions={self.sessions.enabled} "
              f"calendar={self.calendar.enabled} news={self.news.enabled} "
              f"regime={self.regime.enabled}", flush=True)
        print(f"[config] ml: enabled={self.ml.enabled} shadow={self.ml.shadow_mode}", flush=True)
        if isinstance(self.strategy, BreakoutParams):
            s = self.strategy
            print(f"[config] breakout: ema={s.ema_fast}/{s.ema_slow} atr={s.atr_period} "
                  f"atr_min={s.atr_min} atr_pct_min={s.atr_pct_min} "
                  f"4h_gate={s.use_4h_trend_gate} k_sl={s.k_sl} k_tp={s.k_tp}", flush=True)
        else:
            s = self.strategy
            print(f"[config] smc: pivot=h{s.htf_pivot}/l{s.ltf_pivot} "
                  f"poi_min={s.min_poi_score} fresh={s.poi_freshness_bars}b "
                  f"sl_buf={s.sl_buffer_atr_frac}×atr require_choch={s.require_ltf_choch} "
                  f"min_rr={s.min_rr}", flush=True)


# ============================== LOADER ==============================
def _get(d: dict, path: str, default: Any = None, required: bool = False) -> Any:
    """Nested .get with dotted path, e.g. 'risk.kelly.fraction'."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if required:
                raise ConfigError(f"missing required config key: {path}")
            return default
        cur = cur[part]
    return cur


def _load_env() -> dict:
    """Load .env if present. Returns dict (env vars are also in os.environ after this)."""
    try:
        from dotenv import load_dotenv, dotenv_values
        env_path = ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return dotenv_values(env_path)
    except ImportError:
        pass
    return {}


def load_config(strategy_name: Literal["breakout", "smc"]) -> BotConfig:
    """Load config.yaml + .env. Strategy-specific params returned based on strategy_name."""
    if not CONFIG_PATH.exists():
        raise ConfigError(f"config.yaml not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f) or {}

    _load_env()    # ensures os.getenv works for secrets

    # === MT5 ===
    mt5 = MT5Config(
        symbol=_get(raw, "mt5.symbol", "GOLD.i#"),
        poll_seconds=int(_get(raw, "mt5.poll_seconds", 60)),
    )
    mt5_creds = MT5Credentials(
        path=os.getenv("MT5_PATH"),
        login=int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None,
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
    )

    # === Telegram ===
    tg_token_env = _get(raw, "telegram.bot_token_env", "TELEGRAM_BOT_TOKEN")
    tg_chat_env = _get(raw, "telegram.chat_id_env", "TELEGRAM_CHAT_ID")
    telegram = TelegramConfig(
        bot_token=os.getenv(tg_token_env),
        chat_id=os.getenv(tg_chat_env),
    )

    # === Postgres ===
    postgres_enabled = bool(_get(raw, "postgres.enabled", True))
    db_creds = DBCredentials(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME", "postgres"),
    )

    # === Risk ===
    dd_tier_raw = _get(raw, "risk.dd_tiers", [])
    dd_tiers = [DDTier(threshold_pct=float(t["threshold_pct"]),
                       multiplier=float(t["multiplier"])) for t in dd_tier_raw]
    kelly_raw = _get(raw, "risk.kelly", {})
    kelly = KellyConfig(
        enabled=bool(kelly_raw.get("enabled", True)),
        lookback_trades=int(kelly_raw.get("lookback_trades", 30)),
        fraction=float(kelly_raw.get("fraction", 0.25)),
        min_trades_required=int(kelly_raw.get("min_trades_required", 10)),
        max_multiplier=float(kelly_raw.get("max_multiplier", 2.0)),
        min_multiplier=float(kelly_raw.get("min_multiplier", 0.25)),
        default_multiplier=float(kelly_raw.get("default_multiplier", 1.0)),
    )
    risk = RiskConfig(
        risk_per_trade_pct=float(_get(raw, "risk.risk_per_trade_pct", 0.01)),
        max_concurrent_positions=int(_get(raw, "risk.max_concurrent_positions", 1)),
        daily_loss_cap_pct=float(_get(raw, "risk.daily_loss_cap_pct", 0.03)),
        max_drawdown_pct=float(_get(raw, "risk.max_drawdown_pct", 0.15)),
        cooldown_after_consecutive_losses=int(_get(raw, "risk.cooldown_after_consecutive_losses", 2)),
        cooldown_minutes_after_losses=int(_get(raw, "risk.cooldown_minutes_after_losses", 240)),
        reentry_block_minutes=int(_get(raw, "risk.reentry_block_minutes", 120)),
        dd_tiers=dd_tiers,
        kelly=kelly,
    )

    # === Sessions ===
    windows_raw = _get(raw, "sessions.windows", [])
    sessions = SessionsConfig(
        enabled=bool(_get(raw, "sessions.enabled", True)),
        block_weekend=bool(_get(raw, "sessions.block_weekend", True)),
        windows=[SessionWindowConfig(name=w["name"], start_ist=w["start_ist"],
                                      end_ist=w["end_ist"]) for w in windows_raw],
    )

    # === Calendar ===
    calendar = CalendarConfig(
        enabled=bool(_get(raw, "calendar.enabled", True)),
        path=str(_get(raw, "calendar.path", "./data/economic_calendar.json")),
        before_minutes=int(_get(raw, "calendar.before_minutes", 30)),
        after_minutes=int(_get(raw, "calendar.after_minutes", 60)),
        impact_threshold=str(_get(raw, "calendar.impact_threshold", "high")),
        currencies=list(_get(raw, "calendar.currencies", ["USD"])),
    )

    # === News ===
    news = NewsConfig(
        enabled=bool(_get(raw, "news.enabled", True)),
        cache_path=str(_get(raw, "news.cache_path", "./data/.av_news_cache.json")),
        cache_ttl_minutes=int(_get(raw, "news.cache_ttl_minutes", 25)),
        max_age_hours=int(_get(raw, "news.max_age_hours", 6)),
        block_threshold=float(_get(raw, "news.block_threshold", 0.35)),
        api_key=os.getenv("ALPHA_VANTAGE_KEY"),
    )

    # === Regime ===
    regime = RegimeConfig(
        enabled=bool(_get(raw, "regime.enabled", True)),
        adx_period=int(_get(raw, "regime.adx_period", 14)),
        adx_trend_min=float(_get(raw, "regime.adx_trend_min", 25.0)),
        adx_chop_max=float(_get(raw, "regime.adx_chop_max", 20.0)),
        high_vol_pct=float(_get(raw, "regime.high_vol_pct", 0.95)),
        weights=dict(_get(raw, "regime.weights", {})),
    )

    # === ML ===
    ml = MLConfig(
        enabled=bool(_get(raw, "ml.enabled", True)),
        shadow_mode=_resolve_shadow_mode(raw),
        model_path=str(_get(raw, "ml.model_path", "./models/meta_labeler.pkl")),
        meta_path=str(_get(raw, "ml.meta_path", "./models/meta_labeler.meta.json")),
    )

    # === Reporting ===
    reporting = ReportingConfig(
        daily_summary_hour_ist=int(_get(raw, "reporting.daily_summary_hour_ist", 23)),
        daily_summary_minute_ist=int(_get(raw, "reporting.daily_summary_minute_ist", 55)),
    )

    # === Per-strategy params + journal CSV path ===
    journals_raw = _get(raw, "journals", {})
    if strategy_name == "breakout":
        bp_raw = _get(raw, "strategies.breakout", {}, required=True)
        global_stops = _get(raw, "stops", {})
        strategy = BreakoutParams(
            magic=int(bp_raw["magic"]),
            ema_fast=int(bp_raw.get("ema_fast", 50)),
            ema_slow=int(bp_raw.get("ema_slow", 200)),
            atr_period=int(bp_raw.get("atr_period", 14)),
            atr_min=float(bp_raw.get("atr_min", 10.0)),
            atr_pct_min=float(bp_raw.get("atr_pct_min", 0.25)),
            min_trend_strength=float(bp_raw.get("min_trend_strength", 0.0)),
            use_4h_trend_gate=bool(bp_raw.get("use_4h_trend_gate", False)),
            k_sl=float(bp_raw.get("k_sl", global_stops.get("k_sl", 1.5))),
            k_tp=float(bp_raw.get("k_tp", global_stops.get("k_tp", 2.5))),
        )
        journal_csv = str(journals_raw.get("breakout", "./data/mt5_trades.csv"))
    elif strategy_name == "smc":
        sp_raw = _get(raw, "strategies.smc", {}, required=True)
        strategy = SMCParams(
            magic=int(sp_raw["magic"]),
            htf_pivot=int(sp_raw.get("htf_pivot", 2)),
            ltf_pivot=int(sp_raw.get("ltf_pivot", 2)),
            min_impulse_bars=int(sp_raw.get("min_impulse_bars", 3)),
            poi_freshness_bars=int(sp_raw.get("poi_freshness_bars", 60)),
            min_poi_score=int(sp_raw.get("min_poi_score", 2)),
            sl_buffer_atr_frac=float(sp_raw.get("sl_buffer_atr_frac", 0.25)),
            require_ltf_choch=bool(sp_raw.get("require_ltf_choch", False)),
            min_rr=float(sp_raw.get("min_rr", 1.5)),
            atr_period=int(sp_raw.get("atr_period", 14)),
            max_structure_lookback_bars=int(sp_raw.get("max_structure_lookback_bars", 300)),
        )
        journal_csv = str(journals_raw.get("smc", "./data/mt5_smc_trades.csv"))
    else:
        raise ConfigError(f"unknown strategy_name: {strategy_name}")

    cfg = BotConfig(
        strategy_name=strategy_name,
        mt5=mt5,
        mt5_creds=mt5_creds,
        telegram=telegram,
        postgres_enabled=postgres_enabled,
        db_creds=db_creds,
        risk=risk,
        sessions=sessions,
        calendar=calendar,
        news=news,
        regime=regime,
        ml=ml,
        reporting=reporting,
        strategy=strategy,
        journal_csv=journal_csv,
    )
    _validate(cfg)
    return cfg


def _resolve_shadow_mode(raw: dict) -> bool:
    """ML_SHADOW_MODE env var (if set) overrides config — lets user flip from
    shadow to live without a code change. Default: shadow (True)."""
    env = os.getenv("ML_SHADOW_MODE")
    if env is not None:
        return env.lower() in ("true", "1", "yes", "on")
    return bool(_get(raw, "ml.shadow_mode", True))


def _validate(cfg: BotConfig) -> None:
    """Refuse to run with insane config values."""
    errs = []
    if not (0 < cfg.risk.risk_per_trade_pct <= 0.10):
        errs.append(f"risk_per_trade_pct={cfg.risk.risk_per_trade_pct} out of (0, 0.10]")
    if not (0 < cfg.risk.daily_loss_cap_pct <= 0.20):
        errs.append(f"daily_loss_cap_pct={cfg.risk.daily_loss_cap_pct} out of (0, 0.20]")
    if not (0 < cfg.risk.max_drawdown_pct <= 0.50):
        errs.append(f"max_drawdown_pct={cfg.risk.max_drawdown_pct} out of (0, 0.50]")
    if cfg.risk.max_concurrent_positions < 1:
        errs.append("max_concurrent_positions must be >= 1")
    if not cfg.risk.dd_tiers:
        errs.append("dd_tiers is empty — risk system needs at least one tier")
    if cfg.strategy.magic <= 0:
        errs.append(f"strategy.magic={cfg.strategy.magic} invalid")
    if errs:
        raise ConfigError("config validation failed:\n  - " + "\n  - ".join(errs))


# ============================== CLI =================================
if __name__ == "__main__":
    """Run directly to see the parsed config: python _config_loader.py [breakout|smc]"""
    name = sys.argv[1] if len(sys.argv) > 1 else "breakout"
    try:
        cfg = load_config(name)
        print(f"=== Loaded config for strategy={name} ===")
        cfg.print_summary()
        print()
        print(f"MT5 creds: path={cfg.mt5_creds.path is not None}, "
              f"login={cfg.mt5_creds.login is not None}, "
              f"password={'***' if cfg.mt5_creds.password else 'MISSING'}, "
              f"server={cfg.mt5_creds.server}")
        print(f"DB creds:  host={cfg.db_creds.host}, "
              f"password={'***' if cfg.db_creds.password else 'MISSING'}")
        print(f"Telegram:  token={'***' if cfg.telegram.bot_token else 'MISSING'}, "
              f"chat={cfg.telegram.chat_id}")
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        sys.exit(1)
