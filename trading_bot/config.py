from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    lookback_window: int = 96
    test_size: float = 0.2


@dataclass(frozen=True)
class ModelConfig:
    type: str = "random_forest"
    random_state: int = 42
    probability_threshold: float = 0.55
    n_estimators: int = 300
    max_depth: int = 8


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1000.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    risk_per_trade: float = 0.25
    stop_loss_pct: float = 0.02


@dataclass(frozen=True)
class LiveConfig:
    exchange_id: str = "binance"
    dry_run: bool = True
    sandbox: bool = False
    max_position_fraction: float = 0.25
    poll_interval_seconds: float = 30.0
    candle_limit: int = 200
    state_path: str = "reports/live_state.json"
    journal_path: str = "reports/trades.jsonl"
    native_protection: bool = True
    require_native_protection: bool = False
    require_approved_model: bool = True


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_fraction: float = 0.005
    atr_stop_multiplier: float = 2.0
    take_profit_pct: float = 0.04
    trailing_stop_pct: float = 0.015
    break_even_trigger_pct: float = 0.02
    max_daily_loss_fraction: float = 0.03
    max_consecutive_losses: int = 3
    cooldown_minutes: int = 240


@dataclass(frozen=True)
class PortfolioConfig:
    max_total_exposure_fraction: float = 0.5
    max_open_positions: int = 3
    correlation_threshold: float = 0.8
    state_path: str = "reports/portfolio_state.json"


@dataclass(frozen=True)
class ValidationConfig:
    folds: int = 3
    min_accuracy: float = 0.52
    min_roc_auc: float = 0.52
    max_drawdown_pct: float = -20.0


@dataclass(frozen=True)
class BotConfig:
    data: DataConfig
    model: ModelConfig
    backtest: BacktestConfig
    live: LiveConfig
    risk: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def load_config(path: str | Path = "configs/default.yaml") -> BotConfig:
    config_path = Path(path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    config = BotConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        live=LiveConfig(**raw.get("live", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        portfolio=PortfolioConfig(**raw.get("portfolio", {})),
        validation=ValidationConfig(**raw.get("validation", {})),
    )
    config = replace(
        config,
        data=replace(
            config.data,
            symbol=os.environ.get("TRADING_SYMBOL", config.data.symbol),
            timeframe=os.environ.get("TRADING_TIMEFRAME", config.data.timeframe),
        ),
        live=replace(
            config.live,
            exchange_id=os.environ.get("EXCHANGE_ID", config.live.exchange_id),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: BotConfig) -> None:
    if not 0.5 < config.model.probability_threshold < 1:
        raise ValueError("model.probability_threshold must be between 0.5 and 1")
    if not 0 < config.data.test_size < 1:
        raise ValueError("data.test_size must be between 0 and 1")
    if not 0 < config.live.max_position_fraction <= 1:
        raise ValueError("live.max_position_fraction must be between 0 and 1")
    if not 0 < config.backtest.stop_loss_pct < 1:
        raise ValueError("backtest.stop_loss_pct must be between 0 and 1")
    if config.live.poll_interval_seconds <= 0:
        raise ValueError("live.poll_interval_seconds must be positive")
    if config.live.candle_limit < 100:
        raise ValueError("live.candle_limit must be at least 100")
    fractions = {
        "risk.risk_per_trade_fraction": config.risk.risk_per_trade_fraction,
        "risk.max_daily_loss_fraction": config.risk.max_daily_loss_fraction,
        "portfolio.max_total_exposure_fraction": (
            config.portfolio.max_total_exposure_fraction
        ),
    }
    for name, value in fractions.items():
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if config.risk.atr_stop_multiplier <= 0:
        raise ValueError("risk.atr_stop_multiplier must be positive")
    if not 0 < config.risk.take_profit_pct < 1:
        raise ValueError("risk.take_profit_pct must be between 0 and 1")
    if not 0 <= config.risk.trailing_stop_pct < 1:
        raise ValueError("risk.trailing_stop_pct must be between 0 and 1")
    if config.risk.max_consecutive_losses < 1:
        raise ValueError("risk.max_consecutive_losses must be positive")
    if config.portfolio.max_open_positions < 1:
        raise ValueError("portfolio.max_open_positions must be positive")
    if not -1 <= config.portfolio.correlation_threshold <= 1:
        raise ValueError("portfolio.correlation_threshold must be between -1 and 1")
    if config.validation.folds < 2:
        raise ValueError("validation.folds must be at least 2")
