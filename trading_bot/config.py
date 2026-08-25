from __future__ import annotations

from dataclasses import dataclass
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
    max_position_fraction: float = 0.25


@dataclass(frozen=True)
class BotConfig:
    data: DataConfig
    model: ModelConfig
    backtest: BacktestConfig
    live: LiveConfig


def load_config(path: str | Path = "configs/default.yaml") -> BotConfig:
    config_path = Path(path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    return BotConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        live=LiveConfig(**raw.get("live", {})),
    )