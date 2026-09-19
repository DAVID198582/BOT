from __future__ import annotations

from dataclasses import dataclass

from trading_bot.config import RiskConfig


@dataclass(frozen=True)
class PositionPlan:
    amount: float
    notional: float
    stop_pct: float
    stop_price: float
    take_profit_price: float


def position_plan(
    equity: float,
    price: float,
    atr_pct: float,
    fixed_stop_pct: float,
    max_position_fraction: float,
    config: RiskConfig,
) -> PositionPlan:
    if equity <= 0 or price <= 0:
        return PositionPlan(0.0, 0.0, fixed_stop_pct, 0.0, 0.0)
    volatility_stop = max(0.0, atr_pct) * config.atr_stop_multiplier
    stop_pct = max(fixed_stop_pct, volatility_stop)
    risk_budget = equity * config.risk_per_trade_fraction
    risk_limited_notional = risk_budget / stop_pct
    exposure_limited_notional = equity * max_position_fraction
    notional = max(0.0, min(risk_limited_notional, exposure_limited_notional))
    return PositionPlan(
        amount=notional / price,
        notional=notional,
        stop_pct=stop_pct,
        stop_price=price * (1 - stop_pct),
        take_profit_price=price * (1 + config.take_profit_pct),
    )


def managed_stop_price(
    entry_price: float,
    current_stop: float,
    peak_price: float,
    config: RiskConfig,
) -> tuple[float, bool]:
    stop = current_stop
    break_even = peak_price >= entry_price * (1 + config.break_even_trigger_pct)
    if break_even:
        stop = max(stop, entry_price)
    if config.trailing_stop_pct > 0 and peak_price > entry_price:
        stop = max(stop, peak_price * (1 - config.trailing_stop_pct))
    return stop, break_even
