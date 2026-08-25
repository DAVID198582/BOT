from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.config import BacktestConfig


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(features: pd.DataFrame, signals: pd.Series, config: BacktestConfig) -> BacktestResult:
    cash = config.initial_cash
    units = 0.0
    entry_price = 0.0
    entry_fee = 0.0
    position_side = 0
    equity_rows: list[dict[str, float | pd.Timestamp]] = []
    trades: list[dict[str, float | str | pd.Timestamp]] = []

    for row_index, row in features.reset_index(drop=True).iterrows():
        signal = int(signals.iloc[row_index])
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        timestamp = row["timestamp"]

        if position_side != 0:
            stop_price = entry_price * (1 - config.stop_loss_pct if position_side > 0 else 1 + config.stop_loss_pct)
            stopped = low_price <= stop_price if position_side > 0 else high_price >= stop_price
            if stopped:
                cash, pnl = _close_position(cash, units, position_side, entry_price, entry_fee, stop_price, config)
                trades.append(_trade(timestamp, "stop", stop_price, position_side, pnl))
                units = 0.0
                position_side = 0

        if signal != position_side:
            if position_side != 0:
                cash, pnl = _close_position(cash, units, position_side, entry_price, entry_fee, open_price, config)
                trades.append(_trade(timestamp, "exit", open_price, position_side, pnl))
                units = 0.0
                position_side = 0
            if signal != 0:
                allocation = cash * config.risk_per_trade
                execution_price = _entry_price(open_price, signal, config)
                units = allocation / execution_price
                entry_fee = allocation * config.fee_rate
                cash = cash - allocation - entry_fee if signal > 0 else cash + allocation - entry_fee
                entry_price = execution_price
                position_side = signal
                trades.append(_trade(timestamp, "entry", execution_price, position_side, -entry_fee))

        position_value = units * close_price * position_side if position_side != 0 else 0.0
        equity = cash + position_value
        equity_rows.append({"timestamp": timestamp, "equity": equity, "cash": cash, "position": float(position_side)})

    if position_side != 0 and not features.empty:
        last_row = features.iloc[-1]
        cash, pnl = _close_position(cash, units, position_side, entry_price, entry_fee, float(last_row["close"]), config)
        trades.append(_trade(last_row["timestamp"], "final_exit", float(last_row["close"]), position_side, pnl))
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["position"] = 0.0

    equity_curve = pd.DataFrame(equity_rows)
    trades_frame = pd.DataFrame(trades)
    return BacktestResult(equity_curve, trades_frame, performance_metrics(equity_curve, trades_frame, config.initial_cash))


def _entry_price(price: float, side: int, config: BacktestConfig) -> float:
    return price * (1 + config.slippage_rate if side > 0 else 1 - config.slippage_rate)


def _exit_price(price: float, side: int, config: BacktestConfig) -> float:
    return price * (1 - config.slippage_rate if side > 0 else 1 + config.slippage_rate)


def _close_position(
    cash: float,
    units: float,
    side: int,
    entry_price: float,
    entry_fee: float,
    price: float,
    config: BacktestConfig,
) -> tuple[float, float]:
    execution_price = _exit_price(price, side, config)
    exit_notional = units * execution_price
    entry_notional = units * entry_price
    exit_fee = exit_notional * config.fee_rate
    if side > 0:
        new_cash = cash + exit_notional - exit_fee
        pnl = exit_notional - entry_notional - entry_fee - exit_fee
    else:
        new_cash = cash - exit_notional - exit_fee
        pnl = entry_notional - exit_notional - entry_fee - exit_fee
    return new_cash, pnl


def _trade(timestamp: pd.Timestamp, action: str, price: float, side: int, pnl: float) -> dict[str, float | str | pd.Timestamp]:
    return {"timestamp": timestamp, "action": action, "price": price, "side": float(side), "pnl": pnl}


def performance_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, initial_cash: float) -> dict[str, float]:
    if equity_curve.empty:
        return {}
    equity = equity_curve["equity"]
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    drawdown = equity / equity.cummax() - 1
    sharpe = 0.0
    if returns.std() and not np.isnan(returns.std()):
        sharpe = float(returns.mean() / returns.std() * np.sqrt(365 * 24 * 4))
    return {
        "initial_cash": float(initial_cash),
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": float((equity.iloc[-1] / initial_cash - 1) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "sharpe": sharpe,
        "trade_count": float(len(trades)),
    }
