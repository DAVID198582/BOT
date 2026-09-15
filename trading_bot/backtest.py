from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.config import BacktestConfig, RiskConfig
from trading_bot.risk import position_plan


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(
    features: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig,
    risk_config: RiskConfig | None = None,
) -> BacktestResult:
    """
    Backtest signals with fees, slippage, stop-loss and position sizing.

    Critical timing rule
    --------------------
    `signals` are generated from features of bar t (which include the close of t).
    In live trading you only know those features *after* bar t has closed.
    Therefore the signal produced on bar t is executed at the *open of bar t+1*.

    Implementation: we shift the signal series by one bar so that
    actionable_signals[t] == original_signals[t-1].
    """
    cash = config.initial_cash
    units = 0.0
    entry_price = 0.0
    entry_fee = 0.0
    position_stop_pct = config.stop_loss_pct
    position_side = 0
    equity_rows: list[dict[str, float | pd.Timestamp]] = []
    trades: list[dict[str, float | str | pd.Timestamp]] = []

    # Lag signals by one bar to eliminate look-ahead bias.
    # First bar has no actionable signal (stays flat).
    actionable_signals = signals.shift(1).fillna(0).astype(int)

    for row_index, row in features.reset_index(drop=True).iterrows():
        signal = int(actionable_signals.iloc[row_index])
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        timestamp = row["timestamp"]

        # 1. Process the prior bar's signal at this bar's open. The open occurs
        # before this bar's high/low, so signal exits must precede stop checks.
        if signal != position_side:
            if position_side != 0:
                cash, pnl = _close_position(
                    cash, units, position_side, entry_price, entry_fee, open_price, config
                )
                trades.append(_trade(timestamp, "exit", open_price, position_side, pnl))
                units = 0.0
                position_side = 0

            if signal != 0:
                if risk_config is None:
                    allocation = cash * config.risk_per_trade
                    position_stop_pct = config.stop_loss_pct
                else:
                    plan = position_plan(
                        cash,
                        open_price,
                        float(row.get("atr_14", 0.0)),
                        config.stop_loss_pct,
                        config.risk_per_trade,
                        risk_config,
                    )
                    allocation = plan.notional
                    position_stop_pct = plan.stop_pct
                if allocation > 0:
                    execution_price = _entry_price(open_price, signal, config)
                    units = allocation / execution_price
                    entry_fee = allocation * config.fee_rate
                    if signal > 0:
                        cash = cash - allocation - entry_fee
                    else:
                        # Synthetic short: receive proceeds (futures-style accounting)
                        cash = cash + allocation - entry_fee
                    entry_price = execution_price
                    position_side = signal
                    trades.append(
                        _trade(timestamp, "entry", execution_price, position_side, -entry_fee)
                    )

        # 2. Check the current bar's intrabar range after open-price orders. This
        # also allows a position entered above to stop during its entry bar.
        if position_side != 0:
            stop_price = entry_price * (
                1 - position_stop_pct if position_side > 0 else 1 + position_stop_pct
            )
            stopped = (
                low_price <= stop_price if position_side > 0 else high_price >= stop_price
            )
            if stopped:
                # A gap through the stop can only fill at the less favorable open.
                if position_side > 0 and open_price < stop_price:
                    stop_price = open_price
                elif position_side < 0 and open_price > stop_price:
                    stop_price = open_price
                cash, pnl = _close_position(
                    cash, units, position_side, entry_price, entry_fee, stop_price, config
                )
                trades.append(_trade(timestamp, "stop", stop_price, position_side, pnl))
                units = 0.0
                position_side = 0

        # 3. Mark-to-market equity at the close
        position_value = units * close_price * position_side if position_side != 0 else 0.0
        equity = cash + position_value
        equity_rows.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "cash": cash,
                "position": float(position_side),
            }
        )

    # Force flat at the end of the test period
    if position_side != 0 and not features.empty:
        last_row = features.iloc[-1]
        cash, pnl = _close_position(
            cash,
            units,
            position_side,
            entry_price,
            entry_fee,
            float(last_row["close"]),
            config,
        )
        trades.append(
            _trade(
                last_row["timestamp"],
                "final_exit",
                float(last_row["close"]),
                position_side,
                pnl,
            )
        )
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["position"] = 0.0

    equity_curve = pd.DataFrame(equity_rows)
    trades_frame = pd.DataFrame(trades)
    return BacktestResult(
        equity_curve,
        trades_frame,
        performance_metrics(equity_curve, trades_frame, config.initial_cash),
    )


def _entry_price(price: float, side: int, config: BacktestConfig) -> float:
    # Longs pay higher, shorts sell lower (adverse slippage)
    return price * (1 + config.slippage_rate if side > 0 else 1 - config.slippage_rate)


def _exit_price(price: float, side: int, config: BacktestConfig) -> float:
    # Longs sell lower, shorts buy higher (adverse slippage)
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

    if side > 0:  # closing long
        new_cash = cash + exit_notional - exit_fee
        pnl = exit_notional - entry_notional - entry_fee - exit_fee
    else:  # closing short
        new_cash = cash - exit_notional - exit_fee
        pnl = entry_notional - exit_notional - entry_fee - exit_fee

    return new_cash, pnl


def _trade(
    timestamp: pd.Timestamp, action: str, price: float, side: int, pnl: float
) -> dict[str, float | str | pd.Timestamp]:
    return {
        "timestamp": timestamp,
        "action": action,
        "price": price,
        "side": float(side),
        "pnl": pnl,
    }


def performance_metrics(
    equity_curve: pd.DataFrame, trades: pd.DataFrame, initial_cash: float
) -> dict[str, float]:
    if equity_curve.empty:
        return {}
    equity = equity_curve["equity"]
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    drawdown = equity / equity.cummax() - 1
    sharpe = 0.0
    if len(returns) > 1 and returns.std() and not np.isnan(returns.std()):
        # Crypto 15-minute bars: ~ 365 * 24 * 4 periods per year
        sharpe = float(returns.mean() / returns.std() * np.sqrt(365 * 24 * 4))
    return {
        "initial_cash": float(initial_cash),
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": float((equity.iloc[-1] / initial_cash - 1) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "sharpe": sharpe,
        "trade_count": float(len(trades)),
    }
