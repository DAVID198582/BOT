from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from trading_bot.alerts import AlertManager
from trading_bot.bot import Decision, PaperTradingBot
from trading_bot.config import BotConfig
from trading_bot.data import REQUIRED_COLUMNS
from trading_bot.journal import TradeJournal
from trading_bot.portfolio import PortfolioStore
from trading_bot.risk import PositionPlan, managed_stop_price, position_plan


@dataclass
class LiveState:
    symbol: str
    dry_run: bool
    last_candle: str | None = None
    position_amount: float = 0.0
    entry_price: float = 0.0
    entry_fee: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    peak_price: float = 0.0
    break_even_armed: bool = False
    paper_quote_balance: float = 0.0
    realized_pnl: float = 0.0
    daily_realized_pnl: float = 0.0
    day_start_equity: float = 0.0
    trading_day: str | None = None
    consecutive_losses: int = 0
    halted_until: str | None = None
    manual_halt: bool = False
    halt_reason: str | None = None
    native_protection_active: bool = False
    native_protection_mode: str | None = None
    protective_order_ids: list[str] = field(default_factory=list)
    base_balance_floor: float = 0.0
    order_id: str | None = None
    order_side: str | None = None
    order_status: str | None = None
    order_filled_applied: float = 0.0
    order_fee_applied: float = 0.0
    last_market_price: float = 0.0
    last_equity: float = 0.0
    daily_drawdown_pct: float = 0.0
    last_model_probability: float | None = None
    last_updated_at: str | None = None


class StateStore:
    def __init__(self, path: str | Path, symbol: str, dry_run: bool, initial_cash: float) -> None:
        self.path = Path(path)
        self.symbol = symbol
        self.dry_run = dry_run
        self.initial_cash = initial_cash

    def load(self) -> LiveState:
        if not self.path.exists():
            return LiveState(
                symbol=self.symbol,
                dry_run=self.dry_run,
                paper_quote_balance=self.initial_cash if self.dry_run else 0.0,
                day_start_equity=self.initial_cash if self.dry_run else 0.0,
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(LiveState)}
        state = LiveState(**{key: value for key, value in payload.items() if key in allowed})
        if state.symbol != self.symbol or state.dry_run != self.dry_run:
            raise ValueError(
                f"State file {self.path} belongs to a different symbol or trading mode"
            )
        return state

    def save(self, state: LiveState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(asdict(state), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary_path.replace(self.path)


class LiveTradingRunner:
    """Poll complete candles and manage a persistent, spot-only strategy."""

    def __init__(
        self,
        model: Any,
        config: BotConfig,
        exchange: Any,
        *,
        execute: bool = False,
        state_store: StateStore | None = None,
        journal: TradeJournal | None = None,
        alerts: AlertManager | None = None,
        portfolio: PortfolioStore | None = None,
    ) -> None:
        if execute and config.live.dry_run:
            raise ValueError("Live orders require live.dry_run: false in the config")
        self.model = model
        self.config = config
        self.exchange = exchange
        self.execute = execute
        self.dry_run = not execute
        self.store = state_store or StateStore(
            config.live.state_path,
            config.data.symbol,
            self.dry_run,
            config.backtest.initial_cash,
        )
        self.journal = journal or TradeJournal(config.live.journal_path)
        self.alerts = alerts or AlertManager()
        self.portfolio = portfolio or PortfolioStore(
            config.portfolio.state_path, config.portfolio
        )
        self.state = self.store.load()
        self._migrate_state()
        self._returns: list[float] = []

    def run_once(self) -> dict[str, Any]:
        # Reload so dashboard emergency controls are observed without a restart.
        self.state = self.store.load()
        self._migrate_state()
        candles = self._fetch_complete_candles()
        latest = candles.iloc[-1]
        candle_id = pd.Timestamp(latest["timestamp"]).isoformat()
        candle_close = float(latest["close"])
        self._reset_trading_day(candle_close)
        self._returns = (
            candles["close"].pct_change().dropna().tail(96).astype(float).tolist()
        )

        if self.execute:
            self._reconcile_order(candle_close)
            self._reconcile_balance(candle_close)
            self._migrate_state()
        self._update_equity(candle_close)

        if self.state.position_amount > 0 and self.state.entry_price > 0:
            market_price = self._fetch_current_price()
            self.state.peak_price = max(self.state.peak_price, market_price)
            self.state.stop_price, self.state.break_even_armed = managed_stop_price(
                self.state.entry_price,
                self.state.stop_price,
                self.state.peak_price,
                self.config.risk,
            )
            self.store.save(self.state)
            # Client-side trailing can tighten the native static protection. If
            # this process disappears, the original native stop remains active.
            if market_price <= self.state.stop_price:
                return self._exit_result("stop_loss", market_price, candle_id, candle_close)
            if market_price >= self.state.take_profit_price:
                return self._exit_result("take_profit", market_price, candle_id, candle_close)

        if candle_id == self.state.last_candle:
            return {
                "status": "waiting",
                "symbol": self.config.data.symbol,
                "candle": candle_id,
                "position_amount": self.state.position_amount,
                "stop_price": self.state.stop_price or None,
                "reason": "latest completed candle was already processed",
            }

        decision = PaperTradingBot(self.model, self.config).decide(candles)
        self.state.last_model_probability = decision.probability
        self.state.last_candle = candle_id
        self.store.save(self.state)

        action = "hold"
        order: dict[str, Any] | None = None
        reason = decision.action
        if self.state.position_amount <= 0 and decision.signal > 0:
            blocked = self._entry_block_reason()
            if blocked:
                action, reason = "blocked", blocked
                self._record("entry_blocked", {"reason": blocked}, alert=True)
            else:
                action, order = self._buy(candle_close, decision)
        elif self.state.position_amount > 0 and decision.signal < 0:
            action, order, _ = self._sell(candle_close, "model_exit")

        self._sync_portfolio(candle_close)
        self.store.save(self.state)
        result = self._result(
            candle_id, candle_close, decision, action, order, reason=reason
        )
        self._record("decision", result)
        return result

    def record_error(self, error: Exception) -> None:
        self._record(
            "error",
            {"error_type": type(error).__name__, "message": str(error)},
            alert=True,
        )

    def _buy(self, reference_price: float, decision: Decision) -> tuple[str, dict[str, Any]]:
        equity, balance = self._account_equity(reference_price)
        plan = position_plan(
            equity,
            reference_price,
            decision.volatility_pct,
            self.config.backtest.stop_loss_pct,
            self.config.live.max_position_fraction,
            self.config.risk,
        )
        proposed_fraction = plan.notional / equity if equity else 0.0
        allowed, reason = self.portfolio.allow_open(
            self.config.data.symbol, proposed_fraction, self._returns
        )
        if not allowed:
            self._record("entry_blocked", {"reason": reason}, alert=True)
            return "blocked", {"reason": reason}
        if plan.amount <= 0:
            return "hold", {"reason": "account equity is empty"}

        if self.dry_run:
            fee = plan.notional * self.config.backtest.fee_rate
            if plan.notional + fee > self.state.paper_quote_balance:
                return "hold", {"reason": "paper quote balance is too small"}
            self.state.paper_quote_balance -= plan.notional + fee
            self._apply_entry(plan.amount, reference_price, fee, plan)
            summary = {
                "amount": plan.amount,
                "notional": plan.notional,
                "reference_price": reference_price,
                "fee": fee,
            }
        else:
            base, _ = split_symbol(self.config.data.symbol)
            self.state.base_balance_floor = balance_total(balance, base)
            amount = self._precise_amount(plan.amount, reference_price)
            plan = PositionPlan(
                amount,
                amount * reference_price,
                plan.stop_pct,
                plan.stop_price,
                plan.take_profit_price,
            )
            params, protection_mode = self._native_entry_params(plan)
            before_base = balance_total(balance, base)
            order = self.exchange.create_order(
                self.config.data.symbol, "market", "buy", amount, None, params
            )
            after_balance = self.exchange.fetch_balance()
            filled = order_filled(order)
            if filled <= 0:
                filled = max(0.0, balance_total(after_balance, base) - before_base)
            average = float(order.get("average") or reference_price)
            _, quote = split_symbol(self.config.data.symbol)
            fee = order_fee(order, average, base, quote)
            self._track_order(order, "buy", filled, fee)
            self.store.save(self.state)
            ensure_order_progress(order, filled)
            self._apply_entry(filled, average, fee, plan)
            self.state.native_protection_mode = protection_mode
            self.state.native_protection_active = protection_mode == "attached" and bool(
                filled > 0 or self.state.order_id
            )
            self.state.protective_order_ids = protective_order_ids(order)
            self.store.save(self.state)
            if protection_mode == "standalone":
                self._ensure_standalone_protection()
            summary = public_order_summary(order, filled, average, fee)

        self.store.save(self.state)
        self._sync_portfolio(reference_price)
        self._record(
            "entry",
            {
                **summary,
                "stop_price": self.state.stop_price,
                "take_profit_price": self.state.take_profit_price,
                "model_probability": decision.probability,
                "volatility_pct": decision.volatility_pct,
                "native_protection": self.state.native_protection_active,
            },
            alert=True,
        )
        return "buy", summary

    def _sell(
        self, reference_price: float, reason: str
    ) -> tuple[str, dict[str, Any], float]:
        if not self.dry_run:
            self._cancel_pending_entry(reference_price)
        position_before = self.state.position_amount
        entry_fee_before = self.state.entry_fee
        if position_before <= 0:
            return "hold", {"reason": "no managed position"}, 0.0

        if self.dry_run:
            amount = position_before
            proceeds = amount * reference_price
            exit_fee = proceeds * self.config.backtest.fee_rate
            self.state.paper_quote_balance += proceeds - exit_fee
            summary = {
                "amount": amount,
                "reference_price": reference_price,
                "fee": exit_fee,
            }
        else:
            self._cancel_protective_orders()
            base, _ = split_symbol(self.config.data.symbol)
            balance = self.exchange.fetch_balance()
            available = balance_free(balance, base)
            amount = self._precise_amount(
                min(position_before, available), reference_price
            )
            order = self.exchange.create_order(
                self.config.data.symbol, "market", "sell", amount
            )
            filled = order_filled(order)
            if filled <= 0:
                after = self.exchange.fetch_balance()
                filled = max(0.0, available - balance_free(after, base))
            amount = filled
            reference_price = float(order.get("average") or reference_price)
            _, quote = split_symbol(self.config.data.symbol)
            exit_fee = order_fee(order, reference_price, base, quote)
            self._track_order(order, "sell", filled, exit_fee)
            self.store.save(self.state)
            ensure_order_progress(order, filled)
            summary = public_order_summary(order, filled, reference_price, exit_fee)

        entry_fee_share = entry_fee_before * min(1.0, amount / position_before)
        realized = (reference_price - self.state.entry_price) * amount
        realized -= entry_fee_share + exit_fee
        self.state.position_amount = max(0.0, position_before - amount)
        self.state.entry_fee = max(0.0, entry_fee_before - entry_fee_share)
        closed = self.state.position_amount <= 1e-12
        self._register_realized_pnl(realized, closed=closed)
        if closed:
            self._clear_position()
        self.store.save(self.state)
        self._sync_portfolio(reference_price)
        self._record(
            "exit",
            {**summary, "reason": reason, "realized_pnl": realized},
            alert=True,
        )
        return "sell", summary, realized

    def _exit_result(
        self, reason: str, market_price: float, candle_id: str, candle_close: float
    ) -> dict[str, Any]:
        self.state.last_candle = candle_id
        self.store.save(self.state)
        action, order, realized = self._sell(market_price, reason)
        return {
            "status": "processed",
            "mode": "live" if self.execute else "dry-run",
            "symbol": self.config.data.symbol,
            "timeframe": self.config.data.timeframe,
            "candle": candle_id,
            "close": candle_close,
            "market_price": market_price,
            "signal": 0,
            "model_action": reason,
            "action": action,
            "stop_triggered": reason == "stop_loss",
            "realized_pnl": realized,
            "position_amount": self.state.position_amount,
            "paper_quote_balance": self.state.paper_quote_balance if self.dry_run else None,
            "order": order,
        }

    def _result(
        self,
        candle_id: str,
        close: float,
        decision: Decision,
        action: str,
        order: dict[str, Any] | None,
        *,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "status": "processed",
            "mode": "live" if self.execute else "dry-run",
            "symbol": self.config.data.symbol,
            "timeframe": self.config.data.timeframe,
            "candle": candle_id,
            "close": close,
            "signal": decision.signal,
            "model_probability": decision.probability,
            "volatility_pct": decision.volatility_pct,
            "model_action": decision.action,
            "action": action,
            "reason": reason,
            "position_amount": self.state.position_amount,
            "stop_price": self.state.stop_price or None,
            "take_profit_price": self.state.take_profit_price or None,
            "break_even_armed": self.state.break_even_armed,
            "daily_realized_pnl": self.state.daily_realized_pnl,
            "consecutive_losses": self.state.consecutive_losses,
            "halt_reason": self.state.halt_reason,
            "paper_quote_balance": self.state.paper_quote_balance if self.dry_run else None,
            "order": order,
        }

    def _apply_entry(
        self, amount: float, price: float, fee: float, plan: PositionPlan
    ) -> None:
        self.state.position_amount = amount
        self.state.entry_price = price
        self.state.entry_fee = fee
        self.state.stop_price = price * (1 - plan.stop_pct)
        self.state.take_profit_price = price * (1 + self.config.risk.take_profit_pct)
        self.state.peak_price = price
        self.state.break_even_armed = False
        if self.state.day_start_equity <= 0:
            self.state.day_start_equity = plan.notional / self.config.live.max_position_fraction

    def _clear_position(self) -> None:
        self.state.position_amount = 0.0
        self.state.entry_price = 0.0
        self.state.entry_fee = 0.0
        self.state.stop_price = 0.0
        self.state.take_profit_price = 0.0
        self.state.peak_price = 0.0
        self.state.break_even_armed = False
        self.state.native_protection_active = False
        self.state.native_protection_mode = None
        self.state.protective_order_ids = []
        self.state.base_balance_floor = 0.0

    def _register_realized_pnl(self, realized: float, *, closed: bool = True) -> None:
        self.state.realized_pnl += realized
        self.state.daily_realized_pnl += realized
        if closed:
            self.state.consecutive_losses = (
                self.state.consecutive_losses + 1 if realized < 0 else 0
            )
        daily_limit = self.state.day_start_equity * self.config.risk.max_daily_loss_fraction
        if daily_limit > 0 and self.state.daily_realized_pnl <= -daily_limit:
            self.state.halt_reason = "daily loss limit reached"
            tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
            self.state.halted_until = datetime.combine(
                tomorrow, datetime.min.time(), tzinfo=timezone.utc
            ).isoformat()
            self._record("circuit_breaker", {"reason": self.state.halt_reason}, alert=True)
        elif self.state.consecutive_losses >= self.config.risk.max_consecutive_losses:
            self.state.halt_reason = "consecutive loss limit reached"
            self.state.halted_until = (
                datetime.now(timezone.utc)
                + timedelta(minutes=self.config.risk.cooldown_minutes)
            ).isoformat()
            self._record("circuit_breaker", {"reason": self.state.halt_reason}, alert=True)

    def _entry_block_reason(self) -> str | None:
        if self.state.manual_halt:
            return self.state.halt_reason or "manual emergency halt"
        if self.state.order_status == "open":
            return "an exchange order is still open"
        if self.state.halted_until:
            halted_until = datetime.fromisoformat(self.state.halted_until)
            if datetime.now(timezone.utc) < halted_until:
                return self.state.halt_reason or "risk cooldown is active"
            self.state.halted_until = None
            self.state.halt_reason = None
        return None

    def _reset_trading_day(self, reference_price: float) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if self.state.trading_day == today:
            return
        self.state.trading_day = today
        self.state.daily_realized_pnl = 0.0
        self.state.consecutive_losses = 0
        if not self.state.manual_halt:
            self.state.halted_until = None
            self.state.halt_reason = None
        equity, _ = self._account_equity(reference_price)
        self.state.day_start_equity = equity
        self.store.save(self.state)

    def _migrate_state(self) -> None:
        if self.state.position_amount <= 0 or self.state.entry_price <= 0:
            return
        if self.state.stop_price <= 0:
            self.state.stop_price = self.state.entry_price * (
                1 - self.config.backtest.stop_loss_pct
            )
        if self.state.take_profit_price <= 0:
            self.state.take_profit_price = self.state.entry_price * (
                1 + self.config.risk.take_profit_pct
            )
        if self.state.peak_price <= 0:
            self.state.peak_price = self.state.entry_price
        self.store.save(self.state)

    def _update_equity(self, reference_price: float) -> None:
        equity, _ = self._account_equity(reference_price)
        self.state.last_market_price = reference_price
        self.state.last_equity = equity
        self.state.last_updated_at = datetime.now(timezone.utc).isoformat()
        if self.state.day_start_equity > 0:
            self.state.daily_drawdown_pct = (
                equity / self.state.day_start_equity - 1
            ) * 100
            loss_limit = self.config.risk.max_daily_loss_fraction * 100
            if self.state.daily_drawdown_pct <= -loss_limit:
                newly_halted = self.state.halt_reason != "daily equity drawdown limit reached"
                self.state.halt_reason = "daily equity drawdown limit reached"
                tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
                self.state.halted_until = datetime.combine(
                    tomorrow, datetime.min.time(), tzinfo=timezone.utc
                ).isoformat()
                if newly_halted:
                    self.store.save(self.state)
                    self._record(
                        "circuit_breaker",
                        {
                            "reason": self.state.halt_reason,
                            "daily_drawdown_pct": self.state.daily_drawdown_pct,
                        },
                        alert=True,
                    )
        self.store.save(self.state)

    def _account_equity(self, price: float) -> tuple[float, dict[str, Any]]:
        if self.dry_run:
            equity = self.state.paper_quote_balance + self.state.position_amount * price
            return equity, {}
        balance = self.exchange.fetch_balance()
        _, quote = split_symbol(self.config.data.symbol)
        equity = balance_total(balance, quote) + self.state.position_amount * price
        return equity, balance

    def _native_entry_params(self, plan: PositionPlan) -> tuple[dict[str, Any], str | None]:
        if not self.config.live.native_protection:
            if self.config.live.require_native_protection:
                raise RuntimeError("native protection is required but disabled")
            return {}, None
        stop_supported = self._feature_value("stopLoss")
        take_supported = self._feature_value("takeProfit")
        if stop_supported:
            params: dict[str, Any] = {
                "stopLoss": {"triggerPrice": plan.stop_price},
            }
            if take_supported:
                params["takeProfit"] = {"triggerPrice": plan.take_profit_price}
            return params, "attached"
        if self._feature_value("stopLossPrice"):
            return {}, "standalone"
        if self.config.live.require_native_protection:
            raise RuntimeError("exchange does not support native stop-loss orders")
        return {}, None

    def _ensure_standalone_protection(self) -> None:
        if (
            self.state.native_protection_mode != "standalone"
            or self.state.position_amount <= 0
        ):
            return
        self._cancel_protective_orders(clear_mode=False)
        amount = self._precise_amount(
            self.state.position_amount, self.state.entry_price
        )
        order = self.exchange.create_order(
            self.config.data.symbol,
            "market",
            "sell",
            amount,
            None,
            {"stopLossPrice": self.state.stop_price},
        )
        if not order.get("id"):
            raise RuntimeError("exchange did not return an id for the native stop-loss")
        self.state.protective_order_ids = [str(order["id"])]
        self.state.native_protection_active = True
        self.store.save(self.state)
        self._record(
            "native_protection",
            {
                "order_id": order.get("id"),
                "stop_price": self.state.stop_price,
                "amount": amount,
            },
        )

    def _feature_value(self, feature: str) -> bool:
        method = getattr(self.exchange, "feature_value", None)
        if method is None:
            return False
        try:
            return bool(method(self.config.data.symbol, "createOrder", feature))
        except Exception:
            return False

    def _track_order(
        self, order: dict[str, Any], side: str, filled: float, fee: float
    ) -> None:
        self.state.order_id = str(order.get("id")) if order.get("id") else None
        self.state.order_side = side
        self.state.order_status = order.get("status") or (
            "open" if self.state.order_id else None
        )
        self.state.order_filled_applied = filled
        self.state.order_fee_applied = fee

    def _reconcile_order(self, reference_price: float) -> None:
        if not self.state.order_id or self.state.order_status != "open":
            return
        if not getattr(self.exchange, "has", {}).get("fetchOrder"):
            return
        order = self.exchange.fetch_order(
            self.state.order_id, self.config.data.symbol
        )
        filled = order_filled(order)
        base, quote = split_symbol(self.config.data.symbol)
        fee = order_fee(
            order, float(order.get("average") or reference_price), base, quote
        )
        fill_delta = max(0.0, filled - self.state.order_filled_applied)
        fee_delta = max(0.0, fee - self.state.order_fee_applied)
        average = float(order.get("average") or reference_price)
        if fill_delta > 0 and self.state.order_side == "buy":
            previous_cost = self.state.entry_price * self.state.position_amount
            self.state.position_amount += fill_delta
            self.state.entry_price = (
                previous_cost + average * fill_delta
            ) / self.state.position_amount
            self.state.entry_fee += fee_delta
        elif fill_delta > 0 and self.state.order_side == "sell":
            position_before = self.state.position_amount
            entry_fee_share = self.state.entry_fee * min(1.0, fill_delta / position_before)
            realized = (average - self.state.entry_price) * fill_delta
            realized -= entry_fee_share + fee_delta
            self.state.position_amount = max(0.0, position_before - fill_delta)
            self.state.entry_fee = max(0.0, self.state.entry_fee - entry_fee_share)
            closed = self.state.position_amount <= 1e-12
            self._register_realized_pnl(realized, closed=closed)
            if closed:
                self._clear_position()
        self._track_order(order, self.state.order_side or "unknown", filled, fee)
        self.store.save(self.state)
        if self.state.order_side == "buy" and fill_delta > 0:
            self._ensure_standalone_protection()
        self._record(
            "order_reconciled",
            {"order": public_order_summary(order, filled, average, fee)},
        )

    def _reconcile_balance(self, reference_price: float) -> None:
        if self.state.position_amount <= 0 or self.state.base_balance_floor < 0:
            return
        base, _ = split_symbol(self.config.data.symbol)
        balance = self.exchange.fetch_balance()
        managed_amount = max(
            0.0, balance_total(balance, base) - self.state.base_balance_floor
        )
        if managed_amount + 1e-12 >= self.state.position_amount:
            return
        sold = self.state.position_amount - managed_amount
        position_before = self.state.position_amount
        entry_fee_share = self.state.entry_fee * min(1.0, sold / position_before)
        realized = (reference_price - self.state.entry_price) * sold - entry_fee_share
        self.state.position_amount = managed_amount
        self.state.entry_fee = max(0.0, self.state.entry_fee - entry_fee_share)
        self._register_realized_pnl(realized, closed=managed_amount <= 1e-12)
        if managed_amount <= 1e-12:
            self._clear_position()
        self.store.save(self.state)
        self._record(
            "exit" if managed_amount <= 1e-12 else "exchange_reconciliation",
            {
                "amount": sold,
                "reference_price": reference_price,
                "realized_pnl": realized,
                "reason": "exchange protective order reconciliation",
            },
            alert=True,
        )

    def _cancel_pending_entry(self, reference_price: float) -> None:
        if (
            self.state.order_side != "buy"
            or self.state.order_status != "open"
            or not self.state.order_id
        ):
            return
        self.exchange.cancel_order(self.state.order_id, self.config.data.symbol)
        if getattr(self.exchange, "has", {}).get("fetchOrder"):
            self._reconcile_order(reference_price)
        else:
            self.state.order_status = "canceled"
            self._reconcile_balance(reference_price)
        self.store.save(self.state)

    def _cancel_protective_orders(self, *, clear_mode: bool = True) -> None:
        if not self.state.protective_order_ids:
            return
        for order_id in self.state.protective_order_ids:
            try:
                self.exchange.cancel_order(order_id, self.config.data.symbol)
            except Exception as error:
                self._record(
                    "protective_cancel_error",
                    {"order_id": order_id, "error_type": type(error).__name__},
                    alert=True,
                )
                raise RuntimeError("could not cancel a protective order safely") from error
        self.state.protective_order_ids = []
        self.state.native_protection_active = False
        if clear_mode:
            self.state.native_protection_mode = None

    def _sync_portfolio(self, price: float) -> None:
        equity, _ = self._account_equity(price)
        self.state.last_market_price = price
        self.state.last_equity = equity
        self.state.last_updated_at = datetime.now(timezone.utc).isoformat()
        if self.state.day_start_equity > 0:
            self.state.daily_drawdown_pct = (
                equity / self.state.day_start_equity - 1
            ) * 100
        exposure = self.state.position_amount * price
        fraction = exposure / equity if equity > 0 else 0.0
        self.portfolio.record(self.config.data.symbol, fraction, self._returns)
        self.store.save(self.state)

    def _record(self, event: str, payload: dict[str, Any], *, alert: bool = False) -> None:
        safe_payload = {
            "symbol": self.config.data.symbol,
            "mode": "live" if self.execute else "dry-run",
            **payload,
        }
        self.journal.append(event, safe_payload)
        if alert:
            errors = self.alerts.notify(event, safe_payload)
            if errors:
                self.journal.append(
                    "alert_error", {"symbol": self.config.data.symbol, "errors": errors}
                )

    def _fetch_current_price(self) -> float:
        ticker = self.exchange.fetch_ticker(self.config.data.symbol)
        price = ticker.get("last") or ticker.get("bid") or ticker.get("ask")
        if price is None or float(price) <= 0:
            raise RuntimeError("Exchange returned no valid ticker price")
        return float(price)

    def _fetch_complete_candles(self) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(
            self.config.data.symbol,
            timeframe=self.config.data.timeframe,
            limit=self.config.live.candle_limit,
        )
        if not raw:
            raise RuntimeError("Exchange returned no OHLCV candles")
        frame = pd.DataFrame(raw, columns=REQUIRED_COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        duration_ms = int(self.exchange.parse_timeframe(self.config.data.timeframe) * 1000)
        now_ms = int(self.exchange.milliseconds())
        close_times = frame["timestamp"].astype("int64") // 1_000_000 + duration_ms
        frame = frame.loc[close_times <= now_ms].copy()
        if frame.empty:
            raise RuntimeError("Exchange returned no completed candles")
        return frame.reset_index(drop=True)

    def _precise_amount(self, amount: float, reference_price: float) -> float:
        if amount <= 0:
            raise RuntimeError("Available balance is too small to place an order")
        precise = float(
            self.exchange.amount_to_precision(self.config.data.symbol, amount)
        )
        market = self.exchange.market(self.config.data.symbol)
        limits = market.get("limits", {})
        minimum_amount = (limits.get("amount") or {}).get("min")
        minimum_cost = (limits.get("cost") or {}).get("min")
        if precise <= 0 or (minimum_amount and precise < float(minimum_amount)):
            raise RuntimeError("Order amount is below the exchange minimum")
        if minimum_cost and precise * reference_price < float(minimum_cost):
            raise RuntimeError("Order value is below the exchange minimum")
        return precise


def create_exchange(config: BotConfig, *, execute: bool = False) -> Any:
    if execute and config.live.dry_run:
        raise ValueError("Live orders require live.dry_run: false in the config")
    import ccxt

    try:
        exchange_class = getattr(ccxt, config.live.exchange_id)
    except AttributeError as error:
        raise ValueError(f"Unknown ccxt exchange: {config.live.exchange_id}") from error
    options: dict[str, Any] = {"enableRateLimit": True}
    if execute:
        api_key = os.environ.get("EXCHANGE_API_KEY")
        api_secret = os.environ.get("EXCHANGE_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError(
                "EXCHANGE_API_KEY and EXCHANGE_API_SECRET are required with --execute"
            )
        options.update({"apiKey": api_key, "secret": api_secret})
    exchange = exchange_class(options)
    if config.live.sandbox:
        exchange.set_sandbox_mode(True)
    exchange.load_markets()
    return exchange


def run_loop(
    runner: LiveTradingRunner,
    emit: Callable[[dict[str, Any]], None],
    *,
    once: bool = False,
) -> None:
    while True:
        try:
            emit(runner.run_once())
        except Exception as error:
            runner.record_error(error)
            if once:
                raise
            emit(
                {
                    "status": "error",
                    "error": str(error),
                    "retry_in_seconds": runner.config.live.poll_interval_seconds,
                }
            )
        if once:
            return
        time.sleep(runner.config.live.poll_interval_seconds)


def split_symbol(symbol: str) -> tuple[str, str]:
    parts = symbol.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Spot symbol must use BASE/QUOTE format: {symbol}")
    return parts[0], parts[1].split(":", 1)[0]


def balance_total(balance: dict[str, Any], currency: str) -> float:
    nested = balance.get(currency) or {}
    if nested.get("total") is not None:
        return float(nested["total"])
    return float((balance.get("total") or {}).get(currency) or 0.0)


def balance_free(balance: dict[str, Any], currency: str) -> float:
    nested = balance.get(currency) or {}
    if nested.get("free") is not None:
        return float(nested["free"])
    return float((balance.get("free") or {}).get(currency) or 0.0)


def order_filled(order: dict[str, Any]) -> float:
    return float(order.get("filled") or 0.0)


def order_fee(
    order: dict[str, Any], price: float = 1.0, base: str | None = None, quote: str | None = None
) -> float:
    fees = order.get("fees") or []
    if not fees and order.get("fee"):
        fees = [order["fee"]]
    total = 0.0
    for fee in fees:
        cost = float(fee.get("cost") or 0.0)
        currency = fee.get("currency")
        if base and currency == base:
            total += cost * price
        elif not currency or not quote or currency == quote:
            total += cost
    return total


def protective_order_ids(order: dict[str, Any]) -> list[str]:
    primary_id = str(order.get("id")) if order.get("id") else None
    discovered: list[str] = []
    for key in ("stopLossOrder", "takeProfitOrder", "stopLoss", "takeProfit"):
        child = order.get(key)
        if isinstance(child, dict) and child.get("id"):
            discovered.append(str(child["id"]))
    for child in order.get("orders") or []:
        if isinstance(child, dict) and child.get("id"):
            discovered.append(str(child["id"]))
    return sorted({item for item in discovered if item != primary_id})


def ensure_order_progress(order: dict[str, Any], filled: float) -> None:
    status = str(order.get("status") or "open").lower()
    if filled <= 0 and status in {"canceled", "expired", "rejected"}:
        raise RuntimeError(f"Exchange order was {status} without a fill")


def public_order_summary(
    order: dict[str, Any], amount: float, price: float, fee: float = 0.0
) -> dict[str, Any]:
    return {
        "id": order.get("id"),
        "status": order.get("status"),
        "filled": amount,
        "average": price,
        "fee": fee,
    }
