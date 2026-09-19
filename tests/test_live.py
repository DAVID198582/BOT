from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from trading_bot.config import (
    BacktestConfig,
    BotConfig,
    DataConfig,
    LiveConfig,
    ModelConfig,
    PortfolioConfig,
    RiskConfig,
)
from trading_bot.live import LiveTradingRunner, StateStore


class ConstantModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features):
        probability = np.full(len(features), self.probability)
        return np.column_stack((1 - probability, probability))


class FakeExchange:
    def __init__(self, rows: int = 140) -> None:
        self.rows = rows
        self.start = 1_735_689_600_000
        self.ticker_price = 114.0

    def fetch_ohlcv(self, symbol, timeframe, limit):
        values = []
        for index in range(self.rows):
            close = 100 + index * 0.1 + np.sin(index / 3)
            values.append(
                [
                    self.start + index * 900_000,
                    close - 0.1,
                    close + 0.5,
                    close - 0.5,
                    close,
                    1000 + index + np.cos(index / 4) * 10,
                ]
            )
        return values[-limit:]

    def parse_timeframe(self, timeframe):
        return 900

    def milliseconds(self):
        return self.start + self.rows * 900_000

    def fetch_ticker(self, symbol):
        return {"last": self.ticker_price}


def bot_config(state_path: Path) -> BotConfig:
    return BotConfig(
        data=DataConfig(),
        model=ModelConfig(),
        backtest=BacktestConfig(),
        live=LiveConfig(
            state_path=str(state_path),
            journal_path=str(state_path.with_name("journal.jsonl")),
        ),
        portfolio=PortfolioConfig(
            state_path=str(state_path.with_name("portfolio.json"))
        ),
    )


class LiveRunnerTests(unittest.TestCase):
    def test_dry_run_persists_position_and_deduplicates_candle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            config = bot_config(state_path)
            exchange = FakeExchange()
            runner = LiveTradingRunner(ConstantModel(0.9), config, exchange)

            first = runner.run_once()
            duplicate = runner.run_once()

            self.assertEqual(first["action"], "buy")
            self.assertIsInstance(first["order"], dict)
            self.assertIn("amount", first["order"])
            self.assertGreater(first["position_amount"], 0)
            self.assertEqual(first["paper_quote_balance"], 749.75)
            self.assertEqual(duplicate["status"], "waiting")

            restored = LiveTradingRunner(ConstantModel(0.9), config, exchange)
            self.assertGreater(restored.state.position_amount, 0)
            self.assertEqual(restored.run_once()["status"], "waiting")

    def test_negative_signal_exits_only_the_managed_spot_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            config = bot_config(state_path)
            exchange = FakeExchange()
            runner = LiveTradingRunner(ConstantModel(0.9), config, exchange)
            runner.run_once()

            exchange.rows += 1
            runner.model = ConstantModel(0.1)
            result = runner.run_once()

            self.assertEqual(result["action"], "sell")
            self.assertEqual(result["position_amount"], 0.0)
            self.assertGreater(result["paper_quote_balance"], 750.0)

    def test_stop_loss_runs_between_completed_candles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            config = bot_config(state_path)
            exchange = FakeExchange()
            runner = LiveTradingRunner(ConstantModel(0.9), config, exchange)
            entry = runner.run_once()
            exchange.ticker_price = entry["order"]["reference_price"] * 0.97

            result = runner.run_once()

            self.assertEqual(result["model_action"], "stop_loss")
            self.assertTrue(result["stop_triggered"])
            self.assertEqual(result["action"], "sell")
            self.assertEqual(result["position_amount"], 0.0)

    def test_real_execution_requires_config_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = bot_config(Path(directory) / "state.json")
            store = StateStore(
                config.live.state_path,
                config.data.symbol,
                False,
                config.backtest.initial_cash,
            )
            with self.assertRaisesRegex(ValueError, "live.dry_run"):
                LiveTradingRunner(
                    ConstantModel(0.9), config, FakeExchange(), execute=True, state_store=store
                )

    def test_consecutive_loss_circuit_breaker_blocks_next_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = bot_config(Path(directory) / "state.json")
            config = replace(
                config,
                risk=replace(config.risk, max_consecutive_losses=1),
            )
            exchange = FakeExchange()
            runner = LiveTradingRunner(ConstantModel(0.9), config, exchange)
            entry = runner.run_once()
            exchange.ticker_price = entry["order"]["reference_price"] * 0.97
            runner.run_once()
            exchange.rows += 1
            exchange.ticker_price = 200.0

            blocked = runner.run_once()

            self.assertEqual(blocked["action"], "blocked")
            self.assertEqual(blocked["reason"], "consecutive loss limit reached")


class FakeExecutionExchange(FakeExchange):
    def __init__(self) -> None:
        super().__init__()
        self.ticker_price = 114.0
        self.balance = {
            "BTC": {"free": 0.0, "total": 0.0},
            "USDT": {"free": 1000.0, "total": 1000.0},
        }
        self.created_params = None
        self.amount = 0.0

    has = {"fetchOrder": True}

    def feature_value(self, symbol, method, feature):
        return feature in {"stopLoss", "takeProfit"}

    def fetch_balance(self):
        return self.balance

    def amount_to_precision(self, symbol, amount):
        return f"{amount:.8f}"

    def market(self, symbol):
        return {"limits": {"amount": {"min": 0.00001}, "cost": {"min": 1.0}}}

    def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        self.created_params = params
        self.amount = amount
        filled = amount / 2
        self.balance["BTC"]["free"] = filled
        self.balance["BTC"]["total"] = filled
        return {
            "id": "entry-1",
            "status": "open",
            "filled": filled,
            "average": 114.0,
            "stopLossOrder": {"id": "stop-1"},
            "takeProfitOrder": {"id": "take-1"},
        }

    def fetch_order(self, order_id, symbol):
        self.balance["BTC"]["free"] = self.amount
        self.balance["BTC"]["total"] = self.amount
        return {
            "id": order_id,
            "status": "closed",
            "filled": self.amount,
            "average": 114.0,
        }

    def cancel_order(self, order_id, symbol):
        return {"id": order_id, "status": "canceled"}


class ExecutionTests(unittest.TestCase):
    def test_native_protection_and_partial_fill_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            config = bot_config(state_path)
            config = replace(config, live=replace(config.live, dry_run=False))
            exchange = FakeExecutionExchange()
            runner = LiveTradingRunner(
                ConstantModel(0.9), config, exchange, execute=True
            )

            first = runner.run_once()
            first_amount = first["position_amount"]
            second = runner.run_once()

            self.assertIn("stopLoss", exchange.created_params)
            self.assertIn("takeProfit", exchange.created_params)
            self.assertTrue(runner.state.native_protection_active)
            self.assertGreater(second["position_amount"], first_amount)
            self.assertEqual(runner.state.order_status, "closed")

    def test_required_native_protection_fails_before_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            config = bot_config(state_path)
            config = replace(
                config,
                live=replace(
                    config.live,
                    dry_run=False,
                    require_native_protection=True,
                ),
            )
            exchange = FakeExecutionExchange()
            exchange.feature_value = lambda symbol, method, feature: False
            runner = LiveTradingRunner(
                ConstantModel(0.9), config, exchange, execute=True
            )

            with self.assertRaisesRegex(RuntimeError, "does not support"):
                runner.run_once()

            self.assertIsNone(exchange.created_params)

    def test_standalone_native_stop_is_created_when_attached_stop_is_unavailable(self) -> None:
        class StandaloneExchange(FakeExecutionExchange):
            def __init__(self) -> None:
                super().__init__()
                self.orders = []

            def feature_value(self, symbol, method, feature):
                return feature == "stopLossPrice"

            def create_order(
                self, symbol, order_type, side, amount, price=None, params=None
            ):
                self.orders.append({"side": side, "params": params or {}})
                if side == "sell" and params and "stopLossPrice" in params:
                    return {"id": "native-stop-1", "status": "open", "filled": 0.0}
                self.amount = amount
                self.balance["BTC"]["free"] = amount
                self.balance["BTC"]["total"] = amount
                return {
                    "id": "entry-1",
                    "status": "closed",
                    "filled": amount,
                    "average": 114.0,
                }

        with tempfile.TemporaryDirectory() as directory:
            config = bot_config(Path(directory) / "state.json")
            config = replace(config, live=replace(config.live, dry_run=False))
            exchange = StandaloneExchange()
            runner = LiveTradingRunner(
                ConstantModel(0.9), config, exchange, execute=True
            )

            runner.run_once()

            self.assertEqual(runner.state.native_protection_mode, "standalone")
            self.assertEqual(runner.state.protective_order_ids, ["native-stop-1"])
            self.assertIn("stopLossPrice", exchange.orders[1]["params"])

    def test_old_position_state_is_migrated_with_protection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            config = bot_config(state_path)
            state_path.write_text(
                '{"symbol":"BTC/USDT","dry_run":true,'
                '"position_amount":1.0,"entry_price":100.0,'
                '"paper_quote_balance":900.0}',
                encoding="utf-8",
            )

            runner = LiveTradingRunner(ConstantModel(0.9), config, FakeExchange())

            self.assertEqual(runner.state.stop_price, 98.0)
            self.assertEqual(runner.state.take_profit_price, 104.0)


if __name__ == "__main__":
    unittest.main()
