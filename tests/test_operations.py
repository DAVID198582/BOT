from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from trading_bot.alerts import AlertManager
from trading_bot.config import PortfolioConfig
from trading_bot.dashboard import DashboardData
from trading_bot.journal import TradeJournal
from trading_bot.portfolio import PortfolioStore
from trading_bot.risk import managed_stop_price, position_plan
from trading_bot.config import RiskConfig


class RiskTests(unittest.TestCase):
    def test_position_size_is_risk_and_exposure_limited(self) -> None:
        plan = position_plan(1000, 100, 0.01, 0.02, 0.25, RiskConfig())
        volatile_plan = position_plan(1000, 100, 0.05, 0.02, 0.25, RiskConfig())

        self.assertEqual(plan.notional, 250.0)
        self.assertEqual(plan.amount, 2.5)
        self.assertEqual(plan.stop_price, 98.0)
        self.assertEqual(volatile_plan.notional, 50.0)

    def test_trailing_stop_moves_to_break_even_and_never_down(self) -> None:
        stop, break_even = managed_stop_price(100, 98, 104, RiskConfig())

        self.assertTrue(break_even)
        self.assertGreaterEqual(stop, 100)
        lower_stop, _ = managed_stop_price(100, stop, 101, RiskConfig())
        self.assertEqual(lower_stop, stop)


class PortfolioTests(unittest.TestCase):
    def test_correlated_position_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = PortfolioConfig(correlation_threshold=0.8)
            store = PortfolioStore(Path(directory) / "portfolio.json", config)
            returns = [index / 1000 for index in range(30)]
            store.record("BTC/USDT", 0.2, returns)

            allowed, reason = store.allow_open("ETH/USDT", 0.2, returns)

            self.assertFalse(allowed)
            self.assertIn("correlation", reason)


class JournalAndDashboardTests(unittest.TestCase):
    def test_journal_summary_and_dashboard_emergency_halt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = TradeJournal(root / "trades.jsonl")
            journal.append("exit", {"realized_pnl": -5.0})
            journal.append("exit", {"realized_pnl": 8.0})
            state_path = root / "live-state.json"
            state_path.write_text(
                json.dumps({"symbol": "BTC/USDT", "dry_run": True}),
                encoding="utf-8",
            )
            portfolio_path = root / "portfolio-state.json"
            portfolio_path.write_text(json.dumps({"positions": {}}), encoding="utf-8")
            data = DashboardData(
                str(root / "*state.json"),
                str(portfolio_path),
                str(root / "trades.jsonl"),
            )

            changed = data.set_manual_halt(True)
            status = data.status()

            self.assertEqual(changed, 1)
            self.assertTrue(status["states"][0]["manual_halt"])
            self.assertEqual(status["journal"]["realized_pnl"], 3.0)


class AlertTests(unittest.TestCase):
    def test_notification_failures_are_reported_without_raising(self) -> None:
        alerts = AlertManager()
        alerts.telegram_token = "local-test-token"
        alerts.telegram_chat_id = "local-test-chat"
        alerts.smtp_host = None
        alerts._telegram = Mock(side_effect=TimeoutError())

        errors = alerts.notify("entry", {"symbol": "BTC/USDT"})

        self.assertEqual(errors, ["telegram: TimeoutError"])
