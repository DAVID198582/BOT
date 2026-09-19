from __future__ import annotations

import unittest

import pandas as pd

from trading_bot.backtest import run_backtest
from trading_bot.config import BacktestConfig


class BacktestTimingTests(unittest.TestCase):
    def test_new_position_can_stop_during_its_entry_bar(self) -> None:
        timestamps = pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC")
        features = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 100.0, 100.0],
                "high": [101.0, 101.0, 101.0],
                "low": [99.0, 90.0, 99.0],
                "close": [100.0, 95.0, 100.0],
            }
        )
        signals = pd.Series([1, 0, 0])

        result = run_backtest(features, signals, BacktestConfig())

        self.assertEqual(result.trades["action"].tolist(), ["entry", "stop"])
        self.assertEqual(result.trades.iloc[0]["timestamp"], timestamps[1])
        self.assertEqual(result.trades.iloc[1]["timestamp"], timestamps[1])


if __name__ == "__main__":
    unittest.main()
