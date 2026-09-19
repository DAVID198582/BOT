from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trading_bot.features import make_features


def candle_frame(rows: int = 140) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100 + index * 0.1 + np.sin(index / 3)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC"),
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000 + index + np.cos(index / 4) * 10,
        }
    )


class FeatureTests(unittest.TestCase):
    def test_training_drops_unknown_target_but_inference_keeps_latest_bar(self) -> None:
        candles = candle_frame()

        training = make_features(candles)
        inference = make_features(candles, include_target=False)

        self.assertEqual(len(inference), len(training) + 1)
        self.assertEqual(inference.iloc[-1]["timestamp"], candles.iloc[-1]["timestamp"])
        self.assertEqual(training.iloc[-1]["timestamp"], candles.iloc[-2]["timestamp"])
        self.assertNotIn("target", inference.columns)
        self.assertFalse(training["target"].isna().any())

    def test_monotonic_market_still_produces_features(self) -> None:
        candles = candle_frame()
        candles["close"] = np.arange(len(candles), dtype=float) + 100
        candles["open"] = candles["close"] - 0.1
        candles["high"] = candles["close"] + 0.5
        candles["low"] = candles["close"] - 0.5

        inference = make_features(candles, include_target=False)

        self.assertFalse(inference.empty)
        self.assertEqual(inference.iloc[-1]["rsi_14"], 100.0)


if __name__ == "__main__":
    unittest.main()
