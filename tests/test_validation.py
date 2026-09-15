from __future__ import annotations

import unittest
from dataclasses import replace

from tests.test_features import candle_frame
from trading_bot.config import BotConfig, BacktestConfig, DataConfig, LiveConfig, ModelConfig
from trading_bot.features import make_features
from trading_bot.validation import walk_forward_validate


class WalkForwardTests(unittest.TestCase):
    def test_expanding_window_report_has_all_folds_and_approval(self) -> None:
        features = make_features(candle_frame(500))
        config = BotConfig(
            data=DataConfig(),
            model=ModelConfig(n_estimators=10, max_depth=4),
            backtest=BacktestConfig(),
            live=LiveConfig(),
        )
        config = replace(
            config,
            validation=replace(
                config.validation,
                min_accuracy=0.0,
                min_roc_auc=0.0,
                max_drawdown_pct=-100.0,
            ),
        )

        report = walk_forward_validate(features, config)

        self.assertTrue(report["approved"])
        self.assertEqual(len(report["folds"]), config.validation.folds)
        self.assertLess(report["folds"][0]["train_rows"], report["folds"][-1]["train_rows"])
