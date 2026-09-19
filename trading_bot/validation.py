from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_bot.backtest import run_backtest
from trading_bot.config import BotConfig
from trading_bot.models import evaluate_model, predict_signal, train_model


def walk_forward_validate(features: pd.DataFrame, config: BotConfig) -> dict[str, Any]:
    folds = config.validation.folds
    test_rows = len(features) // (folds + 1)
    initial_train_rows = len(features) - folds * test_rows
    if test_rows < 2 or initial_train_rows < 100:
        raise ValueError("not enough feature rows for walk-forward validation")

    results: list[dict[str, Any]] = []
    for fold in range(folds):
        test_start = initial_train_rows + fold * test_rows
        test_end = test_start + test_rows
        # Purge the boundary row because its next-bar target belongs to test data.
        train = features.iloc[: max(0, test_start - 1)].copy()
        test = features.iloc[test_start:test_end].copy()
        model = train_model(train, config.model)
        metrics = evaluate_model(model, test)
        signals = predict_signal(model, test, config.model.probability_threshold)
        backtest = run_backtest(test, signals, config.backtest, config.risk)
        results.append(
            {
                "fold": fold + 1,
                "train_rows": len(train),
                "test_rows": len(test),
                **metrics,
                "return_pct": backtest.metrics.get("total_return_pct", 0.0),
                "max_drawdown_pct": backtest.metrics.get("max_drawdown_pct", 0.0),
            }
        )

    accuracy = float(np.mean([fold["accuracy"] for fold in results]))
    roc_values = [float(fold["roc_auc"]) for fold in results if "roc_auc" in fold]
    roc_auc = float(np.mean(roc_values)) if roc_values else 0.0
    worst_drawdown = float(min(fold["max_drawdown_pct"] for fold in results))
    approved = bool(
        accuracy >= config.validation.min_accuracy
        and roc_auc >= config.validation.min_roc_auc
        and worst_drawdown >= config.validation.max_drawdown_pct
    )
    return {
        "approved": approved,
        "folds": results,
        "summary": {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "worst_drawdown_pct": worst_drawdown,
        },
        "thresholds": {
            "min_accuracy": config.validation.min_accuracy,
            "min_roc_auc": config.validation.min_roc_auc,
            "max_drawdown_pct": config.validation.max_drawdown_pct,
        },
    }
