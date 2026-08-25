from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from trading_bot.config import ModelConfig
from trading_bot.features import FEATURE_COLUMNS


def build_model(config: ModelConfig) -> Pipeline:
    if config.type != "random_forest":
        raise ValueError(f"Unsupported model type: {config.type}")
    classifier = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=config.random_state,
        n_jobs=-1,
    )
    return Pipeline([("scaler", MaxAbsScaler()), ("classifier", classifier)])


def train_model(train_features: pd.DataFrame, config: ModelConfig) -> Pipeline:
    model = build_model(config)
    model.fit(train_features[FEATURE_COLUMNS], train_features["target"])
    return model


def evaluate_model(model: Pipeline, test_features: pd.DataFrame) -> dict[str, float]:
    probabilities = model.predict_proba(test_features[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(test_features["target"], predictions)),
        "precision": float(precision_score(test_features["target"], predictions, zero_division=0)),
    }
    if test_features["target"].nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(test_features["target"], probabilities))
    return metrics


def predict_signal(model: Pipeline, features: pd.DataFrame, threshold: float) -> pd.Series:
    probabilities = model.predict_proba(features[FEATURE_COLUMNS])[:, 1]
    signals = pd.Series(0, index=features.index, dtype=int)
    signals[probabilities >= threshold] = 1
    signals[probabilities <= 1 - threshold] = -1
    return signals


def save_model(model: Pipeline, path: str | Path, metadata: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, output_path)


def load_model(path: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    payload = joblib.load(path)
    return payload["model"], payload.get("metadata", {})

