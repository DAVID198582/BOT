from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.config import BotConfig
from trading_bot.features import FEATURE_COLUMNS, make_features
from trading_bot.models import predict_signal


@dataclass(frozen=True)
class Decision:
    action: str
    signal: int
    close: float
    reason: str
    probability: float | None = None
    volatility_pct: float = 0.0


class PaperTradingBot:
    def __init__(self, model, config: BotConfig) -> None:
        self.model = model
        self.config = config

    def decide(self, candles: pd.DataFrame) -> Decision:
        """
        Produce a paper-trading decision from the latest available candles.

        The signal is generated from features of the most recent complete bar.
        In a real system you would place the corresponding order to be filled
        at the open of the *next* bar (or via market/limit order logic).
        """
        feature_frame = make_features(candles, include_target=False)
        if feature_frame.empty:
            close = float(candles["close"].iloc[-1]) if not candles.empty else float("nan")
            return Decision(
                "hold",
                0,
                close,
                "not enough candles for features",
            )

        latest = feature_frame.tail(1)
        probability = float(
            self.model.predict_proba(latest[FEATURE_COLUMNS])[:, 1][0]
        )
        signal = int(
            predict_signal(self.model, latest, self.config.model.probability_threshold).iloc[0]
        )
        action = {1: "buy_or_hold_long", -1: "sell_or_hold_short", 0: "hold"}[signal]
        return Decision(
            action,
            signal,
            float(latest["close"].iloc[0]),
            "model probability threshold decision (execute next open)",
            probability,
            float(latest["atr_14"].iloc[0]),
        )
