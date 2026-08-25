from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.config import BotConfig
from trading_bot.features import make_features
from trading_bot.models import predict_signal


@dataclass(frozen=True)
class Decision:
    action: str
    signal: int
    close: float
    reason: str


class PaperTradingBot:
    def __init__(self, model, config: BotConfig) -> None:
        self.model = model
        self.config = config

    def decide(self, candles: pd.DataFrame) -> Decision:
        feature_frame = make_features(candles)
        if feature_frame.empty:
            return Decision("hold", 0, float(candles["close"].iloc[-1]), "not enough candles for features")
        latest = feature_frame.tail(1)
        signal = int(predict_signal(self.model, latest, self.config.model.probability_threshold).iloc[0])
        action = {1: "buy_or_hold_long", -1: "sell_or_hold_short", 0: "hold"}[signal]
        return Decision(action, signal, float(latest["close"].iloc[0]), "model probability threshold decision")

