from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1",
    "return_4",
    "return_16",
    "volatility_16",
    "volatility_96",
    "volume_zscore_96",
    "rsi_14",
    "ema_gap_12_26",
    "bb_position_20",
    "atr_14",
    "range_pct",
]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    data["return_1"] = close.pct_change()
    data["return_4"] = close.pct_change(4)
    data["return_16"] = close.pct_change(16)
    data["volatility_16"] = data["return_1"].rolling(16).std()
    data["volatility_96"] = data["return_1"].rolling(96).std()
    data["volume_zscore_96"] = (volume - volume.rolling(96).mean()) / volume.rolling(96).std()
    data["rsi_14"] = rsi(close, 14)
    data["ema_gap_12_26"] = (close.ewm(span=12, adjust=False).mean() / close.ewm(span=26, adjust=False).mean()) - 1
    data["bb_position_20"] = bollinger_position(close, 20)
    data["atr_14"] = atr(high, low, close, 14) / close
    data["range_pct"] = (high - low) / close
    data["target"] = (close.shift(-1) > close).astype(int)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.dropna().reset_index(drop=True)


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = -delta.clip(upper=0).rolling(period).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def bollinger_position(close: pd.Series, period: int) -> pd.Series:
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mean + (2 * std)
    lower = mean - (2 * std)
    return (close - lower) / (upper - lower)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def summarize_market(frame: pd.DataFrame) -> dict[str, float]:
    returns = frame["close"].pct_change().dropna()
    annualization = np.sqrt(365 * 24 * 4)
    drawdown = frame["close"] / frame["close"].cummax() - 1
    return {
        "rows": float(len(frame)),
        "close_start": float(frame["close"].iloc[0]),
        "close_end": float(frame["close"].iloc[-1]),
        "total_return_pct": float((frame["close"].iloc[-1] / frame["close"].iloc[0] - 1) * 100),
        "volatility_annualized_pct": float(returns.std() * annualization * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
    }

