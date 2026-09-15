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


def make_features(frame: pd.DataFrame, *, include_target: bool = True) -> pd.DataFrame:
    """
    Build model features and, when requested, the next-bar direction target.

    All features are computed using information available at the close of bar t.
    The training target is the direction of the *next* bar
    (close[t+1] > close[t]). The final row is kept for inference, but dropped
    during training because its target is not known yet.
    """
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
    volume_mean = volume.rolling(96).mean()
    volume_std = volume.rolling(96).std()
    data["volume_zscore_96"] = ((volume - volume_mean) / volume_std).mask(
        volume_std == 0, 0.0
    )
    data["rsi_14"] = rsi(close, 14)
    data["ema_gap_12_26"] = (
        close.ewm(span=12, adjust=False).mean() / close.ewm(span=26, adjust=False).mean()
    ) - 1
    data["bb_position_20"] = bollinger_position(close, 20)
    data["atr_14"] = atr(high, low, close, 14) / close
    data["range_pct"] = (high - low) / close

    if include_target:
        # Comparing against NaN yields False, so explicitly restore the final
        # unknown target to NaN instead of silently training it as a down bar.
        data["target"] = (close.shift(-1) > close).astype(float)
        if not data.empty:
            data.loc[data.index[-1], "target"] = np.nan
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.dropna().reset_index(drop=True)


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = -delta.clip(upper=0).rolling(period).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    result = result.mask((losses == 0) & (gains > 0), 100.0)
    return result.mask((losses == 0) & (gains == 0), 50.0)


def bollinger_position(close: pd.Series, period: int) -> pd.Series:
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mean + (2 * std)
    lower = mean - (2 * std)
    band_width = upper - lower
    return ((close - lower) / band_width).mask(band_width == 0, 0.5)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def summarize_market(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or len(frame) < 2:
        return {
            "rows": 0.0,
            "close_start": float("nan"),
            "close_end": float("nan"),
            "total_return_pct": float("nan"),
            "volatility_annualized_pct": float("nan"),
            "max_drawdown_pct": float("nan"),
            "error": "DataFrame is empty or has fewer than 2 rows",
        }

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
