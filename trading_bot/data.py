from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    frame = frame[REQUIRED_COLUMNS].copy()
    frame["timestamp"] = _parse_timestamp(frame["timestamp"])
    numeric_columns = ["open", "high", "low", "close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna().sort_values("timestamp").drop_duplicates("timestamp")
    return frame.reset_index(drop=True)


def _parse_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        max_value = float(series.max())
        unit = "ms" if max_value > 10_000_000_000 else "s"
        return pd.to_datetime(series, unit=unit, utc=True)
    return pd.to_datetime(series, utc=True)


def split_train_test(frame: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    split_index = int(len(frame) * (1 - test_size))
    if split_index <= 0 or split_index >= len(frame):
        raise ValueError("not enough rows for the requested train/test split")
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()

