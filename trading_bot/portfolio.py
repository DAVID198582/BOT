from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from trading_bot.config import PortfolioConfig


class PortfolioStore:
    def __init__(self, path: str | Path, config: PortfolioConfig) -> None:
        self.path = Path(path)
        self.config = config

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"positions": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def record(
        self,
        symbol: str,
        exposure_fraction: float,
        returns: list[float] | None = None,
    ) -> None:
        data = self.load()
        positions = data.setdefault("positions", {})
        if exposure_fraction <= 0:
            positions.pop(symbol, None)
        else:
            positions[symbol] = {
                "exposure_fraction": exposure_fraction,
                "returns": (returns or [])[-96:],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        self._save(data)

    def allow_open(
        self,
        symbol: str,
        proposed_fraction: float,
        returns: list[float],
    ) -> tuple[bool, str]:
        positions = self.load().get("positions", {})
        other_positions = {
            key: value for key, value in positions.items() if key != symbol
        }
        if len(other_positions) >= self.config.max_open_positions:
            return False, "maximum open positions reached"
        total = sum(
            float(value.get("exposure_fraction") or 0.0)
            for value in other_positions.values()
        )
        if total + proposed_fraction > self.config.max_total_exposure_fraction:
            return False, "maximum portfolio exposure reached"
        for other_symbol, value in other_positions.items():
            other_returns = value.get("returns") or []
            length = min(len(returns), len(other_returns))
            if length < 20:
                continue
            correlation = float(np.corrcoef(returns[-length:], other_returns[-length:])[0, 1])
            if np.isfinite(correlation) and abs(correlation) >= self.config.correlation_threshold:
                return False, f"correlation limit reached with {other_symbol}"
        return True, "portfolio limits passed"

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)
