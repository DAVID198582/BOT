from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TradeJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record, default=str, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            written = 0
            while written < len(line):
                written += os.write(descriptor, line[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return record

    def records(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def summary(self) -> dict[str, Any]:
        records = self.records(limit=10_000)
        exits = [record for record in records if record.get("event") == "exit"]
        pnl = sum(float(record.get("realized_pnl") or 0.0) for record in exits)
        return {
            "events": len(records),
            "closed_trades": len(exits),
            "realized_pnl": pnl,
            "wins": sum(float(record.get("realized_pnl") or 0.0) > 0 for record in exits),
            "losses": sum(float(record.get("realized_pnl") or 0.0) < 0 for record in exits),
        }
