from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from trading_bot.journal import TradeJournal


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Trading Bot</title><style>
body{font:15px system-ui;background:#0d1117;color:#e6edf3;max-width:1000px;margin:30px auto;padding:0 18px}
h1{font-size:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px}button{padding:10px 16px;margin-right:8px;border:0;border-radius:6px;cursor:pointer}.halt{background:#da3633;color:white}.resume{background:#238636;color:white}pre{white-space:pre-wrap;word-break:break-word}small{color:#8b949e}
</style></head><body><h1>Trading Bot Control</h1><p><button class="halt" onclick="control('halt')">Emergency halt</button><button class="resume" onclick="control('resume')">Resume entries</button></p><div id="summary" class="grid"></div><h2>State</h2><pre id="state">Loading…</pre><small>Emergency halt blocks new entries; it does not liquidate positions.</small><script>
let token=localStorage.getItem('dashboardToken')||'';if(!['localhost','127.0.0.1','::1'].includes(location.hostname)&&!token){token=prompt('Dashboard token')||'';localStorage.setItem('dashboardToken',token)}const headers=()=>({Authorization:'Bearer '+token});
async function refresh(){let r=await fetch('/api/status',{headers:headers()});if(!r.ok){document.getElementById('state').textContent='Unauthorized';return}let d=await r.json();let equity=d.states.reduce((n,s)=>n+Number(s.last_equity||0),0);let drawdown=Math.min(0,...d.states.map(s=>Number(s.daily_drawdown_pct||0)));let confidence=Math.max(0,...d.states.map(s=>Number(s.last_model_probability||0)));document.getElementById('summary').innerHTML=`<div class="card">Health<br><b>${d.healthy?'Healthy':'Stale'}</b></div><div class="card">Equity<br><b>${equity.toFixed(2)}</b></div><div class="card">Daily drawdown<br><b>${drawdown.toFixed(2)}%</b></div><div class="card">Model confidence<br><b>${(confidence*100).toFixed(1)}%</b></div><div class="card">Closed trades<br><b>${d.journal.closed_trades}</b></div><div class="card">Realized P&amp;L<br><b>${Number(d.journal.realized_pnl).toFixed(2)}</b></div>`;document.getElementById('state').textContent=JSON.stringify(d,null,2)}
async function control(action){if(!token&&confirm('Set a dashboard token?')){token=prompt('Dashboard token')||'';localStorage.setItem('dashboardToken',token)}let r=await fetch('/api/'+action,{method:'POST',headers:headers()});if(!r.ok)alert(await r.text());await refresh()}
refresh();setInterval(refresh,5000)
</script></body></html>"""


class DashboardData:
    def __init__(self, state_glob: str, portfolio_path: str, journal_path: str) -> None:
        self.state_glob = state_glob
        self.portfolio_path = Path(portfolio_path)
        self.journal = TradeJournal(journal_path)

    def status(self) -> dict[str, Any]:
        states: list[dict[str, Any]] = []
        for name in sorted(glob.glob(self.state_glob)):
            try:
                payload = json.loads(Path(name).read_text(encoding="utf-8"))
                if "symbol" in payload and "dry_run" in payload:
                    states.append({"path": name, **payload})
            except (OSError, json.JSONDecodeError):
                states.append({"path": name, "error": "unreadable state"})
        portfolio: dict[str, Any] = {"positions": {}}
        if self.portfolio_path.exists():
            try:
                portfolio = json.loads(self.portfolio_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                portfolio = {"error": "unreadable portfolio state"}
        now = datetime.now(timezone.utc)
        for state in states:
            updated = state.get("last_updated_at")
            try:
                age = (now - datetime.fromisoformat(updated)).total_seconds()
                state["heartbeat_age_seconds"] = max(0.0, age)
                state["healthy"] = age <= 120
            except (TypeError, ValueError):
                state["healthy"] = False
        return {
            "healthy": bool(states) and all(state["healthy"] for state in states),
            "states": states,
            "portfolio": portfolio,
            "journal": self.journal.summary(),
            "recent_events": self.journal.records(limit=30),
        }

    def set_manual_halt(self, halted: bool) -> int:
        changed = 0
        for name in sorted(glob.glob(self.state_glob)):
            path = Path(name)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if "symbol" not in payload or "dry_run" not in payload:
                continue
            payload["manual_halt"] = halted
            payload["halt_reason"] = "dashboard emergency halt" if halted else None
            temporary = path.with_suffix(f"{path.suffix}.dashboard.tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(path)
            changed += 1
        self.journal.append("manual_halt" if halted else "manual_resume", {"states": changed})
        return changed


def serve_dashboard(
    data: DashboardData,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    token = os.environ.get("DASHBOARD_TOKEN")
    if host not in {"127.0.0.1", "localhost", "::1"} and not token:
        raise ValueError("DASHBOARD_TOKEN is required when binding beyond localhost")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                if token and self.headers.get("Authorization") != f"Bearer {token}":
                    self.send_error(HTTPStatus.UNAUTHORIZED)
                    return
                self._json(data.status())
            elif self.path == "/healthz":
                status = data.status()
                if status["healthy"]:
                    self._json({"status": "ok"})
                else:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "stale bot heartbeat")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self.send_error(HTTPStatus.UNAUTHORIZED)
                return
            if self.path == "/api/halt":
                self._json({"halted": True, "states": data.set_manual_halt(True)})
            elif self.path == "/api/resume":
                self._json({"halted": False, "states": data.set_manual_halt(False)})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json(self, payload: dict[str, Any]) -> None:
            self._send(json.dumps(payload, default=str), "application/json")

        def _send(self, body: str, content_type: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
