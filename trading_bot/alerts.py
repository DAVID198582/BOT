from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from email.message import EmailMessage
from typing import Any


class AlertManager:
    """Best-effort Telegram and SMTP notifications configured by environment."""

    def __init__(self) -> None:
        self.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        self.smtp_host = os.environ.get("SMTP_HOST")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user = os.environ.get("SMTP_USER")
        self.smtp_password = os.environ.get("SMTP_PASSWORD")
        self.email_from = os.environ.get("ALERT_EMAIL_FROM")
        self.email_to = os.environ.get("ALERT_EMAIL_TO")

    @property
    def enabled(self) -> bool:
        telegram = bool(self.telegram_token and self.telegram_chat_id)
        email = bool(self.smtp_host and self.email_from and self.email_to)
        return telegram or email

    def notify(self, event: str, payload: dict[str, Any]) -> list[str]:
        message = f"Trading bot {event}: {json.dumps(payload, default=str, sort_keys=True)}"
        errors: list[str] = []
        if self.telegram_token and self.telegram_chat_id:
            try:
                self._telegram(message)
            except Exception as error:
                errors.append(f"telegram: {type(error).__name__}")
        if self.smtp_host and self.email_from and self.email_to:
            try:
                self._email(event, message)
            except Exception as error:
                errors.append(f"email: {type(error).__name__}")
        return errors

    def _telegram(self, message: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        body = json.dumps(
            {"chat_id": self.telegram_chat_id, "text": message[:4000]}
        ).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status >= 300:
                raise RuntimeError(f"Telegram returned HTTP {response.status}")

    def _email(self, event: str, message: str) -> None:
        email = EmailMessage()
        email["Subject"] = f"Trading bot: {event}"
        email["From"] = self.email_from
        email["To"] = self.email_to
        email.set_content(message)
        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
            server.starttls()
            if self.smtp_user and self.smtp_password:
                server.login(self.smtp_user, self.smtp_password)
            server.send_message(email)
