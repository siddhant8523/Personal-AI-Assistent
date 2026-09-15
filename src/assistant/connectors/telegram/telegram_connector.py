"""
Telegram Connector (Part 1, Section 8; Part 3, Section 10)
===============================================================
Uses the raw Bot API over HTTPS (no extra SDK dependency beyond `requests`).
In mock mode (no bot token), send() just logs instead of calling out.
"""

from __future__ import annotations
from typing import Any

import requests


class TelegramConnector:
    BASE_URL = "https://api.telegram.org/bot{token}/{method}"
    FILE_URL = "https://api.telegram.org/file/bot{token}/{file_path}"

    def __init__(self, bot_token: str, enabled: bool = False):
        self.bot_token = bot_token
        self.enabled = enabled and bool(bot_token)

    def _url(self, method: str) -> str:
        return self.BASE_URL.format(token=self.bot_token, method=method)

    def _file_url(self, file_path: str) -> str:
        return self.FILE_URL.format(token=self.bot_token, file_path=file_path)

    def get_status(self) -> dict[str, Any]:
        """Returns the real status of the Telegram Bot connector."""
        has_token = bool(self.bot_token)
        return {
            "has_token": has_token,
            "enabled": self.enabled,
            "connected": self.enabled and has_token,
        }

    def get_file(self, file_id: str) -> dict:
        if not self.enabled:
            return {}
        resp = requests.post(self._url("getFile"), json={"file_id": file_id}, timeout=10)
        if resp.ok:
            return resp.json().get("result", {})
        return {}

    def download_file(self, file_path: str) -> bytes:
        if not self.enabled:
            return b""
        resp = requests.get(self._file_url(file_path), timeout=30)
        resp.raise_for_status()
        return resp.content

    def send_message(self, chat_id: str, text: str) -> dict:
        if not self.enabled:
            return {"status": "ok", "detail": f"[MOCK] would send Telegram to {chat_id}: {text}"}

        target_chat_id = str(chat_id)
        if not target_chat_id.replace("-", "").isdigit():
            import os
            default_chat_id = os.environ.get("TELEGRAM_AGENT_CHAT_ID", "")
            if default_chat_id:
                target_chat_id = default_chat_id

        resp = requests.post(self._url("sendMessage"), json={"chat_id": target_chat_id, "text": text}, timeout=10)
        if resp.ok:
            return {"status": "ok", "detail": resp.json()}
        return {"status": "failed", "detail": resp.text}

    def send_chat_action(self, chat_id: str, action: str = "typing") -> dict:
        if not self.enabled:
            return {"status": "ok", "detail": f"[MOCK] would send Telegram chat action {action} to {chat_id}"}

        target_chat_id = str(chat_id)
        if not target_chat_id.replace("-", "").isdigit():
            import os
            default_chat_id = os.environ.get("TELEGRAM_AGENT_CHAT_ID", "")
            if default_chat_id:
                target_chat_id = default_chat_id

        try:
            resp = requests.post(self._url("sendChatAction"), json={"chat_id": target_chat_id, "action": action}, timeout=10)
            if resp.ok:
                return {"status": "ok", "detail": resp.json()}
            return {"status": "failed", "detail": resp.text}
        except Exception as exc:
            return {"status": "failed", "detail": str(exc)}

    def get_updates(self, offset: int | None = None) -> list[dict]:
        if not self.enabled:
            return []
        params = {"timeout": 0}
        if offset is not None:
            params["offset"] = offset
        resp = requests.get(self._url("getUpdates"), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("result", [])
