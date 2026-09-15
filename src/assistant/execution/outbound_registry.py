"""
Outbound Registry
==================
THE FIX (part 1 of 2). See docs/architecture for the full bug writeup.

Every time the assistant sends something out (WhatsApp, Telegram, SMS,
Gmail), the Task Manager registers the outgoing message HERE *before* the
network call is made. The Echo Filter (router/echo_filter.py) checks
inbound events against this registry to recognize "this is my own message
coming back to me" (e.g. WhatsApp/Baileys echoing fromMe:true events),
BEFORE the Conversation Router decides whether something is Agent Chat or
Normal Chat.

Registering before sending (not after) avoids a race where a fast platform
echo could arrive before the registry entry exists.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class OutboundRecord:
    task_id: str
    source: str          # platform, e.g. "whatsapp"
    conversation_id: str
    content_hash: str
    sent_at: float


class OutboundRegistry:
    def __init__(self, window_seconds: int = 120):
        self._window = window_seconds
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str, str], OutboundRecord] = {}

    def register_pending(self, task_id: str, source: str, conversation_id: str, content_hash: str) -> None:
        """Call this BEFORE the connector actually transmits the message."""
        key = (source, conversation_id, content_hash)
        with self._lock:
            self._records[key] = OutboundRecord(
                task_id=task_id,
                source=source,
                conversation_id=conversation_id,
                content_hash=content_hash,
                sent_at=time.time(),
            )

    def matches_recent_outbound(self, source: str, conversation_id: str, content_hash: str) -> OutboundRecord | None:
        """Used by the Echo Filter. Returns the matching record if an
        outbound send with the same (platform, conversation, content) was
        registered within the echo window — meaning the inbound event is
        almost certainly our own message bouncing back, not a new message."""
        key = (source, conversation_id, content_hash)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if time.time() - record.sent_at > self._window:
                del self._records[key]
                return None
            return record

    def sweep_expired(self) -> None:
        cutoff = time.time() - self._window
        with self._lock:
            expired = [k for k, v in self._records.items() if v.sent_at < cutoff]
            for k in expired:
                del self._records[k]
