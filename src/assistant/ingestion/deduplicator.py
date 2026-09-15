"""
Deduplicator
==============
Catches the same underlying message arriving via two independent delivery
paths (e.g. a platform redelivering a webhook). This is deliberately
SEPARATE from the Echo Filter: dedup solves "two events, same message";
the echo filter solves "one event, and it's actually mine" — an
identity problem, not a duplicate-content problem. Conflating them was
part of the original bug (see docs/architecture bug writeup).
"""

from __future__ import annotations

import threading
import time

from assistant.ingestion.unified_message import UnifiedMessage


class Deduplicator:
    def __init__(self, window_seconds: int = 300):
        self._window = window_seconds
        self._lock = threading.Lock()
        self._seen: dict[str, float] = {}

    def is_duplicate(self, message: UnifiedMessage) -> bool:
        key = message.platform_message_id or message.content_hash()
        now = time.time()
        with self._lock:
            self._prune(now)
            if key in self._seen:
                return True
            self._seen[key] = now
            return False

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        expired = [k for k, t in self._seen.items() if t < cutoff]
        for k in expired:
            del self._seen[k]
