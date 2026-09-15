"""
Session Memory
=================
Short-lived, per-conversation context (Part 2, Section 16). E.g. resolving
"the one from work" after the agent already asked "which Rahul?".
In-memory only, with a TTL — not written to disk.
"""

from __future__ import annotations

import time


class SessionMemory:
    def __init__(self, ttl_minutes: int = 60):
        self._ttl = ttl_minutes * 60
        self._store: dict[str, dict] = {}
        self._touched: dict[str, float] = {}

    def get(self, conversation_id: str) -> dict:
        self._expire_if_needed(conversation_id)
        return self._store.get(conversation_id, {})

    def update(self, conversation_id: str, **kwargs) -> None:
        self._store.setdefault(conversation_id, {}).update(kwargs)
        self._touched[conversation_id] = time.time()

    def _expire_if_needed(self, conversation_id: str) -> None:
        last = self._touched.get(conversation_id)
        if last and (time.time() - last) > self._ttl:
            self._store.pop(conversation_id, None)
            self._touched.pop(conversation_id, None)
