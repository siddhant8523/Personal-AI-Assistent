"""
Message Ingestion
====================
Entry point for NORMAL conversation traffic only (Agent Chat bypasses this
by design and goes straight from the Router to the Agent Command Handler).

Flow: normalize -> dedup -> hand off to Message Intelligence.
"""

from __future__ import annotations

import logging
from typing import Callable

from assistant.ingestion.deduplicator import Deduplicator
from assistant.ingestion.unified_message import UnifiedMessage

logger = logging.getLogger("assistant.ingestion")


class MessageIngestion:
    def __init__(self, deduplicator: Deduplicator, on_message: Callable[[UnifiedMessage], None]):
        self._dedup = deduplicator
        self._on_message = on_message

    def ingest(self, message: UnifiedMessage) -> str:
        if self._dedup.is_duplicate(message):
            logger.info("DROPPED (duplicate): %s", message.message_id)
            return "dropped_duplicate"
        self._on_message(message)
        return "ingested"
