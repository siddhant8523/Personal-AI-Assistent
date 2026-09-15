"""
Unified Message Model
======================
Every inbound event (Gmail, Telegram, WhatsApp, SMS, CLI) is converted into
this common structure before it goes anywhere else in the system. The Agent
Core should never need to understand a platform's native format.

See: FINAL SYSTEM ARCHITECTURE — Part 2, Section 3 ("Unified Message Model").
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Origin(str, Enum):
    """Who authored this message. Critical for the echo-filter fix."""
    USER = "USER"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"


class Source(str, Enum):
    CLI = "cli"
    GMAIL = "gmail"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    SMS = "sms"


@dataclass
class UnifiedMessage:
    source: Source
    conversation_id: str          # stable chat/thread identity (drives routing)
    sender: str
    content: str
    recipient: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    attachments: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    origin: Origin = Origin.USER
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # platform-native id, when the connector provides one (used for dedup)
    platform_message_id: Optional[str] = None

    def content_hash(self) -> str:
        """Stable hash of (conversation, content) — used by dedup and the
        echo filter to recognize 'the same message' without relying on a
        platform message id (which not every connector guarantees)."""
        raw = f"{self.conversation_id}|{self.content.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def is_from_self(self, self_identities: set[str]) -> bool:
        return self.sender in self_identities or self.origin == Origin.AGENT
