"""
Conversation Registry
=======================
Maps conversation/chat identity -> AGENT_CHAT | NORMAL.

Routing must be based on conversation identity, not message text
(Part 1, Section 6). This class is the single source of truth for that
mapping, loaded from config/settings.yaml `conversations:`.
"""

from __future__ import annotations

from enum import Enum


class ConversationClass(str, Enum):
    AGENT_CHAT = "AGENT_CHAT"
    NORMAL = "NORMAL"


class ConversationRegistry:
    def __init__(self, mapping: dict[str, str] | None = None, default: ConversationClass = ConversationClass.NORMAL):
        self._mapping: dict[str, ConversationClass] = {
            k: ConversationClass(v) for k, v in (mapping or {}).items()
        }
        self._default = default

    def classify(self, conversation_id: str) -> ConversationClass:
        return self._mapping.get(conversation_id, self._default)

    def set(self, conversation_id: str, cls: ConversationClass) -> None:
        self._mapping[conversation_id] = cls
