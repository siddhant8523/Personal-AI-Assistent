"""
Sender Analysis
==================
Looks up whether the sender is a known/important contact via the User
preferences (Part 2, Section 10 — Personalization of Priority).
"""

from __future__ import annotations

from assistant.ingestion.unified_message import UnifiedMessage
from assistant.memory.user_profile import UserProfile


def sender_importance(message: UnifiedMessage, user_profile: UserProfile) -> int:
    """Returns a small integer score contribution."""
    sender = message.sender.lower()
    for contact in user_profile.important_contacts():
        if contact.lower() in sender or sender in contact.lower():
            return 3
    return 0
