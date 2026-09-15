"""
Message Classifier
======================
Fast, rule-based first pass. Cheap keyword classification so we don't send
every message to the LLM (Part 2, Section 9 — LLM vs Rules/Classifiers).
"""

from __future__ import annotations

from enum import Enum

from assistant.ingestion.unified_message import UnifiedMessage


class Category(str, Enum):
    WORK = "WORK"
    PERSONAL = "PERSONAL"
    FINANCIAL = "FINANCIAL"
    PROMOTIONAL = "PROMOTIONAL"
    OTHER = "OTHER"


_KEYWORDS = {
    Category.FINANCIAL: {"transaction", "bank", "payment", "debited", "credited", "invoice", "otp"},
    Category.WORK: {"meeting", "interview", "deadline", "project", "report", "review", "call me", "urgent"},
    Category.PROMOTIONAL: {"discount", "offer", "sale", "% off", "unsubscribe", "newsletter"},
}


def classify(message: UnifiedMessage) -> Category:
    text = message.content.lower()
    for category, keywords in _KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return Category.PERSONAL if message.source.value in ("whatsapp", "telegram", "sms") else Category.OTHER
