"""
Content / Urgency / Importance Analysis
==========================================
Lightweight rule-based signal extraction. Kept separate per-signal so the
Priority Engine can combine them independently, per the architecture's
Message Intelligence fan-out (classifier -> [sender, content, context,
urgency, importance, noise] -> priority score).
"""

from __future__ import annotations

from assistant.ingestion.unified_message import UnifiedMessage

_URGENT_WORDS = {"urgent", "asap", "immediately", "now", "call me", "emergency", "today"}
_IMPORTANT_WORDS = {"interview", "deadline", "payment due", "contract", "offer letter", "meeting"}
_NOISE_WORDS = {"unsubscribe", "% off", "sale", "limited time", "click here", "win a"}


def urgency_score(message: UnifiedMessage) -> int:
    text = message.content.lower()
    return 3 if any(w in text for w in _URGENT_WORDS) else 0


def importance_score(message: UnifiedMessage) -> int:
    text = message.content.lower()
    return 2 if any(w in text for w in _IMPORTANT_WORDS) else 0


def noise_score(message: UnifiedMessage) -> int:
    text = message.content.lower()
    return 4 if any(w in text for w in _NOISE_WORDS) else 0
