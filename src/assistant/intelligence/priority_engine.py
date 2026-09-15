"""
Priority Engine
==================
priority = sender_importance + content_importance + urgency - noise
(weights configurable via config/settings.yaml)

Result is HIGH / MEDIUM / LOW (Part 2, Section 7-8).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from assistant.ingestion.unified_message import UnifiedMessage
from assistant.intelligence import classifier, content_analysis, sender_analysis
from assistant.memory.user_profile import UserProfile


class PriorityLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class PriorityResult:
    message: UnifiedMessage
    category: classifier.Category
    score: int
    level: PriorityLevel


class PriorityEngine:
    def __init__(self, weights: dict[str, int] | None = None, thresholds: dict[str, int] | None = None):
        self.weights = weights or {
            "sender_importance": 3, "content_importance": 2,
            "urgency": 3, "noise_penalty": -4,
        }
        self.thresholds = thresholds or {"high": 6, "medium": 3}

    def score(self, message: UnifiedMessage, user_profile: UserProfile) -> PriorityResult:
        category = classifier.classify(message)
        sender = sender_analysis.sender_importance(message, user_profile)
        importance = content_analysis.importance_score(message)
        urgency = content_analysis.urgency_score(message)
        noise = content_analysis.noise_score(message)

        total = sender + importance + urgency - noise
        level = (
            PriorityLevel.HIGH if total >= self.thresholds["high"]
            else PriorityLevel.MEDIUM if total >= self.thresholds["medium"]
            else PriorityLevel.LOW
        )
        return PriorityResult(message=message, category=category, score=total, level=level)
