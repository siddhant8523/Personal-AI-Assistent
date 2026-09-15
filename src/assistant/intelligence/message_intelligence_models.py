"""
Message Intelligence Models
===========================
Pydantic schemas for structured extraction and strict validation of
message intelligence from the LLM.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class MessageAnalysisItem(BaseModel):
    message_id: str = Field(..., description="Unique message identifier matching input item")
    category: str = Field(..., description="Category: WORK, PERSONAL, PROMOTIONAL, TRANSACTIONAL, SOCIAL, NEWSLETTER, OTHER")
    intent: str = Field(..., description="Intent: INTERVIEW, MEETING, TASK, NOTIFICATION, INQUIRY, CHAT, SPAM, OTHER")
    importance: Literal["HIGH", "MEDIUM", "LOW"] = Field(..., description="Importance level: HIGH, MEDIUM, LOW")
    urgency: Literal["HIGH", "MEDIUM", "LOW"] = Field(..., description="Urgency level: HIGH, MEDIUM, LOW")
    requires_action: bool = Field(..., description="True if action is required by the user")
    spam: bool = Field(..., description="True if message is promotional spam or bulk marketing")
    scam: bool = Field(..., description="True if message has phishing, fraud, or suspicious risk")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Risk score from 0.0 to 1.0")
    deadline: Optional[str] = Field(default=None, description="Explicit ISO-8601 date/timestamp if mentioned, null otherwise")
    reason: str = Field(..., description="Short explanation for the classification")

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: str) -> str:
        if isinstance(v, str):
            v_clean = v.strip().upper()
            valid = {"WORK", "PERSONAL", "PROMOTIONAL", "TRANSACTIONAL", "SOCIAL", "NEWSLETTER", "OTHER"}
            return v_clean if v_clean in valid else "OTHER"
        return "OTHER"

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, v: str) -> str:
        if isinstance(v, str):
            v_clean = v.strip().upper()
            valid = {"INTERVIEW", "MEETING", "TASK", "NOTIFICATION", "INQUIRY", "CHAT", "SPAM", "OTHER"}
            return v_clean if v_clean in valid else "OTHER"
        return "OTHER"

    @field_validator("importance", "urgency", mode="before")
    @classmethod
    def normalize_level(cls, v: str) -> str:
        if isinstance(v, str):
            v_clean = v.strip().upper()
            if v_clean in ("HIGH", "MEDIUM", "LOW"):
                return v_clean
        raise ValueError(f"Must be HIGH, MEDIUM, or LOW, got: {v}")

    @field_validator("deadline", mode="before")
    @classmethod
    def validate_deadline(cls, v: Optional[str]) -> Optional[str]:
        if not v or v in ("null", "None", ""):
            return None
        return str(v).strip()


class BatchAnalysisResponse(BaseModel):
    items: list[MessageAnalysisItem]
