"""
Message Eligibility Filter (Two-Stage Architecture)
===================================================
Filters inbound messages before insertion into PriorityInbox:
- Stage 1: Hard Exclusion (Self-origin, Agent channels, Agent noise, Automated bots, Auth/OTP noise)
- Stage 2: Priority Eligibility (Validates human communication)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage


@dataclass
class EligibilityResult:
    eligible: bool
    reason: str
    category: str


class SubFilter(Protocol):
    def evaluate(self, message: UnifiedMessage) -> EligibilityResult | None:
        ...


class OriginFilter:
    def evaluate(self, message: UnifiedMessage) -> EligibilityResult | None:
        if message.origin == Origin.AGENT or message.metadata.get("from_me") is True:
            return EligibilityResult(eligible=False, reason="agent_origin", category="self_origin")
        return None


class AgentChannelFilter:
    def __init__(self, conversation_registry=None, whatsapp_connector=None):
        self.conversation_registry = conversation_registry
        self.whatsapp_connector = whatsapp_connector

    def evaluate(self, message: UnifiedMessage) -> EligibilityResult | None:
        cid = (message.conversation_id or "").lower()
        if cid.startswith("cli:") or cid.startswith("telegram_bot:"):
            return EligibilityResult(eligible=False, reason="agent_control_channel", category="agent_chat")

        if self.conversation_registry:
            from assistant.router.conversation_registry import ConversationClass
            if self.conversation_registry.classify(message.conversation_id) == ConversationClass.AGENT_CHAT:
                return EligibilityResult(eligible=False, reason="agent_control_channel", category="agent_chat")

        # Check WhatsApp agent user channel
        agent_wa_id = (str(message.conversation_id) if message.conversation_id else "").lower()
        raw_id = agent_wa_id.removeprefix("whatsapp:")
        if message.source == Source.WHATSAPP:
            if "919172767219" in agent_wa_id or "52909752496163" in agent_wa_id or raw_id == "52909752496163@lid":
                return EligibilityResult(eligible=False, reason="whatsapp_agent_chat", category="agent_chat")
            if self.whatsapp_connector and self.whatsapp_connector.is_agent_user_identity(raw_id, message.sender):
                return EligibilityResult(eligible=False, reason="whatsapp_agent_chat", category="agent_chat")

        return None


class AgentNoiseFilter:
    NOISE_PATTERNS = (
        "proposed action",
        "approved and executed",
        "approved, but failed to execute",
        "waiting for your approval",
        "approval required",
        "there are no pending actions",
        "approve or reject",
        "draft (v",
        "task sent successfully",
        "draft retained",
        "conversational state cleared",
        "i'm ready.",
        "hello. how can i assist",
    )

    def evaluate(self, message: UnifiedMessage) -> EligibilityResult | None:
        content_lower = (message.content or "").lower().strip()
        for pattern in self.NOISE_PATTERNS:
            if pattern in content_lower:
                return EligibilityResult(eligible=False, reason="agent_generated_noise", category="agent_noise")
        return None


class BotSystemFilter:
    KNOWN_BOTS = {
        "botfather",
        "userinfobot",
        "idbot",
        "projectbot",
        "telegram",
    }

    def evaluate(self, message: UnifiedMessage) -> EligibilityResult | None:
        sender_lower = (message.sender or "").lower().strip()
        content_lower = (message.content or "").lower().strip()

        for bot in self.KNOWN_BOTS:
            if bot in sender_lower or bot in content_lower:
                return EligibilityResult(eligible=False, reason="known_bot_sender", category="bot_noise")

        if sender_lower.endswith("bot") or "userinfo" in sender_lower or "idbot" in sender_lower or "userinfobot" in content_lower or "idbot" in content_lower:
            return EligibilityResult(eligible=False, reason="automated_bot_pattern", category="bot_noise")

        return None


class AuthSecurityFilter:
    AUTH_PATTERNS = (
        r"\blogin code:\s*\d+",
        r"\blogin code\b",
        r"\bverification code\b",
        r"\byour otp is\b",
        r"\botp:\s*\d+",
        r"\bdo not give this code to anyone\b",
        r"\bpasscode\b",
        r"\bsecurity code\b",
        r"\b2fa code\b",
    )

    def evaluate(self, message: UnifiedMessage) -> EligibilityResult | None:
        content_lower = (message.content or "").lower()
        for pattern in self.AUTH_PATTERNS:
            if re.search(pattern, content_lower):
                return EligibilityResult(eligible=False, reason="auth_security_noise", category="system_noise")
        return None


class MessageEligibilityFilter:
    """Two-stage filter engine for Priority Inbox eligibility."""

    def __init__(
        self,
        custom_filters: list[SubFilter] | None = None,
        conversation_registry=None,
        whatsapp_connector=None,
    ):
        self.stage1_filters: list[SubFilter] = custom_filters or [
            OriginFilter(),
            AgentChannelFilter(conversation_registry=conversation_registry, whatsapp_connector=whatsapp_connector),
            AgentNoiseFilter(),
            BotSystemFilter(),
            AuthSecurityFilter(),
        ]

    def evaluate(self, message: UnifiedMessage) -> EligibilityResult:
        # Stage 1: Hard Exclusions
        for sub_filter in self.stage1_filters:
            res = sub_filter.evaluate(message)
            if res is not None and not res.eligible:
                return res

        # Stage 2: Priority Eligibility (Passes hard exclusion -> Eligible human message)
        return EligibilityResult(eligible=True, reason="human_communication", category="inbound_human")
