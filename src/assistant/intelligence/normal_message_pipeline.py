"""
Normal Message Pipeline
===========================
Wires: Message Ingestion -> Priority Engine -> Priority Inbox -> (optional)
Agent Core summary for HIGH priority messages. This is what the
Conversation Router's `normal_message_handler` callback points at for
NORMAL-classified conversations. Agent Chat never reaches this file
(that's the whole point of Rule 8 / the routing split).
"""

from __future__ import annotations

import logging

from assistant.agent_core.orchestrator import AgentOrchestrator
from assistant.ingestion.deduplicator import Deduplicator
from assistant.ingestion.message_ingestion import MessageIngestion
from assistant.ingestion.unified_message import UnifiedMessage
from assistant.intelligence.priority_engine import PriorityEngine, PriorityLevel
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.memory.user_profile import UserProfile

from assistant.intelligence.eligibility_filter import MessageEligibilityFilter

logger = logging.getLogger("assistant.normal_pipeline")


class NormalMessagePipeline:
    def __init__(
        self,
        priority_engine: PriorityEngine,
        priority_inbox: PriorityInbox,
        user_profile: UserProfile,
        orchestrator: AgentOrchestrator,
        dedup_window_seconds: int = 300,
        eligibility_filter: MessageEligibilityFilter | None = None,
    ):
        self.priority_engine = priority_engine
        self.priority_inbox = priority_inbox
        self.user_profile = user_profile
        self.orchestrator = orchestrator
        self.eligibility_filter = eligibility_filter or MessageEligibilityFilter()
        self.ingestion = MessageIngestion(Deduplicator(dedup_window_seconds), self._on_normalized_message)

    def handle(self, message: UnifiedMessage) -> str:
        """Called by the Conversation Router for NORMAL-classified
        conversations. Evaluates message eligibility first."""
        eligibility = self.eligibility_filter.evaluate(message)
        if not eligibility.eligible:
            logger.debug("Skipped ineligible message (%s)", eligibility.reason)
            return f"skipped_{eligibility.reason}"

        return self.ingestion.ingest(message)

    def _on_normalized_message(self, message: UnifiedMessage) -> None:
        eligibility = self.eligibility_filter.evaluate(message)
        if not eligibility.eligible:
            return

        result = self.priority_engine.score(message, self.user_profile)
        self.priority_inbox.add(result)
        logger.info(
            "%s | %s | %s -> score=%s level=%s",
            result.message.source.value, result.message.sender, result.category.value,
            result.score, result.level.value,
        )
        if result.level == PriorityLevel.HIGH:
            self.orchestrator.handle_normal_message_summary(
                conversation_id=message.conversation_id,
                level=result.level.value,
                sender=message.sender,
                content=message.content,
            )
