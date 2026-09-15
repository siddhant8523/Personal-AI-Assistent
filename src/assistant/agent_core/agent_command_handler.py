"""
Agent Command Handler (Part 1, Section 6; Part 2, Section 12)
==================================================================
Thin adapter the Conversation Router calls for AGENT_CHAT-classified
messages. Its only job is to hand off to the Orchestrator and route the
reply back out through the originating channel -- it deliberately knows
nothing about priority scoring or message intelligence, because Agent
Chat traffic must never enter that pipeline (Rule 8).
"""

from __future__ import annotations

import logging
from typing import Callable

from assistant.agent_core.orchestrator import AgentOrchestrator
from assistant.channels.presence import NullPresenceIndicator, PresenceIndicator
from assistant.ingestion.unified_message import UnifiedMessage

logger = logging.getLogger("assistant.agent_command_handler")

ReplySender = Callable[[UnifiedMessage, str], None]
PresenceProvider = Callable[[UnifiedMessage], PresenceIndicator]


class AgentCommandHandler:
    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        reply_sender: ReplySender,
        presence_provider: PresenceProvider | None = None,
    ):
        self._orchestrator = orchestrator
        self._reply_sender = reply_sender
        self._presence_provider = presence_provider

    def __call__(self, message: UnifiedMessage) -> None:
        presence = self._presence_provider(message) if self._presence_provider else NullPresenceIndicator()
        presence.start()
        req_id = message.platform_message_id or (message.metadata.get("whatsapp_message_id") if message.metadata else None) or message.message_id
        try:
            logger.info("Agent processing started")
            if message.conversation_id and message.conversation_id.startswith("whatsapp:"):
                logger.debug("[WhatsApp] AgentCommandHandler invocation message_id=%s", req_id)
                logger.debug("[Agent Core] processing WhatsApp agent message")
            else:
                logger.debug("[Agent Core] processing message: conv=%s", message.conversation_id)

            reply_text = self._orchestrator.handle_agent_command(message)
            logger.debug("[AGENT] final response generated task_id=%s", req_id)
            logger.info("Response generated")

            logger.debug("[CHANNEL] sending response task_id=%s", req_id)
            logger.info("Response delivery started")
            self._reply_sender(message, reply_text)
            logger.debug("[CHANNEL] response sent task_id=%s", req_id)
            logger.info("Response delivery completed")
            logger.debug("[CHANNEL] request completed task_id=%s", req_id)
        except Exception as exc:
            logger.error("Request failed: %s", exc)
            raise
        finally:
            presence.stop()
