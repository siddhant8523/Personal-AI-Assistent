"""
Conversation Router
======================
Order of operations, deliberately:

    Connector -> Echo Filter -> Conversation Router -> (Agent Chat | Normal)

The echo filter runs FIRST, before the Agent Chat / Normal Chat branch,
so both paths are protected by one component instead of relying on a
mitigation buried inside a pipeline that Agent Chat bypasses.

See: Part 3, Section 25-27 for the original problem statement, and
docs/architecture/04_bug_and_fix.md for the fix writeup.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from assistant.ingestion.unified_message import UnifiedMessage
from assistant.router.conversation_registry import ConversationClass, ConversationRegistry
from assistant.router.echo_filter import EchoFilter

logger = logging.getLogger("assistant.router")

AgentCommandHandler = Callable[[UnifiedMessage], None]
NormalMessageHandler = Callable[[UnifiedMessage], None]


class ConversationRouter:
    def __init__(
        self,
        registry: ConversationRegistry,
        echo_filter: EchoFilter,
        agent_command_handler: AgentCommandHandler,
        normal_message_handler: NormalMessageHandler,
    ):
        self._registry = registry
        self._echo_filter = echo_filter
        self._agent_command_handler = agent_command_handler
        self._normal_message_handler = normal_message_handler

    def route(self, message: UnifiedMessage) -> str:
        """Returns a short string describing what happened, mainly for
        logging/tests/demo purposes."""

        classification = self._registry.classify(message.conversation_id)

        if message.conversation_id and message.conversation_id.startswith("whatsapp:"):
            from_me = bool(message.metadata.get("from_me", False)) if message.metadata else False
            raw_chat_id = message.metadata.get("raw_chat_id", message.conversation_id.removeprefix("whatsapp:")) if message.metadata else message.conversation_id.removeprefix("whatsapp:")
            resolved_chat_id = message.conversation_id.removeprefix("whatsapp:")
            sender = message.sender
            sender_alt = message.metadata.get("sender_alt", sender) if message.metadata else sender
            is_agent = classification == ConversationClass.AGENT_CHAT

            if self._echo_filter.is_echo(message):
                logger.debug(
                    "[WA ROUTE]\nraw_chat_id=%s\nresolved_chat_id=%s\nsender=%s\nsender_alt=%s\nfrom_me=%s\nis_agent_user_identity=%s\nconversation_type=%s\npipeline_selected=IGNORED",
                    raw_chat_id, resolved_chat_id, sender, sender_alt, str(from_me).lower(), str(is_agent).lower(), classification.value,
                )
                return "dropped_echo"

            pipeline_selected = "AGENT_COMMAND" if classification == ConversationClass.AGENT_CHAT else "NORMAL_MESSAGE"
            logger.debug(
                "[WA ROUTE]\nraw_chat_id=%s\nresolved_chat_id=%s\nsender=%s\nsender_alt=%s\nfrom_me=%s\nis_agent_user_identity=%s\nconversation_type=%s\npipeline_selected=%s",
                raw_chat_id, resolved_chat_id, sender, sender_alt, str(from_me).lower(), str(is_agent).lower(), classification.value, pipeline_selected,
            )
        elif self._echo_filter.is_echo(message):
            logger.debug("DROPPED (self-echo)")
            return "dropped_echo"

        if classification == ConversationClass.AGENT_CHAT:
            if message.conversation_id and message.conversation_id.startswith("whatsapp:"):
                logger.info("[WhatsApp] Message routed to Agent Chat")
            logger.debug("[Conversation Router] AGENT_CHAT")
            self._agent_command_handler(message)
            return "routed_agent_chat"

        logger.debug("[Conversation Router] NORMAL")
        if message.conversation_id and message.conversation_id.startswith("whatsapp:"):
            logger.info("[WhatsApp] Message routed to message processing")
        self._normal_message_handler(message)
        return "routed_normal"
