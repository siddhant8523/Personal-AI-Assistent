"""
Echo / Self-Origin Filter
===========================
THE FIX (part 2 of 2).

Sits between the Connector and the Conversation Router — i.e. BEFORE the
Agent Chat / Normal Chat branch decision, not after it. This is the whole
point: the previous architecture put self-echo prevention inside the
Message Ingestion pipeline, but Agent Chat traffic bypasses that pipeline
entirely (it goes straight from the Router to the Agent Core). That left
the one channel most at risk of a self-triggering loop with no protection.

Two independent signals are checked (either is sufficient to flag an echo):

1. Sender identity — the event's sender is one of the assistant's own
   platform identities (bot id / self number / send-as address).
2. Outbound Registry match — an outbound send with the same
   (platform, conversation, content-hash) was registered within the
   echo window (see execution/outbound_registry.py).

Messages flagged as echoes are dropped silently before reaching either the
Agent Command Handler or the Message Ingestion pipeline — they never touch
Priority Engine, Memory, or the LLM.
"""

from __future__ import annotations

import logging

from assistant.execution.outbound_registry import OutboundRegistry
from assistant.ingestion.unified_message import UnifiedMessage, Origin

logger = logging.getLogger("assistant.router.echo_filter")


class EchoFilter:
    def __init__(self, outbound_registry: OutboundRegistry, self_identities: set[str] | None = None):
        self._registry = outbound_registry
        self._self_identities = self_identities or set()

    def add_self_identity(self, identity: str) -> None:
        self._self_identities.add(identity)

    def is_echo(self, message: UnifiedMessage) -> bool:
        # Signal 1: explicit origin tag or known self-identity as sender
        if message.origin == Origin.AGENT:
            logger.debug("echo dropped (origin=AGENT): %s", message.message_id)
            return True

        if message.sender in self._self_identities:
            logger.debug("echo dropped (sender is self identity %s)", message.sender)
            return True

        # Signal 2: outbound registry match (platform echoed our own send)
        record = self._registry.matches_recent_outbound(
            source=message.source.value if hasattr(message.source, "value") else str(message.source),
            conversation_id=message.conversation_id,
            content_hash=message.content_hash(),
        )
        if record is not None:
            logger.debug(
                "echo dropped (matched outbound task %s in conversation %s)",
                record.task_id, message.conversation_id,
            )
            return True

        return False
