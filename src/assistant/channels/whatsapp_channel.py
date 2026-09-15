"""Optional dedicated WhatsApp Agent Chat channel via the Baileys bridge."""

from __future__ import annotations

import logging
from assistant.channels.presence import PresenceIndicator, WhatsAppPresenceIndicator
from assistant.ingestion.unified_message import UnifiedMessage

logger = logging.getLogger("assistant.channels.whatsapp")


class WhatsAppChannel:
    def __init__(self, connector, router, agent_chat_id: str = "", poll_seconds: float = 2):
        self.connector, self.router = connector, router
        self.agent_chat_id, self.poll_seconds = agent_chat_id, poll_seconds

    def get_presence_indicator(
        self, jid: str | None = None, request_id: str | None = None, interval_seconds: float = 4.0
    ) -> PresenceIndicator:
        target = str(jid or self.agent_chat_id)
        return WhatsAppPresenceIndicator(
            connector=self.connector,
            jid=target,
            request_id=request_id,
            interval_seconds=interval_seconds,
        )

    def start(self) -> None:
        """No separate poller: the shared WhatsApp ingestion worker owns the
        bridge buffer, avoiding races that could drop normal conversations."""
        return None

    def reply(self, original: UnifiedMessage, text: str) -> None:
        target = original.conversation_id.removeprefix("whatsapp:") or self.agent_chat_id
        if target:
            logger.info("[WhatsApp Channel] routing agent message to target %s", target)
            if len(text) > 4096:
                for i in range(0, len(text), 4096):
                    self.connector.send_message(target, text[i:i + 4096])
            else:
                self.connector.send_message(target, text)

