"""
Unit tests for the reusable Presence Indicator system.

Verifies:
- Generic request lifecycle (start -> processing -> send -> stop)
- Long-running processing (heartbeat pulses during slow agent/tool execution)
- Large response delivery (presence remains active across all chunks until final chunk)
- Slow send delivery (presence active while send is delayed)
- Processing failure (presence stops cleanly when agent processing fails)
- Send failure (presence stops cleanly when send fails)
- Cancellation handling (heartbeat cleaned up on cancellation/timeout)
- Multiple simultaneous chats (independent presence tracking per destination)
- Multiple concurrent requests for the same chat (reference-counted request tracking)
- Heartbeat cleanup (no orphaned threads/tasks left behind)
- Telegram-specific typing updates (periodic typing, clean stop)
- WhatsApp-specific composing updates (periodic composing, paused on stop)
- Error handling during cleanup (cleanup failures do not mask original request errors)
- Request lifecycle logging sequence
"""

import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from assistant.agent_core.agent_command_handler import AgentCommandHandler
from assistant.channels.presence import (
    NullPresenceIndicator,
    PresenceCoordinator,
    PresenceIndicator,
    TelegramPresenceIndicator,
    WhatsAppPresenceIndicator,
)
from assistant.connectors.telegram.telegram_connector import TelegramConnector
from assistant.connectors.whatsapp.whatsapp_connector import WhatsAppConnector
from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage


class MockTelegramConnector:
    def __init__(self):
        self.actions = []
        self.lock = threading.Lock()

    def send_chat_action(self, chat_id: str, action: str = "typing"):
        with self.lock:
            self.actions.append((chat_id, action, time.time()))
        return {"status": "ok"}


class MockWhatsAppConnector:
    def __init__(self):
        self.presence_updates = []
        self.lock = threading.Lock()
        self.fail_paused = False

    def send_presence_update(self, chat_id: str, state: str = "composing"):
        with self.lock:
            if state == "paused" and self.fail_paused:
                raise RuntimeError("WhatsApp bridge connection lost")
            self.presence_updates.append((chat_id, state, time.time()))
        return {"status": "ok"}


def test_null_presence_indicator():
    indicator = NullPresenceIndicator("test_dest")
    assert not indicator.is_active
    assert indicator.destination == "test_dest"

    with indicator:
        assert indicator.is_active

    assert not indicator.is_active


def test_telegram_periodic_heartbeat_and_cleanup():
    connector = MockTelegramConnector()
    coordinator = PresenceCoordinator()
    indicator = TelegramPresenceIndicator(
        connector=connector,
        chat_id="12345",
        interval_seconds=0.05,
        coordinator=coordinator,
    )

    indicator.start()
    assert indicator.is_active
    assert coordinator.is_active("telegram", "12345")

    # Verify initial pulse sent immediately
    assert len(connector.actions) >= 1
    assert connector.actions[0][0] == "12345"
    assert connector.actions[0][1] == "typing"

    # Wait for at least 2 more pulses
    time.sleep(0.15)
    with connector.lock:
        pulse_count = len(connector.actions)
    assert pulse_count >= 3

    indicator.stop()
    assert not indicator.is_active
    assert not coordinator.is_active("telegram", "12345")

    # Verify worker thread was terminated and cleaned up
    worker = coordinator.get_worker("telegram", "12345")
    assert worker is None

    # Wait and ensure no further pulses are sent
    final_count = len(connector.actions)
    time.sleep(0.1)
    assert len(connector.actions) == final_count


def test_whatsapp_periodic_heartbeat_and_paused_on_stop():
    connector = MockWhatsAppConnector()
    coordinator = PresenceCoordinator()
    indicator = WhatsAppPresenceIndicator(
        connector=connector,
        jid="user@s.whatsapp.net",
        interval_seconds=0.05,
        coordinator=coordinator,
    )

    indicator.start()
    assert indicator.is_active
    assert coordinator.is_active("whatsapp", "user@s.whatsapp.net")

    # Initial pulse
    assert len(connector.presence_updates) >= 1
    assert connector.presence_updates[0] == ("user@s.whatsapp.net", "composing", connector.presence_updates[0][2])

    time.sleep(0.15)
    with connector.lock:
        composing_count = sum(1 for _, state, _ in connector.presence_updates if state == "composing")
    assert composing_count >= 3

    indicator.stop()
    assert not indicator.is_active
    assert not coordinator.is_active("whatsapp", "user@s.whatsapp.net")

    # Verify final presence state is 'paused'
    with connector.lock:
        last_action = connector.presence_updates[-1]
    assert last_action[0] == "user@s.whatsapp.net"
    assert last_action[1] == "paused"

    # Ensure thread is cleaned up and no more pulses are sent
    time.sleep(0.1)
    with connector.lock:
        final_paused_count = sum(1 for _, state, _ in connector.presence_updates if state == "paused")
    assert final_paused_count == 1


def test_whatsapp_paused_failure_does_not_mask_error():
    connector = MockWhatsAppConnector()
    connector.fail_paused = True  # send_presence_update('paused') will raise
    coordinator = PresenceCoordinator()
    indicator = WhatsAppPresenceIndicator(
        connector=connector,
        jid="user@s.whatsapp.net",
        interval_seconds=0.05,
        coordinator=coordinator,
    )

    indicator.start()
    # stop should catch the error and not re-raise it
    indicator.stop()
    assert not indicator.is_active


def test_multiple_chats_independent_presence():
    tg_connector = MockTelegramConnector()
    wa_connector = MockWhatsAppConnector()
    coordinator = PresenceCoordinator()

    ind_tg_a = TelegramPresenceIndicator(tg_connector, "chat_A", interval_seconds=0.05, coordinator=coordinator)
    ind_tg_b = TelegramPresenceIndicator(tg_connector, "chat_B", interval_seconds=0.05, coordinator=coordinator)
    ind_wa_c = WhatsAppPresenceIndicator(wa_connector, "chat_C", interval_seconds=0.05, coordinator=coordinator)

    ind_tg_a.start()
    ind_tg_b.start()
    ind_wa_c.start()

    assert coordinator.is_active("telegram", "chat_A")
    assert coordinator.is_active("telegram", "chat_B")
    assert coordinator.is_active("whatsapp", "chat_C")

    # Stop chat A
    ind_tg_a.stop()
    assert not coordinator.is_active("telegram", "chat_A")
    assert coordinator.is_active("telegram", "chat_B")
    assert coordinator.is_active("whatsapp", "chat_C")

    # Stop chat B
    ind_tg_b.stop()
    assert not coordinator.is_active("telegram", "chat_B")
    assert coordinator.is_active("whatsapp", "chat_C")

    # Stop chat C
    ind_wa_c.stop()
    assert not coordinator.is_active("whatsapp", "chat_C")


def test_concurrent_requests_same_chat_shared_presence():
    connector = MockWhatsAppConnector()
    coordinator = PresenceCoordinator()

    ind1 = WhatsAppPresenceIndicator(connector, "group_1", request_id="req1", interval_seconds=0.05, coordinator=coordinator)
    ind2 = WhatsAppPresenceIndicator(connector, "group_1", request_id="req2", interval_seconds=0.05, coordinator=coordinator)

    ind1.start()
    ind2.start()

    assert coordinator.is_active("whatsapp", "group_1")
    assert coordinator.is_active("whatsapp", "group_1", request_id="req1")
    assert coordinator.is_active("whatsapp", "group_1", request_id="req2")

    # Stop request 1
    ind1.stop()
    assert not ind1.is_active
    assert ind2.is_active
    # Chat should still be active because request 2 is running
    assert coordinator.is_active("whatsapp", "group_1")
    # 'paused' should NOT have been sent yet
    with connector.lock:
        states = [s for _, s, _ in connector.presence_updates]
    assert "paused" not in states

    # Stop request 2
    ind2.stop()
    assert not ind2.is_active
    assert not coordinator.is_active("whatsapp", "group_1")
    # Now 'paused' should have been sent
    with connector.lock:
        states = [s for _, s, _ in connector.presence_updates]
    assert states[-1] == "paused"


def test_agent_command_handler_full_lifecycle_and_logging(caplog):
    connector = MockTelegramConnector()
    coordinator = PresenceCoordinator()

    def presence_provider(msg):
        cid = msg.conversation_id.removeprefix("telegram_bot:")
        return TelegramPresenceIndicator(connector, cid, request_id=msg.message_id, interval_seconds=0.05, coordinator=coordinator)

    sent_replies = []
    def reply_sender(msg, text):
        sent_replies.append((msg.conversation_id, text))

    orchestrator = MagicMock()
    orchestrator.handle_agent_command.return_value = "Hello from Agent"

    handler = AgentCommandHandler(orchestrator, reply_sender, presence_provider=presence_provider)

    msg = UnifiedMessage(
        source=Source.TELEGRAM,
        conversation_id="telegram_bot:9999",
        sender="user",
        content="Hello",
        origin=Origin.USER,
        message_id="msg_123",
    )

    caplog.set_level(logging.INFO)
    handler(msg)

    assert sent_replies == [("telegram_bot:9999", "Hello from Agent")]
    assert not coordinator.is_active("telegram", "9999")

    # Check log messages sequence
    log_texts = [r.message for r in caplog.records]
    assert "Presence started" in log_texts
    assert "Agent processing started" in log_texts
    assert "Response generated" in log_texts
    assert "Response delivery started" in log_texts
    assert "Response delivery completed" in log_texts
    assert "Presence stopped" in log_texts

    # Verify chronological ordering
    idx_p_start = log_texts.index("Presence started")
    idx_agent = log_texts.index("Agent processing started")
    idx_gen = log_texts.index("Response generated")
    idx_del_start = log_texts.index("Response delivery started")
    idx_del_comp = log_texts.index("Response delivery completed")
    idx_p_stop = log_texts.index("Presence stopped")

    assert idx_p_start < idx_agent < idx_gen < idx_del_start < idx_del_comp < idx_p_stop


def test_long_running_processing():
    connector = MockTelegramConnector()
    coordinator = PresenceCoordinator()

    def presence_provider(msg):
        return TelegramPresenceIndicator(connector, "chat_slow", interval_seconds=0.04, coordinator=coordinator)

    orchestrator = MagicMock()
    def slow_llm(msg):
        time.sleep(0.15)
        return "Slow response"
    orchestrator.handle_agent_command.side_effect = slow_llm

    handler = AgentCommandHandler(orchestrator, lambda m, t: None, presence_provider=presence_provider)
    msg = UnifiedMessage(source=Source.TELEGRAM, conversation_id="telegram_bot:chat_slow", sender="u", content="hi", origin=Origin.USER)

    handler(msg)

    # Verify multiple typing heartbeats were sent during the slow processing
    with connector.lock:
        actions = list(connector.actions)
    assert len(actions) >= 3
    assert not coordinator.is_active("telegram", "chat_slow")


def test_large_response_multiple_chunks():
    connector = MockTelegramConnector()
    coordinator = PresenceCoordinator()
    chunks_sent = []

    def presence_provider(msg):
        return TelegramPresenceIndicator(connector, "chat_large", interval_seconds=0.04, coordinator=coordinator)

    def chunked_reply_sender(msg, text):
        # Simulate delivery of 4 chunks with delays
        for i in range(4):
            assert coordinator.is_active("telegram", "chat_large"), f"Presence stopped early before chunk {i+1}!"
            time.sleep(0.04)
            chunks_sent.append(f"chunk_{i}")

    orchestrator = MagicMock()
    orchestrator.handle_agent_command.return_value = "Long message"

    handler = AgentCommandHandler(orchestrator, chunked_reply_sender, presence_provider=presence_provider)
    msg = UnifiedMessage(source=Source.TELEGRAM, conversation_id="telegram_bot:chat_large", sender="u", content="hi", origin=Origin.USER)

    handler(msg)

    assert len(chunks_sent) == 4
    # After final chunk, presence stops
    assert not coordinator.is_active("telegram", "chat_large")


def test_slow_send_delivery():
    connector = MockWhatsAppConnector()
    coordinator = PresenceCoordinator()

    def presence_provider(msg):
        return WhatsAppPresenceIndicator(connector, "user@jid", interval_seconds=0.04, coordinator=coordinator)

    def slow_reply_sender(msg, text):
        time.sleep(0.12)
        assert coordinator.is_active("whatsapp", "user@jid"), "Presence must remain active during slow network send!"

    orchestrator = MagicMock()
    orchestrator.handle_agent_command.return_value = "Reply"

    handler = AgentCommandHandler(orchestrator, slow_reply_sender, presence_provider=presence_provider)
    msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="whatsapp:user@jid", sender="u", content="hi", origin=Origin.USER)

    handler(msg)

    assert not coordinator.is_active("whatsapp", "user@jid")
    with connector.lock:
        states = [s for _, s, _ in connector.presence_updates]
    # Verify multiple composing pulses occurred, and final state is paused
    assert states.count("composing") >= 3
    assert states[-1] == "paused"


def test_processing_failure_stops_presence(caplog):
    connector = MockTelegramConnector()
    coordinator = PresenceCoordinator()

    def presence_provider(msg):
        return TelegramPresenceIndicator(connector, "fail_chat", interval_seconds=0.05, coordinator=coordinator)

    orchestrator = MagicMock()
    orchestrator.handle_agent_command.side_effect = ValueError("LLM inference error")

    handler = AgentCommandHandler(orchestrator, lambda m, t: None, presence_provider=presence_provider)
    msg = UnifiedMessage(source=Source.TELEGRAM, conversation_id="telegram_bot:fail_chat", sender="u", content="hi", origin=Origin.USER)

    caplog.set_level(logging.INFO)
    with pytest.raises(ValueError, match="LLM inference error"):
        handler(msg)

    # Presence must be stopped despite the exception
    assert not coordinator.is_active("telegram", "fail_chat")

    log_texts = [r.message for r in caplog.records]
    assert "Presence started" in log_texts
    assert any("Request failed" in txt for txt in log_texts)
    assert "Presence stopped" in log_texts


def test_send_failure_stops_presence():
    connector = MockWhatsAppConnector()
    coordinator = PresenceCoordinator()

    def presence_provider(msg):
        return WhatsAppPresenceIndicator(connector, "fail_send_jid", interval_seconds=0.05, coordinator=coordinator)

    def failing_sender(msg, text):
        raise ConnectionError("WhatsApp network down")

    orchestrator = MagicMock()
    orchestrator.handle_agent_command.return_value = "Reply"

    handler = AgentCommandHandler(orchestrator, failing_sender, presence_provider=presence_provider)
    msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="whatsapp:fail_send_jid", sender="u", content="hi", origin=Origin.USER)

    with pytest.raises(ConnectionError, match="WhatsApp network down"):
        handler(msg)

    # Presence must be cleaned up and stopped
    assert not coordinator.is_active("whatsapp", "fail_send_jid")
    with connector.lock:
        states = [s for _, s, _ in connector.presence_updates]
    assert states[-1] == "paused"


def test_cancellation_and_cleanup():
    connector = MockTelegramConnector()
    coordinator = PresenceCoordinator()

    indicator = TelegramPresenceIndicator(connector, "cancelled_chat", interval_seconds=0.05, coordinator=coordinator)
    indicator.start()
    assert coordinator.is_active("telegram", "cancelled_chat")

    # Simulate task cancellation by triggering cleanup
    worker = coordinator.get_worker("telegram", "cancelled_chat")
    assert worker is not None
    assert worker._thread is not None
    thread = worker._thread
    assert thread.is_alive()

    # Cancel via indicator stop
    indicator.stop()
    assert not thread.is_alive()
    assert not coordinator.is_active("telegram", "cancelled_chat")


def test_telegram_connector_send_chat_action_mock():
    conn = TelegramConnector(bot_token="", enabled=False)
    res = conn.send_chat_action("12345", "typing")
    assert res.get("status") == "ok"
    assert "[MOCK]" in res.get("detail", "")


def test_whatsapp_connector_send_presence_update_mock():
    conn = WhatsAppConnector(bridge_url="http://localhost:3000", enabled=False)
    res = conn.send_presence_update("919172767219@s.whatsapp.net", "composing")
    assert res.get("status") == "ok"
    assert "[MOCK]" in res.get("detail", "")

    res_paused = conn.send_presence_update("919172767219@s.whatsapp.net", "paused")
    assert res_paused.get("status") == "ok"
    assert "[MOCK]" in res_paused.get("detail", "")
