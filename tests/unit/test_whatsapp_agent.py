import os
from unittest.mock import MagicMock
import pytest

from assistant.ingestion.adapters import adapt_whatsapp
from assistant.ingestion.unified_message import Source, Origin
from assistant.router.conversation_registry import ConversationClass, ConversationRegistry
from assistant.router.conversation_router import ConversationRouter
from assistant.router.echo_filter import EchoFilter
from assistant.execution.outbound_registry import OutboundRegistry


def test_whatsapp_authorized_user_classification():
    registry = ConversationRegistry()
    registry.set("whatsapp:919172767219@s.whatsapp.net", ConversationClass.AGENT_CHAT)

    assert registry.classify("whatsapp:919172767219@s.whatsapp.net") == ConversationClass.AGENT_CHAT
    assert registry.classify("whatsapp:919888888888@s.whatsapp.net") == ConversationClass.NORMAL


def test_whatsapp_routing_authorized_vs_unauthorized():
    registry = ConversationRegistry()
    registry.set("whatsapp:919172767219@s.whatsapp.net", ConversationClass.AGENT_CHAT)

    outbound_reg = OutboundRegistry()
    echo_filter = EchoFilter(outbound_reg)

    agent_handler = MagicMock()
    normal_handler = MagicMock()

    router = ConversationRouter(
        registry=registry,
        echo_filter=echo_filter,
        agent_command_handler=agent_handler,
        normal_message_handler=normal_handler,
    )

    # 1. Authorized payload
    payload_auth = {
        "id": "1",
        "chat_id": "919172767219@s.whatsapp.net",
        "raw_chat_id": "52909752496163@lid",
        "sender": "52909752496163@lid",
        "text": "Hi Jarvis",
        "from_me": False,
    }
    msg_auth = adapt_whatsapp(payload_auth)
    res_auth = router.route(msg_auth)

    assert res_auth == "routed_agent_chat"
    assert agent_handler.called
    assert not normal_handler.called

    agent_handler.reset_mock()
    normal_handler.reset_mock()

    # 2. Unauthorized payload (Rahul)
    payload_unauth = {
        "id": "2",
        "chat_id": "919888888888@s.whatsapp.net",
        "raw_chat_id": "919888888888@s.whatsapp.net",
        "sender": "919888888888@s.whatsapp.net",
        "text": "Hello",
        "from_me": False,
    }
    msg_unauth = adapt_whatsapp(payload_unauth)
    res_unauth = router.route(msg_unauth)

    assert res_unauth == "routed_normal"
    assert not agent_handler.called
    assert normal_handler.called


def test_whatsapp_echo_filter():
    outbound_reg = OutboundRegistry()
    echo_filter = EchoFilter(outbound_reg)

    payload_echo = {
        "id": "3",
        "chat_id": "919172767219@s.whatsapp.net",
        "sender": "919309026953@s.whatsapp.net",
        "text": "Outbound echo",
        "from_me": True,
    }
    msg_echo = adapt_whatsapp(payload_echo)
    assert echo_filter.is_echo(msg_echo) is True
