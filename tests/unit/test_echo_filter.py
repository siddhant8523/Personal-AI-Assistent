"""Tests for the core architecture fix: the Echo / Self-Origin Filter."""

from assistant.execution.outbound_registry import OutboundRegistry
from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage
from assistant.router.echo_filter import EchoFilter


def test_origin_tagged_message_is_echo():
    registry = OutboundRegistry()
    filt = EchoFilter(registry)
    msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="c1", sender="user", content="hi", origin=Origin.AGENT)
    assert filt.is_echo(msg) is True


def test_known_self_identity_is_echo():
    registry = OutboundRegistry()
    filt = EchoFilter(registry, self_identities={"me"})
    msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="c1", sender="me", content="hi")
    assert filt.is_echo(msg) is True


def test_outbound_registry_match_is_echo():
    """The realistic Baileys case: platform echoes fromMe without any
    origin tag we control -- must still be caught via the registry."""
    registry = OutboundRegistry()
    filt = EchoFilter(registry)

    msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="rahul_chat", sender="me", content="on my way")
    registry.register_pending("TASK-1", source="whatsapp", conversation_id="rahul_chat", content_hash=msg.content_hash())

    assert filt.is_echo(msg) is True


def test_genuine_new_message_is_not_echo():
    registry = OutboundRegistry()
    filt = EchoFilter(registry)
    msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="rahul_chat", sender="Rahul", content="sounds good")
    assert filt.is_echo(msg) is False


def test_expired_outbound_registry_entry_does_not_match():
    registry = OutboundRegistry(window_seconds=0)
    filt = EchoFilter(registry)
    msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="rahul_chat", sender="me", content="on my way")
    registry.register_pending("TASK-1", "whatsapp", "rahul_chat", msg.content_hash())
    import time
    time.sleep(0.01)
    assert filt.is_echo(msg) is False
