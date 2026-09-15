"""Tests for the Conversation Router — verifies echo filter runs BEFORE
the Agent Chat / Normal Chat branch, and routing decisions are by
conversation identity, not message text."""

from assistant.execution.outbound_registry import OutboundRegistry
from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage
from assistant.router.conversation_registry import ConversationClass, ConversationRegistry
from assistant.router.conversation_router import ConversationRouter
from assistant.router.echo_filter import EchoFilter


def make_router():
    registry = ConversationRegistry(mapping={"agent_chat": "AGENT_CHAT", "rahul_chat": "NORMAL"})
    outbound = OutboundRegistry()
    echo_filter = EchoFilter(outbound)
    calls = {"agent": [], "normal": []}
    router = ConversationRouter(
        registry=registry, echo_filter=echo_filter,
        agent_command_handler=lambda m: calls["agent"].append(m),
        normal_message_handler=lambda m: calls["normal"].append(m),
    )
    return router, calls, outbound


def test_agent_chat_routes_to_agent_handler():
    router, calls, _ = make_router()
    msg = UnifiedMessage(source=Source.CLI, conversation_id="agent_chat", sender="user", content="hello")
    assert router.route(msg) == "routed_agent_chat"
    assert len(calls["agent"]) == 1
    assert len(calls["normal"]) == 0


def test_normal_chat_routes_to_normal_handler():
    router, calls, _ = make_router()
    msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="rahul_chat", sender="Rahul", content="hi")
    assert router.route(msg) == "routed_normal"
    assert len(calls["normal"]) == 1


def test_echo_never_reaches_either_handler_even_on_agent_chat():
    """This is the regression test for the bug: an agent-origin echo on the
    Agent Chat conversation must be dropped before reaching the agent
    command handler, not just before the (bypassed) ingestion pipeline."""
    router, calls, _ = make_router()
    echo = UnifiedMessage(
        source=Source.WHATSAPP, conversation_id="agent_chat", sender="me",
        content="agent's own reply", origin=Origin.AGENT,
    )
    assert router.route(echo) == "dropped_echo"
    assert calls["agent"] == []
    assert calls["normal"] == []


def test_text_content_does_not_affect_routing():
    """Routing must be based on conversation identity, not text content."""
    router, calls, _ = make_router()
    msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="rahul_chat", sender="Rahul", content="what are my important messages?")
    router.route(msg)
    assert len(calls["normal"]) == 1
    assert len(calls["agent"]) == 0
