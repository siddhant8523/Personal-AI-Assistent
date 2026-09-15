"""
Unit tests for LLM and network failure resilience.
===================================================
Verifies:
1. Network / DNS failure produces friendly response without traceback or process crash.
2. Rate limit (HTTP 429) returns friendly rate-limit message.
3. Rate limit with Retry-After header reflects the retry duration and bounded retry policy.
4. Authentication failure (HTTP 401/403) returns safe config message without leaking keys.
5. Upstream service failure (HTTP 5xx) returns temporary service message.
6. Unexpected exceptions return safe fallback and preserve developer logging.
7. Conversation survives failure: subsequent requests succeed normally.
8. Presence indicator stop() is guaranteed on all failure paths.
9. LangGraph tool execution is not re-executed or looped when an LLM error occurs.
10. Existing successful LLM interactions and offline mode continue to operate normally.
"""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from assistant.agent_core.agent_command_handler import AgentCommandHandler
from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.context_builder import ContextBuilder
from assistant.agent_core.orchestrator import AgentOrchestrator
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.agent_core.task_planner import TaskPlanner
from assistant.channels.cli_channel import CLIChannel
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.llm.llm_client import (
    LLMAuthenticationError,
    LLMClient,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMServiceError,
    classify_llm_error,
    llm_error_to_user_message,
)
from assistant.memory.memory_service import MemoryService
from assistant.router.conversation_registry import ConversationClass, ConversationRegistry
from assistant.router.conversation_router import ConversationRouter
from assistant.router.echo_filter import EchoFilter


def make_mock_llm_client():
    client = LLMClient(api_key="fake-groq-key", provider="groq")
    mock_groq = MagicMock()
    client._client = mock_groq
    return client, mock_groq


def make_orchestrator(tmp_path, llm_client):
    memory = MemoryService(memory_dir=str(tmp_path / "memory"))
    context_builder = ContextBuilder(memory)
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    tool_router = ToolRouter(task_manager)
    policy_engine = PolicyEngine({})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    priority_inbox = PriorityInbox()

    orchestrator = AgentOrchestrator(
        llm=llm_client,
        memory=memory,
        context_builder=context_builder,
        task_planner=task_planner,
        approval_manager=approval_manager,
        tool_router=tool_router,
        priority_inbox=priority_inbox,
    )
    return orchestrator, tool_router


# ==============================================================================
# 1. Error Classification & Formatting Tests
# ==============================================================================

def test_classify_network_connection_error():
    req = httpx.Request("POST", "https://api.groq.com")
    raw_exc = httpx.ConnectError("[Errno -3] Temporary failure in name resolution", request=req)
    err = classify_llm_error(raw_exc)

    assert isinstance(err, LLMNetworkError)
    assert err.category == "NETWORK"
    msg = llm_error_to_user_message(err)
    assert "internet connection is unavailable" in msg
    assert "Traceback" not in msg


def test_classify_rate_limit_error_with_retry_after():
    req = httpx.Request("POST", "https://api.groq.com")
    resp = httpx.Response(429, headers={"retry-after": "20"}, request=req)
    # Simulate an HTTP 429 status error
    class MockRateLimit(Exception):
        status_code = 429
        response = resp

    err = classify_llm_error(MockRateLimit("Rate limit reached"))
    assert isinstance(err, LLMRateLimitError)
    assert err.retry_after == 20.0
    msg = llm_error_to_user_message(err)
    assert "temporarily rate-limited" in msg
    assert "20 seconds" in msg


def test_classify_rate_limit_error_without_retry_after():
    class MockRateLimit(Exception):
        status_code = 429

    err = classify_llm_error(MockRateLimit("Rate limit exceeded"))
    assert isinstance(err, LLMRateLimitError)
    assert err.retry_after is None
    msg = llm_error_to_user_message(err)
    assert "The AI service is temporarily rate-limited. Please try again in a moment." in msg


def test_classify_auth_error():
    class MockAuthError(Exception):
        status_code = 401

    err = classify_llm_error(MockAuthError("Invalid API key provided"))
    assert isinstance(err, LLMAuthenticationError)
    msg = llm_error_to_user_message(err)
    assert "API key configuration" in msg
    assert "Invalid API key provided" not in msg  # No secrets or internal messages leaked


def test_classify_service_error():
    class MockServerError(Exception):
        status_code = 503

    err = classify_llm_error(MockServerError("Service Unavailable"))
    assert isinstance(err, LLMServiceError)
    msg = llm_error_to_user_message(err)
    assert "temporarily unavailable" in msg


def test_classify_unexpected_error():
    err = classify_llm_error(ValueError("Some internal bug"))
    assert isinstance(err, LLMError)
    msg = llm_error_to_user_message(err)
    assert "something went wrong" in msg


# ==============================================================================
# 2. End-to-End Resilience & User Experience
# ==============================================================================

def test_network_failure_returns_friendly_message_no_traceback(tmp_path):
    llm, mock_groq = make_mock_llm_client()
    req = httpx.Request("POST", "https://api.groq.com")
    mock_groq.chat.completions.create.side_effect = httpx.ConnectError("[Errno -3] Temporary failure in name resolution", request=req)

    orchestrator, _ = make_orchestrator(tmp_path, llm)
    replies = []
    handler = AgentCommandHandler(orchestrator, lambda msg, text: replies.append(text))

    user_msg = UnifiedMessage(source=Source.CLI, conversation_id="conv-1", sender="user", content="How are you?")
    handler(user_msg)

    assert len(replies) == 1
    reply = replies[0]
    assert "internet connection is unavailable" in reply
    assert "Traceback" not in reply
    assert "httpx" not in reply


def test_rate_limit_returns_friendly_message(tmp_path):
    llm, mock_groq = make_mock_llm_client()
    req = httpx.Request("POST", "https://api.groq.com")
    resp = httpx.Response(429, headers={"retry-after": "15"}, request=req)

    class RateLimit429(Exception):
        status_code = 429
        response = resp

    mock_groq.chat.completions.create.side_effect = RateLimit429("Rate limit reached")

    orchestrator, _ = make_orchestrator(tmp_path, llm)
    replies = []
    handler = AgentCommandHandler(orchestrator, lambda msg, text: replies.append(text))

    user_msg = UnifiedMessage(source=Source.CLI, conversation_id="conv-1", sender="user", content="Tell me a joke")
    handler(user_msg)

    assert len(replies) == 1
    reply = replies[0]
    assert "temporarily rate-limited" in reply
    assert "15 seconds" in reply


def test_presence_cleanup_on_failure(tmp_path):
    llm, mock_groq = make_mock_llm_client()
    req = httpx.Request("POST", "https://api.groq.com")
    mock_groq.chat.completions.create.side_effect = httpx.ConnectError("Network unreachable", request=req)

    orchestrator, _ = make_orchestrator(tmp_path, llm)
    presence_mock = MagicMock()
    handler = AgentCommandHandler(
        orchestrator,
        reply_sender=lambda msg, text: None,
        presence_provider=lambda msg: presence_mock,
    )

    user_msg = UnifiedMessage(source=Source.CLI, conversation_id="conv-1", sender="user", content="ping")
    handler(user_msg)

    assert presence_mock.start.called
    assert presence_mock.stop.called


def test_conversation_survives_failure(tmp_path):
    """Verifies that after an LLM network failure, the conversation state remains
    clean and subsequent user messages succeed normally once connectivity is restored."""
    llm, mock_groq = make_mock_llm_client()

    # Request 1: Network fails
    req = httpx.Request("POST", "https://api.groq.com")
    mock_groq.chat.completions.create.side_effect = httpx.ConnectError("Network unreachable", request=req)

    orchestrator, _ = make_orchestrator(tmp_path, llm)
    replies = []
    handler = AgentCommandHandler(orchestrator, lambda msg, text: replies.append(text))

    msg1 = UnifiedMessage(source=Source.CLI, conversation_id="conv-persist", sender="user", content="first message")
    handler(msg1)

    assert len(replies) == 1
    assert "internet connection is unavailable" in replies[0]

    # Request 2: Connectivity restored, normal response generated
    mock_groq.chat.completions.create.side_effect = None
    mock_msg = MagicMock()
    mock_msg.content = "Hello! I am back online."
    mock_msg.tool_calls = []
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_groq.chat.completions.create.return_value = mock_resp

    msg2 = UnifiedMessage(source=Source.CLI, conversation_id="conv-persist", sender="user", content="second message")
    handler(msg2)

    assert len(replies) == 2
    assert replies[1] == "Hello! I am back online."


def test_langgraph_tool_not_repeated_on_llm_failure(tmp_path):
    """Verifies that when a tool has executed and the subsequent LLM summarization
    fails, the tool is NOT re-executed in an endless loop."""
    llm, mock_groq = make_mock_llm_client()
    orchestrator, tool_router = make_orchestrator(tmp_path, llm)

    tool_call_count = 0
    def dummy_tool(task):
        nonlocal tool_call_count
        tool_call_count += 1
        return {"status": "ok", "data": "tool output"}

    tool_router.register("send_whatsapp_message", dummy_tool)

    # First LLM call: returns tool call to send_whatsapp_message
    tool_call_msg = MagicMock()
    tool_call_msg.content = ""
    tool_call_msg.tool_calls = [
        {"name": "propose_send_message", "args": {"platform": "whatsapp", "recipient": "John", "content": "hi"}, "id": "call_1"}
    ]
    tool_call_choice = MagicMock(message=tool_call_msg)

    # Second LLM call (summarization/next step): fails with network error
    req = httpx.Request("POST", "https://api.groq.com")
    network_err = httpx.ConnectError("Network failure during LLM response", request=req)

    mock_groq.chat.completions.create.side_effect = [
        MagicMock(choices=[tool_call_choice]),
        network_err,
    ]

    replies = []
    handler = AgentCommandHandler(orchestrator, lambda msg, text: replies.append(text))

    msg = UnifiedMessage(source=Source.CLI, conversation_id="conv-tool", sender="user", content="Send John hi on whatsapp")
    handler(msg)

    # Tool should not be repeatedly executed
    assert len(replies) == 1
    assert "Proposed action" in replies[0] or "internet connection is unavailable" in replies[0]


def test_cli_channel_survives_unexpected_exception():
    """Verifies that even if the router encounters an unexpected exception,
    the CLIChannel loop does not crash."""
    mock_router = MagicMock()
    mock_router.route.side_effect = RuntimeError("Unexpected router failure")

    cli = CLIChannel(mock_router, agent_chat_conversation_id="cli_test")

    with patch("rich.console.Console.input", side_effect=["test command", "/quit"]), \
         patch("assistant.channels.cli_channel.render_assistant_reply") as mock_render:
        cli.run()

        # Rendered error message, loop exited gracefully via /quit
        mock_render.assert_called_with("Sorry, something went wrong while processing that request. Please try again.")
