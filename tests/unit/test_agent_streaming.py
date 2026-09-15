"""
Unit Tests: Agent Core Streaming Foundation (tests/unit/test_agent_streaming.py)
=============================================================================
Tests deterministic streaming event model, token streaming, tool call preservation,
approval interrupts, LLM resilience during streaming, and synchronous compatibility.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolCallChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool

from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.context_builder import ContextBuilder
from assistant.agent_core.events import (
    AgentStreamEvent,
    sanitize_tool_result,
    sanitize_value,
)
from assistant.agent_core.graph.builder import build_agent_graph
from assistant.agent_core.orchestrator import AgentOrchestrator
from assistant.agent_core.task_planner import TaskPlanner
from assistant.ingestion.unified_message import Source, UnifiedMessage
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.execution.task import TaskState
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.llm.llm_client import (
    ChatGroqCustom,
    LLMAuthenticationError,
    LLMClient,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
)
from assistant.memory.memory_service import MemoryService


# ==============================================================================
# 1. Event Model & Sanitization Tests
# ==============================================================================

def test_event_model_construction_defaults():
    ev = AgentStreamEvent(type="status", message="Testing")
    assert ev.type == "status"
    assert ev.message == "Testing"
    assert ev.timestamp > 0
    assert ev.metadata == {}
    assert ev.content is None
    assert ev.final_response is None
    assert ev.tool_name is None


def test_event_model_factories():
    ev_status = AgentStreamEvent.status("Thinking...")
    assert ev_status.type == "status"
    assert ev_status.message == "Thinking..."

    ev_token = AgentStreamEvent.token("hello ")
    assert ev_token.type == "token"
    assert ev_token.content == "hello "

    ev_tool_start = AgentStreamEvent.tool_start(
        tool_name="set_alarm",
        tool_args={"time": "7:00 AM", "api_key": "secret123"},
        task_id="t-1",
    )
    assert ev_tool_start.type == "tool_start"
    assert ev_tool_start.tool_name == "set_alarm"
    assert ev_tool_start.task_id == "t-1"
    # Verify sanitization
    assert ev_tool_start.tool_args["time"] == "7:00 AM"
    assert ev_tool_start.tool_args["api_key"] == "[REDACTED]"

    ev_tool_complete = AgentStreamEvent.tool_complete(
        tool_name="set_alarm",
        tool_result="Alarm set with Bearer eyJhbGciOiJub25lIn0.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
        task_id="t-1",
        success=True,
    )
    assert ev_tool_complete.type == "tool_complete"
    assert "Bearer [REDACTED]" in ev_tool_complete.tool_result
    assert ev_tool_complete.metadata.get("success") is True

    ev_approval = AgentStreamEvent.approval_required(
        task_id="TASK-99",
        task_type="SEND_MESSAGE",
        target="Alice",
        draft="Hi Alice",
        message="Approve?",
    )
    assert ev_approval.type == "approval_required"
    assert ev_approval.task_id == "TASK-99"
    assert ev_approval.target == "Alice"
    assert ev_approval.draft == "Hi Alice"

    ev_err = AgentStreamEvent.error("Something went wrong", error_code="NET_ERR")
    assert ev_err.type == "error"
    assert ev_err.message == "Something went wrong"
    assert ev_err.metadata.get("error_code") == "NET_ERR"

    ev_comp = AgentStreamEvent.complete("Finished reply")
    assert ev_comp.type == "complete"
    assert ev_comp.final_response == "Finished reply"


def test_event_to_dict_and_json_serializability():
    ev = AgentStreamEvent.tool_start(
        tool_name="test_tool",
        tool_args={"nested": {"token": "secret_xyz", "value": 42}},
    )
    d = ev.to_dict()
    assert isinstance(d, dict)
    assert d["type"] == "tool_start"
    assert d["tool_args"]["nested"]["token"] == "[REDACTED]"
    assert d["tool_args"]["nested"]["value"] == 42

    # Verify JSON serialization does not raise
    serialized = json.dumps(d)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["tool_name"] == "test_tool"


def test_sanitization_helpers():
    # Token / secret sanitization
    data = {
        "api_key": "gsk_123456789012345678901234567890",
        "authorization": "Bearer token12345",
        "password": "pass",
        "safe_key": "safe_val",
        "list_data": [{"secret": "bad"}, "clean"],
    }
    sanitized = sanitize_value(data)
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["authorization"] == "[REDACTED]"
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["safe_key"] == "safe_val"
    assert sanitized["list_data"][0]["secret"] == "[REDACTED]"
    assert sanitized["list_data"][1] == "clean"

    # Result truncation and redaction
    long_result = "Result " + ("x" * 500)
    truncated = sanitize_tool_result(long_result, max_len=100)
    assert len(truncated) <= 100
    assert truncated.endswith("...")

    key_result = "Created with gsk_abcdefghijklmnopqrstuvwxyz123456"
    assert "[REDACTED_KEY]" in sanitize_tool_result(key_result)


# ==============================================================================
# 2. ChatGroqCustom Real Streaming & Resilience Tests
# ==============================================================================

class MockChunkChoice:
    def __init__(self, content=None, tool_calls=None):
        self.delta = MagicMock()
        self.delta.content = content
        self.delta.tool_calls = tool_calls


class MockChunk:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [MockChunkChoice(content=content, tool_calls=tool_calls)]


def test_chat_groq_custom_stream_tokens():
    mock_client = MagicMock()
    chunks = [
        MockChunk(content="Hello"),
        MockChunk(content=" world"),
        MockChunk(content="!"),
    ]
    mock_client.chat.completions.create.return_value = iter(chunks)

    model = ChatGroqCustom(client=mock_client, model_name="test-model")
    messages = [HumanMessage(content="Hi")]
    
    stream_results = list(model._stream(messages))
    assert len(stream_results) == 3
    assert [c.message.content for c in stream_results] == ["Hello", " world", "!"]


def test_chat_groq_custom_stream_tool_call_chunks():
    mock_client = MagicMock()
    tc_delta1 = {
        "id": "call_1",
        "index": 0,
        "function": {"name": "get_weather", "arguments": '{"loc'},
    }
    tc_delta2 = {
        "id": None,
        "index": 0,
        "function": {"name": None, "arguments": 'ation": "Paris"}'},
    }
    chunks = [
        MockChunk(content="", tool_calls=[tc_delta1]),
        MockChunk(content="", tool_calls=[tc_delta2]),
    ]
    mock_client.chat.completions.create.return_value = iter(chunks)

    model = ChatGroqCustom(client=mock_client, model_name="test-model")
    messages = [HumanMessage(content="Weather in Paris?")]
    
    stream_results = list(model._stream(messages))
    assert len(stream_results) == 2

    # Accumulating chunks via LangChain (+) must reconstruct tool_calls
    accumulated = stream_results[0].message + stream_results[1].message
    assert len(accumulated.tool_calls) == 1
    assert accumulated.tool_calls[0]["name"] == "get_weather"
    assert accumulated.tool_calls[0]["args"] == {"location": "Paris"}
    assert accumulated.tool_calls[0]["id"] == "call_1"


def test_chat_groq_custom_stream_retry_before_tokens():
    """Failure before any tokens have been emitted triggers bounded retry."""
    mock_client = MagicMock()
    # First attempt fails with network error
    # Second attempt succeeds
    chunks = [MockChunk(content="Success after retry")]
    mock_client.chat.completions.create.side_effect = [
        ConnectionError("DNS failure"),
        iter(chunks),
    ]

    model = ChatGroqCustom(client=mock_client, model_name="test-model")
    messages = [HumanMessage(content="Test")]

    with patch("time.sleep") as mock_sleep:
        results = list(model._stream(messages))
        assert len(results) == 1
        assert results[0].message.content == "Success after retry"
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == 0.5


def test_chat_groq_custom_stream_no_retry_after_tokens():
    """CRITICAL RULE: Never restart the stream after tokens have already been emitted."""
    mock_client = MagicMock()

    def faulty_generator():
        yield MockChunk(content="Token A")
        yield MockChunk(content="Token B")
        raise ConnectionError("Network dropped mid-stream")

    mock_client.chat.completions.create.return_value = faulty_generator()

    model = ChatGroqCustom(client=mock_client, model_name="test-model")
    messages = [HumanMessage(content="Test")]

    stream_iter = model._stream(messages)
    # First two chunks arrive
    c1 = next(stream_iter)
    assert c1.message.content == "Token A"
    c2 = next(stream_iter)
    assert c2.message.content == "Token B"

    # Mid-stream error must NOT retry from the beginning; it must raise LLMNetworkError
    with pytest.raises(LLMNetworkError):
        next(stream_iter)

    # Verify create was only called ONCE (no restart)
    assert mock_client.chat.completions.create.call_count == 1


def test_chat_groq_custom_rate_limit_resilience():
    """Rate limit with Retry-After <= 2.0s retries once; > 2.0s fails fast."""
    mock_client = MagicMock()

    class RateLimitExc(Exception):
        pass

    RateLimitExc.__name__ = "RateLimitError"
    rl_err = RateLimitExc("Rate limit exceeded. Please try again in 1.5s")

    mock_client.chat.completions.create.side_effect = [
        rl_err,
        iter([MockChunk(content="Recovered")]),
    ]

    model = ChatGroqCustom(client=mock_client, model_name="test-model")
    with patch("time.sleep") as mock_sleep:
        results = list(model._stream([HumanMessage(content="Test")]))
        assert len(results) == 1
        assert results[0].message.content == "Recovered"
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == 1.5


def test_chat_groq_custom_authentication_no_retry():
    """Authentication errors fail immediately without pointless retries."""
    mock_client = MagicMock()

    class AuthExc(Exception):
        pass

    AuthExc.__name__ = "AuthenticationError"
    mock_client.chat.completions.create.side_effect = AuthExc("Invalid API key provided")

    model = ChatGroqCustom(client=mock_client, model_name="test-model")
    with pytest.raises(LLMAuthenticationError):
        list(model._stream([HumanMessage(content="Test")]))

    # Must be called exactly once
    assert mock_client.chat.completions.create.call_count == 1


# ==============================================================================
# 3. Agent Orchestrator Streaming Foundation Tests
# ==============================================================================

@pytest.fixture
def orchestrator_setup(tmp_path):
    memory = MemoryService(memory_dir=str(tmp_path / "memory"))
    llm = LLMClient(api_key="")  # offline mode by default
    context_builder = ContextBuilder(memory)
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    tool_router = ToolRouter(task_manager)
    policy_engine = PolicyEngine({})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    priority_inbox = PriorityInbox()

    orchestrator = AgentOrchestrator(
        llm=llm,
        memory=memory,
        context_builder=context_builder,
        task_planner=task_planner,
        approval_manager=approval_manager,
        tool_router=tool_router,
        priority_inbox=priority_inbox,
    )
    return {
        "orchestrator": orchestrator,
        "llm": llm,
        "task_planner": task_planner,
        "approval_manager": approval_manager,
        "tool_router": tool_router,
    }


def test_stream_agent_command_normal_response(orchestrator_setup):
    """Verify normal streaming produces status, tokens, and exactly one complete event."""
    orch: AgentOrchestrator = orchestrator_setup["orchestrator"]
    llm: LLMClient = orchestrator_setup["llm"]

    # Mock LLM online streaming
    llm._client = MagicMock()
    mock_model = MagicMock()
    mock_chunks = [
        AIMessageChunk(content="Hello"),
        AIMessageChunk(content=" there!"),
        AIMessageChunk(content=" How can I help?"),
    ]
    mock_model.stream.return_value = iter(mock_chunks)

    with patch.object(llm, "get_langchain_model", return_value=mock_model):
        events = list(orch.stream_agent_command("Hello assistant"))

    types = [e.type for e in events]
    assert "status" in types
    assert "token" in types
    assert types.count("complete") == 1

    # Verify status ownership: load_context then agent thinking
    status_events = [e.message for e in events if e.type == "status"]
    assert "Checking context & memory..." in status_events
    assert "Thinking..." in status_events

    # Verify incremental token contents
    token_contents = [e.content for e in events if e.type == "token"]
    assert token_contents == ["Hello", " there!", " How can I help?"]

    # Verify final completion response matches accumulated text
    complete_ev = next(e for e in events if e.type == "complete")
    assert complete_ev.final_response == "Hello there! How can I help?"


def test_stream_agent_command_tool_execution(orchestrator_setup):
    """Verify tool_start, exactly-once tool execution, tool_complete, and completion."""
    orch: AgentOrchestrator = orchestrator_setup["orchestrator"]
    llm: LLMClient = orchestrator_setup["llm"]

    llm._client = MagicMock()

    # Tool call chunk from model
    tc_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            ToolCallChunk(
                name="query_priority_inbox",
                args="{}",
                id="call_priority_1",
                index=0,
            )
        ],
    )
    # After tool execution, second agent step produces conversational answer
    reply_chunk = AIMessageChunk(content="You have no unread priority messages.")

    mock_model = MagicMock()
    mock_model.stream.side_effect = [
        iter([tc_chunk]),
        iter([reply_chunk]),
    ]

    with patch.object(llm, "get_langchain_model", return_value=mock_model):
        events = list(orch.stream_agent_command("Check my priority messages"))

    types = [e.type for e in events]
    assert "tool_start" in types
    assert "tool_complete" in types
    assert types.count("complete") == 1

    tool_start_ev = next(e for e in events if e.type == "tool_start")
    assert tool_start_ev.tool_name == "query_priority_inbox"

    tool_comp_ev = next(e for e in events if e.type == "tool_complete")
    assert tool_comp_ev.tool_name == "query_priority_inbox"
    assert tool_comp_ev.tool_result is not None


def test_stream_agent_command_approval_interrupt_and_resume(orchestrator_setup):
    """Verify approval_required event, no execution before approval, and resume lifecycle."""
    orch: AgentOrchestrator = orchestrator_setup["orchestrator"]
    llm: LLMClient = orchestrator_setup["llm"]
    task_planner: TaskPlanner = orchestrator_setup["task_planner"]

    llm._client = MagicMock()

    # First turn: Agent proposes sending a message
    tc_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            ToolCallChunk(
                name="propose_send_message",
                args='{"platform": "telegram", "recipient": "Bob", "content": "Hello Bob"}',
                id="call_send_1",
                index=0,
            )
        ],
    )
    mock_model = MagicMock()
    mock_model.stream.return_value = iter([tc_chunk])

    msg = UnifiedMessage(
        source=Source.CLI,
        conversation_id="stream_conv_approval",
        sender="user",
        content="Send Hello Bob to Bob on Telegram",
    )

    with patch.object(llm, "get_langchain_model", return_value=mock_model):
        events_1 = list(orch.stream_agent_command(msg))

    types_1 = [e.type for e in events_1]
    assert "approval_required" in types_1
    # MUST NOT emit complete when approval is required
    assert "complete" not in types_1

    approval_ev = next(e for e in events_1 if e.type == "approval_required")
    assert "send" in approval_ev.task_type.lower()
    assert approval_ev.target == "Bob"
    assert "Hello Bob" in approval_ev.draft

    # Verify task is waiting for approval in task_manager and NOT dispatched yet
    waiting_task = task_planner.task_manager._tasks.get(approval_ev.task_id)
    assert waiting_task is not None
    assert waiting_task.execution_state == TaskState.WAITING_FOR_APPROVAL

    # Second turn: User approves via streaming
    approve_msg = UnifiedMessage(
        source=Source.CLI,
        conversation_id="stream_conv_approval",
        sender="user",
        content=f"approve {approval_ev.task_id}",
    )

    with patch.object(orch.tool_router, "dispatch", return_value={"status": "sent", "detail": "sent via telegram"}):
        events_2 = list(orch.stream_agent_command(approve_msg))

    types_2 = [e.type for e in events_2]
    assert "complete" in types_2
    comp_ev = next(e for e in events_2 if e.type == "complete")
    assert "Approved and executed" in comp_ev.final_response


def test_stream_agent_command_unexpected_worker_exception(orchestrator_setup):
    """Unexpected exception in worker yields error event and terminates cleanly without hanging."""
    orch: AgentOrchestrator = orchestrator_setup["orchestrator"]

    with patch.object(orch, "_handle_agent_command_internal", side_effect=RuntimeError("Fatal worker crash")):
        events = list(orch.stream_agent_command("hello", timeout=5.0))

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].message is not None
    assert events[0].metadata.get("error_type") == "RuntimeError"


def test_stream_agent_command_timeout(orchestrator_setup):
    """Timeout waiting on empty event queue terminates generator cleanly."""
    orch: AgentOrchestrator = orchestrator_setup["orchestrator"]

    def slow_worker(*args, **kwargs):
        time.sleep(0.5)

    with patch.object(orch, "_handle_agent_command_internal", side_effect=slow_worker):
        # Short timeout of 0.05s
        events = list(orch.stream_agent_command("hello", timeout=0.05))

    assert len(events) == 1
    assert events[0].type == "error"
    assert "timed out" in events[0].message.lower()


def test_synchronous_regression_unchanged(orchestrator_setup):
    """Existing synchronous handle_agent_command(...) continues to work identically."""
    orch: AgentOrchestrator = orchestrator_setup["orchestrator"]
    msg = UnifiedMessage(
        source=Source.CLI,
        conversation_id="sync_test_conv",
        sender="user",
        content="hello",
    )
    reply = orch.handle_agent_command(msg)
    assert isinstance(reply, str)
    assert len(reply) > 0


def test_request_isolation(orchestrator_setup):
    """Multiple concurrent streaming requests have isolated queues, tokens, and events."""
    orch: AgentOrchestrator = orchestrator_setup["orchestrator"]
    llm: LLMClient = orchestrator_setup["llm"]
    llm._client = MagicMock()

    class StreamModel:
        def stream(self, msgs, **kwargs):
            last_content = str(msgs[-1].content) if msgs else ""
            prefix = "reqB" if "User: Query reqB" in last_content else "reqA"
            return iter([
                AIMessageChunk(content=f"{prefix} token1 "),
                AIMessageChunk(content=f"{prefix} token2"),
            ])

    results = {}

    def run_stream(req_id: str):
        msg = UnifiedMessage(
            source=Source.CLI,
            conversation_id=f"conv_{req_id}",
            sender="user",
            content=f"Query {req_id}",
        )
        events = list(orch.stream_agent_command(msg))
        tokens = [e.content for e in events if e.type == "token"]
        comp = next(e for e in events if e.type == "complete")
        results[req_id] = {"tokens": tokens, "final": comp.final_response}

    with patch.object(llm, "get_langchain_model", return_value=StreamModel()):
        t1 = threading.Thread(target=run_stream, args=("reqA",))
        t2 = threading.Thread(target=run_stream, args=("reqB",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    # Verify no tokens or completions leaked across requests
    assert results["reqA"]["tokens"] == ["reqA token1 ", "reqA token2"]
    assert results["reqA"]["final"] == "reqA token1 reqA token2"

    assert results["reqB"]["tokens"] == ["reqB token1 ", "reqB token2"]
    assert results["reqB"]["final"] == "reqB token1 reqB token2"


def test_stream_agent_command_device_tool_execution(tmp_path):
    """Verify device tools execute exactly once, direct-route to respond_node, and no second LLM call."""
    memory = MemoryService(memory_dir=str(tmp_path / "memory"))
    llm = LLMClient(api_key="mock_key")
    llm._client = MagicMock()
    context_builder = ContextBuilder(memory)
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    tool_router = ToolRouter(task_manager)
    policy_engine = PolicyEngine({})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    priority_inbox = PriorityInbox()

    from assistant.device_gateway.device_gateway import (
        DeviceCommand,
        DeviceGateway,
        DeviceResult,
    )
    validator = MagicMock()
    validator.validate_android.return_value = True
    gateway = DeviceGateway(validator)

    class ImmediateMockDeviceHandle:
        def __init__(self, gw):
            self.gateway = gw
            self.dispatched_commands = []

        def dispatch(self, command: DeviceCommand):
            self.dispatched_commands.append(command)
            res = DeviceResult(task_id=command.task_id, status="ok", detail="Alarm set for 7:00 AM")
            self.gateway.handle_device_result(res)
            return {"status": "ok", "detail": "queued for Android device", "device_task_id": command.task_id}

    handle = ImmediateMockDeviceHandle(gateway)
    gateway.attach_device(handle)

    orchestrator = AgentOrchestrator(
        llm=llm,
        memory=memory,
        context_builder=context_builder,
        task_planner=task_planner,
        approval_manager=approval_manager,
        tool_router=tool_router,
        priority_inbox=priority_inbox,
        device_gateway=gateway,
    )

    tc_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            ToolCallChunk(
                name="set_alarm",
                args='{"time": "7:00 AM"}',
                id="call_alarm_1",
                index=0,
            )
        ],
    )
    mock_model = MagicMock()
    mock_model.stream.return_value = iter([tc_chunk])

    with patch.object(llm, "get_langchain_model", return_value=mock_model):
        events = list(orchestrator.stream_agent_command("Set alarm for 7:00 AM"))

    # Verify model streamed only ONCE (no unnecessary second agent invocation)
    assert mock_model.stream.call_count == 1

    types = [e.type for e in events]
    assert "tool_start" in types
    assert "tool_complete" in types
    assert types.count("complete") == 1

    tool_start_ev = next(e for e in events if e.type == "tool_start")
    assert tool_start_ev.tool_name == "set_alarm"
    assert tool_start_ev.tool_args == {"time": "7:00 AM"}

    tool_complete_ev = next(e for e in events if e.type == "tool_complete")
    assert tool_complete_ev.tool_name == "set_alarm"
    assert "Alarm set for 7:00 AM" in tool_complete_ev.tool_result

    # Final response came directly from tool execution via respond_node
    complete_ev = next(e for e in events if e.type == "complete")
    assert "Alarm set for 7:00 AM" in complete_ev.final_response
