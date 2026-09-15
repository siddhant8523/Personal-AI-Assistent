"""
Focused unit tests for direct LangGraph checkpoint resumption via resume_approval().

Requirements tested:
1. resume_approval() does not construct a new UnifiedMessage.
2. It directly invokes the existing checkpoint resume mechanism via Command(resume=action).
3. No LLM call occurs merely because approval is resumed.
4. The original thread/checkpoint is used.
5. approve executes the pending action once.
6. reject executes zero consequential tools.
7. duplicate approval cannot resume twice.
"""

from unittest.mock import MagicMock, patch
import pytest

from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.context_builder import ContextBuilder
from assistant.agent_core.events import AgentStreamEvent
from assistant.agent_core.orchestrator import AgentOrchestrator
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.agent_core.task_planner import TaskPlanner
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task import TaskState
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.ingestion.unified_message import Source, UnifiedMessage
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.llm.llm_client import LLMClient
from assistant.memory.memory_service import MemoryService
from langgraph.types import Command


def make_test_orchestrator(tmp_path):
    """Constructs a real LangGraph-backed AgentOrchestrator with mock dispatch."""
    memory = MemoryService(memory_dir=str(tmp_path / "memory"))
    llm = LLMClient(api_key="")  # offline mode, deterministic
    context_builder = ContextBuilder(memory)
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    tool_router = ToolRouter(task_manager)

    mock_dispatch = MagicMock(return_value={"status": "ok", "detail": "sent via whatsapp"})
    tool_router.register("send_whatsapp_message", mock_dispatch)
    policy_engine = PolicyEngine({"send_whatsapp_message": {"requires_approval": True}})
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
    return orchestrator, task_manager, mock_dispatch


def _trigger_approval_interrupt(orchestrator, conversation_id="conv-1"):
    """Helper to start an action that pauses in interrupt()."""
    send_msg = UnifiedMessage(
        source=Source.CLI,
        conversation_id=conversation_id,
        sender="user",
        content="send Priya on whatsapp: running late",
    )
    draft_reply = orchestrator.handle_agent_command(send_msg)
    assert "Proposed action" in draft_reply

    config = {"configurable": {"thread_id": conversation_id}}
    snapshot = orchestrator._graph.get_state(config)
    assert snapshot.next, "graph should be paused inside interrupt()"
    pending_id = orchestrator._pending_task_id(snapshot)
    assert pending_id is not None
    return pending_id


def test_resume_approval_does_not_construct_unified_message(tmp_path):
    """Requirement: resume_approval() must NOT construct a new UnifiedMessage."""
    orchestrator, task_manager, _ = make_test_orchestrator(tmp_path)
    task_id = _trigger_approval_interrupt(orchestrator, "conv-no-msg")

    # Patch UnifiedMessage constructor to ensure it is never called
    with patch(
        "assistant.ingestion.unified_message.UnifiedMessage",
        side_effect=AssertionError("UnifiedMessage was constructed inside resume_approval!"),
    ), patch.object(
        orchestrator,
        "stream_agent_command",
        side_effect=AssertionError("stream_agent_command was called!"),
    ):
        events = list(orchestrator.resume_approval(conversation_id="conv-no-msg", task_id=task_id, action="approve"))

    assert len(events) > 0
    assert any(e.type == "complete" for e in events)
    assert task_manager.get(task_id).execution_state == TaskState.COMPLETED


def test_resume_approval_directly_invokes_checkpoint_resume_mechanism(tmp_path):
    """Requirement: Directly resumes the checkpoint using Command(resume=action)."""
    orchestrator, task_manager, _ = make_test_orchestrator(tmp_path)
    task_id = _trigger_approval_interrupt(orchestrator, "conv-direct-invoke")

    real_invoke = orchestrator._graph.invoke
    invoke_spy = MagicMock(side_effect=real_invoke)
    orchestrator._graph.invoke = invoke_spy

    events = list(orchestrator.resume_approval(conversation_id="conv-direct-invoke", task_id=task_id, action="approve"))

    assert invoke_spy.call_count == 1
    call_args, call_kwargs = invoke_spy.call_args
    assert isinstance(call_args[0], Command)
    assert call_args[0].resume == "approve"
    assert call_kwargs["config"]["configurable"]["thread_id"] == "conv-direct-invoke"
    assert task_manager.get(task_id).execution_state == TaskState.COMPLETED


def test_no_llm_call_occurs_when_approval_resumed(tmp_path):
    """Requirement: No LLM call occurs merely because approval is resumed."""
    orchestrator, task_manager, _ = make_test_orchestrator(tmp_path)
    task_id = _trigger_approval_interrupt(orchestrator, "conv-no-llm")

    # Spy on LLM generate
    with patch.object(
        orchestrator.llm,
        "generate",
        side_effect=AssertionError("LLM.generate called during approval resumption!"),
    ):
        events = list(orchestrator.resume_approval(conversation_id="conv-no-llm", task_id=task_id, action="approve"))

    assert len(events) > 0
    assert task_manager.get(task_id).execution_state == TaskState.COMPLETED


def test_original_thread_checkpoint_is_used(tmp_path):
    """Requirement: The exact existing conversation_id / thread_id / checkpoint identity is preserved."""
    orchestrator, task_manager, _ = make_test_orchestrator(tmp_path)
    thread_id = "thread-xyz-789"
    task_id = _trigger_approval_interrupt(orchestrator, thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    snapshot_before = orchestrator._graph.get_state(config)
    assert snapshot_before.next

    events = list(orchestrator.resume_approval(conversation_id=thread_id, task_id=task_id, action="approve"))

    # Verify state after resume
    snapshot_after = orchestrator._graph.get_state(config)
    assert not snapshot_after.next, "Checkpoint on original thread should have finished execution"
    assert task_manager.get(task_id).execution_state == TaskState.COMPLETED


def test_approve_executes_pending_action_once(tmp_path):
    """Requirement: approve executes the already-approved pending action exactly once."""
    orchestrator, task_manager, mock_dispatch = make_test_orchestrator(tmp_path)
    task_id = _trigger_approval_interrupt(orchestrator, "conv-exec-once")

    assert mock_dispatch.call_count == 0

    events = list(orchestrator.resume_approval(conversation_id="conv-exec-once", task_id=task_id, action="approve"))

    # Dispatched exactly once
    assert mock_dispatch.call_count == 1
    assert task_manager.get(task_id).execution_state == TaskState.COMPLETED
    complete_events = [e for e in events if e.type == "complete"]
    assert len(complete_events) == 1
    assert "Approved and executed" in complete_events[0].final_response


def test_reject_executes_zero_consequential_tools(tmp_path):
    """Requirement: Reject must result in zero consequential tool execution."""
    orchestrator, task_manager, mock_dispatch = make_test_orchestrator(tmp_path)
    task_id = _trigger_approval_interrupt(orchestrator, "conv-reject-zero")

    assert mock_dispatch.call_count == 0

    events = list(orchestrator.resume_approval(conversation_id="conv-reject-zero", task_id=task_id, action="reject"))

    # Zero tools executed
    assert mock_dispatch.call_count == 0
    assert task_manager.get(task_id).execution_state == TaskState.REJECTED
    complete_events = [e for e in events if e.type == "complete"]
    assert len(complete_events) == 1
    assert "Rejected" in complete_events[0].final_response


def test_duplicate_approval_cannot_resume_twice(tmp_path):
    """Requirement: duplicate approval cannot resume twice."""
    orchestrator, task_manager, mock_dispatch = make_test_orchestrator(tmp_path)
    task_id = _trigger_approval_interrupt(orchestrator, "conv-dup-guard")

    # First approval -> success
    events1 = list(orchestrator.resume_approval(conversation_id="conv-dup-guard", task_id=task_id, action="approve"))
    assert mock_dispatch.call_count == 1
    assert task_manager.get(task_id).execution_state == TaskState.COMPLETED

    # Second approval attempt on the same task -> blocked
    events2 = list(orchestrator.resume_approval(conversation_id="conv-dup-guard", task_id=task_id, action="approve"))
    assert mock_dispatch.call_count == 1, "Tool dispatch must NOT run a second time"
    error_events = [e for e in events2 if e.type == "error"]
    assert len(error_events) == 1
    assert f"Task {task_id} has already been completed" in error_events[0].message


def test_resume_approval_flexible_arguments(tmp_path):
    """Verify resume_approval accepts both (conv, task_id, action) and (conv, action, task_id)."""
    orchestrator, task_manager, mock_dispatch = make_test_orchestrator(tmp_path)

    # Positional order: (conversation_id, task_id, action)
    t1 = _trigger_approval_interrupt(orchestrator, "conv-arg1")
    events1 = list(orchestrator.resume_approval("conv-arg1", t1, "approve"))
    assert task_manager.get(t1).execution_state == TaskState.COMPLETED

    # Positional order: (conversation_id, action, task_id)
    t2 = _trigger_approval_interrupt(orchestrator, "conv-arg2")
    events2 = list(orchestrator.resume_approval("conv-arg2", "reject", t2))
    assert task_manager.get(t2).execution_state == TaskState.REJECTED

    # Keyword arguments
    t3 = _trigger_approval_interrupt(orchestrator, "conv-arg3")
    events3 = list(orchestrator.resume_approval(conversation_id="conv-arg3", action="approve", task_id=t3))
    assert task_manager.get(t3).execution_state == TaskState.COMPLETED
