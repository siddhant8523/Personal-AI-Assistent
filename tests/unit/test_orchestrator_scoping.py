import pytest
from unittest.mock import MagicMock

from assistant.agent_core.orchestrator import AgentOrchestrator
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task import TaskState
from assistant.execution.task_manager import TaskManager


def test_get_pending_task_conversation_isolation():
    task_manager = TaskManager(OutboundRegistry())
    task_planner = MagicMock()
    task_planner.task_manager = task_manager

    # Task for Conversation A
    task_a = task_manager.create_task("send_sms", target="conv_A", parameters={"conversation_id": "conv_A"}, requires_approval=True)
    task_a.transition(TaskState.WAITING_FOR_APPROVAL)

    orchestrator = MagicMock()
    orchestrator.task_planner = task_planner
    orchestrator._get_pending_task = AgentOrchestrator._get_pending_task.__get__(orchestrator)

    # Conversation A should find task_a
    assert orchestrator._get_pending_task("conv_A") == task_a

    # Conversation B should NOT find task_a even though it is the only pending task in the system
    assert orchestrator._get_pending_task("conv_B") is None


def test_get_pending_task_multiple_conversations():
    task_manager = TaskManager(OutboundRegistry())
    task_planner = MagicMock()
    task_planner.task_manager = task_manager

    task_a = task_manager.create_task("send_sms", target="conv_A", parameters={"conversation_id": "conv_A"}, requires_approval=True)
    task_a.transition(TaskState.WAITING_FOR_APPROVAL)

    task_b = task_manager.create_task("make_call", target="conv_B", parameters={"conversation_id": "conv_B"}, requires_approval=True)
    task_b.transition(TaskState.WAITING_FOR_APPROVAL)

    orchestrator = MagicMock()
    orchestrator.task_planner = task_planner
    orchestrator._get_pending_task = AgentOrchestrator._get_pending_task.__get__(orchestrator)

    assert orchestrator._get_pending_task("conv_A") == task_a
    assert orchestrator._get_pending_task("conv_B") == task_b
    assert orchestrator._get_pending_task("conv_C") is None
