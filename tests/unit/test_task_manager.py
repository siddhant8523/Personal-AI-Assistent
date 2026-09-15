import pytest

from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task import InvalidTransition, TaskState
from assistant.execution.task_manager import TaskManager


def test_task_lifecycle_happy_path():
    tm = TaskManager(OutboundRegistry())
    task = tm.create_task("send_whatsapp_message", target="Rahul", parameters={"content": "hi"}, requires_approval=True)
    assert task.execution_state == TaskState.PLANNED

    tm.request_approval(task, draft="hi")
    assert task.execution_state == TaskState.WAITING_FOR_APPROVAL

    tm.approve(task)
    assert task.execution_state == TaskState.APPROVED

    tm.mark_executing(task)
    assert task.execution_state == TaskState.EXECUTING

    tm.mark_completed(task, result={"status": "ok"})
    assert task.execution_state == TaskState.COMPLETED


def test_rejected_task_cannot_execute():
    tm = TaskManager(OutboundRegistry())
    task = tm.create_task("send_sms", target="Rahul", parameters={}, requires_approval=True)
    tm.request_approval(task, draft="hi")
    tm.reject(task)
    assert task.execution_state == TaskState.REJECTED
    with pytest.raises(InvalidTransition):
        tm.mark_executing(task)


def test_outbound_registered_before_send_for_send_capabilities():
    registry = OutboundRegistry()
    tm = TaskManager(registry)
    task = tm.create_task("send_whatsapp_message", target="Rahul", parameters={"content": "on my way"}, requires_approval=False)
    tm.register_outbound_if_needed(task, source="whatsapp", conversation_id="rahul_chat", content_hash="abc123")
    assert registry.matches_recent_outbound("whatsapp", "rahul_chat", "abc123") is not None
