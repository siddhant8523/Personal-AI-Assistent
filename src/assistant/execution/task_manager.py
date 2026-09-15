"""
Task Manager (Part 3, Section 2-6)
======================================
Converts an approved plan into an executable task, tracks lifecycle state,
and is where the Outbound Registry gets its entry written -- BEFORE the
actual send -- for every outbound capability.
"""

from __future__ import annotations

import json
import time
from typing import Any
from contextvars import ContextVar


from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task import Task, TaskState
from assistant.storage.db import get_connection


OUTBOUND_CAPABILITIES = {
    "send_whatsapp_message", "send_telegram_message", "send_sms", "send_email", "send_file",
    "send_whatsapp_file", "send_telegram_file", "send_gmail_attachment",
}


current_conversation_id: ContextVar[str] = ContextVar("current_conversation_id", default="")


class TaskManager:
    def __init__(self, outbound_registry: OutboundRegistry):
        self._outbound_registry = outbound_registry
        self._tasks: dict[str, Task] = {}

    def create_task(
        self,
        task_type: str,
        target: str,
        parameters: dict[str, Any] | None = None,
        requires_approval: bool = True,
    ) -> Task:
        parameters = dict(parameters or {})
        cid = current_conversation_id.get()
        if cid and "conversation_id" not in parameters:
            parameters["conversation_id"] = cid

        task = Task(task_type=task_type, target=target, parameters=parameters, requires_approval=requires_approval)
        task.transition(TaskState.PLANNED)
        self._tasks[task.task_id] = task
        self._persist(task)
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def find_waiting_approval(self, task_type: str, target: str, draft: str) -> Task | None:
        """Used by the LangGraph tool layer (graph/tools.py) to make
        `interrupt()`-based tools idempotent: LangGraph replays a node's
        code from the top when resuming a pending interrupt, so the
        tool would otherwise call TaskPlanner.plan() again on every
        approve/reject turn and mint a second task. Looking up an
        already-pending task with the same shape lets the tool reuse it
        instead of creating a duplicate."""
        for task in reversed(list(self._tasks.values())):
            if (
                task.task_type == task_type
                and task.target == target
                and task.draft == draft
                and task.execution_state == TaskState.WAITING_FOR_APPROVAL
            ):
                return task
        return None

    def request_approval(self, task: Task, draft: str) -> None:
        task.draft = draft
        task.transition(TaskState.WAITING_FOR_APPROVAL)
        self._persist(task)

    def approve(self, task: Task) -> None:
        task.transition(TaskState.APPROVED)
        self._persist(task)

    def reject(self, task: Task) -> None:
        task.transition(TaskState.REJECTED)
        self._persist(task)

    def register_outbound_if_needed(self, task: Task, source: str, conversation_id: str, content_hash: str) -> None:
        """Called by the Tool Router right before it actually transmits an
        outbound message -- this is what lets the Echo Filter recognize the
        platform's own echo of this send later."""
        if task.task_type in OUTBOUND_CAPABILITIES:
            self._outbound_registry.register_pending(task.task_id, source, conversation_id, content_hash)

    def mark_executing(self, task: Task) -> None:
        task.transition(TaskState.EXECUTING)
        self._persist(task)

    def mark_completed(self, task: Task, result) -> None:
        task.result = result
        task.transition(TaskState.VERIFYING)
        task.transition(TaskState.COMPLETED)
        self._persist(task)

    def mark_failed(self, task: Task, error: str) -> None:
        task.error = error
        if task.execution_state != TaskState.FAILED:
            task.transition(TaskState.FAILED)
        self._persist(task)


    def _persist(self, task: Task) -> None:
        conn = get_connection()
        conn.execute(
            "INSERT INTO tasks (task_id, task_type, target, parameters, state, draft, created_at, updated_at, result, error) "
            "VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(task_id) DO UPDATE SET state=excluded.state, draft=excluded.draft, "
            "updated_at=excluded.updated_at, result=excluded.result, error=excluded.error",
            (
                task.task_id, task.task_type, task.target, json.dumps(task.parameters),
                task.execution_state.value, task.draft, task.created_at, task.updated_at,
                json.dumps(task.result) if task.result is not None else None, task.error,
            ),
        )
        conn.commit()
