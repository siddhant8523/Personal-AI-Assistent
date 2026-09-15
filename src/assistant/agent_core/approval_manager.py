"""
Approval Manager (Part 3, Section 4-7)
==========================================
Holds tasks in WAITING_FOR_APPROVAL and only allows approval when it is
bound to an exact task_id + action + target + draft version (Rule 6 --
a generic "yes" is never sufficient on its own; the caller must reference
a task_id, which the CLI/channel prompts the user with).
"""

from __future__ import annotations

from assistant.execution.task import Task
from assistant.execution.task_manager import TaskManager


class UnknownTaskError(Exception):
    pass


class ApprovalManager:
    def __init__(self, task_manager: TaskManager):
        self._task_manager = task_manager

    def propose(self, task: Task, draft: str) -> str:
        self._task_manager.request_approval(task, draft)
        ref = task.approval_reference()
        return (
            f"Proposed action ({ref['task_id']}):\n"
            f"{task.task_type} -> {ref['target']}\n\n"
            f"Draft (v{ref['draft_version']}):\n{draft}\n\n"
            f"Reply: approve {ref['task_id']}  |  reject {ref['task_id']} (or simply 'approve' / 'yes' | 'reject' / 'no')"
        )

    def approve(self, task_id: str) -> Task:
        task = self._task_manager.get(task_id)
        if task is None:
            raise UnknownTaskError(task_id)
        self._task_manager.approve(task)
        return task

    def reject(self, task_id: str) -> Task:
        task = self._task_manager.get(task_id)
        if task is None:
            raise UnknownTaskError(task_id)
        self._task_manager.reject(task)
        return task
