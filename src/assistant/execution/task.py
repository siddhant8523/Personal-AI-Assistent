"""
Task model + lifecycle.
See: Part 3, Section 2-3 (Task Manager, Task Lifecycle).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskState(str, Enum):
    CREATED = "CREATED"
    PLANNED = "PLANNED"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    WAITING_FOR_DEVICE = "WAITING_FOR_DEVICE"


# Legal transitions — used to guard against invalid state jumps.
ALLOWED_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.CREATED: {TaskState.PLANNED, TaskState.FAILED},
    TaskState.PLANNED: {TaskState.WAITING_FOR_APPROVAL, TaskState.APPROVED, TaskState.FAILED},
    TaskState.WAITING_FOR_APPROVAL: {TaskState.APPROVED, TaskState.REJECTED},
    TaskState.APPROVED: {TaskState.EXECUTING, TaskState.FAILED},
    TaskState.EXECUTING: {TaskState.VERIFYING, TaskState.FAILED, TaskState.WAITING_FOR_DEVICE},
    TaskState.WAITING_FOR_DEVICE: {TaskState.EXECUTING, TaskState.FAILED},
    TaskState.VERIFYING: {TaskState.COMPLETED, TaskState.FAILED},
    TaskState.FAILED: {TaskState.EXECUTING},   # retry
    TaskState.REJECTED: set(),
    TaskState.COMPLETED: set(),
}


class InvalidTransition(Exception):
    pass


@dataclass
class Task:
    task_type: str                       # e.g. "send_whatsapp_message"
    target: str                          # e.g. resolved contact / platform id
    parameters: dict[str, Any] = field(default_factory=dict)
    origin: str = "user"                 # what request created this task
    task_id: str = field(default_factory=lambda: f"TASK-{uuid.uuid4().hex[:8]}")
    approval_state: TaskState = TaskState.CREATED
    execution_state: TaskState = TaskState.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[Any] = None
    error: Optional[str] = None
    draft: Optional[str] = None
    draft_version: int = 1
    requires_approval: bool = True

    def transition(self, new_state: TaskState) -> None:
        current = self.execution_state
        if current == new_state:
            return
        if new_state not in ALLOWED_TRANSITIONS.get(current, set()):
            raise InvalidTransition(f"{self.task_id}: {current} -> {new_state} not allowed")
        self.execution_state = new_state
        self.approval_state = new_state
        self.updated_at = time.time()


    def approval_reference(self) -> dict[str, Any]:
        """What a real approval must be bound to (Rule 6) — never a bare 'yes'."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "target": self.target,
            "draft_version": self.draft_version,
        }
