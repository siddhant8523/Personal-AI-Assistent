"""
Tool Router (Part 3, Section 32-33)
=======================================
Resolves an abstract capability (send_message, send_file, ...) to the
concrete connector implementation. The Agent Core only ever asks for a
capability -- it never imports a connector directly.

This is also where OutboundRegistry gets populated (via TaskManager)
immediately before the actual transmit call, which is what makes the
Echo Filter fix work.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

from assistant.execution.task import Task
from assistant.execution.task_manager import TaskManager
from assistant.execution.verification import verify_connector_result

from assistant.security.capability_validator import normalize_android_capability, REV_ANDROID_CAPABILITY_ALIASES

logger = logging.getLogger("assistant.tool_router")


class ToolRouter:
    def __init__(self, task_manager: TaskManager):
        self._task_manager = task_manager
        self._capabilities: dict[str, Callable[[Task], dict]] = {}
        self.on_tool_start: Any = None
        self.on_tool_finish: Any = None

    def register(self, capability: str, handler: Callable[[Task], dict]) -> None:
        self._capabilities[capability] = handler

    def dispatch(self, task: Task, source: str, conversation_id: str) -> dict:
        handler = self._capabilities.get(task.task_type)
        if handler is None:
            norm = normalize_android_capability(task.task_type)
            handler = self._capabilities.get(norm)
        if handler is None:
            rev = REV_ANDROID_CAPABILITY_ALIASES.get(task.task_type)
            if rev:
                handler = self._capabilities.get(rev)
        if handler is None:
            self._task_manager.mark_failed(task, f"No handler registered for {task.task_type}")
            return {"status": "failed", "detail": "no handler"}

        if callable(self.on_tool_start):
            try:
                self.on_tool_start(task.task_type, task)
            except Exception as exc:
                logger.warning("on_tool_start callback failed for task %s: %s", task.task_id, exc)

        # Register the outbound send BEFORE transmitting so the Echo Filter
        # can recognize the platform's echo of this exact send later.
        content_for_hash = task.parameters.get("content", "") or task.draft or ""
        content_hash = hashlib.sha256(f"{conversation_id}|{content_for_hash.strip()}".encode()).hexdigest()[:24]
        self._task_manager.register_outbound_if_needed(task, source, conversation_id, content_hash)

        self._task_manager.mark_executing(task)
        try:
            raw_result = handler(task)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool execution failed for %s", task.task_id)
            self._task_manager.mark_failed(task, str(exc))
            raw_result = {"status": "failed", "detail": str(exc)}

        if callable(self.on_tool_finish):
            try:
                self.on_tool_finish(task.task_type, task, raw_result)
            except Exception as exc:
                logger.warning("on_tool_finish callback failed for task %s: %s", task.task_id, exc)


        verification = verify_connector_result(raw_result)

        if verification.success:
            self._task_manager.mark_completed(task, raw_result)
        else:
            self._task_manager.mark_failed(task, verification.detail)

        return raw_result
