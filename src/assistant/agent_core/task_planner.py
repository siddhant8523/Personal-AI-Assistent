"""
Task Planner (Part 1, Section 7.5)
======================================
Converts a parsed Intent into a concrete Task the Task Manager can track.
"""

from __future__ import annotations

from assistant.agent_core.intent_understanding import Intent
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.execution.task import Task
from assistant.execution.task_manager import TaskManager

_INTENT_TO_CAPABILITY = {
    ("SEND_MESSAGE", "whatsapp"): "send_whatsapp_message",
    ("SEND_MESSAGE", "telegram"): "send_telegram_message",
    ("SEND_MESSAGE", "sms"): "send_sms",
    ("SEND_EMAIL", "gmail"): "send_email",
}


class TaskPlanner:
    def __init__(self, task_manager: TaskManager, policy_engine: PolicyEngine, whatsapp_connector=None):
        self._task_manager = task_manager
        self._policy = policy_engine
        self._whatsapp_connector = whatsapp_connector

    @property
    def task_manager(self) -> TaskManager:
        return self._task_manager

    def capability_for(self, intent: Intent) -> str | None:
        return _INTENT_TO_CAPABILITY.get((intent.intent, intent.platform))

    def _resolve_whatsapp_target(self, target: str) -> str:
        t = (target or "").strip()
        if not t:
            return t
        if t.endswith("@s.whatsapp.net") or t.endswith("@g.us") or t.endswith("@lid"):
            return t
        if self._whatsapp_connector:
            res_jid, err = self._whatsapp_connector.resolve_recipient(t)
            if res_jid:
                return res_jid
        return t

    def plan(self, intent: Intent) -> Task | None:
        capability = self.capability_for(intent)
        if capability is None:
            return None

        target = intent.recipient or ""
        if capability in ("send_whatsapp_message", "send_whatsapp_file") or intent.platform == "whatsapp":
            target = self._resolve_whatsapp_target(target)

        requires_approval = self._policy.requires_approval(capability)
        return self._task_manager.create_task(
            task_type=capability,
            target=target,
            parameters={"content": intent.content or ""},
            requires_approval=requires_approval,
        )

    def plan_capability(self, capability: str, target: str, parameters: dict) -> Task:
        """Create a task for a validated non-text operation (for example a
        file attachment) while preserving the same policy gate as messages."""
        if capability in ("send_whatsapp_message", "send_whatsapp_file"):
            target = self._resolve_whatsapp_target(target)
        return self._task_manager.create_task(
            task_type=capability, target=target, parameters=parameters,
            requires_approval=self._policy.requires_approval(capability),
        )
