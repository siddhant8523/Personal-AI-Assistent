"""
Agent Orchestrator — LangGraph-backed
=========================================
Public interface is unchanged (`__init__` takes the same arguments,
`handle_agent_command` and `handle_normal_message_summary` have the same
signatures) so nothing downstream -- agent_command_handler.py, main.py's
wiring, tests -- needs to change. Internally, `handle_agent_command` runs
the compiled LangGraph graph (agent_core/graph/) instead of a hand-written
if/elif chain.

Approval is real LangGraph human-in-the-loop, not a second orchestrator
hiding behind a node name:

  1. A "send" turn runs the graph. If the capability requires approval,
     a tool calls `interrupt(...)` (graph/tools.py). `graph.invoke(...)`
     returns with an `__interrupt__` entry instead of a final `reply`,
     and the graph is left PAUSED in the checkpointer under this
     conversation's thread_id.
  2. `handle_agent_command` turns that into an approval prompt.
  3. The next "approve TASK-x" / "reject TASK-x" turn does NOT start a
     new graph run. It resumes the SAME paused run with
     `graph.invoke(Command(resume="approve"), config=...)`, which
     continues execution inside the interrupted tool call, dispatches
     through the EXISTING ToolRouter, and only then reaches respond_node.

The LLM never decides whether to bypass approval -- the pause is a
Python-level suspension of the graph, not a prompt instruction.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from typing import Any, Callable, Iterator

from langgraph.types import Command

from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.context_builder import ContextBuilder
from assistant.agent_core.events import AgentStreamEvent
from assistant.agent_core.graph.builder import build_agent_graph
from assistant.agent_core.task_planner import TaskPlanner
from assistant.execution.tool_router import ToolRouter
from assistant.files.attachment_manager import AttachmentManager
from assistant.files.file_resolver import FileResolver
from assistant.ingestion.unified_message import Source, UnifiedMessage
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.llm.llm_client import (
    LLMClient,
    classify_llm_error,
    llm_error_to_user_message,
)
from assistant.memory.memory_service import MemoryService
from assistant.tools.web_information import WebInformationService

logger = logging.getLogger("assistant.orchestrator")

_APPROVE_WORDS = {"approve", "approves", "approved", "yes", "y", "confirm", "confirmed", "ok", "sure", "do it", "proceed", "accept"}
_REJECT_WORDS = {"reject", "rejects", "rejected", "no", "n", "deny", "denied", "cancel", "cancelled", "dont", "don't", "stop"}
_RESUME_PATTERN = re.compile(r"^(approve|reject)(?:\s+(\S+))?", re.IGNORECASE)


def _parse_approval_input(text: str) -> tuple[str | None, str | None]:
    text = text.strip()
    match = _RESUME_PATTERN.match(text)
    if match:
        action = match.group(1).lower()
        task_id = match.group(2)
        return action, task_id

    cleaned = text.lower().strip().rstrip(".!")
    if cleaned in _APPROVE_WORDS:
        return "approve", None
    if cleaned in _REJECT_WORDS:
        return "reject", None

    return None, None


class AgentOrchestrator:
    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryService,
        context_builder: ContextBuilder,
        task_planner: TaskPlanner,
        approval_manager: ApprovalManager,
        tool_router: ToolRouter,
        priority_inbox: PriorityInbox,
        telegram_connector=None,
        telegram_personal=None,
        whatsapp_connector=None,
        gmail_connector=None,
        file_resolver: FileResolver | None = None,
        attachment_manager: AttachmentManager | None = None,
        web_information: WebInformationService | None = None,
        device_gateway=None,
    ):
        self.llm = llm
        self.memory = memory
        self.context_builder = context_builder
        self.task_planner = task_planner
        self.approval_manager = approval_manager
        self.tool_router = tool_router
        self.priority_inbox = priority_inbox
        self.web_information = web_information or WebInformationService()
        self.device_gateway = device_gateway

        self._graph = build_agent_graph(
            llm=llm, memory=memory, context_builder=context_builder, task_planner=task_planner,
            approval_manager=approval_manager, tool_router=tool_router, priority_inbox=priority_inbox,
            telegram_connector=telegram_connector, telegram_personal=telegram_personal,
            whatsapp_connector=whatsapp_connector, gmail_connector=gmail_connector,
            file_resolver=file_resolver, attachment_manager=attachment_manager,
            web_information=self.web_information, device_gateway=device_gateway,
        )

    def _get_pending_task(self, conversation_id: str):
        from assistant.execution.task import TaskState
        cid = str(conversation_id or "").strip()
        if not cid:
            return None
        matching_tasks = []
        for task in self.task_planner.task_manager._tasks.values():
            if task.execution_state == TaskState.WAITING_FOR_APPROVAL:
                target_str = str(getattr(task, "target", ""))
                params = getattr(task, "parameters", {}) or {}
                conv_param = params.get("conversation_id") if isinstance(params, dict) else None
                source_conv = params.get("source_conversation_id") if isinstance(params, dict) else None
                if (target_str == cid or cid in target_str or conv_param == cid or source_conv == cid or task.task_id.endswith(cid)):
                    matching_tasks.append(task)
        if matching_tasks:
            return matching_tasks[-1]
        return None


    def handle_agent_command(self, message: UnifiedMessage) -> str:
        """Entry point for Agent Chat traffic (routed directly here by the
        Conversation Router, bypassing Message Ingestion by design --
        unchanged from before)."""
        from assistant.execution.task_manager import current_conversation_id
        token = current_conversation_id.set(message.conversation_id)
        try:
            return self._handle_agent_command_internal(message)
        finally:
            current_conversation_id.reset(token)

    def stream_agent_command(
        self,
        message: UnifiedMessage | str,
        timeout: float = 120.0,
    ) -> Iterator[AgentStreamEvent]:
        """Streaming entry point for Agent Chat traffic.

        Executes the agent graph in a dedicated, request-scoped worker thread,
        streaming real-time structured events (status, token, tool_start, tool_complete,
        approval_required, error, complete) back to the caller via an iterator.
        """
        if isinstance(message, str):
            msg = UnifiedMessage(
                source=Source.CLI,
                conversation_id="cli",
                sender="user",
                content=message,
            )
        else:
            msg = message

        from assistant.execution.task_manager import current_conversation_id

        event_queue: queue.Queue[AgentStreamEvent | object] = queue.Queue()
        sentinel = object()

        def worker() -> None:
            token = current_conversation_id.set(msg.conversation_id)
            try:
                self._handle_agent_command_internal(msg, event_callback=event_queue.put)
            except Exception as exc:
                classified = classify_llm_error(exc)
                logger.warning(
                    "[Stream Worker] unexpected failure category=%s error=%s: %s",
                    classified.category,
                    type(exc).__name__,
                    exc,
                )
                friendly_reply = llm_error_to_user_message(classified)
                event_queue.put(
                    AgentStreamEvent.error(
                        message=friendly_reply,
                        metadata={"error_type": type(exc).__name__, "category": classified.category},
                    )
                )
            finally:
                current_conversation_id.reset(token)
                event_queue.put(sentinel)

        worker_thread = threading.Thread(
            target=worker,
            name=f"stream-worker-{msg.conversation_id}",
            daemon=True,
        )
        worker_thread.start()

        try:
            while True:
                try:
                    item = event_queue.get(timeout=timeout)
                except queue.Empty:
                    yield AgentStreamEvent.error("Request timed out waiting for agent response.")
                    break
                if item is sentinel:
                    break
                if isinstance(item, AgentStreamEvent):
                    yield item
        finally:
            pass

    def resume_approval(
        self,
        conversation_id: str,
        task_id: str | None = None,
        action: str = "approve",
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> Iterator[AgentStreamEvent]:
        """Resumes a paused LangGraph approval checkpoint directly via Command(resume=action).

        Directly resumes the existing checkpoint thread without constructing a
        UnifiedMessage, without invoking the LLM to interpret the approval, and
        without routing through the normal user-message input path.
        """
        if "action" in kwargs:
            action = kwargs.pop("action")
        if "task_id" in kwargs:
            task_id = kwargs.pop("task_id")

        norm_action = str(action).strip().lower()
        norm_task_id = str(task_id).strip().lower() if task_id else ""

        # Normalize positional arguments if (conversation_id, action, task_id) was provided
        if norm_task_id in ("approve", "reject") and norm_action not in ("approve", "reject"):
            action, task_id = task_id, action
        elif norm_task_id in ("approve", "reject") and norm_action in ("approve", "reject") and "action" not in kwargs:
            action = task_id
            task_id = None

        from assistant.execution.task_manager import current_conversation_id

        event_queue: queue.Queue[AgentStreamEvent | object] = queue.Queue()
        sentinel = object()

        def worker() -> None:
            token = current_conversation_id.set(conversation_id)
            try:
                self._resume_approval_internal(
                    conversation_id=conversation_id,
                    action=action,
                    task_id=task_id,
                    event_callback=event_queue.put,
                )
            except Exception as exc:
                classified = classify_llm_error(exc)
                logger.warning(
                    "[Resume Approval Worker] error=%s: %s",
                    type(exc).__name__,
                    exc,
                )
                friendly_reply = llm_error_to_user_message(classified)
                event_queue.put(
                    AgentStreamEvent.error(
                        message=friendly_reply,
                        metadata={"error_type": type(exc).__name__},
                    )
                )
            finally:
                current_conversation_id.reset(token)
                event_queue.put(sentinel)

        worker_thread = threading.Thread(
            target=worker,
            name=f"resume-worker-{conversation_id}",
            daemon=True,
        )
        worker_thread.start()

        try:
            while True:
                try:
                    item = event_queue.get(timeout=timeout)
                except queue.Empty:
                    yield AgentStreamEvent.error("Request timed out waiting for approval resumption.")
                    break
                if item is sentinel:
                    break
                if isinstance(item, AgentStreamEvent):
                    yield item
        finally:
            pass

    def _resume_approval_internal(
        self,
        conversation_id: str,
        action: str,
        task_id: str | None = None,
        event_callback: Callable[[AgentStreamEvent], None] | None = None,
    ) -> str:
        """Internal worker executing direct LangGraph checkpoint resumption."""
        config: dict[str, Any] = {"configurable": {"thread_id": conversation_id}}
        if event_callback:
            config["configurable"]["event_callback"] = event_callback

        norm_action = str(action).strip().lower()
        if norm_action not in ("approve", "reject"):
            norm_action = "approve" if "approve" in norm_action else "reject"

        # Check existing checkpoint snapshot
        snapshot = self._graph.get_state(config)
        is_paused = bool(snapshot.next)

        # Duplicate protection via task execution state
        from assistant.execution.task import TaskState
        existing_task = self.task_planner.task_manager.get(task_id) if task_id else None
        if existing_task and existing_task.execution_state in (
            TaskState.APPROVED,
            TaskState.REJECTED,
            TaskState.COMPLETED,
            TaskState.FAILED,
        ):
            state_desc = existing_task.execution_state.value.lower()
            err_msg = f"Task {task_id} has already been {state_desc}."
            if event_callback:
                event_callback(AgentStreamEvent.error(message=err_msg))
            return err_msg

        if is_paused:
            pending_id = self._pending_task_id(snapshot)
            if task_id and pending_id and pending_id != task_id:
                err_msg = (
                    f"Pending approval task is {pending_id}, which does not match requested task {task_id}."
                )
                if event_callback:
                    event_callback(AgentStreamEvent.error(message=err_msg))
                return err_msg

            # Directly resume the LangGraph checkpoint via Command(resume=norm_action)
            # NO UnifiedMessage is created. NO LLM call occurs.
            result = self._graph.invoke(Command(resume=norm_action), config=config)

            # Clear pause state from checkpointer storage if execution completed
            new_snapshot = self._graph.get_state(config)
            if not new_snapshot.next and hasattr(self._graph.checkpointer, "storage"):
                self._graph.checkpointer.storage.pop(conversation_id, None)

            reply = self._extract_reply(result, None)
            if event_callback:
                interrupts = result.get("__interrupt__")
                if interrupts:
                    payload = interrupts[0].value
                    event_callback(
                        AgentStreamEvent.approval_required(
                            task_id=str(payload.get("task_id", "")),
                            task_type=str(payload.get("task_type", "")),
                            target=str(payload.get("target", "")),
                            draft=str(payload.get("draft", "")),
                            message=reply,
                        )
                    )
                elif not result.get("error"):
                    if reply:
                        event_callback(AgentStreamEvent.token(content=reply))
                    event_callback(AgentStreamEvent.complete(final_response=reply))
            return reply
        else:
            # Checkpoint not paused - check if pending task exists in TaskManager (fallback path)
            pending_task = self._get_pending_task(conversation_id)
            if pending_task and (not task_id or pending_task.task_id == task_id):
                if norm_action == "approve":
                    approved = self.approval_manager.approve(pending_task.task_id)
                    res = self.tool_router.dispatch(approved, source="cli", conversation_id=conversation_id)
                    status = res.get("status") if isinstance(res, dict) else None
                    detail = res.get("detail") if isinstance(res, dict) else str(res)
                    if status in ("ok", "success", "sent"):
                        reply = f"Approved and executed {pending_task.task_id}: {detail or 'sent'}"
                    else:
                        reply = f"Approved, but failed to execute {pending_task.task_id}: {detail or 'Execution failed'}"
                else:
                    self.approval_manager.reject(pending_task.task_id)
                    reply = f"Rejected {pending_task.task_id}."
                if event_callback:
                    if reply:
                        event_callback(AgentStreamEvent.token(content=reply))
                    event_callback(AgentStreamEvent.complete(final_response=reply))
                return reply
            else:
                err_msg = (
                    f"No pending approval checkpoint found for conversation '{conversation_id}'."
                    if not task_id
                    else f"I don't have a pending approval checkpoint for task '{task_id}' in conversation '{conversation_id}'."
                )
                if event_callback:
                    event_callback(AgentStreamEvent.error(message=err_msg))
                return err_msg

    def _handle_agent_command_internal(
        self,
        message: UnifiedMessage,
        event_callback: Callable[[AgentStreamEvent], None] | None = None,
    ) -> str:
        config: dict[str, Any] = {"configurable": {"thread_id": message.conversation_id}}
        if event_callback:
            config["configurable"]["event_callback"] = event_callback
        text = message.content.strip()

        if text.lower() in ("/clear", "clear"):
            if hasattr(self._graph.checkpointer, "storage"):
                self._graph.checkpointer.storage.pop(message.conversation_id, None)
            from assistant.execution.task import TaskState
            for t in list(self.task_planner.task_manager._tasks.values()):
                if t.execution_state == TaskState.WAITING_FOR_APPROVAL:
                    self.approval_manager.reject(t.task_id)
            reply = "Conversational state cleared."
            if event_callback:
                event_callback(AgentStreamEvent.complete(final_response=reply))
            return reply

        action, task_id = _parse_approval_input(text)

        snapshot = self._graph.get_state(config)
        is_paused = bool(snapshot.next)
        pending_task = self._get_pending_task(message.conversation_id)

        if action:
            if is_paused:
                pending_id = self._pending_task_id(snapshot)
                if task_id and pending_id and pending_id != task_id:
                    reply = (
                        f"I have a different task ({pending_id}) awaiting your approval. "
                        f"Reply 'approve {pending_id}' or 'reject {pending_id}'."
                    )
                    if event_callback:
                        event_callback(AgentStreamEvent.complete(final_response=reply))
                    return reply
                result = self._graph.invoke(Command(resume=action), config=config)
                # Clear pause state from checkpointer storage if execution completed
                new_snapshot = self._graph.get_state(config)
                if not new_snapshot.next and hasattr(self._graph.checkpointer, "storage"):
                    self._graph.checkpointer.storage.pop(message.conversation_id, None)
                reply = self._extract_reply(result, message)
                if event_callback:
                    interrupts = result.get("__interrupt__")
                    if interrupts:
                        payload = interrupts[0].value
                        event_callback(
                            AgentStreamEvent.approval_required(
                                task_id=str(payload.get("task_id", "")),
                                task_type=str(payload.get("task_type", "")),
                                target=str(payload.get("target", "")),
                                draft=str(payload.get("draft", "")),
                                message=reply,
                            )
                        )
                    elif not result.get("error"):
                        event_callback(AgentStreamEvent.complete(final_response=reply))
                return reply
            elif pending_task:
                if task_id and pending_task.task_id != task_id:
                    reply = f"I don't have a pending task with id {task_id} in this chat."
                    if event_callback:
                        event_callback(AgentStreamEvent.complete(final_response=reply))
                    return reply
                if action == "approve":
                    approved = self.approval_manager.approve(pending_task.task_id)
                    res = self.tool_router.dispatch(approved, source=message.source.name.lower(), conversation_id=message.conversation_id)
                    status = res.get("status") if isinstance(res, dict) else None
                    detail = res.get("detail") if isinstance(res, dict) else str(res)
                    if status in ("ok", "success", "sent"):
                        reply = f"Approved and executed {pending_task.task_id}: {detail or 'sent'}"
                    else:
                        reply = f"Approved, but failed to execute {pending_task.task_id}: {detail or 'Execution failed'}"
                else:
                    self.approval_manager.reject(pending_task.task_id)
                    reply = f"Rejected {pending_task.task_id}."
                if event_callback:
                    event_callback(AgentStreamEvent.complete(final_response=reply))
                return reply
            else:
                if task_id:
                    reply = f"I don't have a pending task with id {task_id} in this chat."
                else:
                    reply = "There are no pending actions awaiting approval in this chat."
                if event_callback:
                    event_callback(AgentStreamEvent.complete(final_response=reply))
                return reply

        if is_paused:
            result = self._graph.invoke(Command(resume={"action": "pass", "text": text}), config=config)
            new_snapshot = self._graph.get_state(config)
            if not new_snapshot.next and hasattr(self._graph.checkpointer, "storage"):
                self._graph.checkpointer.storage.pop(message.conversation_id, None)
            reply = self._extract_reply(result, message)
            if event_callback:
                interrupts = result.get("__interrupt__")
                if interrupts:
                    payload = interrupts[0].value
                    event_callback(
                        AgentStreamEvent.approval_required(
                            task_id=str(payload.get("task_id", "")),
                            task_type=str(payload.get("task_type", "")),
                            target=str(payload.get("target", "")),
                            draft=str(payload.get("draft", "")),
                            message=reply,
                        )
                    )
                elif not result.get("error"):
                    event_callback(AgentStreamEvent.complete(final_response=reply))
            return reply

        result = self._graph.invoke(
            {"conversation_id": message.conversation_id, "input_text": text}, config=config,
        )
        reply = self._extract_reply(result, message)
        if event_callback:
            interrupts = result.get("__interrupt__")
            if interrupts:
                payload = interrupts[0].value
                event_callback(
                    AgentStreamEvent.approval_required(
                        task_id=str(payload.get("task_id", "")),
                        task_type=str(payload.get("task_type", "")),
                        target=str(payload.get("target", "")),
                        draft=str(payload.get("draft", "")),
                        message=reply,
                    )
                )
            elif not result.get("error"):
                event_callback(AgentStreamEvent.complete(final_response=reply))
        return reply

    def handle_normal_message_summary(self, conversation_id: str, level: str, sender: str, content: str) -> None:
        """Called by the Normal Chat / Message Intelligence pipeline for
        HIGH priority messages. Unchanged: this is a passive background
        summarization call, not an agent action, so it doesn't go through
        the graph -- same as before."""
        if not self.llm.online:
            return
        self.llm.generate(
            system="Summarize this message in one short sentence for a priority inbox.",
            user_message=f"From: {sender}\n{content}",
            max_tokens=80,
        )

    @staticmethod
    def _pending_task_id(snapshot) -> str | None:
        for task in getattr(snapshot, "tasks", ()) or ():
            for i in getattr(task, "interrupts", None) or ():
                payload = i.value
                if isinstance(payload, dict) and "task_id" in payload:
                    return payload["task_id"]
        return None

    @staticmethod
    def _get_channel_name(message: UnifiedMessage | None) -> str:
        if not message:
            return "CLI"
        cid = message.conversation_id or ""
        src = message.source
        if cid.startswith("telegram_bot:") or src == Source.TELEGRAM:
            return "Telegram Bot"
        if cid.startswith("whatsapp:") or src == Source.WHATSAPP:
            return "WhatsApp Bot"
        if cid.startswith("sms:") or src == Source.SMS:
            return "SMS"
        if cid.startswith("gmail:") or src == Source.GMAIL:
            return "Gmail"
        return "CLI"

    @classmethod
    def _extract_reply(cls, result: dict, message: UnifiedMessage | None = None) -> str:
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value
            task_type = payload.get("task_type", "")
            target = payload.get("target", "")
            draft = payload.get("draft", "")
            channel_name = cls._get_channel_name(message)
            return (
                f"Proposed action via {channel_name}:\n"
                f"{task_type} → {target}\n\n"
                f"Draft:\n{draft}\n\n"
                f"Approve or reject?"
            )
        return result.get("reply") or "(no response generated)"

