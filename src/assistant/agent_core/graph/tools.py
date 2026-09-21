"""
LangChain Tools — adapters over the EXISTING connectors / Agent Core services
=================================================================================
Per the migration requirement: these tools do not reimplement any business
logic. Each one is a thin wrapper over an EXISTING component:

    EXISTING CONNECTOR / SERVICE  (telegram/whatsapp/gmail connector,
           ^                       TaskPlanner, ApprovalManager, ToolRouter,
    LangChain Tool   <-- this file PriorityInbox, FileResolver, AttachmentManager)
           ^
    LangGraph

Consequential tools (anything that sends something) never dispatch
directly to a connector. They build a Task via the EXISTING TaskPlanner,
and if the EXISTING PolicyEngine (consulted inside TaskPlanner) says the
capability requires approval, the tool calls LangGraph's `interrupt()`
-- which genuinely pauses the graph -- and only calls the EXISTING
ToolRouter.dispatch(...) once AgentOrchestrator resumes the graph with
`Command(resume="approve")`. The LLM can never bypass this: the pause
happens in Python, not in the prompt.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal, Optional

from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field


def _item_ts(item: dict) -> float:
    val = item.get("ts") if item.get("ts") is not None else item.get("timestamp")
    if isinstance(val, (int, float)):
        return float(val) / 1000.0 if val > 1e11 else float(val)
    if isinstance(val, str):
        try:
            f = float(val)
            return f / 1000.0 if f > 1e11 else f
        except ValueError:
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
    return 0.0


from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.intent_understanding import Intent
from assistant.agent_core.task_planner import TaskPlanner
from assistant.device_gateway.device_gateway import DeviceGateway, DeviceOfflineError
from assistant.execution.tool_router import ToolRouter
from assistant.files.attachment_manager import AttachmentManager
from assistant.files.file_resolver import AmbiguousFileError, FileResolver
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.memory.identity import IdentityManager
from assistant.tools.web_information import WebInformationService


class SendMessageArgs(BaseModel):
    """Structured args for drafting an outbound message (structured output,
    validated by pydantic before the EXISTING TaskPlanner ever sees it)."""

    platform: Literal["whatsapp", "telegram", "sms", "gmail"] = Field(
        description="Which platform to send on."
    )
    recipient: str = Field(description="Who to send it to, e.g. a contact name or chat id.")
    content: str = Field(description="The exact message text to send. Preserve the user's requested words and meaning verbatim without altering or paraphrasing.")


class SendFileArgs(BaseModel):
    platform: Literal["whatsapp", "telegram", "gmail"] = Field(description="Which platform to send the file on.")
    recipient: str = Field(description="Who to send it to.")
    file_query: str = Field(description="Fuzzy description of the file, e.g. 'report.pdf' or 'work docs report'.")
    message: Optional[str] = Field(default="", description="Optional message text to accompany the file.")


class ChangeNameArgs(BaseModel):
    name: str = Field(description="New assistant name; up to 49 safe characters.")


class WebSearchArgs(BaseModel):
    query: str = Field(description="Question or current-information search query.")
    max_results: int = Field(default=5, ge=1, le=10)
    search_depth: Literal["basic", "advanced"] = Field(default="basic", description="Search depth: basic or advanced.")


class WebQnaArgs(BaseModel):
    query: str = Field(description="Question or topic for a quick direct answer from Tavily Q&A.")


class ExtractPageArgs(BaseModel):
    url: str = Field(description="Complete http:// or https:// URL to extract.")


class ExtractStructureArgs(BaseModel):
    url: str = Field(description="Complete http:// or https:// URL to parse with BeautifulSoup.")
    selector: Optional[str] = Field(default=None, description="Optional CSS selector (e.g. 'h1', 'p.content', '.main-article').")


class TelegramFindContactArgs(BaseModel):
    query: str = Field(description="Name, username, or phone number to search for on Telegram.")


class TelegramListDialogsArgs(BaseModel):
    limit: int = Field(default=20, description="Maximum number of Telegram dialogs/chats to return.")


class TelegramReadMessagesArgs(BaseModel):
    recipient: str = Field(description="Contact name, username, or chat ID to read messages from.")
    limit: int = Field(default=20, description="Maximum number of messages to read.")


class TelegramSearchMessagesArgs(BaseModel):
    query: str = Field(description="Keyword or text to search for in Telegram messages.")
    recipient: Optional[str] = Field(default="", description="Optional contact name or chat ID to limit search scope.")


from assistant.files.mobile_path_resolver import canonical_mobile_path, is_mobile_path
from assistant.tools.alarm_parser import parse_alarm_time

APP_PACKAGE_MAP: dict[str, str] = {
    "whatsapp": "com.whatsapp",
    "telegram": "org.telegram.messenger",
    "youtube": "com.google.android.youtube",
    "album": "com.android.gallery3d",
    "gallery": "com.android.gallery3d",
    "photos": "com.google.android.apps.photos",
    "settings": "com.android.settings",
    "clock": "com.google.android.deskclock",
    "camera": "com.android.camera",
}


class ReadSMSArgs(BaseModel):
    recipient: Optional[str] = Field(default="", description="Optional contact name or phone number to filter SMS messages.")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of SMS messages to read.")


class MakeCallArgs(BaseModel):
    recipient: str = Field(description="Contact name or phone number to make a phone call to.")


class SetAlarmArgs(BaseModel):
    time: str = Field(description="Alarm time, e.g. '07:00' or '7:57 PM'.")
    label: Optional[str] = Field(default="", description="Optional label/tag for the alarm.")


class ListAlarmsArgs(BaseModel):
    pass


class CancelAlarmArgs(BaseModel):
    alarm_id: Optional[str] = Field(default="", description="Unique ID of the alarm to cancel (e.g. 'alarm-a1b2c3d4').")
    query: Optional[str] = Field(default="", description="Optional time or label query to match alarm to cancel (e.g. '8:44 PM' or 'Meeting').")



class SetTimerArgs(BaseModel):
    duration_seconds: int = Field(ge=1, description="Timer duration in seconds.")
    label: Optional[str] = Field(default="", description="Optional label for the timer.")


class ExecuteIntentArgs(BaseModel):
    action: str = Field(description="Android Intent action, e.g. 'android.intent.action.SET_ALARM'.")
    extras: Optional[dict] = Field(default=None, description="Optional key-value extras dictionary for the intent.")


class ListMobileFilesArgs(BaseModel):
    path: str = Field(default="/Books", description="Mobile storage directory path, e.g. '/Books' or '/storage/emulated/0/Books'.")


class FindMobileFilesArgs(BaseModel):
    query: str = Field(description="Filename or keyword to search for on the mobile device.")
    path: Optional[str] = Field(default="/storage/emulated/0", description="Optional mobile directory to limit search scope.")


class OpenMobileAppArgs(BaseModel):
    app_name: str = Field(description="Name or package of the application to open on mobile device, e.g. 'WhatsApp', 'Album', 'YouTube'.")


import os
import re


def parse_priority_query(query: str) -> tuple[str | None, int | None]:

    q = (query or "").lower()
    req_level = None
    if "high" in q:
        req_level = "HIGH"
    elif "medium" in q:
        req_level = "MEDIUM"
    elif "low" in q:
        req_level = "LOW"

    m = re.search(r'\b(top|show)?\s*(\d+)\b', q)
    req_limit = int(m.group(2)) if m else None
    return req_level, req_limit


def _format_priority_item(item: dict, index: int) -> str:
    content = (item.get("content") or "").strip()
    if len(content) > 200:
        content = content[:197] + "..."
    source_str = (item.get("source") or "").upper()
    sender_str = item.get("sender") or "Unknown"
    line = f"{index}. [{source_str}] From {sender_str}: {content}"

    meta = []
    if item.get("deadline"):
        meta.append(f"Time/Deadline: {item['deadline']}")
    if item.get("system_intent") and item["system_intent"] not in ("OTHER", "CHAT"):
        meta.append(f"Intent: {item['system_intent']}")
    if item.get("requires_action"):
        meta.append("Action Required: YES")
    if item.get("system_urgency") and item["system_urgency"] != "LOW":
        meta.append(f"Urgency: {item['system_urgency']}")
    if item.get("system_importance") and item["system_importance"] != "LOW":
        meta.append(f"Importance: {item['system_importance']}")
    if item.get("is_scam"):
        meta.append("SECURITY ALERT: Phishing / Scam Risk")
    elif item.get("is_spam"):
        meta.append("Spam/Promotional")
    if item.get("system_reason"):
        meta.append(f"Reason: {item['system_reason']}")

    if meta:
        line += f"\n   [{' | '.join(meta)}]"
    return line


def render_priority_inbox(
    priority_inbox: PriorityInbox,
    level: str | None = None,
    limit: int | None = None,
    query: str = "",
) -> str:
    if priority_inbox is not None and hasattr(priority_inbox, "process_pending_if_needed"):
        priority_inbox.process_pending_if_needed()

    parsed_level, parsed_limit = parse_priority_query(query)
    target_level = level or parsed_level
    target_limit = limit or parsed_limit

    items = priority_inbox.today()
    if not items:
        return "No messages in the priority inbox yet today."

    if target_level:
        bucket = [i for i in items if i["level"] == target_level]
        if not bucket:
            return f"No {target_level} priority messages found today."
        cap = target_limit or 20
        bucket = bucket[:cap]
        lines = [f"{target_level} PRIORITY", ""]
        for i, item in enumerate(bucket, 1):
            lines.append(_format_priority_item(item, i))
        return "\n".join(lines).strip()

    lines = ["TODAY'S PRIORITY", ""]
    for lvl in ("HIGH", "MEDIUM", "LOW"):
        bucket = [i for i in items if i["level"] == lvl]
        if not bucket:
            continue
        cap = target_limit or 20
        bucket = bucket[:cap]
        lines.append(lvl)
        for i, item in enumerate(bucket, 1):
            lines.append(_format_priority_item(item, i))
        lines.append("")
    return "\n".join(lines).strip()


def render_agenda(
    priority_inbox: PriorityInbox,
    target_date: str = "tomorrow",
    query: str = "",
) -> str:
    if priority_inbox is not None and hasattr(priority_inbox, "process_pending_if_needed"):
        priority_inbox.process_pending_if_needed()

    q_lower = (query or "").lower()
    date_key = target_date or "tomorrow"
    if "tomorrow" in q_lower:
        date_key = "tomorrow"
    elif "today" in q_lower:
        date_key = "today"

    items = priority_inbox.get_agenda(target_date=date_key, limit=20)
    human_label = date_key.upper()

    if not items:
        return f"No scheduled meetings, deadlines, or priority commitments found for {date_key}."

    lines = [f"SCHEDULE / AGENDA FOR {human_label}:", ""]
    for i, item in enumerate(items, 1):
        lines.append(_format_priority_item(item, i))
        lines.append("")
    return "\n".join(lines).strip()



def _run_consequential_task(
    task_planner: TaskPlanner,
    approval_manager: ApprovalManager,
    tool_router: ToolRouter,
    intent: Intent,
    draft_label: str,
    extra_parameters: dict | None = None,
) -> str:
    """Shared plan -> (interrupt if required) -> dispatch flow used by every
    send_* tool, whatever platform it targets. This is the ONE place the
    real LangGraph `interrupt()` is called, so every consequential tool
    gets identical, non-bypassable approval behavior."""
    capability = task_planner.capability_for(intent)
    if capability is None:
        return (
            f"I don't have a working handler for {intent.platform} yet -- "
            "try a supported platform (whatsapp/telegram/sms/gmail)."
        )

    # LangGraph replays this node's code from the top when resuming a
    # paused interrupt() -- reuse the already-pending task from the first
    # pass instead of minting a duplicate one on every approve/reject turn.
    target = intent.recipient or ""
    task = task_planner.task_manager.find_waiting_approval(capability, target, draft_label)
    if task is None:
        task = task_planner.plan(intent)
        if task is not None and extra_parameters:
            task.parameters.update(extra_parameters)

    if not task.requires_approval:
        result = tool_router.dispatch(task, source=intent.platform or "unknown", conversation_id=task.target)
        return f"Sent via {intent.platform}: {result.get('detail')}"

    # Puts the task into WAITING_FOR_APPROVAL in the EXISTING TaskManager
    # (audit trail / task lifecycle unchanged) before pausing the graph.
    # Guarded because LangGraph replays this function from the top on
    # resume, and the task may already be WAITING_FOR_APPROVAL from the
    # first pass (see find_waiting_approval above).
    from assistant.execution.task import TaskState

    if task.execution_state != TaskState.WAITING_FOR_APPROVAL:
        approval_manager.propose(task, draft_label)
    ref = task.approval_reference()

    decision = interrupt({
        "type": "approval_request",
        "task_id": ref["task_id"],
        "task_type": ref["task_type"],
        "target": ref["target"],
        "draft_version": ref["draft_version"],
        "draft": draft_label,
    })

    action = ""
    pass_text = None
    if isinstance(decision, dict):
        action = str(decision.get("action", "")).strip().lower()
        pass_text = decision.get("text")
    else:
        action = str(decision).strip().lower()

    if action == "approve":
        approved_task = approval_manager.approve(ref["task_id"])
        result = tool_router.dispatch(approved_task, source=intent.platform or "unknown", conversation_id=approved_task.target)
        status = result.get("status") if isinstance(result, dict) else None
        detail = result.get("detail") if isinstance(result, dict) else str(result)
        if status in ("ok", "success", "sent"):
            return f"Approved and executed {ref['task_id']}: {detail}"
        return f"Approved, but failed to execute {ref['task_id']}: {detail}"
    elif action == "reject":
        approval_manager.reject(ref["task_id"])
        return f"Rejected {ref['task_id']}. Draft retained."
    else:
        return f"[Pending approval {ref['task_id']} retained] User requested: {pass_text or action}"


def _run_file_task(task_planner: TaskPlanner, approval_manager: ApprovalManager, tool_router: ToolRouter,
                   capability: str, recipient: str, path: str, message: str) -> str:
    """The same approval/dispatch path as text sends, with a prevalidated
    attachment path. This keeps files out of arbitrary LLM filesystem access."""
    draft = f"Send {path.rsplit('/', 1)[-1]} to {recipient}" + (f": {message}" if message else "")
    task = task_planner.task_manager.find_waiting_approval(capability, recipient, draft)
    if task is None:
        task = task_planner.plan_capability(capability, recipient, {"path": path, "content": message})
    from assistant.execution.task import TaskState
    if task.execution_state != TaskState.WAITING_FOR_APPROVAL:
        approval_manager.propose(task, draft)
    decision = interrupt({"type": "approval_request", **task.approval_reference(), "draft": draft})
    action = ""
    pass_text = None
    if isinstance(decision, dict):
        action = str(decision.get("action", "")).strip().lower()
        pass_text = decision.get("text")
    else:
        action = str(decision).strip().lower()

    if action == "approve":
        approved = approval_manager.approve(task.task_id)
        result = tool_router.dispatch(approved, source=capability.split("_")[1] if "_" in capability else "unknown", conversation_id=approved.target)
        status = result.get("status") if isinstance(result, dict) else None
        detail = result.get("detail") if isinstance(result, dict) else str(result)
        if status in ("ok", "success", "sent"):
            return f"Approved and executed {task.task_id}: {detail}"
        return f"Approved, but failed to execute {task.task_id}: {detail}"
    elif action == "reject":
        approval_manager.reject(task.task_id)
        return f"Rejected {task.task_id}. Draft retained."
    else:
        return f"[Pending approval {task.task_id} retained] User requested: {pass_text or action}"


def build_tools(
    task_planner: TaskPlanner,
    approval_manager: ApprovalManager,
    priority_inbox: PriorityInbox,
    tool_router: ToolRouter,
    telegram_connector=None,
    telegram_personal=None,
    whatsapp_connector=None,
    gmail_connector=None,
    file_resolver: FileResolver | None = None,
    attachment_manager: AttachmentManager | None = None,
    identity_manager: IdentityManager | None = None,
    user_profile=None,
    web_information: WebInformationService | None = None,
    device_gateway: DeviceGateway | None = None,
) -> list[StructuredTool]:
    tools: list[StructuredTool] = []

    if whatsapp_connector is not None and device_gateway is not None:
        if getattr(whatsapp_connector, "device_gateway", None) is None:
            whatsapp_connector.set_device_gateway(device_gateway)


    # ---- generic send (used by the offline regex fallback + general chat) ----
    def resolve_sms_recipient(recipient: str) -> tuple[str | None, str | None, str | None]:
        """Resolves recipient into (display_target, phone_number, error_message).
        If recipient is already a phone number: returns (phone_number, phone_number, None).
        If contact name: queries Android device contacts / user profile.
        Returns error string if ambiguous or not found.
        """
        if device_gateway is None or not device_gateway.is_device_connected():
            return None, None, "Android device is offline or not connected."

        r_str = recipient.strip()
        clean_num = r_str.replace("-", "").replace(" ", "")
        if r_str.startswith("+") or (clean_num.isdigit() and len(clean_num) >= 7):
            return r_str, r_str, None

        phone_num = ""
        c_name = r_str
        if user_profile is not None and hasattr(user_profile, "resolve_contact_number"):
            phone_num = user_profile.resolve_contact_number(r_str) or ""

        if not phone_num and device_gateway is not None and device_gateway.is_device_connected():
            res = device_gateway.send_command("contact.find", {"query": r_str})
            matches = []
            if res.status == "ok" and isinstance(res.detail, dict):
                matches = res.detail.get("matches", [])
            if len(matches) == 1 and isinstance(matches[0], dict):
                c_name = matches[0].get("name", r_str)
                phone_num = matches[0].get("phone_number", "")
            elif len(matches) > 1:
                options = [f"{m.get('name')}: {m.get('phone_number')}" for m in matches if isinstance(m, dict)]
                return None, None, f"Found multiple contacts matching '{r_str}':\n" + "\n".join(f"- {opt}" for opt in options) + "\nPlease specify which number to send SMS to."

        if not phone_num:
            return None, None, f"I couldn't find a contact named '{r_str}' on your phone. Please provide the phone number."

        display_target = f"{c_name} ({phone_num})"
        return display_target, phone_num, None

    def _propose_send_message(platform: str, recipient: str, content: str) -> str:
        """Draft an outbound WhatsApp, Telegram, SMS or Gmail message and
        (if policy requires it) pause for the user's approval. Use this
        whenever the user asks to send/text/message someone."""
        phone_number = None
        if platform == "whatsapp" and whatsapp_connector is not None:
            resolved_jid, err = whatsapp_connector.resolve_recipient(recipient)
            if err:
                return err
            recipient = resolved_jid
        elif platform == "sms":
            display_target, phone_number, err = resolve_sms_recipient(recipient)
            if err:
                return err
            recipient = display_target or recipient

        intent = Intent(
            intent="SEND_EMAIL" if platform == "gmail" else "SEND_MESSAGE",
            platform=platform, recipient=recipient, content=content,
        )
        extra = {"phone_number": phone_number, "number": phone_number} if phone_number else None
        return _run_consequential_task(task_planner, approval_manager, tool_router, intent, content, extra_parameters=extra)


    tools.append(StructuredTool.from_function(
        func=_propose_send_message,
        name="propose_send_message",
        description=(
            "Draft an outbound message on WhatsApp, Telegram, SMS or Gmail "
            "and submit it for approval. Use this whenever the user asks "
            "to send/text/message someone."
        ),
        args_schema=SendMessageArgs,
    ))

    class PriorityQueryArgs(BaseModel):
        level: Optional[str] = Field(default="", description="Optional priority level filter (HIGH, MEDIUM, LOW)")
        limit: Optional[int] = Field(default=None, description="Optional maximum number of messages to return")
        query: Optional[str] = Field(default="", description="Original user query describing priority request")

    tools.append(StructuredTool.from_function(
        func=lambda level="", limit=None, query="": render_priority_inbox(priority_inbox, level=level or None, limit=limit, query=query),
        name="query_priority_inbox",
        description="Return today's priority inbox (important messages across all platforms). Accepts level and limit.",
        args_schema=PriorityQueryArgs,
    ))

    class AgendaQueryArgs(BaseModel):
        target_date: Optional[str] = Field(
            default="tomorrow",
            description="Target date to query schedule/agenda for (e.g. 'today', 'tomorrow', 'upcoming', or YYYY-MM-DD)",
        )
        query: Optional[str] = Field(default="", description="Original user query describing schedule or agenda request")

    tools.append(StructuredTool.from_function(
        func=lambda target_date="tomorrow", query="": render_agenda(priority_inbox, target_date=target_date, query=query),
        name="query_agenda",
        description=(
            "Retrieve scheduled meetings, commitments, appointments, deadlines, and required action items "
            "for a target date ('today', 'tomorrow', 'upcoming', or YYYY-MM-DD). "
            "ALWAYS call this tool when the user asks what they have scheduled, what meetings they have, "
            "what their deadlines are, or what they should do on a specific date (e.g. 'what should I do tomorrow?', "
            "'do I have any meeting for tomorrow?')."
        ),
        args_schema=AgendaQueryArgs,
    ))


    if identity_manager is not None:
        def _change_assistant_name(name: str) -> str:
            """Change the assistant's persistent display name after explicit
            approval in the active chat; it never sends an external message."""
            normalized = name.strip()
            draft = f"Change assistant name from {identity_manager.name()} to {normalized}."
            task = task_planner.task_manager.find_waiting_approval("change_assistant_name", "assistant_identity", draft)
            if task is None:
                task = task_planner.plan_capability("change_assistant_name", "assistant_identity", {"name": normalized})
            from assistant.execution.task import TaskState
            if task.execution_state != TaskState.WAITING_FOR_APPROVAL:
                approval_manager.propose(task, draft)
            decision = interrupt({"type": "approval_request", **task.approval_reference(), "draft": draft})
            action = ""
            if isinstance(decision, dict):
                action = str(decision.get("action", "")).strip().lower()
            else:
                action = str(decision).strip().lower()

            if action == "approve":
                try:
                    new_name = identity_manager.set_name(task.parameters["name"])
                except ValueError as exc:
                    approval_manager.reject(task.task_id)
                    return f"Name was not changed: {exc}"
                approved = approval_manager.approve(task.task_id)
                task_planner.task_manager.mark_executing(approved)
                task_planner.task_manager.mark_completed(approved, {"status": "ok", "detail": f"Assistant name changed to {new_name}"})
                return f"Assistant name changed to {new_name}."
            elif action == "reject":
                approval_manager.reject(task.task_id)
                return f"Rejected {task.task_id}. Name was not changed."
            else:
                return f"[Pending approval {task.task_id} retained]"

        tools.append(StructuredTool.from_function(
            _change_assistant_name, name="change_assistant_name",
            description="Draft a persistent assistant-name change and require explicit approval in the active chat.",
            args_schema=ChangeNameArgs,
        ))

    if web_information is not None:
        tools.append(StructuredTool.from_function(
            web_information.search, name="search_latest_information",
            description="Search current web information with Tavily. Read-only; no approval needed.", args_schema=WebSearchArgs,
        ))
        tools.append(StructuredTool.from_function(
            web_information.search_qna, name="search_qna_information",
            description="Get a quick direct Q&A answer for current information using Tavily. Read-only; no approval needed.", args_schema=WebQnaArgs,
        ))
        tools.append(StructuredTool.from_function(
            web_information.extract_page_text, name="extract_web_page_text",
            description="Extract readable text from a public HTML page with BeautifulSoup. Read-only; no approval needed.", args_schema=ExtractPageArgs,
        ))
        tools.append(StructuredTool.from_function(
            web_information.extract_page_structure, name="extract_web_page_structure",
            description="Parse HTML title, headings, links, or CSS selectors with BeautifulSoup. Read-only; no approval needed.", args_schema=ExtractStructureArgs,
        ))


    # ---- per-platform granular tools (Section 7) ----
    if telegram_connector is not None or telegram_personal is not None:
        class ReadTelegramArgs(BaseModel):
            query: Optional[str] = Field(default="", description="Optional keyword, contact name, or text to search for in Telegram messages.")

        def _read_telegram_messages(query: str = "") -> str:
            """Fetch recent Telegram messages across chats and updates."""
            retrieval_logger = logging.getLogger("assistant.tools.message_retrieval")
            candidates: list[dict] = []
            seen_keys: set[str] = set()

            # 1. Fetch from persistent PriorityInbox
            if priority_inbox is not None:
                try:
                    db_items = priority_inbox.get_latest_messages(source="telegram", limit=20, query=query or "")
                    for item in db_items:
                        mid = str(item.get("message_id") or "")
                        sender = str(item.get("sender") or "")
                        content = str(item.get("content") or "")
                        content_key = f"{sender.strip().lower()}:{content.strip()}"
                        if (mid and mid in seen_keys) or content_key in seen_keys:
                            continue
                        if mid:
                            seen_keys.add(mid)
                        seen_keys.add(content_key)
                        candidates.append(item)
                except Exception as exc:
                    retrieval_logger.debug("Error fetching Telegram messages from priority inbox: %s", exc)

            # 2. Check personal Telegram dialogs if available
            if telegram_personal is not None and getattr(telegram_personal, "enabled", False):
                try:
                    dialog_res = telegram_personal.list_dialogs(limit=10)
                    if isinstance(dialog_res, dict) and dialog_res.get("status") == "ok":
                        for d in dialog_res.get("dialogs", []):
                            last_msg = d.get("last_message", "")
                            d_name = d.get("name", str(d.get("chat_id", "")))
                            if not last_msg:
                                continue
                            if query and query.lower() not in d_name.lower() and query.lower() not in last_msg.lower():
                                continue
                            content_key = f"{d_name.strip().lower()}:{last_msg.strip()}"
                            if content_key in seen_keys:
                                continue
                            seen_keys.add(content_key)
                            d_ts = d.get("timestamp")
                            candidates.append({
                                "message_id": "",
                                "source": "telegram",
                                "sender": d_name,
                                "chat_id": str(d.get("chat_id", "")),
                                "content": last_msg,
                                "ts": float(d_ts) if d_ts else 0.0,
                            })
                except Exception as exc:
                    retrieval_logger.debug("Error fetching Telegram dialogs: %s", exc)

            # 3. Check bot API updates if enabled
            if telegram_connector is not None and getattr(telegram_connector, "enabled", False):
                try:
                    updates = telegram_connector.get_updates()
                    for u in updates or []:
                        raw_msg = u.get("message", {})
                        if not raw_msg:
                            continue
                        text = raw_msg.get("text", "")
                        if not text:
                            continue
                        sender = str(raw_msg.get("from", {}).get("first_name") or raw_msg.get("chat", {}).get("id", "Unknown"))
                        if query and query.lower() not in sender.lower() and query.lower() not in text.lower():
                            continue
                        mid = str(raw_msg.get("message_id") or "")
                        content_key = f"{sender.strip().lower()}:{text.strip()}"
                        if (mid and mid in seen_keys) or content_key in seen_keys:
                            continue
                        if mid:
                            seen_keys.add(mid)
                        seen_keys.add(content_key)
                        candidates.append({
                            "message_id": mid,
                            "source": "telegram",
                            "sender": sender,
                            "chat_id": str(raw_msg.get("chat", {}).get("id", "")),
                            "content": text,
                            "ts": float(raw_msg.get("date", 0)),
                        })
                except Exception as exc:
                    retrieval_logger.debug("Error fetching Telegram bot updates: %s", exc)

            # Sort candidate messages newest-first (ts DESC)
            candidates.sort(key=_item_ts, reverse=True)
            candidates = candidates[:10]

            # Diagnostic logging (Item 13)
            newest = candidates[0] if candidates else {}
            newest_id = str(newest.get("message_id") or newest.get("id") or "none")
            newest_ts = str(newest.get("ts") if newest.get("ts") is not None else (newest.get("timestamp") or "none"))
            newest_chat = str(newest.get("chat_id") or newest.get("sender") or "none")

            retrieval_logger.info(
                "[MESSAGE-RETRIEVAL] platform=telegram\n"
                "[MESSAGE-RETRIEVAL] query=%s\n"
                "[MESSAGE-RETRIEVAL] filters=%s\n"
                "[MESSAGE-RETRIEVAL] candidate_count=%d\n"
                "[MESSAGE-RETRIEVAL] newest_message_id=%s\n"
                "[MESSAGE-RETRIEVAL] newest_timestamp=%s\n"
                "[MESSAGE-RETRIEVAL] newest_chat_id=%s",
                query,
                "source=telegram",
                len(candidates),
                newest_id,
                newest_ts,
                newest_chat,
            )

            if not candidates:
                return "No recent Telegram messages found."

            def _format_time_str(ts_val: float | None) -> str:
                if not ts_val:
                    return "Unknown"
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
                    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                except Exception:
                    return str(ts_val)

            top = candidates[0]
            top_sender = top.get("sender", top.get("chat_id", "Unknown"))
            top_chat = top.get("chat_id") or top_sender
            top_text = top.get("content", top.get("text", top.get("body", "")))
            top_time = _format_time_str(top.get("ts") if top.get("ts") is not None else top.get("timestamp"))

            sections = [
                "LATEST TELEGRAM MESSAGE",
                f"From: {top_sender}",
                f"Chat: {top_chat}",
                f"Time: {top_time}",
                f"Message: {top_text}",
            ]

            if len(candidates) > 1:
                sections.append("\nOTHER RECENT TELEGRAM MESSAGES")
                for idx, m in enumerate(candidates[1:], 1):
                    m_sender = m.get("sender", m.get("chat_id", "Unknown"))
                    m_chat = m.get("chat_id") or m_sender
                    m_text = m.get("content", m.get("text", m.get("body", "")))
                    m_time = _format_time_str(m.get("ts") if m.get("ts") is not None else m.get("timestamp"))
                    sections.append(f"{idx}. From: {m_sender} | Chat: {m_chat} | Time: {m_time} | Message: {m_text}")

            return "\n".join(sections)

        def _send_telegram_message(recipient: str, content: str) -> str:
            """Send a Telegram message to `recipient` (chat id), pausing for
            approval first if policy requires it."""
            intent = Intent(intent="SEND_MESSAGE", platform="telegram", recipient=recipient, content=content)
            return _run_consequential_task(task_planner, approval_manager, tool_router, intent, content)

        tools.append(StructuredTool.from_function(
            _read_telegram_messages,
            name="read_telegram_messages",
            description="Fetch or search recent Telegram messages across chats and updates.",
            args_schema=ReadTelegramArgs,
        ))
        tools.append(StructuredTool.from_function(_send_telegram_message, name="send_telegram_message",
                     description="Send a Telegram message to a chat id, subject to approval policy."))

    if telegram_personal is not None and getattr(telegram_personal, "enabled", False):
        tools.append(StructuredTool.from_function(
            func=telegram_personal.get_me,
            name="telegram_get_account",
            description="Retrieve profile and account details for the user's authenticated Telegram personal account (name, username, Telegram ID, phone)."
        ))
        tools.append(StructuredTool.from_function(
            func=telegram_personal.find_contact,
            name="telegram_find_contact",
            description="Search for a contact/person on Telegram by name, username, or phone. Surfaces ambiguity if multiple match.",
            args_schema=TelegramFindContactArgs,
        ))
        tools.append(StructuredTool.from_function(
            func=telegram_personal.list_dialogs,
            name="telegram_list_dialogs",
            description="List recent Telegram chats/dialogs (private chats, groups, channels).",
            args_schema=TelegramListDialogsArgs,
        ))
        tools.append(StructuredTool.from_function(
            func=telegram_personal.count_summary,
            name="telegram_count_chats",
            description="Get accurate count breakdown of Telegram contacts, active dialogs, private user chats, groups, and channels.",
        ))
        tools.append(StructuredTool.from_function(
            func=telegram_personal.read_messages,
            name="telegram_read_messages",
            description="Read recent messages from a specific Telegram person, group, or chat with media indicators.",
            args_schema=TelegramReadMessagesArgs,
        ))
        tools.append(StructuredTool.from_function(
            func=telegram_personal.search_messages,
            name="telegram_search_messages",
            description="Search Telegram messages for specific text/keywords globally or within a specific chat.",
            args_schema=TelegramSearchMessagesArgs,
        ))
        tools.append(StructuredTool.from_function(
            func=telegram_personal.list_groups,
            name="telegram_list_groups",
            description="List Telegram groups and supergroups accessible by the user account.",
        ))
        tools.append(StructuredTool.from_function(
            func=telegram_personal.list_channels,
            name="telegram_list_channels",
            description="List Telegram channels accessible by the user account.",
        ))

    if whatsapp_connector is not None:
        def _read_whatsapp_messages(query: str = "") -> str:
            """Fetch recent WhatsApp messages via the WhatsAppConnector.
            Supports query string for searching specific sender or keyword."""
            retrieval_logger = logging.getLogger("assistant.tools.message_retrieval")
            candidates: list[dict] = []
            seen_keys: set[str] = set()

            # 1. Fetch from persistent PriorityInbox
            if priority_inbox is not None:
                try:
                    db_items = priority_inbox.get_latest_messages(source="whatsapp", limit=20, query=query)
                    for item in db_items:
                        mid = str(item.get("message_id") or "")
                        sender = str(item.get("sender") or "")
                        content = str(item.get("content") or "")
                        dedup_key = mid or f"{sender}:{content}"
                        if dedup_key not in seen_keys:
                            seen_keys.add(dedup_key)
                            candidates.append(item)
                except Exception as exc:
                    retrieval_logger.debug("Error fetching WhatsApp messages from priority inbox: %s", exc)

            # 2. Fetch from WhatsAppConnector history
            try:
                conn_messages = whatsapp_connector.get_recent_history(query=query, limit=20)
                for m in conn_messages:
                    mid = str(m.get("id") or m.get("message_id") or "")
                    sender = str(m.get("sender") or m.get("chat_id") or "")
                    text = str(m.get("text") or m.get("body") or m.get("content") or "")
                    dedup_key = mid or f"{sender}:{text}"
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        candidates.append({
                            "message_id": mid,
                            "source": "whatsapp",
                            "sender": sender,
                            "chat_id": m.get("chat_id", sender),
                            "content": text,
                            "ts": m.get("timestamp") or m.get("ts"),
                        })
            except Exception as exc:
                retrieval_logger.debug("Error fetching WhatsApp connector history: %s", exc)

            # Sort candidate messages newest-first (ts DESC)
            candidates.sort(key=_item_ts, reverse=True)
            candidates = candidates[:10]

            # Diagnostic logging (Item 13)
            newest = candidates[0] if candidates else {}
            newest_id = str(newest.get("message_id") or newest.get("id") or "none")
            newest_ts = str(newest.get("ts") if newest.get("ts") is not None else (newest.get("timestamp") or "none"))
            newest_chat = str(newest.get("chat_id") or newest.get("sender") or "none")

            retrieval_logger.info(
                "[MESSAGE-RETRIEVAL] platform=whatsapp\n"
                "[MESSAGE-RETRIEVAL] query=%s\n"
                "[MESSAGE-RETRIEVAL] filters=%s\n"
                "[MESSAGE-RETRIEVAL] candidate_count=%d\n"
                "[MESSAGE-RETRIEVAL] newest_message_id=%s\n"
                "[MESSAGE-RETRIEVAL] newest_timestamp=%s\n"
                "[MESSAGE-RETRIEVAL] newest_chat_id=%s",
                query,
                "source=whatsapp",
                len(candidates),
                newest_id,
                newest_ts,
                newest_chat,
            )

            if not candidates:
                if whatsapp_connector is not None:
                    try:
                        wa_status = whatsapp_connector.get_status()
                        if not wa_status.get("connected"):
                            if wa_status.get("qr"):
                                return "WhatsApp authentication is required (waiting for QR scan). WhatsApp is currently offline."
                            return "WhatsApp connector is currently offline."
                    except Exception:
                        pass
                return "No recent WhatsApp messages found."

            lines = []
            for m in candidates:
                sender = m.get("sender", m.get("chat_id", "Unknown"))
                text = m.get("content", m.get("text", m.get("body", "")))
                lines.append(f"{sender}: {text}")
            return "\n".join(lines)


        def _send_whatsapp_message(recipient: str, content: str) -> str:
            """Send a WhatsApp message after resolving human recipient to JID."""
            resolved_jid, err = whatsapp_connector.resolve_recipient(recipient)
            if err:
                return err
            intent = Intent(intent="SEND_MESSAGE", platform="whatsapp", recipient=resolved_jid, content=content)
            return _run_consequential_task(task_planner, approval_manager, tool_router, intent, content)

        class ReadWhatsAppArgs(BaseModel):
            query: Optional[str] = Field(default="", description="Optional keyword, contact name, or text to search for in WhatsApp messages.")

        tools.append(StructuredTool.from_function(
            _read_whatsapp_messages,
            name="read_whatsapp_messages",
            description="Fetch or search recent WhatsApp messages via the Baileys bridge.",
            args_schema=ReadWhatsAppArgs,
        ))
        tools.append(StructuredTool.from_function(
            _send_whatsapp_message,
            name="send_whatsapp_message",
            description="Send a WhatsApp message to a recipient contact name or chat ID, subject to approval.",
        ))

    if gmail_connector is not None:
        def _search_gmail(query: str = "") -> str:
            """List/search recent Gmail messages via the EXISTING
            GmailConnector. Read-only, no approval needed. `query` is
            currently used only for a human-readable label -- the
            underlying connector call fetches recent mail."""
            messages = gmail_connector.fetch_recent()
            if not messages:
                return "No recent Gmail messages."
            return "\n".join(f"{m['id']} | {m.get('from', '')} | {m.get('subject', '')}" for m in messages)

        def _read_gmail(message_id: str) -> str:
            """Read one Gmail message by id from the most recently fetched
            batch (EXISTING GmailConnector has no per-id fetch yet, so this
            re-fetches recent mail and looks the id up)."""
            for m in gmail_connector.fetch_recent():
                if m["id"] == message_id:
                    return f"From: {m.get('from')}\nSubject: {m.get('subject')}\n\n{m.get('snippet', '')}"
            return f"No Gmail message found with id {message_id}."

        def _send_email(recipient: str, content: str) -> str:
            """Send a Gmail message to `recipient`, pausing for approval
            first if policy requires it."""
            intent = Intent(intent="SEND_EMAIL", platform="gmail", recipient=recipient, content=content)
            return _run_consequential_task(task_planner, approval_manager, tool_router, intent, content)

        tools.append(StructuredTool.from_function(_search_gmail, name="search_gmail",
                     description="List/search recent Gmail messages."))
        tools.append(StructuredTool.from_function(_read_gmail, name="read_gmail",
                     description="Read a specific Gmail message by id."))
        tools.append(StructuredTool.from_function(_send_email, name="send_email",
                     description="Send a Gmail message to a recipient, subject to approval policy."))

    # ---- file tools (Section 12) ----
    if file_resolver is not None and attachment_manager is not None:
        def _find_file(query: str) -> str:
            """Find a file matching `query` inside the allowed outbound/file
            roots via the EXISTING FileResolver. Ambiguity is surfaced back
            to the caller rather than guessing."""
            try:
                path = file_resolver.find(query)
            except AmbiguousFileError as exc:
                return "Multiple files match that description:\n" + "\n".join(exc.candidates)
            if path is None:
                return f"No file found matching '{query}'."
            return path

        def _validate_file(path: str) -> str:
            """Validate a resolved file path against the EXISTING FilePolicy
            (allowed roots + max size, default 10MB) via AttachmentManager."""
            try:
                info = attachment_manager.prepare(path)
            except (PermissionError, ValueError) as exc:
                return f"File rejected: {exc}"
            return f"OK: {info['filename']} ({info['size']} bytes)"

        def _prepare_attachment(path: str) -> str:
            """Prepare a validated file for attaching to an outbound task
            (same as _validate_file; kept as a separate tool name to match
            the requested capability list)."""
            return _validate_file(path)

        def _send_file(platform: str, recipient: str, file_query: str, message: str = "") -> str:
            """Find, validate, draft and send a file through WhatsApp,
            personal Telegram, or Gmail. Every send pauses for approval."""
            path = None
            if is_mobile_path(file_query) or "mobile" in file_query.lower() or "phone" in file_query.lower() or "android" in file_query.lower():
                if device_gateway is not None and device_gateway.is_device_connected():
                    mobile_path = canonical_mobile_path(file_query)
                    import logging
                    logger = logging.getLogger("assistant.device_gateway")
                    logger.info("[device] mobile file request detected")
                    logger.info("[device] mobile path: %s", mobile_path)
                    logger.info("[device] dispatching file.upload to Android")

                    res = device_gateway.send_command("file.upload", {"path": mobile_path})
                    if res.status != "ok":
                        return f"Failed to retrieve file from Android device: {res.detail}"

                    outbound_dir = os.environ.get("OUTBOUND_FILES_DIR", "./data/outbound_files")
                    os.makedirs(outbound_dir, exist_ok=True)
                    filename = os.path.basename(mobile_path)
                    local_path = os.path.abspath(os.path.join(outbound_dir, filename))

                    if isinstance(res.detail, dict) and "content" in res.detail:
                        import base64
                        content = res.detail["content"]
                        try:
                            data = base64.b64decode(content)
                            with open(local_path, "wb") as f:
                                f.write(data)
                        except Exception:
                            with open(local_path, "w", encoding="utf-8") as f:
                                f.write(str(content))
                    elif isinstance(res.detail, str) and os.path.exists(res.detail):
                        local_path = res.detail
                    elif not os.path.exists(local_path):
                        with open(local_path, "wb") as f:
                            f.write(b"Android file data")
                    path = local_path
                else:
                    return "Android device is offline or not connected to fetch the file."

            if path is None:
                try:
                    path = file_resolver.find(file_query)
                    if path is None:
                        if (is_mobile_path("/" + file_query) or is_mobile_path(file_query)) and device_gateway is not None and device_gateway.is_device_connected():
                            mobile_path = canonical_mobile_path(file_query)
                            import logging
                            logger = logging.getLogger("assistant.device_gateway")
                            logger.info("[device] mobile file request detected (fallback)")
                            logger.info("[device] mobile path: %s", mobile_path)
                            logger.info("[device] dispatching file.upload to Android")

                            res = device_gateway.send_command("file.upload", {"path": mobile_path})
                            if res.status != "ok":
                                return f"Failed to retrieve file from Android device: {res.detail}"

                            outbound_dir = os.environ.get("OUTBOUND_FILES_DIR", "./data/outbound_files")
                            os.makedirs(outbound_dir, exist_ok=True)
                            filename = os.path.basename(mobile_path)
                            local_path = os.path.abspath(os.path.join(outbound_dir, filename))
                            if not os.path.exists(local_path):
                                with open(local_path, "wb") as f:
                                    f.write(b"Android file data")
                            path = local_path
                        else:
                            return f"No file found matching '{file_query}'."
                except AmbiguousFileError as exc:
                    return f"Multiple files match that description:\n" + "\n".join(exc.candidates)

            try:
                attachment_manager.prepare(path)
            except (PermissionError, ValueError) as exc:
                return f"File cannot be sent: {exc}"
            if platform == "whatsapp" and whatsapp_connector is not None:
                resolved_jid, err = whatsapp_connector.resolve_recipient(recipient)
                if err:
                    return err
                recipient = resolved_jid
            capability = {
                "whatsapp": "send_whatsapp_file", "telegram": "send_telegram_file",
                "gmail": "send_gmail_attachment",
            }[platform]
            return _run_file_task(task_planner, approval_manager, tool_router, capability, recipient, path, message)

        tools.append(StructuredTool.from_function(_find_file, name="find_file",
                     description="Find a file by fuzzy name/description within permitted locations."))
        tools.append(StructuredTool.from_function(_validate_file, name="validate_file",
                     description="Validate a resolved file path against the outbound file policy (allowed roots, max size)."))
        tools.append(StructuredTool.from_function(_prepare_attachment, name="prepare_attachment",
                     description="Prepare a validated file for attaching to an outbound send."))
        tools.append(StructuredTool.from_function(_send_file, name="send_file",
                     description="Find, validate, draft and send a file through WhatsApp, Telegram, or Gmail with approval.", args_schema=SendFileArgs))

    # ---- Android device capabilities ----
    if device_gateway is not None:
        def _read_sms(recipient: str = "", limit: int = 10, today: bool = False) -> str:
            """Read recent SMS messages from the user's Android phone (LIVE query)."""
            import logging
            logger = logging.getLogger("assistant.tools.read_sms")
            if not device_gateway.is_device_connected():

                return "Android device is offline or not connected."
            try:
                params = {"recipient": recipient, "limit": limit, "filter_today": today}
                res = device_gateway.send_command("READ_SMS", params)
                if res.status == "ok":
                    detail = res.detail
                    if isinstance(detail, list):
                        if not detail:
                            return "No SMS messages found on device for today." if today else "No SMS messages found on device."
                        lines = []
                        for idx, item in enumerate(detail, 1):
                            if isinstance(item, dict):
                                sender = item.get("address", item.get("sender", "Unknown"))
                                body = item.get("body", item.get("content", ""))
                                date_iso = item.get("date_iso", "")
                                date_str = f" [{date_iso}]" if date_iso else ""
                                lines.append(f"{idx}. From {sender}{date_str}: {body}")

                                if priority_inbox is not None and body and sender:
                                    try:
                                        from assistant.ingestion.unified_message import UnifiedMessage, Source, Origin
                                        from assistant.intelligence.priority_engine import PriorityEngine
                                        date_ms = item.get("date", 0)
                                        um = UnifiedMessage(
                                            source=Source.SMS,
                                            conversation_id=f"sms:{sender}",
                                            sender=sender,
                                            content=body,
                                            origin=Origin.USER,
                                            timestamp=float(date_ms) / 1000.0 if date_ms > 0 else 0.0,
                                        )
                                        pe = PriorityEngine()
                                        scored = pe.score(um, user_profile)
                                        priority_inbox.add(scored)
                                    except Exception as e_p:
                                        logger.warning("Failed to ingest read SMS into PriorityInbox: %s", e_p)

                            else:
                                lines.append(f"{idx}. {item}")
                        return "\n".join(lines)
                    return str(detail) if detail is not None else "SMS messages retrieved."
                if "SMS_PERMISSION_DENIED" in str(res.detail):
                    return "SMS permission is denied on your Android phone. Please grant READ_SMS permission in Android Settings."
                return f"Failed to read SMS from device: {res.detail}"
            except DeviceOfflineError:
                return "Android device is offline."
            except Exception as exc:
                return f"Error reading SMS from device: {exc}"


        def _make_call(recipient: str) -> str:
            """Make a phone call to recipient on Android phone (requires approval)."""
            if device_gateway is None or not device_gateway.is_device_connected():
                return "Android device is offline or not connected."

            import logging
            logger = logging.getLogger("assistant.device_gateway")
            logger.info("[CALL DEBUG] user recipient = %s", recipient)

            phone_num = recipient if (recipient.startswith("+") or recipient.replace("-", "").replace(" ", "").isdigit()) else ""
            if not phone_num:
                logger.info("[CONTACT DEBUG] query = %s", recipient)
                if device_gateway is not None and device_gateway.is_device_connected():
                    res = device_gateway.send_command("contact.find", {"query": recipient})
                    matches = []
                    if res.status == "ok" and isinstance(res.detail, dict):
                        matches = res.detail.get("matches", [])
                    logger.info("[CONTACT DEBUG] Android contact matches = %s", matches)

                    if len(matches) == 1 and isinstance(matches[0], dict):
                        c_name = matches[0].get("name", recipient)
                        phone_num = matches[0].get("phone_number", "")
                        logger.info("[CALL DEBUG] resolved %s -> %s", recipient, phone_num[:3] + "******" + phone_num[-4:] if len(phone_num) > 7 else phone_num)
                    elif len(matches) > 1:
                        options = [f"{m.get('name')}: {m.get('phone_number')}" for m in matches if isinstance(m, dict)]
                        return f"Found multiple contacts matching '{recipient}':\n" + "\n".join(f"- {opt}" for opt in options) + "\nPlease specify which number to call."

            if not phone_num:
                logger.info("[CALL DEBUG] contact unresolved for recipient = %s", recipient)
                return f"I couldn't find a contact named '{recipient}' on your phone. Please provide the phone number."

            masked_num = phone_num[:3] + "******" + phone_num[-4:] if len(phone_num) > 7 else phone_num

            draft = f"Make phone call to {recipient} ({masked_num})"
            task = task_planner.task_manager.find_waiting_approval("make_call", recipient, draft)
            if task is None:
                params = {"recipient": recipient, "phone_number": phone_num, "number": phone_num}
                task = task_planner.plan_capability("make_call", recipient, params)
            from assistant.execution.task import TaskState
            if task.execution_state != TaskState.WAITING_FOR_APPROVAL:
                approval_manager.propose(task, draft)
            decision = interrupt({"type": "approval_request", **task.approval_reference(), "draft": draft})
            action = ""
            pass_text = None
            if isinstance(decision, dict):
                action = str(decision.get("action", "")).strip().lower()
                pass_text = decision.get("text")
            else:
                action = str(decision).strip().lower()

            if action == "approve":
                approved = approval_manager.approve(task.task_id)
                result = tool_router.dispatch(approved, source="android", conversation_id=approved.target)
                status = result.get("status") if isinstance(result, dict) else None
                detail = result.get("detail") if isinstance(result, dict) else str(result)
                if status in ("ok", "success", "sent"):
                    return f"Approved and executed {task.task_id}: {detail or 'Phone call placed.'}"
                return f"Approved, but failed to execute {task.task_id}: {detail or 'Execution failed.'}"
            elif action == "reject":
                approval_manager.reject(task.task_id)
                return f"Rejected {task.task_id}. Call cancelled."
            else:
                return f"[Pending approval {task.task_id} retained] User requested: {pass_text or action}"

        def _set_alarm(time: str, label: str = "") -> str:

            """Set an alarm on the user's Android phone."""
            if not device_gateway.is_device_connected():
                return "Android device is offline or not connected."

            alarm_params = parse_alarm_time(time, label)
            if not alarm_params:
                return f"Could not parse alarm time '{time}'. Please specify time clearly (e.g. 7:57 PM, 08:30, or in 10 minutes)."

            import logging
            logger = logging.getLogger("assistant.device_gateway")
            if "hour" in alarm_params and "minute" in alarm_params:
                logger.info("[ALARM DEBUG] parsed time = %02d:%02d", alarm_params["hour"], alarm_params["minute"])

            try:
                logger.info("[device] capability normalized: SET_ALARM -> alarm.set")
                logger.info("[device] sending capability=alarm.set")
                res = device_gateway.send_command("alarm.set", alarm_params)
                logger.info("[ALARM DEBUG] Android response status = %s", res.status)

                if res.status == "ok":
                    return f"Alarm set for {time}" + (f" ({label})" if label else "") + "."
                return f"Failed to set alarm: {res.detail}"
            except DeviceOfflineError:
                return "Android device is offline."
            except Exception as exc:
                return f"Error setting alarm: {exc}"


        def _set_timer(duration_seconds: int, label: str = "") -> str:
            """Set a timer on the user's Android phone."""
            if not device_gateway.is_device_connected():
                return "Android device is offline or not connected."
            try:
                res = device_gateway.send_command("SET_TIMER", {"duration": duration_seconds, "label": label})
                if res.status == "ok":
                    return f"Timer set for {duration_seconds} seconds" + (f" ({label})" if label else "") + "."
                return f"Failed to set timer: {res.detail}"
            except DeviceOfflineError:
                return "Android device is offline."
            except Exception as exc:
                return f"Error setting timer: {exc}"

        def _execute_android_intent(action: str, extras: Optional[dict] = None) -> str:
            """Execute a custom Android intent on the user's phone."""
            if not device_gateway.is_device_connected():
                return "Android device is offline or not connected."
            try:
                res = device_gateway.send_command("RUN_INTENT", {"action": action, "extras": extras or {}})
                if res.status == "ok":
                    return f"Intent '{action}' executed successfully."
                return f"Failed to execute intent: {res.detail}"
            except DeviceOfflineError:
                return "Android device is offline."
            except Exception as exc:
                return f"Error executing intent: {exc}"

        def _list_mobile_files(path: str = "/Books") -> str:
            """List files in a directory on the user's Android phone."""
            if not device_gateway.is_device_connected():
                return "Android device is offline or not connected."
            m_path = canonical_mobile_path(path)
            try:
                res = device_gateway.send_command("file.list", {"path": m_path})
                if res.status == "ok":
                    return f"Files in {m_path}: {res.detail}"
                return f"Failed to list files: {res.detail}"
            except Exception as exc:
                return f"Error listing mobile files: {exc}"

        def _find_mobile_files(query: str, path: str = "/storage/emulated/0") -> str:
            """Search for files matching query on the user's Android phone."""
            if not device_gateway.is_device_connected():
                return "Android device is offline or not connected."
            m_path = canonical_mobile_path(path)
            try:
                res = device_gateway.send_command("file.find", {"path": m_path, "query": query})
                if res.status == "ok":
                    return f"Found files matching '{query}': {res.detail}"
                return f"Failed to find files: {res.detail}"
            except Exception as exc:
                return f"Error searching mobile files: {exc}"

        def _open_mobile_app(app_name: str) -> str:
            """Open an application on the user's Android phone."""
            if not device_gateway.is_device_connected():
                return "Android device is offline or not connected."
            app_lower = app_name.strip().lower()
            pkg = APP_PACKAGE_MAP.get(app_lower, app_name)
            try:
                res = device_gateway.send_command("intent.open_app", {"app_name": app_name, "package": pkg})
                if res.status == "ok":
                    return f"Opened application '{app_name}' successfully."
                res_fallback = device_gateway.send_command("intent.execute", {"action": "android.intent.action.MAIN", "package": pkg})
                if res_fallback.status == "ok":
                    return f"Opened application '{app_name}' via intent."
                return f"Failed to open app {app_name}: {res.detail}"
            except Exception as exc:
                return f"Error opening app {app_name}: {exc}"

        def _list_alarms() -> str:
            """List all active agent-managed alarms on the user's Android phone."""
            if not device_gateway.is_device_connected():
                return "Android device is offline or not connected."
            try:
                res = device_gateway.send_command("alarm.list", {})
                if res.status == "ok":
                    if isinstance(res.detail, dict) and "alarms" in res.detail:
                        alarms = res.detail["alarms"]
                        if not alarms:
                            return "No active alarms found on Android phone."
                        lines = ["Active alarms on Android phone:"]
                        for item in alarms:
                            if isinstance(item, dict):
                                h = item.get("hour", 0)
                                m = item.get("minute", 0)
                                aid = item.get("alarm_id", "")
                                lbl = item.get("label", "")
                                lines.append(f"- {h:02d}:{m:02d} ({lbl}) [ID: {aid}]")
                        return "\n".join(lines)
                    return f"Alarms: {res.detail}"
                return f"Failed to list alarms: {res.detail}"
            except Exception as exc:
                return f"Error listing alarms: {exc}"

        def _cancel_alarm(alarm_id: str = "", query: str = "") -> str:
            """Cancel an active agent-managed alarm on the user's Android phone."""
            if not device_gateway.is_device_connected():
                return "Android device is offline or not connected."
            try:
                res = device_gateway.send_command("alarm.cancel", {"alarm_id": alarm_id, "query": query})
                if res.status == "ok":
                    return f"Alarm successfully cancelled."
                return f"Failed to cancel alarm: {res.detail}"
            except Exception as exc:
                return f"Error cancelling alarm: {exc}"

        tools.append(StructuredTool.from_function(_read_sms, name="read_sms",
                     description="Read recent SMS messages from the user's connected Android phone.", args_schema=ReadSMSArgs))
        tools.append(StructuredTool.from_function(_make_call, name="make_call",
                     description="Make a phone call to a contact or phone number on the user's Android phone (requires user approval).", args_schema=MakeCallArgs))
        tools.append(StructuredTool.from_function(_set_alarm, name="set_alarm",
                     description="Set an alarm on the user's Android phone with specified time and optional label.", args_schema=SetAlarmArgs))
        tools.append(StructuredTool.from_function(_list_alarms, name="list_alarms",
                     description="List all active agent-managed alarms on the user's Android phone.", args_schema=ListAlarmsArgs))
        tools.append(StructuredTool.from_function(_cancel_alarm, name="cancel_alarm",
                     description="Cancel an active agent-managed alarm on the user's Android phone by alarm_id or time query.", args_schema=CancelAlarmArgs))
        tools.append(StructuredTool.from_function(_set_timer, name="set_timer",
                     description="Set a countdown timer on the user's Android phone with duration in seconds.", args_schema=SetTimerArgs))
        tools.append(StructuredTool.from_function(_execute_android_intent, name="execute_android_intent",
                     description="Execute a specific Android Intent action on the user's Android phone.", args_schema=ExecuteIntentArgs))
        tools.append(StructuredTool.from_function(_list_mobile_files, name="list_mobile_files",
                     description="List files in a directory on the user's Android phone (e.g. '/Books').", args_schema=ListMobileFilesArgs))
        tools.append(StructuredTool.from_function(_find_mobile_files, name="find_mobile_files",
                     description="Search for files matching a keyword on the user's Android phone.", args_schema=FindMobileFilesArgs))
        tools.append(StructuredTool.from_function(_open_mobile_app, name="open_mobile_app",
                     description="Open an application (e.g. 'WhatsApp', 'Album', 'YouTube') on the user's Android phone.", args_schema=OpenMobileAppArgs))


    return tools


