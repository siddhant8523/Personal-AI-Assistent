"""
Graph Nodes
==============
Each node is a small function over AgentState. Every node that touches
business logic delegates to an EXISTING component -- ContextBuilder,
ApprovalManager (via tools.py), TaskPlanner (via tools.py), ToolRouter
(via tools.py). Nothing here reimplements approval, policy, or execution;
it only sequences them.

Approval is intentionally NOT a node in this graph anymore. Previously
"approve TASK-x" / "reject TASK-x" was detected by a regex node and routed
to a hand-rolled `approval_node`, which is really just a second orchestrator
if/elif in disguise -- not an actual graph pause. Real human-in-the-loop
approval now happens via LangGraph's `interrupt()`, called from inside
consequential tools in tools.py, which genuinely suspends graph execution
(checkpointed) until AgentOrchestrator resumes it with `Command(resume=...)`.
See agent_core/orchestrator.py for the resume logic.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from assistant.agent_core.events import AgentStreamEvent

from assistant.agent_core.context_builder import ContextBuilder
from assistant.agent_core.intent_understanding import parse_intent
from assistant.agent_core.graph.state import AgentState
from assistant.llm.llm_client import (
    LLMClient,
    classify_llm_error,
    llm_error_to_user_message,
)
from assistant.llm.provider_router import _normalize_content_to_text
from assistant.memory.memory_service import MemoryService
from assistant.memory.memory_writer import write_interaction

logger = logging.getLogger("assistant.agent_core.graph")


def make_load_context_node(context_builder: ContextBuilder, memory: MemoryService) -> Callable[..., dict]:
    def load_context_node(state: AgentState, config: RunnableConfig = None) -> dict:
        if config and isinstance(config, dict):
            callback = config.get("configurable", {}).get("event_callback")
            if callback:
                callback(AgentStreamEvent.status("Checking context & memory..."))
        context = context_builder.build(state["conversation_id"], state["input_text"])
        write_interaction(memory, state["conversation_id"], state["input_text"])
        return {"context_summary": context.summary}
    return load_context_node


def make_agent_node(llm: LLMClient, tools: list[StructuredTool], soul_text: str | Callable[[], str]) -> Callable[..., dict]:
    """Understand Request + Plan + Select Tool.

    Online: real LangChain tool-calling via ChatMistralAI.bind_tools(...).
    Offline (no MISTRAL_API_KEY): falls back to the same deterministic
    regex intent parser the project always had, so the app still runs
    end-to-end with zero external calls -- same behavior as before,
    just reached via a graph node instead of an if/elif chain.
    """

    def agent_node(state: AgentState, config: RunnableConfig = None) -> dict:
        text = state["input_text"]
        logger.info("[LangGraph] processing")
        current_soul = soul_text() if callable(soul_text) else soul_text

        callback = None
        if config and isinstance(config, dict):
            callback = config.get("configurable", {}).get("event_callback")

        if callback:
            callback(AgentStreamEvent.status("Thinking..."))

        if llm.online:
            model = llm.get_langchain_model(tools=tools)
            system_instructions = (
                f"{current_soul}\n\n"
                "## Output Formatting Rules (CRITICAL - PLAIN TEXT ONLY):\n"
                "- All normal user-facing responses MUST be written in clean, human-readable plain text without any Markdown formatting.\n"
                "- Do NOT use Markdown tables, bold text (**), italic text (*), headings (#), bullet or numbered list syntax, backticks (`), code blocks, Markdown links, or pipes (|).\n"
                "- Formulate responses using natural sentences, conversational paragraphs, and simple line breaks instead.\n\n"
                "## Tool Execution & Direct Action Rules (CRITICAL - NEVER ASK CONVERSATIONAL PERMISSION INSTEAD OF CALLING TOOLS):\n"
                "- When the user requests an action (such as setting an alarm, setting a timer, placing a phone call, sending a message, checking SMS, opening an app, finding files), YOU MUST DIRECTLY AND IMMEDIATELY INVOKE THE CORRESPONDING TOOL.\n"
                "- NEVER ask conversational confirmation questions (e.g. 'Would you like me to set an alarm?', 'Do you want me to set a timer?', 'Do you want me to place a call to ...?', 'Should I send this?').\n"
                "- The system enforces approval through tool interrupts: when you invoke a tool that requires approval (such as `make_call`, `propose_send_message`, `send_file`), the tool automatically pauses execution and presents the formal draft to the user for approval. If you ask conversational questions in chat instead of calling the tool, the system cannot process the request.\n"
                "- Safe device actions (such as `set_alarm`, `set_timer`, `list_alarms`, `cancel_alarm`, `open_mobile_app`) and read-only tools execute directly without requiring approval. Call their tools immediately.\n\n"
                "## Tool Invocation & Messaging Rules:\n"
                "- PRIORITY INBOX & AGENDA PLANNING (CRITICAL):\n"
                "  * Use `query_priority_inbox` when the user asks to view their priority messages, important emails, priority inbox, or asks 'what is important?', 'what should I know?', 'what did I miss?', 'are there any urgent messages?'.\n"
                "  * ALWAYS invoke `query_agenda` when the user asks about their schedule, meetings, deadlines, commitments, or plans for a specific date (e.g. 'do I have any meeting for tomorrow?', 'what should I do tomorrow?', 'what are my deadlines tomorrow?', 'what is scheduled tomorrow?', 'what do I need to handle tomorrow?').\n"
                "  * NATURAL-LANGUAGE SYNTHESIS: After tool execution returns priority or agenda information, you MUST formulate a natural, helpful conversational response. Explain confirmed commitments and deadlines clearly, specify which messages require user action, and distinguish confirmed meetings from suggestions. NEVER regurgitate raw template text.\n"
                "  * When planning or answering about a day's schedule (e.g. 'what should I do tomorrow?'):\n"
                "    1. Inspect the retrieved commitments from `query_agenda` first.\n"
                "    2. Present actual confirmed meetings, appointments, and deadlines FIRST, noting exact times and details.\n"
                "    3. Clearly distinguish confirmed commitments from suggestions.\n"
                "    4. NEVER invent appointments, meetings, or calendar events.\n"
                "    5. ONLY after actual commitments are clearly presented may you suggest optional free-time activities for open gaps.\n\n"
                "- PRIMARY OUTBOUND INTENT OVERRIDE (CRITICAL):\n"
                "  * When a user request specifies a SEND / EMAIL / MESSAGE intent + recipient + requested information (e.g. 'Send the latest news about India to Sidd on Telegram', 'Send the latest news about India to Rahul on WhatsApp', 'Email me the latest news about India', 'Send me today\\'s AI news on Telegram', 'Send Rahul the latest cricket news'):\n"
                "  * Retrieving, searching for, or generating that information is ONLY an intermediate step.\n"
                "  * After tool execution returns the information, your PRIMARY GOAL remains submitting the requested outbound message via `propose_send_message` (or the appropriate send tool) containing the retrieved/formatted content, and submitting it for user approval.\n"
                "  * NEVER display the retrieved information as a final reply to the user when the user requested sending it to a recipient.\n"
                "- WHATSAPP MESSAGES TOOLS:\n"
                "  * Use `read_whatsapp_messages` whenever asked to read, show, check, or search WhatsApp messages (e.g. 'read my whatsapp messages', 'any meeting message on whatsapp', 'any message from Rahul on whatsapp').\n"
                "- TELEGRAM PERSONAL ACCOUNT TOOLS:\n"
                "  * Use `read_telegram_messages` whenever asked for recent, latest, or last Telegram messages without a specific contact (e.g. 'what is the last message in telegram', 'read my telegram messages', 'check telegram').\n"
                "  * Use `telegram_get_account` when asked about user's own Telegram account info.\n"
                "  * Use `telegram_find_contact` when searching for a person on Telegram.\n"
                "  * Use `telegram_list_dialogs` when asked to show Telegram chats or conversations.\n"
                "  * Use `telegram_count_chats` when asked how many chats/contacts/groups/channels exist on Telegram.\n"
                "  * Use `telegram_read_messages` when asked to read recent messages with a specific person or chat.\n"
                "  * Use `telegram_search_messages` when asked to search Telegram messages for keywords.\n"
                "  * Use `telegram_list_groups` for listing Telegram groups.\n"
                "  * Use `telegram_list_channels` for listing Telegram channels.\n"
                "- LITERAL VS GENERATED MESSAGE CONTENT (CRITICAL):\n"
                "  * If the user provides explicit text to send (e.g. 'Send Sidd: Hello how are you'), pass the text exact verbatim.\n"
                "  * If the user asks you to WRITE, DRAFT, GENERATE, SEARCH FOR, or FETCH a message (e.g. 'Send Sidd a 10-line paragraph about machine learning' or 'Send Rahul the latest news'), YOU MUST GENERATE / FORMAT the complete, high-quality content text FIRST, and pass THAT full content as the `content` parameter when calling the send tool!\n"
                "- GENERAL & CROSS-PLATFORM MESSAGE INQUIRIES (CRITICAL):\n"
                "  * When the user asks 'what is important?', 'what should I know?', 'what did I miss?', 'is there any new message?', 'check my messages', 'any latest message', 'what are my latest messages?': YOU MUST query the priority system (`query_priority_inbox` or `query_agenda`) first to inspect unified, prioritized cross-platform messages.\n"
                "  * NEVER claim or hallucinate that SMS or messages were checked unless you actually invoked the corresponding tool(s).\n"
                "- ANDROID DEVICE TOOLS:\n"
                "  * Use `set_alarm` IMMEDIATELY when asked to set an alarm on the phone (e.g. 'set alarm for 10:54 am today', 'set an alarm for 7:57 PM', 'set an alarm for 7 AM'). Pass the exact requested time string (e.g. '10:54 AM', '7:57 PM', '07:57', '8:30 AM'). NEVER ask conversational confirmation before calling `set_alarm`.\n"
                "  * Use `set_timer` IMMEDIATELY when asked to set a timer on the phone (e.g. 'set timer for 1 minute', 'set timer foe 1 minut', 'set timer for 10 second'). Convert duration to total seconds (e.g. 1 minute -> duration_seconds=60, 10 seconds -> duration_seconds=10) and pass `duration_seconds`. NEVER ask conversational confirmation before calling `set_timer`.\n"
                "  * Use `make_call` IMMEDIATELY when asked to call someone or make a phone call (e.g. 'call to 7798298569', 'call John', 'make a call to Maosi'). Pass recipient. NEVER ask conversational confirmation in chat before calling `make_call`—the tool will automatically pause and generate the formal approval prompt.\n"
                "  * Use `read_sms` ALWAYS whenever asked to read, show, list, check, or fetch SMS / text messages from the phone (e.g. 'read my SMS', 'read my text messages', 'check SMS', 'do check latest SMS', 'what was last incoming message'). NEVER answer from memory or claim SMS was checked without calling `read_sms`.\n"
                "  * Use `propose_send_message` with platform='sms' when asked to send an SMS or text message.\n"
                "  * Use `list_alarms` when asked to view, list, or show active alarms on the phone (e.g. 'show my alarms', 'list my alarms').\n"
                "  * Use `cancel_alarm` when asked to cancel, remove, or stop an alarm on the phone (e.g. 'cancel my 8:44 pm alarm', 'delete alarm alarm-xyz').\n"
                "  * Use `open_mobile_app` when asked to open an application on mobile (e.g. 'open WhatsApp in mobile', 'open Album on phone', 'open YouTube'). Never state that you cannot open apps on mobile.\n"
                "  * Use `list_mobile_files` or `find_mobile_files` when asked to check, list, or search files on the phone (e.g. 'what files are in Books?', 'find cptopic in Books'). Mobile storage paths starting with /Books or Books refer to /storage/emulated/0/Books on Android.\n"
                "  * Use `execute_android_intent` when asked to execute a custom Android intent."

            )
            existing = list(state.get("messages", []))
            user_content = f"User: {text}"
            if not existing:
                messages = [
                    SystemMessage(content=system_instructions),
                    HumanMessage(content=f"{state.get('context_summary', '')}\n\n{user_content}"),
                ]
            else:
                if isinstance(existing[0], SystemMessage):
                    existing[0] = SystemMessage(content=system_instructions)
                else:
                    existing.insert(0, SystemMessage(content=system_instructions))

                last_msg = existing[-1]
                if isinstance(last_msg, AIMessage):
                    existing.append(HumanMessage(content=user_content))
                elif isinstance(last_msg, ToolMessage):
                    # ToolMessage was produced in this turn; model will respond directly to it
                    pass
                elif isinstance(last_msg, HumanMessage) and last_msg.content != user_content:
                    existing.append(HumanMessage(content=user_content))

                messages = existing

            if callback:
                try:
                    accumulated: AIMessageChunk | None = None
                    for chunk in model.stream(messages, config=config):
                        # chunk.content may be a list of content blocks (Gemini) or a str.
                        # Always normalize to plain text before passing to callbacks.
                        chunk_text = _normalize_content_to_text(chunk.content) if chunk.content else ""
                        if chunk_text:
                            callback(AgentStreamEvent.token(content=chunk_text))
                        if accumulated is None:
                            accumulated = chunk
                        else:
                            accumulated = accumulated + chunk

                    ai_message = accumulated if accumulated is not None else AIMessage(content="")
                    tool_calls = getattr(ai_message, "tool_calls", None) or []
                    result: dict[str, Any] = {"messages": messages + [ai_message], "tool_calls": tool_calls}
                    if not tool_calls:
                        result["reply"] = ai_message.content
                    return result
                except Exception as exc:
                    classified = classify_llm_error(exc)
                    logger.warning(
                        "[LLM] streaming failed in agent_node category=%s error=%s: %s",
                        classified.category,
                        type(exc).__name__,
                        exc,
                    )
                    friendly_reply = llm_error_to_user_message(classified)
                    callback(AgentStreamEvent.error(message=friendly_reply))
                    fallback_msg = AIMessage(content=friendly_reply)
                    return {
                        "messages": messages + [fallback_msg],
                        "tool_calls": [],
                        "reply": friendly_reply,
                        "error": friendly_reply,
                    }
            else:
                try:
                    ai_message: AIMessage = model.invoke(messages)
                    tool_calls = ai_message.tool_calls or []
                    result: dict[str, Any] = {"messages": messages + [ai_message], "tool_calls": tool_calls}
                    if not tool_calls:
                        result["reply"] = ai_message.content
                    return result
                except Exception as exc:
                    classified = classify_llm_error(exc)
                    logger.warning(
                        "[LLM] invocation failed in agent_node category=%s error=%s: %s",
                        classified.category,
                        type(exc).__name__,
                        exc,
                    )
                    friendly_reply = llm_error_to_user_message(classified)
                    fallback_msg = AIMessage(content=friendly_reply)
                    return {
                        "messages": messages + [fallback_msg],
                        "tool_calls": [],
                        "reply": friendly_reply,
                    }

        # --- offline fallback: same regex-based intent parsing as before ---
        existing_msgs = list(state.get("messages", []))
        user_content = f"User: {text}"
        if not existing_msgs:
            existing_msgs = [
                SystemMessage(content=current_soul),
                HumanMessage(content=user_content),
            ]
        elif not isinstance(existing_msgs[-1], HumanMessage):
            existing_msgs.append(HumanMessage(content=user_content))

        tool_msgs = [m for m in existing_msgs if isinstance(m, ToolMessage)]
        if tool_msgs:
            reply = tool_msgs[-1].content
            return {"messages": existing_msgs + [AIMessage(content=str(reply))], "tool_calls": [], "reply": reply}

        intent = parse_intent(text, llm=None)
        if intent.intent == "QUERY_PRIORITY":
            return {"messages": existing_msgs, "tool_calls": [{"name": "query_priority_inbox", "args": {}, "id": "offline-1"}]}
        if intent.intent in ("SEND_MESSAGE", "SEND_EMAIL"):
            return {"messages": existing_msgs, "tool_calls": [{
                "name": "propose_send_message",
                "args": {"platform": intent.platform, "recipient": intent.recipient, "content": intent.content},
                "id": "offline-1",
            }]}

        gen_reply = llm.generate(system=current_soul, user_message=text)
        if callback and gen_reply:
            callback(AgentStreamEvent.token(content=gen_reply))
        return {"messages": existing_msgs + [AIMessage(content=gen_reply)], "tool_calls": [], "reply": gen_reply}

    return agent_node


def make_tool_exec_node(tools: list[StructuredTool]) -> Callable[..., dict]:
    """Executes the tool(s) the agent selected. If a tool calls LangGraph's
    `interrupt()` (e.g. a consequential send needing approval), execution of
    this node -- and the whole graph -- pauses right here; LangGraph's
    checkpointer preserves everything already written to state so the graph
    can resume exactly at this point once AgentOrchestrator calls
    `graph.invoke(Command(resume=...), config=...)`."""

    tools_by_name = {t.name: t for t in tools}

    def tool_exec_node(state: AgentState, config: RunnableConfig = None) -> dict:
        callback = None
        if config and isinstance(config, dict):
            callback = config.get("configurable", {}).get("event_callback")

        results = []
        tool_messages = []
        executed_tools = []
        for call in state.get("tool_calls", []):
            tool_name = call.get("name", "")
            tool_args = call.get("args", {})
            executed_tools.append(tool_name)
            tool = tools_by_name.get(tool_name)
            if tool is None:
                results.append(f"[error: unknown tool {tool_name}]")
                continue

            task_id = tool_args.get("task_id") if isinstance(tool_args, dict) else None
            if callback:
                callback(AgentStreamEvent.tool_start(tool_name=tool_name, tool_args=tool_args, task_id=task_id))

            output = tool.invoke(call["args"])
            results.append(str(output))
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call.get("id", ""), name=tool_name))

            if callback:
                callback(AgentStreamEvent.tool_complete(tool_name=tool_name, tool_result=str(output), task_id=task_id))
        
        current_messages = list(state.get("messages", []))
        reply_str = "\n".join(results) if results else state.get("reply", "")
        return {
            "messages": current_messages + tool_messages,
            "tool_calls": [],
            "reply": reply_str,
            "executed_tools": executed_tools,
        }

    return tool_exec_node


def strip_markdown_formatting(text: str) -> str:
    """Ensure all normal user-facing responses are returned in clean plain text."""
    if not text:
        return text

    # Remove bold (**text** or __text__)
    text = re.sub(r"\*{2}(.*?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.*?)_{2}", r"\1", text)

    # Remove italics (*text* or _text_)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)

    # Remove Markdown headings (# Heading)
    text = re.sub(r"^\s*#+\s+", "", text, flags=re.MULTILINE)

    # Remove backticks (`code` or ```code block```)
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("```", ""), text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove Markdown links [label](url) -> label (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    # Remove Markdown bullet point symbols (- or *)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)

    # Convert Markdown table lines (| col1 | col2 |) to plain text
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*\|?[\s\-\:|]+\|\s*$", line):
            continue
        if "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            line = ". ".join(parts)
        lines.append(line)

    return "\n".join(lines).strip()


def respond_node(state: AgentState) -> dict:
    reply = state.get("reply")
    if reply:
        # reply is always built from str(output) in tool_exec_node, so it's already a str.
        return {"reply": strip_markdown_formatting(reply)}
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "content") and messages[-1].content:
        # Normalize list content (Gemini) before passing to regex-based strip_markdown_formatting.
        raw = messages[-1].content
        text = _normalize_content_to_text(raw) if isinstance(raw, list) else str(raw)
        return {"reply": strip_markdown_formatting(text)}
    return {"reply": "(no response generated)"}
