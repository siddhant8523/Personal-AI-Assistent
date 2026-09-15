"""
JOE AI — Streamlit Application (streamlit_app.py)
================================================
Modern presentation layer for the Personal AI Assistant.
Supports interactive human-in-the-loop approvals via LangGraph resume mechanism.
Strictly decoupled from backend orchestration, tool routing, connectors, and memory.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterator

import streamlit as st

from assistant.agent_core.events import AgentStreamEvent
from assistant.ingestion.unified_message import Source, UnifiedMessage
from assistant.main import configure_logging
from assistant.runtime import AssistantRuntime, get_runtime
from assistant.ui.styles import (
    ConnectionStatus,
    apply_custom_css,
    render_approval_html,
    render_empty_state_html,
    render_error_html,
    render_header_html,
    render_thinking_html,
    render_tool_html,
    sanitize_error_message,
)


def resolve_connection_status(name: str, status_data: Any) -> ConnectionStatus:
    """Safely converts runtime status data into a unified ConnectionStatus component."""
    if isinstance(status_data, ConnectionStatus):
        return status_data
    if isinstance(status_data, dict):
        status = status_data.get("status", "connected" if status_data.get("connected") else "disconnected")
        label = status_data.get("label")
        return ConnectionStatus(name=name, status=status, label=label)
    if isinstance(status_data, bool):
        return ConnectionStatus(name=name, status=status_data)
    if hasattr(status_data, "status"):
        return ConnectionStatus(name=name, status=getattr(status_data, "status", "offline"))
    return ConnectionStatus(name=name, status="offline")


@st.cache_resource
def get_cached_runtime() -> AssistantRuntime:
    """Returns a single, process-wide AssistantRuntime instance.

    Streamlit's cache_resource ensures this is executed only once across
    reruns and sessions, preventing duplicate initialization and duplicate
    background worker threads.
    """
    configure_logging()
    runtime = get_runtime()
    if not runtime.is_started:
        runtime.start()
    return runtime


def init_session_state() -> None:
    """Initializes UI session state for the chat interface."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = f"streamlit_{uuid.uuid4().hex[:8]}"
    if "approval_states" not in st.session_state:
        st.session_state["approval_states"] = {}


def render_approval_placeholder(approval: dict[str, Any]) -> None:
    """Renders the approval card for a given approval dictionary."""
    tid = approval.get("task_id", "")
    status = st.session_state.get("approval_states", {}).get(tid, {}).get("status", "pending")
    st.markdown(
        render_approval_html(
            task_type=approval.get("task_type", ""),
            target=approval.get("target", ""),
            draft=approval.get("draft", ""),
            task_id=tid,
            status=status,
        ),
        unsafe_allow_html=True,
    )


def render_interactive_approval_card(approval: dict[str, Any], runtime: AssistantRuntime) -> None:
    """Renders an interactive approval card with Approve/Reject controls.

    Enforces duplicate-action protection and handles resolution state.
    """
    tid = approval.get("task_id", "")
    appr_states = st.session_state.setdefault("approval_states", {})
    if tid not in appr_states:
        appr_states[tid] = {
            "status": "pending",
            "task_id": tid,
            "task_type": approval.get("task_type", ""),
            "target": approval.get("target", ""),
            "draft": approval.get("draft", ""),
            "conversation_id": approval.get("conversation_id", st.session_state.get("conversation_id", "")),
        }

    current_state = appr_states[tid]
    status = current_state.get("status", "pending")

    # Render structured HTML card
    st.markdown(
        render_approval_html(
            task_type=current_state.get("task_type", ""),
            target=current_state.get("target", ""),
            draft=current_state.get("draft", ""),
            task_id=tid,
            status=status,
        ),
        unsafe_allow_html=True,
    )

    # If pending, provide Approve and Reject buttons
    if status == "pending":
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("✓ Approve", key=f"approve_{tid}", type="primary", use_container_width=True):
                process_approval_decision(tid, "approve", runtime)
        with col2:
            if st.button("✕ Reject", key=f"reject_{tid}", type="secondary", use_container_width=True):
                process_approval_decision(tid, "reject", runtime)


def process_approval_decision(
    task_id: str,
    action: str,  # "approve" or "reject"
    runtime: AssistantRuntime,
) -> None:
    """Safely resumes the paused LangGraph execution via the orchestrator.

    Does NOT directly execute tools from Streamlit.
    Protects against duplicate clicks using state checks.
    """
    appr_states = st.session_state.get("approval_states", {})
    state = appr_states.get(task_id)
    if not state or state.get("status") != "pending":
        # Already processed or currently processing - ignore duplicate action
        return

    # Mark state as processing to disable duplicate clicks
    state["status"] = "processing"
    conversation_id = state.get("conversation_id") or st.session_state.get("conversation_id", "")

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        tools_container = st.container()
        response_placeholder = st.empty()

        thinking_placeholder.markdown(render_thinking_html(), unsafe_allow_html=True)

        has_error = False
        final_content = ""
        tools_executed: list[str] = []

        try:
            event_stream = runtime.orchestrator.resume_approval(
                conversation_id=conversation_id,
                action=action,
                task_id=task_id,
            )
            final_content, tools_executed, new_approval, has_error = handle_stream_events(
                event_stream=event_stream,
                thinking_placeholder=thinking_placeholder,
                tools_container=tools_container,
                response_placeholder=response_placeholder,
            )
        except Exception as exc:
            has_error = True
            thinking_placeholder.empty()
            error_msg = sanitize_error_message(str(exc))
            response_placeholder.markdown(render_error_html(error_msg), unsafe_allow_html=True)
            final_content = f"Sorry, I couldn't process that approval: {error_msg}"

        # Update resolution state
        if has_error:
            state["status"] = "failed"
            state["error"] = final_content
        elif action == "approve":
            state["status"] = "approved"
            state["result"] = final_content
        else:
            state["status"] = "rejected"
            state["result"] = final_content

        # Append assistant turn to chat history
        if final_content:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": final_content,
                "tools": tools_executed,
            })

    # Refresh UI so card updates and buttons disappear
    st.rerun()


@st.dialog("WhatsApp Connection")
def show_whatsapp_dialog(runtime: AssistantRuntime) -> None:
    wa_status = runtime.get_whatsapp_status()
    wa_conn = resolve_connection_status("WhatsApp", wa_status)
    st.markdown("#### WhatsApp Bridge")
    st.markdown(f"**Status:** {wa_conn.to_html(include_name=False)}", unsafe_allow_html=True)
    if wa_status.get("connected"):
        if wa_status.get("agent_chat_id"):
            st.markdown(f"Agent Chat: `{wa_status['agent_chat_id']}`")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Restart Bridge", key="wa_dlg_restart", use_container_width=True):
                runtime.whatsapp_connector.stop_bridge()
                runtime.whatsapp_connector.start_bridge(force=True)
                st.rerun()
        with col2:
            if st.button("Close", key="wa_dlg_close", use_container_width=True):
                st.rerun()
    else:
        if wa_status.get("running"):
            st.info("Baileys bridge process is running. Awaiting authentication...")
            qr_code = wa_status.get("qr")
            if qr_code:
                st.markdown("**Scan with WhatsApp on your phone:**")
                st.caption("WhatsApp → Linked Devices → Link a Device")
                try:
                    import urllib.parse
                    qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=220x220&data={urllib.parse.quote(qr_code)}"
                    st.image(qr_url, width=220)
                except Exception:
                    pass
                st.code(qr_code, language="text")
        else:
            st.caption("The Baileys bridge process is currently stopped.")
            if st.button("Start WhatsApp", key="wa_dlg_start", type="primary", use_container_width=True):
                runtime.whatsapp_connector.start_bridge(force=True)
                st.rerun()

        if st.button("Close", key="wa_dlg_close", use_container_width=True):
            st.rerun()


@st.dialog("Telegram Connection")
def show_telegram_dialog(runtime: AssistantRuntime) -> None:
    tg_status = runtime.get_telegram_status()
    tg_conn = resolve_connection_status("Telegram", tg_status)
    st.markdown("#### Telegram Account & Bot")
    st.markdown(f"**Status:** {tg_conn.to_html(include_name=False)}", unsafe_allow_html=True)
    if tg_status.get("connected"):
        if tg_status.get("bot", {}).get("connected"):
            st.markdown("✓ **Bot API**: Configured & Active")
        if tg_status.get("user", {}).get("connected"):
            st.markdown("✓ **Personal Account (Telethon)**: Authorized")
    else:
        user_info = tg_status.get("user", {})
        bot_info = tg_status.get("bot", {})
        if not bot_info.get("has_token") and not user_info.get("has_credentials"):
            st.caption("Neither `TELEGRAM_BOT_TOKEN` nor `TELEGRAM_API_ID` are configured in `.env`.")
        elif not user_info.get("authorized") and user_info.get("has_credentials"):
            st.caption(f"Personal account session `{user_info.get('session_file')}` requires interactive login.")
            st.info("Run `python scripts/setup_telegram_user.py` in your terminal to complete Telethon login.")

    if st.button("Close", key="tg_dlg_close", use_container_width=True):
        st.rerun()


@st.dialog("Android Device Gateway")
def show_android_dialog(runtime: AssistantRuntime) -> None:
    and_status = runtime.get_android_status()
    and_conn = resolve_connection_status("Android", and_status)
    st.markdown("#### Android Companion Device")
    st.markdown(f"**Status:** {and_conn.to_html(include_name=False)}", unsafe_allow_html=True)
    if and_status.get("connected"):
        dev_name = and_status.get("device_name") or "Android Phone"
        st.markdown(f"Device: **{dev_name}**")
    else:
        if and_status.get("server_running"):
            st.caption("WebSocket server is active. Awaiting device connection.")
        else:
            st.caption("WebSocket server is offline.")

    st.markdown("**Gateway WebSocket URL:**")
    st.code(and_status.get("ws_url", "ws://localhost:8765"), language="text")
    st.caption("Enter this address into the Android JOE AI companion app to connect.")

    if st.button("Close", key="and_dlg_close", use_container_width=True):
        st.rerun()


def render_sidebar(runtime: AssistantRuntime) -> None:
    """Renders the minimal, compact sidebar with New Chat and real connection status."""
    with st.sidebar:
        # Sleek New Chat button
        if st.button("＋ New Chat", use_container_width=True, type="primary"):
            st.session_state["messages"] = []
            st.session_state["conversation_id"] = f"streamlit_{uuid.uuid4().hex[:8]}"
            st.session_state["approval_states"] = {}
            st.rerun()

        st.markdown('<div class="sidebar-section-title">Connections</div>', unsafe_allow_html=True)

        chat_status_data = (
            runtime.get_chat_status()
            if hasattr(runtime, "get_chat_status") and callable(runtime.get_chat_status)
            else bool(runtime and getattr(runtime, "is_started", False))
        )
        chat_conn = resolve_connection_status("Chat", chat_status_data)

        wa_status = runtime.get_whatsapp_status()
        tg_status = runtime.get_telegram_status()
        and_status = runtime.get_android_status()

        wa_conn = resolve_connection_status("WhatsApp", wa_status)
        tg_conn = resolve_connection_status("Telegram", tg_status)
        and_conn = resolve_connection_status("Android", and_status)

        # 1. Chat connection status item
        st.markdown(
            f'<div class="sidebar-connection-item">'
            f'<span class="sidebar-conn-label">Chat</span>'
            f'{chat_conn.to_html(include_name=False)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 2. Interactive Connector buttons
        if st.button(f"WhatsApp   {wa_conn.dot} {wa_conn.sidebar_label}", key="btn_wa", use_container_width=True):
            show_whatsapp_dialog(runtime)

        if st.button(f"Telegram   {tg_conn.dot} {tg_conn.sidebar_label}", key="btn_tg", use_container_width=True):
            show_telegram_dialog(runtime)

        if st.button(f"Android   {and_conn.dot} {and_conn.sidebar_label}", key="btn_android", use_container_width=True):
            show_android_dialog(runtime)


def handle_stream_events(
    event_stream: Iterator[AgentStreamEvent],
    thinking_placeholder: Any,
    tools_container: Any,
    response_placeholder: Any,
) -> tuple[str, list[str], dict[str, Any] | None, bool]:
    """Consumes AgentStreamEvents and renders them in the modern UI.

    Maintains isolated tool-progress placeholders per tool call to prevent
    concurrent or sequential tools from overwriting each other.

    Returns:
        tuple of (final_content, tools_executed, approval_info, has_error)
    """
    accumulated_tokens: list[str] = []
    final_content: str = ""
    tools_executed: list[str] = []
    tool_placeholders: dict[str, tuple[Any, str]] = {}
    approval_info: dict[str, Any] | None = None
    has_error = False

    for event in event_stream:
        if event.type == "status":
            thinking_placeholder.markdown(render_thinking_html(), unsafe_allow_html=True)

        elif event.type == "tool_start":
            thinking_placeholder.empty()
            tool_name = event.tool_name or "tool"
            key = event.task_id or f"{tool_name}_{len(tools_executed)}"
            tool_ph = tools_container.empty()
            tool_placeholders[key] = (tool_ph, tool_name)
            tool_ph.markdown(render_tool_html(tool_name, completed=False), unsafe_allow_html=True)
            tools_executed.append(tool_name)

        elif event.type == "tool_complete":
            tool_name = event.tool_name or "tool"
            key = event.task_id or f"{tool_name}_{len(tools_executed) - 1}"
            if key in tool_placeholders:
                tool_ph, original_name = tool_placeholders[key]
                tool_ph.markdown(render_tool_html(original_name, completed=True), unsafe_allow_html=True)

        elif event.type == "token":
            thinking_placeholder.empty()
            if event.content:
                accumulated_tokens.append(event.content)
                response_placeholder.markdown("".join(accumulated_tokens) + "▌")

        elif event.type == "approval_required":
            thinking_placeholder.empty()
            for _k, (_ph, _orig_name) in tool_placeholders.items():
                _ph.markdown(render_tool_html(_orig_name, pending=True), unsafe_allow_html=True)
            tid = event.task_id or f"task_{uuid.uuid4().hex[:6]}"
            approval_info = {
                "task_id": tid,
                "task_type": event.task_type or "",
                "target": event.target or "",
                "draft": event.draft or "",
                "conversation_id": st.session_state.get("conversation_id", ""),
            }
            # Record into session state approval registry
            appr_states = st.session_state.setdefault("approval_states", {})
            if tid not in appr_states:
                appr_states[tid] = {
                    "status": "pending",
                    **approval_info,
                }
            response_placeholder.markdown(
                render_approval_html(
                    task_type=approval_info["task_type"],
                    target=approval_info["target"],
                    draft=approval_info["draft"],
                    task_id=approval_info["task_id"],
                    status="pending",
                ),
                unsafe_allow_html=True,
            )

        elif event.type == "error":
            thinking_placeholder.empty()
            has_error = True
            clean_err = sanitize_error_message(event.message or "Error")
            response_placeholder.markdown(render_error_html(clean_err), unsafe_allow_html=True)
            break

        elif event.type == "complete":
            thinking_placeholder.empty()
            final_content = event.final_response if event.final_response is not None else "".join(accumulated_tokens)
            if final_content and not approval_info:
                response_placeholder.markdown(final_content)

    if not final_content and accumulated_tokens and not approval_info:
        final_content = "".join(accumulated_tokens)

    return final_content, tools_executed, approval_info, has_error


def main() -> None:
    st.set_page_config(
        page_title="JOE AI — Personal Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. Apply sleek dark theme
    apply_custom_css()

    # 2. Access singleton runtime and initialize UI state
    runtime = get_cached_runtime()
    init_session_state()

    # 3. Render sidebar
    render_sidebar(runtime)

    # 4. Render header (strictly derived from runtime network and started availability)
    chat_status_data = (
        runtime.get_chat_status()
        if hasattr(runtime, "get_chat_status") and callable(runtime.get_chat_status)
        else bool(runtime and getattr(runtime, "is_started", False))
    )
    chat_conn = resolve_connection_status("Chat", chat_status_data)
    st.markdown(render_header_html(chat_conn), unsafe_allow_html=True)

    # 5. Render conversation history or empty state
    messages = st.session_state.get("messages", [])
    if not messages:
        st.markdown(render_empty_state_html(), unsafe_allow_html=True)
    else:
        for msg in messages:
            with st.chat_message(msg["role"]):
                approval = msg.get("approval")
                appr_tid = approval.get("task_id") if approval else None
                appr_state = st.session_state.get("approval_states", {}).get(appr_tid, {}) if appr_tid else {}
                appr_status = appr_state.get("status", "pending") if approval else None
                appr_result = str(appr_state.get("result", ""))

                # Render tool executions
                for tool_name in msg.get("tools", []):
                    if approval:
                        if appr_status == "pending":
                            st.markdown(render_tool_html(tool_name, pending=True), unsafe_allow_html=True)
                        elif appr_status == "approved" and not appr_state.get("error") and not appr_result.startswith("Approved, but failed") and "failed" not in appr_result.lower():
                            st.markdown(render_tool_html(tool_name, completed=True), unsafe_allow_html=True)
                        else:
                            st.markdown(render_tool_html(tool_name, failed=True), unsafe_allow_html=True)
                    else:
                        st.markdown(render_tool_html(tool_name, completed=True), unsafe_allow_html=True)

                # Render interactive approval card if present
                if approval:
                    render_interactive_approval_card(approval, runtime)

                # Render text response
                if msg.get("content"):
                    st.markdown(msg["content"])

    # 6. Accept user input
    prompt = st.chat_input("Ask Joe anything...")
    if prompt:
        prompt_text = prompt.strip()
        if not prompt_text:
            return

        # Record and render user message
        st.session_state["messages"].append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Build UnifiedMessage
        unified_msg = UnifiedMessage(
            source=Source.CLI,
            conversation_id=st.session_state["conversation_id"],
            sender="user",
            content=prompt_text,
        )

        # Stream assistant response
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            tools_container = st.container()
            response_placeholder = st.empty()

            try:
                event_stream = runtime.orchestrator.stream_agent_command(unified_msg)
                final_content, tools_executed, approval_info, has_error = handle_stream_events(
                    event_stream=event_stream,
                    thinking_placeholder=thinking_placeholder,
                    tools_container=tools_container,
                    response_placeholder=response_placeholder,
                )
            except Exception as exc:
                has_error = True
                thinking_placeholder.empty()
                clean_err = sanitize_error_message(str(exc))
                response_placeholder.markdown(render_error_html(clean_err), unsafe_allow_html=True)
                final_content, tools_executed, approval_info = "", [], None

            # Record assistant turn into history
            if not has_error and (final_content or approval_info or tools_executed):
                assistant_entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": final_content,
                    "tools": tools_executed,
                }
                if approval_info:
                    assistant_entry["approval"] = approval_info
                st.session_state["messages"].append(assistant_entry)

                # If an approval was emitted, rerun so interactive buttons appear immediately
                if approval_info:
                    st.rerun()


if __name__ == "__main__":
    main()
