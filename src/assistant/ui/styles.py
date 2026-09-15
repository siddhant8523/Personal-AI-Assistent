"""
UI Styling & Presentation Helpers (src/assistant/ui/styles.py)
==============================================================
Provides modern dark interface styling, tool progress formatting,
sanitized error presentations, and component templates for the Streamlit UI.
"""

from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st

# Custom Dark Theme CSS
CUSTOM_CSS = """
<style>
/* Global page styling */
.stApp {
    background-color: #0E1117;
    color: #E6EDF3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Header Container */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    margin-bottom: 1.5rem;
}

.header-brand {
    display: flex;
    flex-direction: column;
}

.app-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F0F6FC;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.2;
}

.app-subtitle {
    font-size: 0.8rem;
    color: #8B949E;
    margin-top: 2px;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    background: #0D1117;
    border: 1px solid #30363D;
}

.status-badge.online,
.status-badge.connected {
    color: #3FB950;
    border-color: #23863666;
}

.status-badge.connecting,
.status-badge.reconnecting {
    color: #E3B341;
    border-color: #D2992266;
}

.status-badge.offline,
.status-badge.disconnected {
    color: #8B949E;
    border-color: #30363D;
}

.status-badge.auth-required,
.status-badge.error {
    color: #FF7B72;
    border-color: #F8514966;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.status-dot.online,
.status-dot.connected {
    background-color: #2EA043;
    box-shadow: 0 0 8px rgba(46, 160, 67, 0.6);
}

.status-dot.connecting,
.status-dot.reconnecting {
    background-color: #D29922;
    box-shadow: 0 0 8px rgba(210, 153, 34, 0.6);
    animation: pulse 1.5s infinite ease-in-out;
}

.status-dot.offline,
.status-dot.disconnected {
    background-color: #6E7681;
}

.status-dot.auth-required,
.status-dot.error {
    background-color: #F85149;
    box-shadow: 0 0 8px rgba(248, 81, 73, 0.6);
}

.sidebar-connection-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 12px;
    border-radius: 8px;
    margin-bottom: 6px;
    background: #0D1117;
    border: 1px solid #30363D;
}

.sidebar-conn-label {
    font-size: 0.88rem;
    font-weight: 500;
    color: #C9D1D9;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #161B22 !important;
    border-right: 1px solid #30363D;
}

.sidebar-section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #8B949E;
    letter-spacing: 0.5px;
    margin: 1.2rem 0 0.5rem 0;
}

.sidebar-item-placeholder {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    font-size: 0.85rem;
    color: #C9D1D9;
    border-radius: 6px;
    margin-bottom: 4px;
    background: transparent;
    transition: background 0.15s ease;
}

.sidebar-item-placeholder:hover {
    background: #21262D;
}

.badge-muted {
    font-size: 0.7rem;
    color: #8B949E;
    background: #21262D;
    padding: 2px 7px;
    border-radius: 10px;
    border: 1px solid #30363D;
}

/* Thinking Indicator */
.thinking-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 6px;
    background: #161B22;
    border: 1px solid #30363D;
    color: #58A6FF;
    font-size: 0.88rem;
    margin-bottom: 0.5rem;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #58A6FF;
    animation: pulse 1.5s infinite ease-in-out;
}

@keyframes pulse {
    0% { transform: scale(0.85); opacity: 0.5; }
    50% { transform: scale(1.15); opacity: 1; }
    100% { transform: scale(0.85); opacity: 0.5; }
}

/* Tool Progress Cards */
.tool-card {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 4px 0 8px 0;
    font-size: 0.86rem;
    font-family: monospace;
}

.tool-card.running {
    background: #161B22;
    border: 1px solid #388BFD55;
    color: #58A6FF;
}

.tool-card.completed {
    background: #161B22;
    border: 1px solid #23863655;
    color: #3FB950;
}

.tool-card.pending {
    background: #161B22;
    border: 1px solid #D2992288;
    color: #E3B341;
}

.tool-card.failed {
    background: #161B22;
    border: 1px solid #F8514988;
    color: #F85149;
}

.tool-icon {
    font-size: 0.95rem;
}

/* Approval Card */
.approval-card {
    background: #161B22;
    border: 1px solid #D2992288;
    border-left: 4px solid #D29922;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 0;
    color: #E6EDF3;
}

.approval-card.processing {
    border-color: #388BFD88;
    border-left-color: #58A6FF;
}

.approval-card.approved {
    border-color: #23863688;
    border-left-color: #2EA043;
}

.approval-card.rejected {
    border-color: #6E768188;
    border-left-color: #8B949E;
}

.approval-card.failed {
    border-color: #F8514988;
    border-left-color: #F85149;
}

.approval-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.approval-title {
    font-weight: 600;
    color: #E3B341;
    font-size: 0.95rem;
}

.approval-title.processing { color: #58A6FF; }
.approval-title.approved { color: #3FB950; }
.approval-title.rejected { color: #8B949E; }
.approval-title.failed { color: #FF7B72; }

.approval-badge {
    background: #2D2200;
    color: #E3B341;
    border: 1px solid #D2992266;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 500;
}

.approval-badge.processing {
    background: #0C2D6B;
    color: #58A6FF;
    border-color: #388BFD66;
}

.approval-badge.approved {
    background: #033A16;
    color: #3FB950;
    border-color: #23863666;
}

.approval-badge.rejected {
    background: #21262D;
    color: #8B949E;
    border-color: #6E768166;
}

.approval-badge.failed {
    background: #490202;
    color: #FF7B72;
    border-color: #F8514966;
}

.approval-details {
    font-size: 0.88rem;
    line-height: 1.5;
    color: #C9D1D9;
}

.approval-row {
    margin: 3px 0;
}

.approval-label {
    color: #8B949E;
    font-weight: 500;
    margin-right: 6px;
}

.approval-draft {
    margin-top: 8px;
    padding: 8px 12px;
    background: #0D1117;
    border-radius: 6px;
    border: 1px solid #30363D;
    font-family: monospace;
    font-size: 0.84rem;
    color: #F0F6FC;
}

.approval-footer {
    margin-top: 10px;
    font-size: 0.75rem;
    color: #8B949E;
    font-style: italic;
}

/* Error Card */
.error-card {
    background: #161B22;
    border: 1px solid #F8514966;
    border-left: 4px solid #F85149;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #FF7B72;
    font-size: 0.9rem;
}

/* Empty State / Subtle Welcome */
.empty-state {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 40vh;
    text-align: center;
}

.empty-prompt {
    font-size: 1.45rem;
    font-weight: 500;
    color: #484F58;
    letter-spacing: -0.3px;
}

/* Sidebar button styling */
[data-testid="stSidebar"] button {
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background-color: #21262D !important;
    color: #F0F6FC !important;
    border: 1px solid #30363D !important;
}

[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
    background-color: #30363D !important;
    border-color: #8B949E !important;
    color: #FFFFFF !important;
}

/* Connection Modal & Info Styles */
.conn-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 500;
}
.conn-status-badge.connected {
    background: #033A16;
    color: #3FB950;
    border: 1px solid #238636;
}
.conn-status-badge.disconnected {
    background: #21262D;
    color: #8B949E;
    border: 1px solid #30363D;
}
.conn-info-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 14px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #C9D1D9;
}
.conn-url-code {
    background: #0D1117;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 6px 10px;
    font-family: monospace;
    font-size: 0.85rem;
    color: #58A6FF;
    word-break: break-all;
    margin-top: 6px;
}
</style>
"""


def apply_custom_css() -> None:
    """Injects the custom dark interface CSS into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Friendly Tool Names
# ------------------------------------------------------------------------------

_TOOL_NAME_MAP: dict[str, tuple[str, str]] = {
    "query_priority_inbox": ("Checking priority inbox...", "Priority inbox checked"),
    "query_agenda": ("Checking your agenda...", "Agenda checked"),
    "send_message": ("Preparing message...", "Message prepared"),
    "send_whatsapp_message": ("Preparing WhatsApp message...", "WhatsApp message prepared"),
    "send_telegram_message": ("Preparing Telegram message...", "Telegram message prepared"),
    "send_email": ("Preparing email...", "Email prepared"),
    "read_sms": ("Reading messages...", "Messages read"),
    "send_sms": ("Preparing SMS...", "SMS prepared"),
    "sms.send": ("Preparing SMS...", "SMS prepared"),
    "sms.read": ("Reading messages...", "Messages read"),
    "find_contact": ("Looking up contact...", "Contact found"),
    "contact.find": ("Looking up contact...", "Contact found"),
    "make_call": ("Placing phone call...", "Phone call placed"),
    "call.make": ("Placing phone call...", "Phone call placed"),
    "set_alarm": ("Setting alarm...", "Alarm set"),
    "alarm.set": ("Setting alarm...", "Alarm set"),
    "list_alarms": ("Retrieving alarms...", "Alarms retrieved"),
    "alarm.list": ("Retrieving alarms...", "Alarms retrieved"),
    "cancel_alarm": ("Cancelling alarm...", "Alarm cancelled"),
    "alarm.cancel": ("Cancelling alarm...", "Alarm cancelled"),
    "set_timer": ("Setting timer...", "Timer set"),
    "timer.set": ("Setting timer...", "Timer set"),
    "search_web": ("Searching the web...", "Web search complete"),
    "read_web_page": ("Reading web page...", "Web page read"),
    "read_file": ("Reading file...", "File read"),
    "file.read": ("Reading file...", "File read"),
    "upload_file": ("Uploading file...", "File uploaded"),
    "file.upload": ("Uploading file...", "File uploaded"),
    "file.list": ("Listing files...", "Files listed"),
    "file.find": ("Searching files...", "Files found"),
    "intent.open_app": ("Opening app...", "App opened"),
    "intent.execute": ("Executing intent...", "Intent executed"),
}


def get_tool_friendly_name(tool_name: str, completed: bool = False) -> str:
    """Transforms raw tool names into polished, human-friendly descriptions.

    Args:
        tool_name: Identifier of the tool (e.g., 'query_priority_inbox').
        completed: If True, returns completed text; otherwise running text.

    Returns:
        A readable description string.
    """
    if not tool_name:
        return "Tool completed" if completed else "Running tool..."

    clean_name = str(tool_name).strip()
    if clean_name in _TOOL_NAME_MAP:
        running_text, completed_text = _TOOL_NAME_MAP[clean_name]
        return completed_text if completed else running_text

    # Fallback for unknown tools: convert snake_case or dot.case to words
    words = clean_name.replace("_", " ").replace(".", " ").strip()
    if completed:
        return f"Completed {words}"
    return f"Running {words}..."


# ------------------------------------------------------------------------------
# Error Sanitization
# ------------------------------------------------------------------------------

def sanitize_error_message(error: Any) -> str:
    """Strips Python tracebacks, raw exception types, and internal paths.

    Returns a clean, user-friendly message suitable for presentation.
    """
    if not error:
        return "An unexpected error occurred. Please try again."

    err_text = str(error).strip()

    # Check for raw traceback indicators
    if "Traceback (most recent call last)" in err_text or 'File "' in err_text:
        return "I couldn't reach the AI service right now. Please try again in a moment."

    # Remove raw exception class names like "groq.APIConnectionError: "
    err_text = re.sub(r"^[a-zA-Z0-9_\.]*(?:Error|Exception|Fault):\s*", "", err_text)

    # Check for network/connection failure patterns
    lower = err_text.lower()
    if any(p in lower for p in ("connection error", "failed to establish", "timeout", "rate limit")):
        return "I couldn't reach the AI service right now. Please try again in a moment."

    return err_text or "An unexpected error occurred. Please try again."


# ------------------------------------------------------------------------------
# Unified Connection Status Component
# ------------------------------------------------------------------------------

class ConnectionStatus:
    """Reusable connection status component for Chat, WhatsApp, Telegram, and Android."""

    def __init__(
        self,
        name: str,
        status: str | bool,
        details: str | None = None,
        label: str | None = None,
    ):
        self.name = (name or "").strip()
        self.details = details

        # Normalize input status
        if isinstance(status, bool):
            raw = "online" if (status and self.name.lower() == "chat") else ("connected" if status else "disconnected")
        else:
            raw = (str(status) if status is not None else "").strip().lower()

        self.raw_status = raw

        # State classification
        if raw in ("online", "chat_online"):
            self.css_class = "online"
            self.dot = "🟢"
            self.label = label or "Online"
            self.is_connected = True
        elif raw in ("connected", "ready"):
            self.css_class = "connected"
            self.dot = "🟢"
            self.label = label or "Connected"
            self.is_connected = True
        elif raw in ("connecting", "reconnecting"):
            self.css_class = "connecting"
            self.dot = "🟡"
            self.label = label or ("Reconnecting" if "reconnect" in raw else "Connecting")
            self.is_connected = False
        elif raw in ("offline", "network_offline"):
            self.css_class = "offline"
            self.dot = "🔴"
            self.label = label or "Offline"
            self.is_connected = False
        elif raw in ("disconnected", "not_connected", "not connected"):
            self.css_class = "disconnected"
            self.dot = "🔴"
            self.label = label or "Not connected"
            self.is_connected = False
        elif raw in ("auth_required", "auth required", "unauthorized"):
            self.css_class = "auth-required"
            self.dot = "⚠️"
            self.label = label or "Auth Required"
            self.is_connected = False
        elif raw in ("error", "failed"):
            self.css_class = "error"
            self.dot = "⚠️"
            self.label = label or "Error"
            self.is_connected = False
        else:
            self.css_class = "offline"
            self.dot = "🔴"
            self.label = label or raw.capitalize() or "Offline"
            self.is_connected = False

    @property
    def sidebar_label(self) -> str:
        """Label formatted for sidebar buttons."""
        return self.label

    def to_html(self, include_name: bool = False) -> str:
        """Renders the unified status badge HTML using the shared dark theme."""
        text = f"{self.name}: {self.label}" if include_name else self.label
        safe_text = html.escape(text)
        return (
            f'<div class="status-badge {self.css_class}">'
            f'<span class="status-dot {self.css_class}"></span>'
            f'<span>{safe_text}</span>'
            f'</div>'
        )

    def __str__(self) -> str:
        return self.to_html(include_name=False)

    def __repr__(self) -> str:
        return f"ConnectionStatus(name={self.name!r}, status={self.css_class!r}, label={self.label!r})"


# ------------------------------------------------------------------------------
# HTML Renderers
# ------------------------------------------------------------------------------

def render_header_html(
    chat_status: ConnectionStatus | bool | None = None,
    connectors: list[ConnectionStatus] | None = None,
    *,
    is_online: bool | None = None,
) -> str:
    """Renders the top application header with the real-time runtime status."""
    target_status = is_online if is_online is not None else (chat_status if chat_status is not None else True)

    if isinstance(target_status, bool):
        chat_conn = ConnectionStatus(name="Chat", status="online" if target_status else "offline")
    elif isinstance(target_status, ConnectionStatus):
        chat_conn = target_status
    else:
        chat_conn = ConnectionStatus(name="Chat", status=str(target_status))

    if not connectors:
        badge_html = chat_conn.to_html(include_name=False)
    else:
        badges = [chat_conn.to_html(include_name=True)] + [c.to_html(include_name=True) for c in connectors]
        badge_html = f'<div class="header-badges-group">{"".join(badges)}</div>'

    return f"""
    <div class="header-container">
        <div class="header-brand">
            <h1 class="app-title">JOE AI</h1>
            <span class="app-subtitle">Personal AI Assistant</span>
        </div>
        {badge_html}
    </div>
    """


def render_thinking_html() -> str:
    """Renders the subtle animated thinking indicator."""
    return """
    <div class="thinking-indicator">
        <span class="pulse-dot"></span>
        <span>Thinking...</span>
    </div>
    """


def render_tool_html(
    tool_name: str,
    completed: bool = False,
    pending: bool = False,
    failed: bool = False,
) -> str:
    """Renders a single tool card transitioning between running, pending, completed, and failed."""
    friendly_label = get_tool_friendly_name(tool_name, completed=completed)
    if failed:
        icon = "✕"
        css_class = "failed"
        safe_label = html.escape(friendly_label)
    elif pending:
        icon = "⏳"
        css_class = "pending"
        running_label = get_tool_friendly_name(tool_name, completed=False)
        safe_label = html.escape(running_label) + " (Awaiting approval)"
    elif completed:
        icon = "✓"
        css_class = "completed"
        safe_label = html.escape(friendly_label)
    else:
        icon = "◌"
        css_class = "running"
        safe_label = html.escape(friendly_label)
    return f"""
    <div class="tool-card {css_class}">
        <span class="tool-icon">{icon}</span>
        <span>{safe_label}</span>
    </div>
    """


def render_approval_html(
    task_type: str,
    target: str,
    draft: str,
    task_id: str | None = None,
    status: str = "pending",
) -> str:
    """Renders the structured human-in-the-loop approval card with resolution state."""
    safe_action = html.escape(task_type or "Action")
    safe_target = html.escape(target or "N/A")
    safe_task_id = html.escape(task_id or "N/A")
    safe_draft = html.escape(draft or "None")

    norm_status = (status or "pending").strip().lower()
    if norm_status == "approved":
        title_text = "✓ Action Approved"
        badge_text = "Approved"
        footer_text = "Action was approved and resumed."
        card_class = "approved"
    elif norm_status == "rejected":
        title_text = "✕ Action Rejected"
        badge_text = "Rejected"
        footer_text = "Action was rejected. No message or tool was executed."
        card_class = "rejected"
    elif norm_status == "processing":
        title_text = "◌ Processing Decision..."
        badge_text = "Processing"
        footer_text = "Communicating with Agent Core..."
        card_class = "processing"
    elif norm_status == "failed":
        title_text = "⚠️ Approval Failed"
        badge_text = "Failed"
        footer_text = "Failed to process decision. No action was executed."
        card_class = "failed"
    else:
        title_text = "⚠️ Approval Required"
        badge_text = "Awaiting approval"
        footer_text = "Action is paused awaiting user confirmation. No action was executed."
        card_class = ""

    return f"""
    <div class="approval-card {card_class}">
        <div class="approval-header">
            <span class="approval-title {card_class}">{title_text}</span>
            <span class="approval-badge {card_class}">{badge_text}</span>
        </div>
        <div class="approval-details">
            <div class="approval-row"><span class="approval-label">Action:</span><b>{safe_action}</b></div>
            <div class="approval-row"><span class="approval-label">Target:</span>{safe_target}</div>
            <div class="approval-row"><span class="approval-label">Task ID:</span><code>{safe_task_id}</code></div>
            <div class="approval-draft">{safe_draft}</div>
        </div>
        <div class="approval-footer">
            {footer_text}
        </div>
    </div>
    """


def render_error_html(error_message: str) -> str:
    """Renders a sanitized user-friendly error card."""
    sanitized = sanitize_error_message(error_message)
    safe_text = html.escape(sanitized)
    return f"""
    <div class="error-card">
        {safe_text}
    </div>
    """


def render_empty_state_html() -> str:
    """Renders a subtle, clean prompt when no conversation messages exist."""
    return """
    <div class="empty-state">
        <div class="empty-prompt">How can I help you today?</div>
    </div>
    """
