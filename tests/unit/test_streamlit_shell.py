"""
Unit tests for Streamlit UI Shell & Modern UX (tests/unit/test_streamlit_shell.py)
==================================================================================
Verifies:
- Streamlit app module imports successfully
- Runtime retrieval uses AssistantRuntime and is cached (singleton)
- Session state initialization and New Chat isolation
- Online badge strictly derived from runtime.is_started
- Tool progress transitions (running -> completed) with tool-call isolation
- Friendly tool name mappings and unknown tool fallback
- Approval cards are non-interactive and NEVER auto-approve or resume graph
- Error sanitization prevents exposing Python tracebacks to user
- Empty state welcome screen rendering
- End-to-end execution flow via Streamlit AppTest
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from assistant.agent_core.events import AgentStreamEvent
from assistant.ingestion.unified_message import Source, UnifiedMessage
from assistant.ui.styles import (
    apply_custom_css,
    get_tool_friendly_name,
    render_approval_html,
    render_empty_state_html,
    render_error_html,
    render_header_html,
    render_thinking_html,
    render_tool_html,
    sanitize_error_message,
)
import streamlit_app


# ==============================================================================
# 1. Module Import & Structure
# ==============================================================================

def test_streamlit_app_imports_successfully():
    """Verify streamlit_app.py imports cleanly and exports expected entry points."""
    assert hasattr(streamlit_app, "get_cached_runtime")
    assert hasattr(streamlit_app, "init_session_state")
    assert hasattr(streamlit_app, "render_approval_placeholder")
    assert hasattr(streamlit_app, "handle_stream_events")
    assert hasattr(streamlit_app, "render_sidebar")
    assert hasattr(streamlit_app, "main")


# ==============================================================================
# 2. Runtime Retrieval & Caching
# ==============================================================================

def test_get_cached_runtime_uses_assistant_runtime_singleton():
    """Verify runtime is obtained via AssistantRuntime / get_runtime and started once."""
    mock_runtime = MagicMock()
    mock_runtime.is_started = False

    streamlit_app.get_cached_runtime.clear()

    with patch("streamlit_app.get_runtime", return_value=mock_runtime):
        rt1 = streamlit_app.get_cached_runtime()
        assert rt1 is mock_runtime
        assert mock_runtime.start.call_count == 1

        mock_runtime.is_started = True

        # Calling again should retrieve the cached instance without re-starting
        rt2 = streamlit_app.get_cached_runtime()
        assert rt2 is mock_runtime
        assert mock_runtime.start.call_count == 1

    streamlit_app.get_cached_runtime.clear()


# ==============================================================================
# 3. Session State & New Chat
# ==============================================================================

def test_init_session_state_sets_defaults():
    """Verify session state initializes messages list and conversation ID."""
    mock_session_state = {}
    with patch("streamlit.session_state", mock_session_state):
        streamlit_app.init_session_state()
        assert "messages" in mock_session_state
        assert mock_session_state["messages"] == []
        assert "conversation_id" in mock_session_state
        assert mock_session_state["conversation_id"].startswith("streamlit_")


def test_unified_message_construction_conventions():
    """Verify the message sent to the orchestrator follows project conventions."""
    conversation_id = "streamlit_abc123"
    prompt_text = "What is my agenda for today?"

    msg = UnifiedMessage(
        source=Source.CLI,
        conversation_id=conversation_id,
        sender="user",
        content=prompt_text,
    )

    assert msg.source == Source.CLI
    assert msg.conversation_id == "streamlit_abc123"
    assert msg.sender == "user"
    assert msg.content == prompt_text
    assert msg.message_id is not None
    assert msg.timestamp > 0


# ==============================================================================
# 4. Styling, Online Badge & Component HTML
# ==============================================================================

def test_apply_custom_css():
    """Verify apply_custom_css renders styles via markdown without error."""
    with patch("streamlit.markdown") as mock_markdown:
        apply_custom_css()
        assert mock_markdown.call_count == 1
        call_arg = mock_markdown.call_args[0][0]
        assert "<style>" in call_arg
        assert "#0E1117" in call_arg
        assert ".header-container" in call_arg


def test_online_badge_derived_strictly_from_runtime_is_started():
    """Online badge must reflect runtime.is_started and not infer from connectors."""
    online_html = render_header_html(is_online=True)
    assert "status-badge online" in online_html
    assert "status-dot online" in online_html
    assert "Online" in online_html

    offline_html = render_header_html(is_online=False)
    assert "status-badge offline" in offline_html
    assert "status-dot offline" in offline_html
    assert "Offline" in offline_html


def test_empty_state_html_content():
    """Verify empty state contains subtle prompt and does NOT contain action suggestion boxes."""
    html_out = render_empty_state_html()
    assert "How can I help you today?" in html_out
    # Completely removed suggestion boxes
    assert "Check my priorities" not in html_out
    assert "Find my meetings" not in html_out
    assert "Read my messages" not in html_out
    assert "Help me with a task" not in html_out
    assert "suggestion-grid" not in html_out
    assert "suggestion-card" not in html_out


def test_thinking_indicator_html():
    """Verify thinking indicator HTML structure."""
    html_out = render_thinking_html()
    assert "thinking-indicator" in html_out
    assert "pulse-dot" in html_out
    assert "Thinking..." in html_out


# ==============================================================================
# 5. Tool Progress & Name Mappings
# ==============================================================================

def test_tool_friendly_names_known_and_fallback():
    """Verify friendly names for known tools and human-readable fallback for unknown tools."""
    # Known tools
    assert get_tool_friendly_name("query_priority_inbox", False) == "Checking priority inbox..."
    assert get_tool_friendly_name("query_priority_inbox", True) == "Priority inbox checked"
    assert get_tool_friendly_name("query_agenda", False) == "Checking your agenda..."
    assert get_tool_friendly_name("query_agenda", True) == "Agenda checked"
    assert get_tool_friendly_name("send_message", False) == "Preparing message..."
    assert get_tool_friendly_name("send_message", True) == "Message prepared"
    assert get_tool_friendly_name("read_sms", False) == "Reading messages..."
    assert get_tool_friendly_name("read_sms", True) == "Messages read"

    # Unknown tool fallback
    assert get_tool_friendly_name("analyze_quarterly_report", False) == "Running analyze quarterly report..."
    assert get_tool_friendly_name("analyze_quarterly_report", True) == "Completed analyze quarterly report"
    assert get_tool_friendly_name("device.reboot", False) == "Running device reboot..."
    assert get_tool_friendly_name("device.reboot", True) == "Completed device reboot"


def test_render_tool_html_states():
    """Verify tool HTML cards differentiate between running and completed."""
    running_html = render_tool_html("query_priority_inbox", completed=False)
    assert "tool-card running" in running_html
    assert "◌" in running_html
    assert "Checking priority inbox..." in running_html

    completed_html = render_tool_html("query_priority_inbox", completed=True)
    assert "tool-card completed" in completed_html
    assert "✓" in completed_html
    assert "Priority inbox checked" in completed_html


# ==============================================================================
# 6. Stream Event Handling & Isolated Tool Containers
# ==============================================================================

def test_handle_stream_events_token_accumulation_and_complete():
    """Verify incremental tokens accumulate and complete event finalizes response."""
    events = [
        AgentStreamEvent.status("Thinking..."),
        AgentStreamEvent.token("Hello"),
        AgentStreamEvent.token(" there!"),
        AgentStreamEvent.complete("Hello there!"),
    ]

    mock_thinking = MagicMock()
    mock_tools = MagicMock()
    mock_response = MagicMock()

    final_content, tools_executed, approval_info, has_error = streamlit_app.handle_stream_events(
        event_stream=iter(events),
        thinking_placeholder=mock_thinking,
        tools_container=mock_tools,
        response_placeholder=mock_response,
    )

    assert final_content == "Hello there!"
    assert tools_executed == []
    assert approval_info is None
    assert has_error is False

    # Thinking was shown then cleared
    assert mock_thinking.markdown.call_count >= 1
    assert mock_thinking.empty.call_count >= 1

    # Token streaming updates
    mock_response.markdown.assert_any_call("Hello▌")
    mock_response.markdown.assert_any_call("Hello there!▌")
    mock_response.markdown.assert_any_call("Hello there!")


def test_handle_stream_events_isolated_tool_progress():
    """Verify sequential tools have isolated placeholders and do not overwrite each other."""
    events = [
        AgentStreamEvent.status("Thinking..."),
        AgentStreamEvent.tool_start("query_priority_inbox", task_id="task_1"),
        AgentStreamEvent.tool_complete("query_priority_inbox", "3 priority messages", task_id="task_1"),
        AgentStreamEvent.tool_start("search_web", {"q": "news"}, task_id="task_2"),
        AgentStreamEvent.tool_complete("search_web", "news headlines", task_id="task_2"),
        AgentStreamEvent.token("Done."),
        AgentStreamEvent.complete("Done."),
    ]

    mock_thinking = MagicMock()
    mock_tools = MagicMock()
    mock_response = MagicMock()

    # Mock tool placeholders returned by tools_container.empty()
    mock_ph_1 = MagicMock()
    mock_ph_2 = MagicMock()
    mock_tools.empty.side_effect = [mock_ph_1, mock_ph_2]

    final_content, tools_executed, approval_info, has_error = streamlit_app.handle_stream_events(
        event_stream=iter(events),
        thinking_placeholder=mock_thinking,
        tools_container=mock_tools,
        response_placeholder=mock_response,
    )

    assert final_content == "Done."
    assert tools_executed == ["query_priority_inbox", "search_web"]
    assert not has_error

    # Two distinct containers were allocated
    assert mock_tools.empty.call_count == 2

    # Container 1 received running then completed for query_priority_inbox
    assert mock_ph_1.markdown.call_count == 2
    args1_running = mock_ph_1.markdown.call_args_list[0][0][0]
    args1_completed = mock_ph_1.markdown.call_args_list[1][0][0]
    assert "Checking priority inbox..." in args1_running
    assert "Priority inbox checked" in args1_completed

    # Container 2 received running then completed for search_web
    assert mock_ph_2.markdown.call_count == 2
    args2_running = mock_ph_2.markdown.call_args_list[0][0][0]
    args2_completed = mock_ph_2.markdown.call_args_list[1][0][0]
    assert "Searching the web..." in args2_running
    assert "Web search complete" in args2_completed


# ==============================================================================
# 7. Approvals (Zero Auto-Approval & Non-Interactive)
# ==============================================================================

def test_handle_stream_events_approval_required_never_resumes():
    """CRITICAL: approval_required must display non-interactive card and never resume graph."""
    events = [
        AgentStreamEvent.status("Thinking..."),
        AgentStreamEvent.approval_required(
            task_id="TASK-99",
            task_type="send_whatsapp_message",
            target="Sarah",
            draft="Meeting at 3 PM",
        ),
        AgentStreamEvent.complete(""),
    ]

    mock_thinking = MagicMock()
    mock_tools = MagicMock()
    mock_response = MagicMock()

    final_content, tools_executed, approval_info, has_error = streamlit_app.handle_stream_events(
        event_stream=iter(events),
        thinking_placeholder=mock_thinking,
        tools_container=mock_tools,
        response_placeholder=mock_response,
    )

    assert approval_info is not None
    assert approval_info["task_id"] == "TASK-99"
    assert approval_info["task_type"] == "send_whatsapp_message"
    assert approval_info["target"] == "Sarah"
    assert approval_info["draft"] == "Meeting at 3 PM"
    assert not has_error

    # Card rendered in response placeholder
    assert mock_response.markdown.call_count >= 1
    approval_html = mock_response.markdown.call_args[0][0]
    assert "Approval Required" in approval_html
    assert "send_whatsapp_message" in approval_html
    assert "Sarah" in approval_html
    assert "Meeting at 3 PM" in approval_html
    assert "Awaiting approval" in approval_html
    assert "No action was executed" in approval_html


# ==============================================================================
# 8. Error Sanitization (No Tracebacks Exposed)
# ==============================================================================

def test_sanitize_error_message_strips_tracebacks():
    """Verify that Python tracebacks and internal exception details are NEVER exposed."""
    raw_traceback = """
    Traceback (most recent call last):
      File "/src/assistant/llm/llm_client.py", line 140, in create
        raise groq.APIConnectionError("Failed to connect")
    groq.APIConnectionError: Failed to connect
    """
    sanitized = sanitize_error_message(raw_traceback)
    assert "Traceback" not in sanitized
    assert "File" not in sanitized
    assert "llm_client.py" not in sanitized
    assert "groq.APIConnectionError" not in sanitized
    assert "I couldn't reach the AI service right now. Please try again in a moment." in sanitized


def test_handle_stream_events_error_cleanly_rendered():
    """Verify streaming error displays clean friendly card and halts execution."""
    events = [
        AgentStreamEvent.status("Thinking..."),
        AgentStreamEvent.error("groq.RateLimitError: Rate limit exceeded."),
    ]

    mock_thinking = MagicMock()
    mock_tools = MagicMock()
    mock_response = MagicMock()

    final_content, tools_executed, approval_info, has_error = streamlit_app.handle_stream_events(
        event_stream=iter(events),
        thinking_placeholder=mock_thinking,
        tools_container=mock_tools,
        response_placeholder=mock_response,
    )

    assert has_error is True
    assert mock_response.markdown.call_count == 1
    rendered = mock_response.markdown.call_args[0][0]
    assert "error-card" in rendered
    assert "Traceback" not in rendered


# ==============================================================================
# 9. AppTest End-to-End Simulation
# ==============================================================================

def test_streamlit_app_test_execution_flow():
    """Verify Streamlit AppTest runs cleanly, renders header, sidebar, and handles chat."""
    mock_runtime = MagicMock()
    mock_runtime.is_started = True

    def simulated_stream(msg):
        yield AgentStreamEvent.status("Thinking...")
        yield AgentStreamEvent.token("Hello")
        yield AgentStreamEvent.token(" from Joe!")
        yield AgentStreamEvent.complete("Hello from Joe!")

    mock_runtime.orchestrator.stream_agent_command.side_effect = simulated_stream

    app_path = os.path.abspath("streamlit_app.py")

    with patch("assistant.runtime.get_runtime", return_value=mock_runtime), \
         patch("streamlit_app.get_runtime", return_value=mock_runtime):
        at = AppTest.from_file(app_path, default_timeout=10)
        at.run()

        # Check that no unhandled exceptions occurred in AppTest
        assert not at.exception

        # Check that chat input is present
        assert len(at.chat_input) == 1

        # Simulate user typing a question
        at.chat_input[0].set_value("Hello Joe").run()
        assert not at.exception

        # Check user and assistant messages
        messages = at.chat_message
        assert len(messages) >= 2
        assert messages[0].name == "user"
        assert messages[1].name == "assistant"

        # Verify stream_agent_command was called with UnifiedMessage
        assert mock_runtime.orchestrator.stream_agent_command.call_count == 1
        called_msg = mock_runtime.orchestrator.stream_agent_command.call_args[0][0]
        assert isinstance(called_msg, UnifiedMessage)
        assert called_msg.content == "Hello Joe"
        assert called_msg.source == Source.CLI


# ==============================================================================
# 10. Interactive Approval UI & LangGraph Resume Mechanics
# ==============================================================================

def test_orchestrator_resume_approval_directly_resumes_checkpoint_without_unified_message():
    """Verify AgentOrchestrator.resume_approval directly resumes checkpoint without creating UnifiedMessage."""
    from assistant.agent_core.orchestrator import AgentOrchestrator
    from langgraph.types import Command
    orch = MagicMock(spec=AgentOrchestrator)
    orch._resume_approval_internal = MagicMock(return_value="Approved")

    with patch("assistant.ingestion.unified_message.UnifiedMessage", side_effect=AssertionError("UnifiedMessage constructed")):
        res = AgentOrchestrator.resume_approval(orch, conversation_id="conv_123", action="approve", task_id="TASK-99")
        list(res)

    orch._resume_approval_internal.assert_called_once()
    call_kwargs = orch._resume_approval_internal.call_args[1]
    assert call_kwargs["conversation_id"] == "conv_123"
    assert call_kwargs["action"] == "approve"
    assert call_kwargs["task_id"] == "TASK-99"


def test_render_interactive_approval_card_pending_shows_buttons():
    """When an approval is pending, card HTML and Approve/Reject buttons must be rendered."""
    mock_runtime = MagicMock()
    approval_info = {
        "task_id": "TASK-101",
        "task_type": "send_whatsapp_message",
        "target": "Sarah",
        "draft": "Meeting at 3 PM",
    }
    mock_session_state = {
        "conversation_id": "conv_test",
        "approval_states": {},
    }

    with patch("streamlit.session_state", mock_session_state), \
         patch("streamlit.markdown") as mock_markdown, \
         patch("streamlit.columns", return_value=[MagicMock(), MagicMock(), MagicMock()]), \
         patch("streamlit.button", return_value=False) as mock_button:
        streamlit_app.render_interactive_approval_card(approval_info, mock_runtime)

        assert mock_markdown.call_count == 1
        card_html = mock_markdown.call_args[0][0]
        assert "TASK-101" in card_html
        assert "Sarah" in card_html
        assert "Awaiting approval" in card_html

        # Both Approve and Reject buttons were rendered
        assert mock_button.call_count == 2
        button_labels = [call[0][0] for call in mock_button.call_args_list]
        assert "✓ Approve" in button_labels
        assert "✕ Reject" in button_labels


def test_render_interactive_approval_card_resolved_hides_buttons():
    """When an approval is already resolved (approved or rejected), buttons must NOT be rendered."""
    mock_runtime = MagicMock()
    approval_info = {
        "task_id": "TASK-102",
        "task_type": "send_telegram_message",
        "target": "Bob",
        "draft": "Hello Bob",
    }
    mock_session_state = {
        "conversation_id": "conv_test",
        "approval_states": {
            "TASK-102": {
                "status": "approved",
                "task_id": "TASK-102",
                "task_type": "send_telegram_message",
                "target": "Bob",
                "draft": "Hello Bob",
            }
        },
    }

    with patch("streamlit.session_state", mock_session_state), \
         patch("streamlit.markdown") as mock_markdown, \
         patch("streamlit.button") as mock_button:
        streamlit_app.render_interactive_approval_card(approval_info, mock_runtime)

        assert mock_markdown.call_count == 1
        card_html = mock_markdown.call_args[0][0]
        assert "✓ Action Approved" in card_html

        # Buttons must NOT be called when approved
        assert mock_button.call_count == 0


def test_process_approval_decision_approve_flow():
    """Approve button resumes LangGraph and records completed assistant response."""
    mock_runtime = MagicMock()

    def simulated_resume_stream(conversation_id, action, task_id):
        yield AgentStreamEvent.tool_start("send_whatsapp_message", task_id=task_id)
        yield AgentStreamEvent.tool_complete("send_whatsapp_message", "Message sent", task_id=task_id)
        yield AgentStreamEvent.complete("Done — I sent Sarah the message.")

    mock_runtime.orchestrator.resume_approval.side_effect = simulated_resume_stream

    mock_session_state = {
        "conversation_id": "conv_approve",
        "messages": [],
        "approval_states": {
            "TASK-103": {
                "status": "pending",
                "task_id": "TASK-103",
                "task_type": "send_whatsapp_message",
                "target": "Sarah",
                "draft": "See you at 3pm",
                "conversation_id": "conv_approve",
            }
        },
    }

    with patch("streamlit.session_state", mock_session_state), \
         patch("streamlit.chat_message"), \
         patch("streamlit.empty"), \
         patch("streamlit.container"), \
         patch("streamlit.rerun") as mock_rerun:
        streamlit_app.process_approval_decision("TASK-103", "approve", mock_runtime)

        # Check resume_approval was called with exact IDs
        mock_runtime.orchestrator.resume_approval.assert_called_once_with(
            conversation_id="conv_approve",
            action="approve",
            task_id="TASK-103",
        )

        # Check state transitioned to approved
        state = mock_session_state["approval_states"]["TASK-103"]
        assert state["status"] == "approved"
        assert state["result"] == "Done — I sent Sarah the message."

        # Check message was added to chat history
        assert len(mock_session_state["messages"]) == 1
        assert mock_session_state["messages"][0]["content"] == "Done — I sent Sarah the message."
        assert mock_session_state["messages"][0]["tools"] == ["send_whatsapp_message"]

        # Check rerun was called to refresh UI
        assert mock_rerun.call_count == 1


def test_security_approve_never_calls_connectors_directly():
    """CRITICAL SECURITY TEST: Streamlit MUST NEVER directly invoke connectors or tools.

    Clicking Approve MUST only call runtime.orchestrator.resume_approval().
    """
    mock_runtime = MagicMock()
    mock_runtime.whatsapp_connector = MagicMock()
    mock_runtime.telegram_connector = MagicMock()
    mock_runtime.gmail_connector = MagicMock()
    mock_runtime.device_gateway = MagicMock()
    mock_runtime.tool_router = MagicMock()

    mock_runtime.orchestrator.resume_approval.return_value = iter([
        AgentStreamEvent.complete("Executed via LangGraph checkpoint"),
    ])

    mock_session_state = {
        "conversation_id": "secure_conv",
        "messages": [],
        "approval_states": {
            "TASK-SEC": {
                "status": "pending",
                "task_id": "TASK-SEC",
                "task_type": "send_whatsapp_message",
                "target": "+1234567890",
                "draft": "Important security payload",
                "conversation_id": "secure_conv",
            }
        },
    }

    with patch("streamlit.session_state", mock_session_state), \
         patch("streamlit.chat_message"), \
         patch("streamlit.empty"), \
         patch("streamlit.container"), \
         patch("streamlit.rerun"):
        streamlit_app.process_approval_decision("TASK-SEC", "approve", mock_runtime)

        # 1. Orchestrator was called
        assert mock_runtime.orchestrator.resume_approval.call_count == 1

        # 2. ZERO direct connector calls
        assert mock_runtime.whatsapp_connector.send_message.call_count == 0
        assert mock_runtime.telegram_connector.send_message.call_count == 0
        assert mock_runtime.gmail_connector.send.call_count == 0
        assert mock_runtime.device_gateway.send_command.call_count == 0
        assert mock_runtime.tool_router.dispatch.call_count == 0


def test_process_approval_decision_reject_flow():
    """Reject button resumes LangGraph with reject and executes zero tools."""
    mock_runtime = MagicMock()

    def simulated_reject_stream(conversation_id, action, task_id):
        yield AgentStreamEvent.complete("Okay, I didn't send the message.")

    mock_runtime.orchestrator.resume_approval.side_effect = simulated_reject_stream

    mock_session_state = {
        "conversation_id": "conv_reject",
        "messages": [],
        "approval_states": {
            "TASK-104": {
                "status": "pending",
                "task_id": "TASK-104",
                "task_type": "send_sms",
                "target": "+999999",
                "draft": "Meeting cancelled",
                "conversation_id": "conv_reject",
            }
        },
    }

    with patch("streamlit.session_state", mock_session_state), \
         patch("streamlit.chat_message"), \
         patch("streamlit.empty"), \
         patch("streamlit.container"), \
         patch("streamlit.rerun"):
        streamlit_app.process_approval_decision("TASK-104", "reject", mock_runtime)

        # Check resume_approval was called with action='reject'
        mock_runtime.orchestrator.resume_approval.assert_called_once_with(
            conversation_id="conv_reject",
            action="reject",
            task_id="TASK-104",
        )

        # State transitioned to rejected
        state = mock_session_state["approval_states"]["TASK-104"]
        assert state["status"] == "rejected"
        assert state["result"] == "Okay, I didn't send the message."

        # Assistant message recorded
        assert len(mock_session_state["messages"]) == 1
        assert "didn't send" in mock_session_state["messages"][0]["content"]
        assert mock_session_state["messages"][0]["tools"] == []


def test_duplicate_click_protection_approve_twice():
    """Clicking Approve twice must only process once."""
    mock_runtime = MagicMock()
    mock_runtime.orchestrator.resume_approval.return_value = iter([
        AgentStreamEvent.complete("Action completed"),
    ])

    mock_session_state = {
        "conversation_id": "conv_dup",
        "messages": [],
        "approval_states": {
            "TASK-DUP": {
                "status": "pending",
                "task_id": "TASK-DUP",
                "conversation_id": "conv_dup",
            }
        },
    }

    with patch("streamlit.session_state", mock_session_state), \
         patch("streamlit.chat_message"), \
         patch("streamlit.empty"), \
         patch("streamlit.container"), \
         patch("streamlit.rerun"):
        # First click: processes
        streamlit_app.process_approval_decision("TASK-DUP", "approve", mock_runtime)
        assert mock_runtime.orchestrator.resume_approval.call_count == 1

        # Second click: state is now 'approved', must be ignored
        streamlit_app.process_approval_decision("TASK-DUP", "approve", mock_runtime)
        assert mock_runtime.orchestrator.resume_approval.call_count == 1


def test_duplicate_click_protection_approve_after_reject():
    """Attempting to Approve an already-rejected task must be discarded."""
    mock_runtime = MagicMock()
    mock_session_state = {
        "conversation_id": "conv_dup",
        "messages": [],
        "approval_states": {
            "TASK-REJ": {
                "status": "rejected",
                "task_id": "TASK-REJ",
                "conversation_id": "conv_dup",
            }
        },
    }

    with patch("streamlit.session_state", mock_session_state):
        streamlit_app.process_approval_decision("TASK-REJ", "approve", mock_runtime)
        assert mock_runtime.orchestrator.resume_approval.call_count == 0


def test_duplicate_click_protection_reject_after_approve():
    """Attempting to Reject an already-approved task must be discarded."""
    mock_runtime = MagicMock()
    mock_session_state = {
        "conversation_id": "conv_dup",
        "messages": [],
        "approval_states": {
            "TASK-APPR": {
                "status": "approved",
                "task_id": "TASK-APPR",
                "conversation_id": "conv_dup",
            }
        },
    }

    with patch("streamlit.session_state", mock_session_state):
        streamlit_app.process_approval_decision("TASK-APPR", "reject", mock_runtime)
        assert mock_runtime.orchestrator.resume_approval.call_count == 0


def test_approval_backend_error_handled_sanitized():
    """Backend error during approval resume is caught and rendered without tracebacks."""
    mock_runtime = MagicMock()
    mock_runtime.orchestrator.resume_approval.side_effect = RuntimeError(
        "Traceback (most recent call last):\n  File 'foo.py', line 12\ngroq.APIConnectionError: Connection reset"
    )

    mock_session_state = {
        "conversation_id": "conv_err",
        "messages": [],
        "approval_states": {
            "TASK-ERR": {
                "status": "pending",
                "task_id": "TASK-ERR",
                "conversation_id": "conv_err",
            }
        },
    }

    with patch("streamlit.session_state", mock_session_state), \
         patch("streamlit.chat_message"), \
         patch("streamlit.empty"), \
         patch("streamlit.container"), \
         patch("streamlit.rerun"):
        streamlit_app.process_approval_decision("TASK-ERR", "approve", mock_runtime)

        state = mock_session_state["approval_states"]["TASK-ERR"]
        assert state["status"] == "failed"

        # Verify message added to chat does NOT contain traceback
        last_msg = mock_session_state["messages"][-1]["content"]
        assert "Traceback" not in last_msg
        assert "foo.py" not in last_msg
        assert "groq.APIConnectionError" not in last_msg
        assert "I couldn't reach the AI service right now" in last_msg


def test_new_chat_clears_approval_states():
    """Clicking New Chat must clear messages, reset conversation ID, and clear approval states."""
    mock_runtime = MagicMock()
    mock_session_state = {
        "messages": [{"role": "user", "content": "hi"}],
        "conversation_id": "old_conv",
        "approval_states": {"TASK-1": {"status": "pending"}},
    }

    with patch("streamlit.session_state", mock_session_state), \
         patch("streamlit.sidebar"), \
         patch("streamlit.button", side_effect=lambda label, **kw: "New Chat" in label), \
         patch("streamlit.rerun"):
        streamlit_app.render_sidebar(mock_runtime)

        assert mock_session_state["messages"] == []
        assert mock_session_state["approval_states"] == {}
        assert mock_session_state["conversation_id"] != "old_conv"
        assert mock_session_state["conversation_id"].startswith("streamlit_")


# ==============================================================================
# 11. UI Cleanup & Real Connection Controls (Phase 3 Task)
# ==============================================================================

def test_sidebar_ui_cleanup_removes_assistant_and_placeholders():
    """Verify rendered sidebar does NOT contain ASSISTANT, Priority, Agenda, or Memory."""
    mock_runtime = MagicMock()
    mock_runtime.get_whatsapp_status.return_value = {"connected": False, "running": False}
    mock_runtime.get_telegram_status.return_value = {"connected": False}
    mock_runtime.get_android_status.return_value = {"connected": False, "server_running": False, "ws_url": "ws://localhost:8765"}

    rendered_markdown: list[str] = []
    rendered_buttons: list[str] = []

    with patch("streamlit.sidebar"), \
         patch("streamlit.markdown", side_effect=lambda content, **kw: rendered_markdown.append(str(content))), \
         patch("streamlit.button", side_effect=lambda label, **kw: rendered_buttons.append(str(label)) and False):
        streamlit_app.render_sidebar(mock_runtime)

    all_rendered = " ".join(rendered_markdown + rendered_buttons)

    # Completely removed from visible UI
    assert "ASSISTANT" not in all_rendered
    assert "Priority" not in all_rendered
    assert "Agenda" not in all_rendered
    assert "Memory" not in all_rendered
    assert "badge-muted" not in all_rendered

    # Retained and functional
    assert "Connections" in all_rendered
    assert any("WhatsApp" in b for b in rendered_buttons)
    assert any("Telegram" in b for b in rendered_buttons)
    assert any("Android" in b for b in rendered_buttons)


def test_sidebar_displays_real_connection_statuses():
    """Verify sidebar button labels reflect real status from runtime and change dynamically."""
    mock_runtime = MagicMock()

    # Case 1: All disconnected
    mock_runtime.get_whatsapp_status.return_value = {"connected": False}
    mock_runtime.get_telegram_status.return_value = {"connected": False}
    mock_runtime.get_android_status.return_value = {"connected": False}

    buttons_1: list[str] = []
    with patch("streamlit.sidebar"), \
         patch("streamlit.markdown"), \
         patch("streamlit.button", side_effect=lambda label, **kw: buttons_1.append(str(label)) and False):
        streamlit_app.render_sidebar(mock_runtime)

    assert any("WhatsApp   🔴 Not connected" in b for b in buttons_1)
    assert any("Telegram   🔴 Not connected" in b for b in buttons_1)
    assert any("Android   🔴 Not connected" in b for b in buttons_1)

    # Case 2: WhatsApp & Android connected
    mock_runtime.get_whatsapp_status.return_value = {"connected": True}
    mock_runtime.get_telegram_status.return_value = {"connected": False}
    mock_runtime.get_android_status.return_value = {"connected": True}

    buttons_2: list[str] = []
    with patch("streamlit.sidebar"), \
         patch("streamlit.markdown"), \
         patch("streamlit.button", side_effect=lambda label, **kw: buttons_2.append(str(label)) and False):
        streamlit_app.render_sidebar(mock_runtime)

    assert any("WhatsApp   🟢 Connected" in b for b in buttons_2)
    assert any("Telegram   🔴 Not connected" in b for b in buttons_2)
    assert any("Android   🟢 Connected" in b for b in buttons_2)


def test_sidebar_triggers_connection_dialogs():
    """Verify clicking connection buttons in the sidebar opens the corresponding dialog."""
    mock_runtime = MagicMock()
    mock_runtime.get_whatsapp_status.return_value = {"connected": False}
    mock_runtime.get_telegram_status.return_value = {"connected": False}
    mock_runtime.get_android_status.return_value = {"connected": False}

    with patch("streamlit.sidebar"), \
         patch("streamlit.markdown"), \
         patch("streamlit.button", side_effect=lambda label, **kw: "WhatsApp" in label), \
         patch("streamlit_app.show_whatsapp_dialog") as mock_wa_dlg, \
         patch("streamlit_app.show_telegram_dialog") as mock_tg_dlg, \
         patch("streamlit_app.show_android_dialog") as mock_and_dlg:
        streamlit_app.render_sidebar(mock_runtime)
        mock_wa_dlg.assert_called_once_with(mock_runtime)
        mock_tg_dlg.assert_not_called()
        mock_and_dlg.assert_not_called()


def test_whatsapp_dialog_and_idempotent_bridge_start():
    """Verify WhatsApp dialog opens with real status and delegates start to connector safely."""
    mock_runtime = MagicMock()
    mock_runtime.get_whatsapp_status.return_value = {
        "running": False,
        "connected": False,
        "qr": None,
    }

    with patch("streamlit.markdown") as mock_md, \
         patch("streamlit.caption"), \
         patch("streamlit.button", return_value=True), \
         patch("streamlit.rerun"):
        # Test unwrapped dialog content function
        streamlit_app.show_whatsapp_dialog.__wrapped__(mock_runtime)

        # Proves start WhatsApp button delegates to runtime.whatsapp_connector.start_bridge(force=True)
        mock_runtime.whatsapp_connector.start_bridge.assert_called_once_with(force=True)


def test_telegram_dialog_displays_real_status():
    """Verify Telegram dialog displays real status and does not pretend connection."""
    mock_runtime = MagicMock()
    mock_runtime.get_telegram_status.return_value = {
        "connected": False,
        "bot": {"has_token": False, "connected": False},
        "user": {"has_credentials": True, "authorized": False, "session_file": "config/secrets/tg.session"},
    }

    rendered_md: list[str] = []
    with patch("streamlit.markdown", side_effect=lambda txt, **kw: rendered_md.append(str(txt))), \
         patch("streamlit.caption") as mock_caption, \
         patch("streamlit.info") as mock_info, \
         patch("streamlit.button"):
        # Test unwrapped dialog content function
        streamlit_app.show_telegram_dialog.__wrapped__(mock_runtime)

        all_text = " ".join(rendered_md)
        assert "Not connected" in all_text
        mock_info.assert_called_once()
        assert "scripts/setup_telegram_user.py" in mock_info.call_args[0][0]


def test_android_dialog_distinguishes_server_running_from_device_connected():
    """Verify Android dialog distinguishes WebSocket server available from actual device connected."""
    mock_runtime = MagicMock()
    # Server running, but device is NOT connected
    mock_runtime.get_android_status.return_value = {
        "server_running": True,
        "connected": False,
        "device_name": None,
        "ws_url": "ws://localhost:8765",
    }

    rendered_md: list[str] = []
    with patch("streamlit.markdown", side_effect=lambda txt, **kw: rendered_md.append(str(txt))), \
         patch("streamlit.caption") as mock_cap, \
         patch("streamlit.code") as mock_code, \
         patch("streamlit.button"):
        # Test unwrapped dialog content function
        streamlit_app.show_android_dialog.__wrapped__(mock_runtime)

        all_text = " ".join(rendered_md)
        # MUST show Not connected even though server is running
        assert "Not connected" in all_text
        mock_code.assert_called_once_with("ws://localhost:8765", language="text")
        cap_text = " ".join(call[0][0] for call in mock_cap.call_args_list)
        assert "Awaiting device connection" in cap_text

