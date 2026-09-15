"""Comprehensive regression tests for Mistral message ordering, multi-turn state, and pending approval flows across CLI and Telegram channels."""

from unittest.mock import MagicMock
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from assistant.ingestion.unified_message import Source, UnifiedMessage
from assistant.execution.task import TaskState
from tests.unit.test_langgraph_orchestrator import make_orchestrator


def test_consecutive_cli_messages(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)
    
    msg1 = UnifiedMessage(source=Source.CLI, conversation_id="cli_1", sender="user", content="hello")
    reply1 = orchestrator.handle_agent_command(msg1)
    assert reply1 is not None

    msg2 = UnifiedMessage(source=Source.CLI, conversation_id="cli_1", sender="user", content="how are you?")
    reply2 = orchestrator.handle_agent_command(msg2)
    assert reply2 is not None

    config = {"configurable": {"thread_id": "cli_1"}}
    snapshot = orchestrator._graph.get_state(config)
    messages = snapshot.values.get("messages", [])
    assert len(messages) >= 4
    assert isinstance(messages[-1], AIMessage)


def test_consecutive_telegram_bot_messages(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)
    
    msg1 = UnifiedMessage(source=Source.TELEGRAM, conversation_id="telegram_bot:100", sender="user", content="hi telegram")
    reply1 = orchestrator.handle_agent_command(msg1)
    assert reply1 is not None

    msg2 = UnifiedMessage(source=Source.TELEGRAM, conversation_id="telegram_bot:100", sender="user", content="tell me a joke")
    reply2 = orchestrator.handle_agent_command(msg2)
    assert reply2 is not None

    config = {"configurable": {"thread_id": "telegram_bot:100"}}
    snapshot = orchestrator._graph.get_state(config)
    messages = snapshot.values.get("messages", [])
    assert len(messages) >= 4
    assert isinstance(messages[-1], AIMessage)


def test_clear_conversational_state(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)

    msg1 = UnifiedMessage(source=Source.CLI, conversation_id="cli_clear", sender="user", content="hello chat")
    orchestrator.handle_agent_command(msg1)

    clear_msg = UnifiedMessage(source=Source.CLI, conversation_id="cli_clear", sender="user", content="clear")
    reply_clear = orchestrator.handle_agent_command(clear_msg)
    assert "cleared" in reply_clear.lower()

    msg2 = UnifiedMessage(source=Source.CLI, conversation_id="cli_clear", sender="user", content="new start")
    reply2 = orchestrator.handle_agent_command(msg2)
    assert reply2 is not None


def test_pending_approval_with_approve(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    send_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_app", sender="user", content="send Rahul on whatsapp: meeting at 5")
    draft_reply = orchestrator.handle_agent_command(send_msg)
    assert "Proposed action" in draft_reply

    approve_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_app", sender="user", content="approve")
    exec_reply = orchestrator.handle_agent_command(approve_msg)
    assert "Approved and executed" in exec_reply


def test_pending_approval_with_yes(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    send_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_yes", sender="user", content="send Rahul on whatsapp: meeting at 6")
    orchestrator.handle_agent_command(send_msg)

    yes_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_yes", sender="user", content="yes")
    exec_reply = orchestrator.handle_agent_command(yes_msg)
    assert "Approved and executed" in exec_reply


def test_pending_approval_with_reject(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    send_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_rej", sender="user", content="send Rahul on whatsapp: meeting at 7")
    orchestrator.handle_agent_command(send_msg)

    reject_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_rej", sender="user", content="reject")
    rej_reply = orchestrator.handle_agent_command(reject_msg)
    assert "Rejected" in rej_reply


def test_pending_approval_with_no(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    send_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_no", sender="user", content="send Rahul on whatsapp: meeting at 8")
    orchestrator.handle_agent_command(send_msg)

    no_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_no", sender="user", content="no")
    rej_reply = orchestrator.handle_agent_command(no_msg)
    assert "Rejected" in rej_reply


def test_pending_approval_with_normal_text(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    send_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_norm", sender="user", content="send Rahul on whatsapp: meeting at 9")
    draft_reply = orchestrator.handle_agent_command(send_msg)
    assert "Proposed action" in draft_reply

    # Normal message while task is pending
    norm_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_norm", sender="user", content="hello")
    norm_reply = orchestrator.handle_agent_command(norm_msg)
    assert "There's a pending approval" not in norm_reply

    # Pending task must still exist
    pending_tasks = [t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]
    assert len(pending_tasks) == 1

    # Subsequent approve must work cleanly
    approve_msg = UnifiedMessage(source=Source.CLI, conversation_id="c_norm", sender="user", content="yes")
    exec_reply = orchestrator.handle_agent_command(approve_msg)
    assert "Approved and executed" in exec_reply


def test_separate_cli_and_telegram_pending_approvals(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    # CLI proposal
    cli_msg = UnifiedMessage(source=Source.CLI, conversation_id="cli_sep", sender="user", content="send Rahul on whatsapp: CLI text")
    orchestrator.handle_agent_command(cli_msg)

    # Telegram proposal
    tg_msg = UnifiedMessage(source=Source.TELEGRAM, conversation_id="telegram_bot:999", sender="user_tg", content="send Priya on whatsapp: Telegram text")
    orchestrator.handle_agent_command(tg_msg)

    # Both pending tasks must exist independently
    pending_tasks = [t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]
    assert len(pending_tasks) == 2

    # Approving Telegram does not affect CLI
    exec_tg = orchestrator.handle_agent_command(UnifiedMessage(source=Source.TELEGRAM, conversation_id="telegram_bot:999", sender="user_tg", content="approve"))
    assert "Approved and executed" in exec_tg

    remaining_pending = [t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]
    assert len(remaining_pending) == 1
    assert remaining_pending[0].target == "Rahul"
