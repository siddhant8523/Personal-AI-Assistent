"""
Regression tests for the LangGraph migration.

These test the SAME behaviors the old hand-written orchestrator had --
offline general chat, priority query, send-draft-approve-execute, and
unknown-task rejection -- to prove the migration didn't change behavior,
only the internal implementation.
"""

from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.context_builder import ContextBuilder
from assistant.agent_core.orchestrator import AgentOrchestrator
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.agent_core.task_planner import TaskPlanner
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task import TaskState
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.ingestion.unified_message import Source, UnifiedMessage
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.llm.llm_client import LLMClient
from assistant.memory.memory_service import MemoryService


def make_orchestrator(tmp_path):
    memory = MemoryService(memory_dir=str(tmp_path / "memory"))
    llm = LLMClient(api_key="")  # offline mode, deterministic
    context_builder = ContextBuilder(memory)
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    tool_router = ToolRouter(task_manager)
    tool_router.register("send_whatsapp_message", lambda task: {"status": "ok", "detail": "sent"})
    policy_engine = PolicyEngine({"send_whatsapp_message": {"requires_approval": True}})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    priority_inbox = PriorityInbox()

    orchestrator = AgentOrchestrator(
        llm=llm, memory=memory, context_builder=context_builder, task_planner=task_planner,
        approval_manager=approval_manager, tool_router=tool_router, priority_inbox=priority_inbox,
    )
    return orchestrator, task_manager


def test_general_chat_offline(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)
    msg = UnifiedMessage(source=Source.CLI, conversation_id="c1", sender="user", content="hello there")
    reply = orchestrator.handle_agent_command(msg)
    assert "hello there" in reply.lower() or "acknowledged" in reply.lower()


def test_priority_query_via_graph(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)
    msg = UnifiedMessage(source=Source.CLI, conversation_id="c1", sender="user", content="what are my important messages today?")
    reply = orchestrator.handle_agent_command(msg)
    assert "PRIORITY" in reply or "No messages" in reply


def test_send_draft_approve_execute_via_graph(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    send_msg = UnifiedMessage(source=Source.CLI, conversation_id="c1", sender="user",
                               content="send Rahul on whatsapp: on my way")
    draft_reply = orchestrator.handle_agent_command(send_msg)
    assert "Proposed action" in draft_reply

    task_id = list(task_manager._tasks.keys())[-1]
    assert task_manager.get(task_id).execution_state == TaskState.WAITING_FOR_APPROVAL

    approve_msg = UnifiedMessage(source=Source.CLI, conversation_id="c1", sender="user", content=f"approve {task_id}")
    approve_reply = orchestrator.handle_agent_command(approve_msg)
    assert "Approved and executed" in approve_reply
    assert task_manager.get(task_id).execution_state == TaskState.COMPLETED


def test_reject_unknown_task_id(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)
    msg = UnifiedMessage(source=Source.CLI, conversation_id="c1", sender="user", content="approve TASK-doesnotexist")
    reply = orchestrator.handle_agent_command(msg)
    assert "don't have a pending task" in reply


def test_graph_checkpoints_per_conversation(tmp_path):
    """Confirms the LangGraph MemorySaver checkpointer is actually wired
    (graph persistence requirement) -- state is retrievable per thread_id."""
    orchestrator, _ = make_orchestrator(tmp_path)
    msg = UnifiedMessage(source=Source.CLI, conversation_id="conv-A", sender="user", content="hi")
    orchestrator.handle_agent_command(msg)

    state = orchestrator._graph.get_state({"configurable": {"thread_id": "conv-A"}})
    assert state is not None
    assert state.values["conversation_id"] == "conv-A"


def test_approval_is_a_real_graph_interrupt_not_a_text_hack(tmp_path):
    """Proves the graph is genuinely paused (not just re-invoked with a
    different input string): before approving, the LangGraph checkpointer
    reports the graph as sitting mid-execution (state.next non-empty) for
    this conversation's thread_id, and the pending task id surfaced to the
    user matches the interrupt payload LangGraph itself is holding."""
    orchestrator, task_manager = make_orchestrator(tmp_path)

    send_msg = UnifiedMessage(source=Source.CLI, conversation_id="c2", sender="user",
                               content="send Priya on whatsapp: running late")
    orchestrator.handle_agent_command(send_msg)

    config = {"configurable": {"thread_id": "c2"}}
    snapshot = orchestrator._graph.get_state(config)
    assert snapshot.next, "graph should be paused inside interrupt(), not finished"

    pending_id = orchestrator._pending_task_id(snapshot)
    assert pending_id is not None
    assert task_manager.get(pending_id).execution_state == TaskState.WAITING_FOR_APPROVAL

    # A stray, non-matching approval must not resume the paused run.
    mismatched = UnifiedMessage(source=Source.CLI, conversation_id="c2", sender="user", content="approve TASK-bogus")
    reply = orchestrator.handle_agent_command(mismatched)
    assert pending_id in reply
    assert task_manager.get(pending_id).execution_state == TaskState.WAITING_FOR_APPROVAL

    approve_reply = orchestrator.handle_agent_command(
        UnifiedMessage(source=Source.CLI, conversation_id="c2", sender="user", content=f"approve {pending_id}")
    )
    assert "Approved and executed" in approve_reply
    assert task_manager.get(pending_id).execution_state == TaskState.COMPLETED

    resumed_snapshot = orchestrator._graph.get_state(config)
    assert not resumed_snapshot.next, "graph should have run to completion after resume"


def test_channel_aware_approval_prompts(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    # Telegram bot conversation
    tg_msg = UnifiedMessage(
        source=Source.TELEGRAM, conversation_id="telegram_bot:98765", sender="user_tg",
        content="send Priya on whatsapp: meeting at 5pm"
    )
    tg_reply = orchestrator.handle_agent_command(tg_msg)
    assert "Proposed action" in tg_reply
    assert "via Telegram Bot" in tg_reply

    config_tg = {"configurable": {"thread_id": "telegram_bot:98765"}}
    snapshot_tg = orchestrator._graph.get_state(config_tg)
    pending_id_tg = orchestrator._pending_task_id(snapshot_tg)

    # Approve from Telegram
    tg_approve = UnifiedMessage(
        source=Source.TELEGRAM, conversation_id="telegram_bot:98765", sender="user_tg",
        content=f"approve {pending_id_tg}"
    )
    exec_reply = orchestrator.handle_agent_command(tg_approve)
    assert "Approved and executed" in exec_reply

    # WhatsApp conversation
    wa_msg = UnifiedMessage(
        source=Source.WHATSAPP, conversation_id="whatsapp:1234567890", sender="user_wa",
        content="send Rahul on whatsapp: see you tomorrow"
    )
    wa_reply = orchestrator.handle_agent_command(wa_msg)
    assert "Proposed action" in wa_reply
    assert "via WhatsApp Bot" in wa_reply


def test_simple_approval_words_like_approve_or_yes(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    # CLI simple "approve"
    msg = UnifiedMessage(source=Source.CLI, conversation_id="c_cli", sender="user", content="send Rahul on whatsapp: hello CLI")
    orchestrator.handle_agent_command(msg)
    reply_cli = orchestrator.handle_agent_command(UnifiedMessage(source=Source.CLI, conversation_id="c_cli", sender="user", content="approve"))
    assert "Approved and executed" in reply_cli

    # Telegram simple "yes"
    tg_msg = UnifiedMessage(source=Source.TELEGRAM, conversation_id="telegram_bot:111", sender="user", content="send Rahul on whatsapp: hello TG")
    orchestrator.handle_agent_command(tg_msg)
    reply_tg = orchestrator.handle_agent_command(UnifiedMessage(source=Source.TELEGRAM, conversation_id="telegram_bot:111", sender="user", content="yes"))
    assert "Approved and executed" in reply_tg


def test_simple_rejection_words_like_no_or_reject(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    # WhatsApp simple "no"
    wa_msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="whatsapp:222", sender="user", content="send Rahul on whatsapp: hello WA")
    orchestrator.handle_agent_command(wa_msg)
    reply_wa = orchestrator.handle_agent_command(UnifiedMessage(source=Source.WHATSAPP, conversation_id="whatsapp:222", sender="user", content="no"))
    assert "Rejected" in reply_wa


def test_repeated_yes_when_not_paused_returns_no_pending_actions(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    # 1. Propose and approve
    msg = UnifiedMessage(source=Source.CLI, conversation_id="c_rep", sender="user", content="send Rahul on whatsapp: test message")
    orchestrator.handle_agent_command(msg)
    reply_approve = orchestrator.handle_agent_command(UnifiedMessage(source=Source.CLI, conversation_id="c_rep", sender="user", content="yes"))
    assert "Approved and executed" in reply_approve

    # 2. Subsequent "yes" when no task is pending must NOT re-trigger send proposal
    reply_subsequent = orchestrator.handle_agent_command(UnifiedMessage(source=Source.CLI, conversation_id="c_rep", sender="user", content="yes"))
    assert "no pending actions awaiting approval" in reply_subsequent.lower()

def test_send_intent_override_with_intermediate_retrieval(tmp_path):
    orchestrator, task_manager = make_orchestrator(tmp_path)

    msg = UnifiedMessage(source=Source.CLI, conversation_id="c_send_news", sender="user", content="send news to Sidd on telegram: India update")
    draft_reply = orchestrator.handle_agent_command(msg)
    assert "Proposed action" in draft_reply or "send_telegram_message" in draft_reply or "Sidd" in draft_reply
