import os
import sqlite3
import pytest
from unittest.mock import MagicMock, patch

from assistant.ingestion.unified_message import Source, Origin, UnifiedMessage
from assistant.intelligence import classifier
from assistant.intelligence.priority_engine import PriorityEngine, PriorityResult, PriorityLevel
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.intelligence.normal_message_pipeline import NormalMessagePipeline
from assistant.intelligence.eligibility_filter import MessageEligibilityFilter
from assistant.memory.user_profile import UserProfile
from assistant.router.conversation_registry import ConversationRegistry, ConversationClass
from assistant.router.conversation_router import ConversationRouter
from assistant.router.echo_filter import EchoFilter
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.agent_core.orchestrator import AgentOrchestrator
from assistant.agent_core.graph.tools import render_priority_inbox, parse_priority_query
from assistant.connectors.whatsapp.whatsapp_connector import WhatsAppConnector
from assistant.channels.telegram_channel import TelegramChannel
from assistant.storage.db import get_connection, reset_connection


@pytest.fixture(autouse=True)
def setup_tmp_db(tmp_path):
    db_file = str(tmp_path / "test_assistant.db")
    os.environ["DATABASE_PATH"] = db_file
    reset_connection()
    yield
    reset_connection()


def test_agent_lid_and_phone_classification():
    registry = ConversationRegistry({
        "whatsapp:52909752496163@lid": "AGENT_CHAT",
        "whatsapp:919172767219@s.whatsapp.net": "AGENT_CHAT",
    })

    # 1. LID classification
    assert registry.classify("whatsapp:52909752496163@lid") == ConversationClass.AGENT_CHAT

    # 2. Phone JID classification
    assert registry.classify("whatsapp:919172767219@s.whatsapp.net") == ConversationClass.AGENT_CHAT

    # 3. Normal user classification
    assert registry.classify("whatsapp:919888888888@s.whatsapp.net") == ConversationClass.NORMAL


def test_whatsapp_agent_isolation_from_history_and_deduplication():
    conn = WhatsAppConnector("http://localhost:3000", enabled=True)
    os.environ["WHATSAPP_AGENT_CHAT_ID"] = "919172767219@s.whatsapp.net"

    items = [
        {"id": "msg1", "chat_id": "52909752496163@lid", "sender": "52909752496163@lid", "text": "Hi Jarvis", "from_me": False},
        {"id": "msg2", "chat_id": "919888888888@s.whatsapp.net", "sender": "Rahul", "text": "Can we meet tomorrow?", "from_me": False},
        {"id": "msg3", "chat_id": "919172767219@s.whatsapp.net", "sender": "self", "text": "Hello, how can I assist?", "from_me": True},
    ]

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = items
        mock_get.return_value = mock_resp

        fetched = conn.fetch_recent()
        assert len(fetched) == 3

        history = conn.get_recent_history()
        assert len(history) == 1
        assert history[0]["sender"] == "Rahul"

        # Duplicate polling check
        conn.fetch_recent()
        history_after_dup = conn.get_recent_history()
        assert len(history_after_dup) == 1


def test_whatsapp_phone_number_normalization():
    conn = WhatsAppConnector("http://localhost:3000", enabled=False)

    # 10-digit Indian number
    jid1, err1 = conn.resolve_recipient("9172767219")
    assert jid1 == "919172767219@s.whatsapp.net"
    assert err1 is None

    # Full 12-digit Indian number without double country code
    jid2, err2 = conn.resolve_recipient("919172767219")
    assert jid2 == "919172767219@s.whatsapp.net"
    assert err2 is None

    # Already valid JID
    jid3, err3 = conn.resolve_recipient("919888888888@s.whatsapp.net")
    assert jid3 == "919888888888@s.whatsapp.net"
    assert err3 is None


def test_message_eligibility_filter_hard_exclusions():
    f = MessageEligibilityFilter()

    # Rejected bot / noise items
    assert f.evaluate(UnifiedMessage(source=Source.TELEGRAM, conversation_id="c1", sender="BotFather", content="Done!")).eligible is False
    assert f.evaluate(UnifiedMessage(source=Source.TELEGRAM, conversation_id="c2", sender="Telegram", content="Login code: 27985")).eligible is False
    assert f.evaluate(UnifiedMessage(source=Source.WHATSAPP, conversation_id="c3", sender="system", content="Approved and executed TASK-123")).eligible is False
    assert f.evaluate(UnifiedMessage(source=Source.WHATSAPP, conversation_id="whatsapp:919172767219@s.whatsapp.net", sender="user", content="hello")).eligible is False
    assert f.evaluate(UnifiedMessage(source=Source.WHATSAPP, conversation_id="c4", sender="me", content="sent", origin=Origin.AGENT)).eligible is False

    # Accepted human messages
    assert f.evaluate(UnifiedMessage(source=Source.WHATSAPP, conversation_id="whatsapp:919888888888@s.whatsapp.net", sender="Rahul", content="Can we meet tomorrow at 9am?")).eligible is True


def test_priority_inbox_hard_limit_and_deduplication():
    inbox = PriorityInbox()

    msg = UnifiedMessage(source=Source.WHATSAPP, conversation_id="whatsapp:123", sender="Rahul", content="Urgent meeting", message_id="m1")
    res = PriorityResult(message=msg, category=classifier.Category.WORK, score=90, level=PriorityLevel.HIGH)

    inbox.add(res)
    inbox.add(res)  # Duplicate polling insertion

    today_items = inbox.today()
    assert len(today_items) == 1

    # Hard limits per bucket (20 max)
    for i in range(30):
        m = UnifiedMessage(source=Source.GMAIL, conversation_id=f"gmail:{i}", sender=f"s_{i}", content=f"High {i}", message_id=f"msg_{i}")
        inbox.add(PriorityResult(message=m, category=classifier.Category.WORK, score=80 + i, level=PriorityLevel.HIGH))

    high_items = [item for item in inbox.today() if item["level"] == "HIGH"]
    assert len(high_items) == 20


def test_priority_query_formatting():
    inbox = PriorityInbox()

    for i in range(10):
        m_high = UnifiedMessage(source=Source.WHATSAPP, conversation_id="c1", sender=f"sender_{i}", content=f"High msg {i}", message_id=f"h_{i}")
        inbox.add(PriorityResult(message=m_high, category=classifier.Category.WORK, score=90, level=PriorityLevel.HIGH))

        m_med = UnifiedMessage(source=Source.TELEGRAM, conversation_id="c2", sender=f"sender_{i}", content=f"Med msg {i}", message_id=f"m_{i}")
        inbox.add(PriorityResult(message=m_med, category=classifier.Category.WORK, score=50, level=PriorityLevel.MEDIUM))

    output_high = render_priority_inbox(inbox, query="top 5 high priority")
    assert "HIGH PRIORITY" in output_high
    assert "MEDIUM" not in output_high
    assert "1. whatsapp" in output_high
    assert "5. whatsapp" in output_high
    assert "6. whatsapp" not in output_high


def test_agent_pn_and_lid_isolation():
    registry = ConversationRegistry({
        "whatsapp:919172767219@s.whatsapp.net": "AGENT_CHAT",
        "whatsapp:919172767219": "AGENT_CHAT",
        "whatsapp:52909752496163@lid": "AGENT_CHAT",
        "whatsapp:52909752496163": "AGENT_CHAT",
    })
    outbound_reg = OutboundRegistry()
    echo_filter = EchoFilter(outbound_reg, self_identities={"me", "assistant"})

    agent_handler = MagicMock()
    normal_handler = MagicMock()
    router = ConversationRouter(
        registry=registry,
        echo_filter=echo_filter,
        agent_command_handler=agent_handler,
        normal_message_handler=normal_handler,
    )

    # 1. Agent PN message -> Agent Command Handler (NOT normal ingestion)
    msg_agent_pn = UnifiedMessage(
        source=Source.WHATSAPP,
        conversation_id="whatsapp:919172767219@s.whatsapp.net",
        sender="919172767219@s.whatsapp.net",
        content="hello agent",
        origin=Origin.USER,
    )
    res_pn = router.route(msg_agent_pn)
    assert res_pn == "routed_agent_chat"
    assert agent_handler.called
    assert not normal_handler.called

    # Verify Agent PN message is NOT PriorityInbox eligible
    elig_filter = MessageEligibilityFilter(conversation_registry=registry)
    assert elig_filter.evaluate(msg_agent_pn).eligible is False

    agent_handler.reset_mock()
    normal_handler.reset_mock()

    # 2. Agent LID -> correctly identified as Agent Chat
    msg_agent_lid = UnifiedMessage(
        source=Source.WHATSAPP,
        conversation_id="whatsapp:52909752496163@lid",
        sender="52909752496163@lid",
        content="hello agent via LID",
        origin=Origin.USER,
    )
    res_lid = router.route(msg_agent_lid)
    assert res_lid == "routed_agent_chat"
    assert agent_handler.called
    assert not normal_handler.called
    assert elig_filter.evaluate(msg_agent_lid).eligible is False

    agent_handler.reset_mock()
    normal_handler.reset_mock()

    # 3. Agent PN outbound echo -> NOT Agent Command Handler, NOT normal ingestion
    msg_echo = UnifiedMessage(
        source=Source.WHATSAPP,
        conversation_id="whatsapp:919172767219@s.whatsapp.net",
        sender="919172767219@s.whatsapp.net",
        content="Agent response text",
        origin=Origin.AGENT,
        metadata={"from_me": True},
    )
    res_echo = router.route(msg_echo)
    assert res_echo == "dropped_echo"
    assert not agent_handler.called
    assert not normal_handler.called


def test_normal_whatsapp_routing_and_priority():
    registry = ConversationRegistry({
        "whatsapp:919172767219@s.whatsapp.net": "AGENT_CHAT",
    })
    outbound_reg = OutboundRegistry()
    echo_filter = EchoFilter(outbound_reg)

    agent_handler = MagicMock()
    normal_handler = MagicMock()
    router = ConversationRouter(
        registry=registry,
        echo_filter=echo_filter,
        agent_command_handler=agent_handler,
        normal_message_handler=normal_handler,
    )

    # 1. Rahul (normal contact) -> normal ingestion, PriorityInbox eligible
    msg_rahul = UnifiedMessage(
        source=Source.WHATSAPP,
        conversation_id="whatsapp:919888888888@s.whatsapp.net",
        sender="Rahul",
        content="Meeting tomorrow at 10am",
        origin=Origin.USER,
    )
    res_rahul = router.route(msg_rahul)
    assert res_rahul == "routed_normal"
    assert not agent_handler.called
    assert normal_handler.called

    elig_filter = MessageEligibilityFilter(conversation_registry=registry)
    assert elig_filter.evaluate(msg_rahul).eligible is True

    # 2. Normal outbound WhatsApp echo -> filtered
    msg_normal_echo = UnifiedMessage(
        source=Source.WHATSAPP,
        conversation_id="whatsapp:919888888888@s.whatsapp.net",
        sender="self",
        content="Sure Rahul, see you then!",
        origin=Origin.AGENT,
        metadata={"from_me": True},
    )
    res_normal_echo = router.route(msg_normal_echo)
    assert res_normal_echo == "dropped_echo"


def test_whatsapp_recipient_resolution_all_cases():
    conn = WhatsAppConnector("http://localhost:3000", enabled=False)
    # Inject mock recent history for contact resolution
    conn._recent_messages_history = [
        {"chat_id": "919876543210@s.whatsapp.net", "sender": "maosi", "text": "hi"},
        {"chat_id": "919876543211@s.whatsapp.net", "sender": "Vector", "text": "hey"},
    ]

    # "maosi" -> resolved JID
    jid_maosi, err_m = conn.resolve_recipient("maosi")
    assert jid_maosi == "919876543210@s.whatsapp.net"
    assert err_m is None

    # "Vector" -> resolved JID
    jid_vector, err_v = conn.resolve_recipient("Vector")
    assert jid_vector == "919876543211@s.whatsapp.net"
    assert err_v is None

    # "9172767219" -> 919172767219@s.whatsapp.net
    jid_10, err_10 = conn.resolve_recipient("9172767219")
    assert jid_10 == "919172767219@s.whatsapp.net"
    assert err_10 is None

    # "9172767219@s.whatsapp.net" -> unchanged
    jid_full, err_f = conn.resolve_recipient("9172767219@s.whatsapp.net")
    assert jid_full == "9172767219@s.whatsapp.net"
    assert err_f is None


def test_approval_flow_recipient_resolved_before_task():
    from assistant.execution.task_manager import TaskManager
    from assistant.execution.tool_router import ToolRouter
    from assistant.agent_core.policy_engine import PolicyEngine
    from assistant.agent_core.task_planner import TaskPlanner
    from assistant.agent_core.approval_manager import ApprovalManager
    from assistant.agent_core.graph.tools import build_tools

    outbound_reg = OutboundRegistry()
    tm = TaskManager(outbound_reg)
    tr = ToolRouter(tm)
    pe = PolicyEngine({"capabilities": {"send_whatsapp_message": {"requires_approval": True}}})
    tp = TaskPlanner(tm, pe)
    am = ApprovalManager(tm)
    conn = WhatsAppConnector("http://localhost:3000", enabled=False)
    conn._recent_messages_history = [
        {"chat_id": "919876543210@s.whatsapp.net", "sender": "maosi", "text": "hi"},
    ]

    sent_recipient = []
    tr.register("send_whatsapp_message", lambda task: (sent_recipient.append(task.target), {"status": "ok", "detail": "sent"})[1])

    pi = PriorityInbox()
    tools = build_tools(tp, am, pi, tr, whatsapp_connector=conn)
    send_tool = next(t for t in tools if t.name == "send_whatsapp_message")

    # 1. User requests send -> recipient resolved BEFORE task creation
    try:
        send_tool.invoke({"recipient": "maosi", "content": "hi"})
    except Exception:
        pass  # Interrupt expected in graph context

    # 2. task.target = resolved JID BEFORE approval
    tasks = list(tm._tasks.values())
    assert len(tasks) == 1
    task = tasks[0]
    assert task.target == "919876543210@s.whatsapp.net"

    # 3. Approval -> ToolRouter -> same resolved JID -> connector
    approved_task = am.approve(task.task_id)
    res = tr.dispatch(approved_task, source="whatsapp", conversation_id=approved_task.target)

    assert res["status"] == "ok"
    assert sent_recipient == ["919876543210@s.whatsapp.net"]


def test_task_planner_whatsapp_target_boundary_enforcement():
    from assistant.execution.task_manager import TaskManager
    from assistant.agent_core.policy_engine import PolicyEngine
    from assistant.agent_core.task_planner import TaskPlanner
    from assistant.agent_core.intent_understanding import Intent

    outbound_reg = OutboundRegistry()
    tm = TaskManager(outbound_reg)
    pe = PolicyEngine({"capabilities": {"send_whatsapp_message": {"requires_approval": True}}})
    conn = WhatsAppConnector("http://localhost:3000", enabled=False)
    conn._recent_messages_history = [
        {"chat_id": "919876543210@s.whatsapp.net", "sender": "maosi", "text": "hi"},
        {"chat_id": "919876543211@s.whatsapp.net", "sender": "Vector", "text": "hey"},
    ]

    tp = TaskPlanner(tm, pe, whatsapp_connector=conn)

    # 1. Plan for "maosi"
    intent_maosi = Intent(intent="SEND_MESSAGE", platform="whatsapp", recipient="maosi", content="hi")
    task_m = tp.plan(intent_maosi)
    assert task_m is not None
    assert task_m.target == "919876543210@s.whatsapp.net"

    # 2. Plan for "Vector"
    intent_vector = Intent(intent="SEND_MESSAGE", platform="whatsapp", recipient="Vector", content="hello")
    task_v = tp.plan(intent_vector)
    assert task_v is not None
    assert task_v.target == "919876543211@s.whatsapp.net"


def test_non_agent_lid_classification():
    registry = ConversationRegistry({
        "whatsapp:919172767219@s.whatsapp.net": "AGENT_CHAT",
        "whatsapp:52909752496163@lid": "AGENT_CHAT",
    })

    conn = WhatsAppConnector("http://localhost:3000", enabled=False)
    # Agent identity checks
    assert conn.is_agent_user_identity("52909752496163@lid") is True
    assert conn.is_agent_user_identity("919172767219@s.whatsapp.net") is True

    # Arbitrary LID contact must NOT be classified as agent user identity
    arbitrary_lid = "280534429257745@lid"
    assert conn.is_agent_user_identity(arbitrary_lid) is False
    assert registry.classify(f"whatsapp:{arbitrary_lid}") == ConversationClass.NORMAL


def test_whatsapp_deduplication_by_message_id(requests_mock=None):
    from unittest.mock import patch, MagicMock

    conn = WhatsAppConnector("http://localhost:3000", enabled=True)

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.raise_for_status = lambda: None
    mock_resp.json.return_value = [
        {"id": "ABC123", "chat_id": "919876543210@s.whatsapp.net", "sender": "maosi", "text": "hello", "from_me": False}
    ]

    with patch("requests.get", return_value=mock_resp):
        # 1st fetch: new message -> returned
        items1 = conn.fetch_recent()
        assert len(items1) == 1
        assert items1[0]["id"] == "ABC123"

        # 2nd fetch with same message ID -> duplicate ignored
        items2 = conn.fetch_recent()
        assert len(items2) == 0


def test_whatsapp_two_identical_texts_different_ids():
    from unittest.mock import patch, MagicMock

    conn = WhatsAppConnector("http://localhost:3000", enabled=True)

    mock_resp1 = MagicMock()
    mock_resp1.raise_for_status = lambda: None
    mock_resp1.json.return_value = [
        {"id": "ABC123", "chat_id": "919876543210@s.whatsapp.net", "sender": "maosi", "text": "hello", "from_me": False}
    ]

    mock_resp2 = MagicMock()
    mock_resp2.raise_for_status = lambda: None
    mock_resp2.json.return_value = [
        {"id": "XYZ789", "chat_id": "919876543210@s.whatsapp.net", "sender": "maosi", "text": "hello", "from_me": False}
    ]

    with patch("requests.get", side_effect=[mock_resp1, mock_resp2]):
        # 1st message with text "hello"
        items1 = conn.fetch_recent()
        assert len(items1) == 1
        assert items1[0]["id"] == "ABC123"

        # 2nd separate message with same text "hello" but different ID "XYZ789"
        items2 = conn.fetch_recent()
        assert len(items2) == 1
        assert items2[0]["id"] == "XYZ789"



