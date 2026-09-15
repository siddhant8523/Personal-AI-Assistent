import logging
import time
from unittest.mock import MagicMock

import pytest

from assistant.agent_core.graph.tools import build_tools
from assistant.ingestion.normalizer import (
    normalize_telegram,
    normalize_whatsapp,
    parse_payload_timestamp,
)
from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage
from assistant.intelligence import classifier
from assistant.intelligence.priority_engine import (
    PriorityLevel,
    PriorityResult,
)

from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.router.conversation_registry import ConversationClass, ConversationRegistry
from assistant.router.conversation_router import ConversationRouter
from assistant.router.echo_filter import EchoFilter
import os
from assistant.storage.db import get_connection, reset_connection


@pytest.fixture(autouse=True)
def setup_tmp_db(tmp_path):
    db_file = str(tmp_path / "test_assistant.db")
    os.environ["DATABASE_PATH"] = db_file
    reset_connection()
    yield
    reset_connection()



def make_priority_result(
    source: Source,
    sender: str,
    content: str,
    ts: float,
    chat_id: str = "chat1",
    msg_id: str = "mid1",
    score: int = 50,
    level: PriorityLevel = PriorityLevel.MEDIUM,
) -> PriorityResult:
    um = UnifiedMessage(
        source=source,
        conversation_id=f"{source.value}:{chat_id}",
        sender=sender,
        content=content,
        timestamp=ts,
        origin=Origin.USER,
        message_id=msg_id,
        platform_message_id=msg_id,
    )
    return PriorityResult(
        message=um,
        score=score,
        level=level,
        category=classifier.Category.PERSONAL,
    )



def test_telegram_receive_store_and_latest_message_returned():
    inbox = PriorityInbox()
    t_now = time.time()
    inbox.add(make_priority_result(Source.TELEGRAM, "Maosi", "Hello how are you?", t_now, msg_id="tg_1"))

    # Build tools with priority inbox
    task_planner = MagicMock()
    approval_mgr = MagicMock()
    tool_router = MagicMock()
    telegram_conn = MagicMock()
    telegram_conn.enabled = False

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_mgr,
        priority_inbox=inbox,
        tool_router=tool_router,
        telegram_connector=telegram_conn,
    )
    read_tg = next(t for t in tools if t.name == "read_telegram_messages")
    res = read_tg.invoke({})

    assert "LATEST TELEGRAM MESSAGE" in res
    assert "From: Maosi" in res
    assert "Message: Hello how are you?" in res


def test_whatsapp_receive_store_and_latest_message_returned():
    inbox = PriorityInbox()
    t_now = time.time()
    inbox.add(make_priority_result(Source.WHATSAPP, "Rahul", "Are we meeting today?", t_now, msg_id="wa_1"))

    task_planner = MagicMock()
    approval_mgr = MagicMock()
    tool_router = MagicMock()
    whatsapp_conn = MagicMock()
    whatsapp_conn.get_recent_history.return_value = []

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_mgr,
        priority_inbox=inbox,
        tool_router=tool_router,
        whatsapp_connector=whatsapp_conn,
    )
    read_wa = next(t for t in tools if t.name == "read_whatsapp_messages")
    res = read_wa.invoke({})

    assert "Rahul: Are we meeting today?" in res


def test_read_messages_still_returned_and_unread_status_does_not_affect():
    inbox = PriorityInbox()
    t_old = time.time() - 3600 * 48  # 48 hours ago
    inbox.add(make_priority_result(Source.TELEGRAM, "Alice", "Read message 2 days ago", t_old, msg_id="tg_old"))

    tools = build_tools(
        task_planner=MagicMock(),
        approval_manager=MagicMock(),
        priority_inbox=inbox,
        tool_router=MagicMock(),
        telegram_connector=MagicMock(enabled=False),
    )
    read_tg = next(t for t in tools if t.name == "read_telegram_messages")
    res = read_tg.invoke({})

    # Must still be returned even if 48 hours old and already read
    assert "LATEST TELEGRAM MESSAGE" in res
    assert "From: Alice" in res
    assert "Message: Read message 2 days ago" in res


def test_multiple_messages_ordered_newest_first():
    inbox = PriorityInbox()
    base_t = time.time()
    inbox.add(make_priority_result(Source.WHATSAPP, "UserA", "Message 1 (oldest)", base_t + 10, msg_id="wa_1"))
    inbox.add(make_priority_result(Source.WHATSAPP, "UserB", "Message 2 (middle)", base_t + 20, msg_id="wa_2"))
    inbox.add(make_priority_result(Source.WHATSAPP, "UserC", "Message 3 (newest)", base_t + 30, msg_id="wa_3"))

    whatsapp_conn = MagicMock()
    whatsapp_conn.get_recent_history.return_value = []

    tools = build_tools(
        task_planner=MagicMock(),
        approval_manager=MagicMock(),
        priority_inbox=inbox,
        tool_router=MagicMock(),
        whatsapp_connector=whatsapp_conn,
    )
    read_wa = next(t for t in tools if t.name == "read_whatsapp_messages")
    res = read_wa.invoke({})

    lines = res.strip().split("\n")
    assert len(lines) == 3
    # Newest must be first
    assert "UserC: Message 3 (newest)" in lines[0]
    assert "UserB: Message 2 (middle)" in lines[1]
    assert "UserA: Message 1 (oldest)" in lines[2]


def test_messages_from_different_chats_handled():
    inbox = PriorityInbox()
    base_t = time.time()
    inbox.add(make_priority_result(Source.TELEGRAM, "ChatFamily", "Dinner at 8", base_t + 10, chat_id="chat_fam", msg_id="tg_f"))
    inbox.add(make_priority_result(Source.TELEGRAM, "Colleague", "PR review ready", base_t + 20, chat_id="chat_work", msg_id="tg_w"))

    tools = build_tools(
        task_planner=MagicMock(),
        approval_manager=MagicMock(),
        priority_inbox=inbox,
        tool_router=MagicMock(),
        telegram_connector=MagicMock(enabled=False),
    )
    read_tg = next(t for t in tools if t.name == "read_telegram_messages")
    res = read_tg.invoke({})

    assert "LATEST TELEGRAM MESSAGE" in res
    assert "From: Colleague" in res
    assert "Message: PR review ready" in res
    assert "OTHER RECENT TELEGRAM MESSAGES" in res
    assert "From: ChatFamily" in res
    assert "Message: Dinner at 8" in res
    assert res.index("Colleague") < res.index("ChatFamily")


def test_platform_filtering_works():
    inbox = PriorityInbox()
    t_now = time.time()
    inbox.add(make_priority_result(Source.TELEGRAM, "TgSender", "Tg message content", t_now, msg_id="tg_x"))
    inbox.add(make_priority_result(Source.WHATSAPP, "WaSender", "Wa message content", t_now + 1, msg_id="wa_x"))

    tools = build_tools(
        task_planner=MagicMock(),
        approval_manager=MagicMock(),
        priority_inbox=inbox,
        tool_router=MagicMock(),
        telegram_connector=MagicMock(enabled=False),
        whatsapp_connector=MagicMock(get_recent_history=MagicMock(return_value=[])),
    )
    read_tg = next(t for t in tools if t.name == "read_telegram_messages")
    read_wa = next(t for t in tools if t.name == "read_whatsapp_messages")

    tg_res = read_tg.invoke({})
    wa_res = read_wa.invoke({})

    # Telegram retrieval must NOT include WhatsApp messages
    assert "From: TgSender" in tg_res
    assert "Message: Tg message content" in tg_res
    assert "WaSender" not in tg_res

    # WhatsApp retrieval must NOT include Telegram messages
    assert "WaSender: Wa message content" in wa_res
    assert "TgSender" not in wa_res


def test_agent_chat_filtering_does_not_accidentally_remove_all_normal_messages():
    registry = ConversationRegistry({
        "telegram_bot:12345": "AGENT_CHAT",
        "whatsapp:919172767219@s.whatsapp.net": "AGENT_CHAT",
    })
    from assistant.execution.outbound_registry import OutboundRegistry
    echo_filter = EchoFilter(OutboundRegistry(window_seconds=60))
    agent_handler = MagicMock()
    normal_handler = MagicMock()

    router = ConversationRouter(
        registry=registry,
        echo_filter=echo_filter,
        agent_command_handler=agent_handler,
        normal_message_handler=normal_handler,
    )

    # 1. Agent command message
    agent_msg = UnifiedMessage(
        source=Source.WHATSAPP,
        conversation_id="whatsapp:919172767219@s.whatsapp.net",
        sender="919172767219@s.whatsapp.net",
        content="what is the last message on whatsapp ?",
    )
    res_agent = router.route(agent_msg)
    assert res_agent == "routed_agent_chat"
    assert agent_handler.call_count == 1
    assert normal_handler.call_count == 0

    # 2. Normal contact message
    contact_msg = UnifiedMessage(
        source=Source.WHATSAPP,
        conversation_id="whatsapp:919876543210@s.whatsapp.net",
        sender="Friend",
        content="Hey let's catch up",
    )
    res_normal = router.route(contact_msg)
    assert res_normal == "routed_normal"
    assert normal_handler.call_count == 1


def test_message_retrieval_diagnostic_logging(caplog):
    inbox = PriorityInbox()
    inbox.add(make_priority_result(Source.WHATSAPP, "Rahul", "Hi", 1700000000.0, msg_id="wa_diag_1"))

    tools = build_tools(
        task_planner=MagicMock(),
        approval_manager=MagicMock(),
        priority_inbox=inbox,
        tool_router=MagicMock(),
        whatsapp_connector=MagicMock(get_recent_history=MagicMock(return_value=[])),
    )
    read_wa = next(t for t in tools if t.name == "read_whatsapp_messages")

    with caplog.at_level(logging.INFO, logger="assistant.tools.message_retrieval"):
        read_wa.invoke({"query": "Rahul"})

    log_text = caplog.text
    assert "[MESSAGE-RETRIEVAL] platform=whatsapp" in log_text
    assert "[MESSAGE-RETRIEVAL] query=Rahul" in log_text
    assert "[MESSAGE-RETRIEVAL] candidate_count=1" in log_text
    assert "[MESSAGE-RETRIEVAL] newest_message_id=wa_diag_1" in log_text
    assert "[MESSAGE-RETRIEVAL] newest_timestamp=1700000000.0" in log_text
    assert "[MESSAGE-RETRIEVAL] newest_chat_id=Rahul" in log_text


def test_timestamp_normalization_handling():
    # ISO string timestamp
    iso_payload = {
        "chat_id": "123",
        "sender": "Alice",
        "text": "Hello",
        "timestamp": "2026-09-12T13:45:00.000Z",
    }
    um_wa = normalize_whatsapp(iso_payload)
    assert isinstance(um_wa.timestamp, float)
    assert um_wa.timestamp > 1.7e9

    # Millisecond int timestamp
    ms_payload = {
        "chat_id": "456",
        "from_id": "Bob",
        "text": "Hi",
        "timestamp": 1789207514424,
    }
    um_tg = normalize_telegram(ms_payload)
    assert isinstance(um_tg.timestamp, float)
    assert 1789207500 < um_tg.timestamp < 1789207600


def test_maosi_identified_as_latest_telegram_message():
    from assistant.connectors.telegram.telegram_user_client import TelegramUserClient
    inbox = PriorityInbox()
    maosi_ts = 1787846178.0
    inbox.add(make_priority_result(Source.TELEGRAM, "Maosi", "Hello how are ?", maosi_ts, chat_id="Maosi", msg_id="7acf6c25-14d0-4db2-afab-39353d02cafe"))

    telegram_personal = TelegramUserClient(api_id="123", api_hash="abc", session_file="test.session", enabled=False)
    telegram_personal.enabled = True
    telegram_personal.list_dialogs = MagicMock(return_value={
        "status": "ok",
        "dialogs": [
            {
                "name": "Asish",
                "chat_id": 1001,
                "last_message": "Hey",
                "timestamp": 1787000000.0,
            },
            {
                "name": "BotFather",
                "chat_id": 93372553,
                "last_message": "Done! Congratulations on your new bot",
                "timestamp": 1786000000.0,
            },
        ],
    })

    tools = build_tools(
        task_planner=MagicMock(),
        approval_manager=MagicMock(),
        priority_inbox=inbox,
        tool_router=MagicMock(),
        telegram_personal=telegram_personal,
    )
    read_tg = next(t for t in tools if t.name == "read_telegram_messages")
    res = read_tg.invoke({})

    assert "LATEST TELEGRAM MESSAGE" in res
    assert "From: Maosi" in res
    assert "Chat: Maosi" in res
    assert "Message: Hello how are ?" in res

    latest_block = res.split("OTHER RECENT TELEGRAM MESSAGES")[0]
    assert "Maosi" in latest_block
    assert "Hello how are ?" in latest_block
    assert "BotFather" not in latest_block


def test_older_botfather_message_cannot_be_selected_as_latest_merely_by_position():
    from assistant.connectors.telegram.telegram_user_client import TelegramUserClient
    inbox = PriorityInbox()
    inbox.add(make_priority_result(Source.TELEGRAM, "Maosi", "Hello how are ?", 1787846178.0, chat_id="Maosi", msg_id="tg_latest"))

    telegram_personal = TelegramUserClient(api_id="123", api_hash="abc", session_file="test.session", enabled=False)
    telegram_personal.enabled = True
    telegram_personal.list_dialogs = MagicMock(return_value={
        "status": "ok",
        "dialogs": [
            {
                "name": "Projectbot",
                "chat_id": 2002,
                "last_message": "I'm not sure what you'd like me to do...",
                "timestamp": 1787500000.0,
            },
            {
                "name": "BotFather",
                "chat_id": 93372553,
                "last_message": "Done! Congratulations on your new bot",
                "timestamp": 1786000000.0,
            },
        ],
    })

    tools = build_tools(
        task_planner=MagicMock(),
        approval_manager=MagicMock(),
        priority_inbox=inbox,
        tool_router=MagicMock(),
        telegram_personal=telegram_personal,
    )
    read_tg = next(t for t in tools if t.name == "read_telegram_messages")
    res = read_tg.invoke({})

    # 1. Output must explicitly label the latest message
    assert "LATEST TELEGRAM MESSAGE\nFrom: Maosi" in res
    # 2. BotFather must NOT be in the latest section
    lines = res.strip().split("\n")
    assert lines[0] == "LATEST TELEGRAM MESSAGE"
    assert lines[1] == "From: Maosi"
    assert lines[4] == "Message: Hello how are ?"
    # 3. BotFather must be explicitly listed under OTHER RECENT TELEGRAM MESSAGES with its real timestamp
    assert "OTHER RECENT TELEGRAM MESSAGES" in res
    assert "From: BotFather" in res
    assert "Message: Done! Congratulations on your new bot" in res
    # 4. Ordering must place Maosi first, followed by Projectbot, followed by BotFather
    pos_maosi = res.index("Maosi")
    pos_proj = res.index("Projectbot")
    pos_botf = res.index("BotFather")
    assert pos_maosi < pos_proj < pos_botf


def test_telegram_retrieval_contains_explicit_timestamps_and_order():
    inbox = PriorityInbox()
    t1 = 1780000000.0
    t2 = 1785000000.0
    t3 = 1789000000.0
    inbox.add(make_priority_result(Source.TELEGRAM, "UserOld", "Old message", t1, msg_id="t1"))
    inbox.add(make_priority_result(Source.TELEGRAM, "UserMid", "Mid message", t2, msg_id="t2"))
    inbox.add(make_priority_result(Source.TELEGRAM, "UserNew", "New message", t3, msg_id="t3"))

    tools = build_tools(
        task_planner=MagicMock(),
        approval_manager=MagicMock(),
        priority_inbox=inbox,
        tool_router=MagicMock(),
        telegram_connector=MagicMock(enabled=False),
    )
    read_tg = next(t for t in tools if t.name == "read_telegram_messages")
    res = read_tg.invoke({})

    # Explicit timestamps
    assert "Time:" in res
    assert "UTC" in res
    # Chronological ordering (newest first)
    pos_new = res.index("UserNew")
    pos_mid = res.index("UserMid")
    pos_old = res.index("UserOld")
    assert pos_new < pos_mid < pos_old


def test_query_based_telegram_retrieval():
    inbox = PriorityInbox()
    base_t = time.time()
    inbox.add(make_priority_result(Source.TELEGRAM, "Maosi", "Hello how are ?", base_t + 30, msg_id="q1"))
    inbox.add(make_priority_result(Source.TELEGRAM, "Alice", "Lunch meeting tomorrow at 1pm", base_t + 20, msg_id="q2"))
    inbox.add(make_priority_result(Source.TELEGRAM, "Bob", "Project status report", base_t + 10, msg_id="q3"))

    tools = build_tools(
        task_planner=MagicMock(),
        approval_manager=MagicMock(),
        priority_inbox=inbox,
        tool_router=MagicMock(),
        telegram_connector=MagicMock(enabled=False),
    )
    read_tg = next(t for t in tools if t.name == "read_telegram_messages")

    # Search for specific keyword
    res_lunch = read_tg.invoke({"query": "lunch"})
    assert "Alice" in res_lunch
    assert "Lunch meeting tomorrow" in res_lunch
    assert "Maosi" not in res_lunch
    assert "Bob" not in res_lunch

    # Search by contact name
    res_maosi = read_tg.invoke({"query": "Maosi"})
    assert "Maosi" in res_maosi
    assert "Hello how are ?" in res_maosi
    assert "Alice" not in res_maosi
    assert "Bob" not in res_maosi

