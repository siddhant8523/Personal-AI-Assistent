"""Unit tests for Telegram Personal Account capabilities and tools."""

import os
from unittest.mock import MagicMock
from assistant.connectors.telegram.telegram_user_client import TelegramUserClient
from assistant.agent_core.graph.tools import build_tools, TelegramFindContactArgs, TelegramReadMessagesArgs


def test_telegram_personal_disabled_by_default():
    client = TelegramUserClient(api_id="", api_hash="", session_file="", enabled=False)
    assert client.enabled is False
    assert client.get_me() == {"status": "failed", "detail": "Personal Telegram is not authenticated"}
    assert client.find_contact("Sidd") == {"status": "failed", "detail": "Personal Telegram is not authenticated"}
    assert client.list_dialogs() == {"status": "failed", "detail": "Personal Telegram is not authenticated"}
    assert client.count_summary() == {"status": "failed", "detail": "Personal Telegram is not authenticated"}
    assert client.read_messages("Sidd") == {"status": "failed", "detail": "Personal Telegram is not authenticated"}
    assert client.search_messages("meeting") == {"status": "failed", "detail": "Personal Telegram is not authenticated"}
    assert client.list_groups() == {"status": "failed", "detail": "Personal Telegram is not authenticated"}
    assert client.list_channels() == {"status": "failed", "detail": "Personal Telegram is not authenticated"}


def test_telegram_personal_tool_registration():
    mock_planner = MagicMock()
    mock_approval = MagicMock()
    mock_inbox = MagicMock()
    mock_router = MagicMock()
    
    client = TelegramUserClient(api_id="123", api_hash="abc", session_file="test.session", enabled=False)
    client.enabled = True

    tools = build_tools(
        task_planner=mock_planner,
        approval_manager=mock_approval,
        priority_inbox=mock_inbox,
        tool_router=mock_router,
        telegram_personal=client,
    )

    tool_names = [t.name for t in tools]
    expected = [
        "telegram_get_account",
        "telegram_find_contact",
        "telegram_list_dialogs",
        "telegram_count_chats",
        "telegram_read_messages",
        "telegram_search_messages",
        "telegram_list_groups",
        "telegram_list_channels",
    ]
    for name in expected:
        assert name in tool_names, f"Missing tool: {name}"
