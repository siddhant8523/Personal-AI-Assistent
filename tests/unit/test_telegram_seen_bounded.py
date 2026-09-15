import pytest
from assistant.connectors.telegram.telegram_user_client import TelegramUserClient


def test_telegram_seen_messages_bounded():
    client = TelegramUserClient(api_id="", api_hash="", session_file="", enabled=False, max_seen=3)

    client._mark_seen(("chat1", "msg1"))
    client._mark_seen(("chat1", "msg2"))
    client._mark_seen(("chat1", "msg3"))

    assert client._is_seen(("chat1", "msg1")) is True
    assert client._is_seen(("chat1", "msg2")) is True
    assert client._is_seen(("chat1", "msg3")) is True
    assert len(client._seen_messages) == 3

    # Add 4th message -> 1st message ("chat1", "msg1") must be evicted
    client._mark_seen(("chat1", "msg4"))

    assert len(client._seen_messages) == 3
    assert client._is_seen(("chat1", "msg1")) is False
    assert client._is_seen(("chat1", "msg4")) is True
