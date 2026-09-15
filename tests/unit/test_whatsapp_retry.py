"""
Unit tests for WhatsApp message store, retry handling, and recipient stability.

Verifies:
1. Outbound message is stored.
2. getMessage(key) retrieves a stored outbound message.
3. Unknown message ID returns undefined.
4. Inbound message is stored.
5. Store remains bounded.
6. Retry counter cache exists and functions.
7. Recreated socket accesses the same in-process store.
8. Existing WhatsApp sending still works.
9. Existing receiving still works.
10. Presence implementation remains unchanged.
11. Existing tests continue passing.
"""

import os
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from assistant.connectors.whatsapp.whatsapp_connector import WhatsAppConnector
from assistant.ingestion.adapters import adapt_whatsapp
from assistant.ingestion.unified_message import Source, Origin


def test_node_bridge_message_store_and_retry_unit_tests():
    """Executes the Node.js test suite for BoundedMessageStore and getMessage()."""
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bridge_dir = os.path.join(root_dir, "src", "assistant", "connectors", "whatsapp", "baileys_bridge")
    test_file = os.path.join(bridge_dir, "test_retry_store.js")

    proc = subprocess.run(
        ["node", test_file],
        cwd=bridge_dir,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"Node tests failed: {proc.stderr}\n{proc.stdout}"
    assert "All Baileys Bridge message store and retry tests passed successfully!" in proc.stdout


def test_whatsapp_connector_outbound_and_inbound_intact():
    """Verify WhatsAppConnector sending and receiving mechanisms continue working."""
    connector = WhatsAppConnector(bridge_url="http://localhost:3000", enabled=False)

    # Sending in mock mode
    send_res = connector.send_message("919172767219@s.whatsapp.net", "Hello test")
    assert send_res.get("status") == "ok"
    assert "[MOCK]" in send_res.get("detail", "")

    # Inbound normalization
    raw = {
        "id": "MSG_1001",
        "chat_id": "919172767219@s.whatsapp.net",
        "sender": "919172767219@s.whatsapp.net",
        "text": "Hello assistant",
        "from_me": False,
    }
    unified = adapt_whatsapp(raw)
    assert unified.source == Source.WHATSAPP
    assert unified.origin == Origin.USER
    assert unified.content == "Hello assistant"
    assert unified.conversation_id == "whatsapp:919172767219@s.whatsapp.net"


def test_presence_implementation_remains_unchanged():
    """Verify WhatsAppPresenceIndicator continues functioning with send_presence_update."""
    from assistant.channels.presence import WhatsAppPresenceIndicator, PresenceCoordinator

    mock_conn = MagicMock()
    coord = PresenceCoordinator()
    indicator = WhatsAppPresenceIndicator(mock_conn, "919172767219@s.whatsapp.net", coordinator=coord)

    indicator.start()
    assert indicator.is_active
    assert mock_conn.send_presence_update.called
    assert mock_conn.send_presence_update.call_args[0] == ("919172767219@s.whatsapp.net", "composing")

    indicator.stop()
    assert not indicator.is_active
    assert mock_conn.send_presence_update.call_args[0] == ("919172767219@s.whatsapp.net", "paused")
