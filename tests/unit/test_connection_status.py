"""
Unit Tests for Truthful Connection Status Handling (tests/unit/test_connection_status.py)
========================================================================================
Validates:
A. Internet available: Chat = Online (when runtime is started)
B. Internet unavailable: Chat = Offline (even when runtime is started)
C. Previously connected connector + network loss: Connector must not remain falsely Connected
D. Internet restored: Chat returns Online automatically
E. Connector reconnecting: Show Connecting/Reconnecting rather than Connected prematurely
F. Actual connector disconnected while internet is available:
   Chat = Online, WhatsApp = Disconnected, Telegram = Connected, Android = Connected
G. Streamlit rerun does not resurrect stale Connected/Online states
H. Android Status: "Connected" strictly requires a live, authenticated WebSocket DeviceHandle
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from assistant.runtime import AssistantRuntime
from assistant.ui.styles import ConnectionStatus, render_header_html
# pyrefly: ignore [missing-import]
import streamlit_app


# ==============================================================================
# Helper Mock Factories
# ==============================================================================

def make_test_runtime(network_online: bool = True, started: bool = True):
    """Creates a mock AssistantRuntime with controllable network and service states."""
    runtime = MagicMock(spec=AssistantRuntime)
    runtime.is_started = started
    runtime._started = started

    # Dynamic network availability
    runtime.is_network_available.side_effect = lambda force_refresh=False: network_online

    # Default Chat status logic
    def _chat_status(force_refresh=False):
        if not runtime.is_started or not network_online:
            return {"connected": False, "status": "offline", "label": "Offline"}
        return {"connected": True, "status": "online", "label": "Online"}

    runtime.get_chat_status.side_effect = _chat_status

    # Default connector status values
    runtime.get_whatsapp_status.return_value = {
        "running": True,
        "connected": network_online,
        "status": "connected" if network_online else "offline",
        "label": "Connected" if network_online else "Offline",
    }
    runtime.get_telegram_status.return_value = {
        "connected": network_online,
        "status": "connected" if network_online else "offline",
        "label": "Connected" if network_online else "Offline",
        "bot": {"connected": network_online},
        "user": {"connected": network_online},
    }
    runtime.get_android_status.return_value = {
        "server_running": True,
        "connected": network_online,
        "status": "connected" if network_online else "offline",
        "label": "Connected" if network_online else "Offline",
        "device_name": "Pixel 8" if network_online else None,
    }

    return runtime


# ==============================================================================
# Test Scenario A: Internet Available -> Chat = Online
# ==============================================================================

def test_scenario_a_internet_available_chat_online():
    """Chat must show 'Online' only when runtime is operational and network connectivity is available."""
    runtime = make_test_runtime(network_online=True, started=True)

    status_data = runtime.get_chat_status()
    chat_conn = streamlit_app.resolve_connection_status("Chat", status_data)

    assert chat_conn.is_connected is True
    assert chat_conn.css_class == "online"
    assert chat_conn.label == "Online"
    assert chat_conn.dot == "🟢"

    html = render_header_html(chat_conn)
    assert "status-badge online" in html
    assert "status-dot online" in html
    assert "Online" in html


# ==============================================================================
# Test Scenario B: Internet Unavailable -> Chat = Offline
# ==============================================================================

def test_scenario_b_internet_unavailable_chat_offline():
    """When network/internet connectivity is lost, Chat must transition to 'Offline' even if app is started."""
    runtime = make_test_runtime(network_online=False, started=True)

    status_data = runtime.get_chat_status()
    chat_conn = streamlit_app.resolve_connection_status("Chat", status_data)

    assert chat_conn.is_connected is False
    assert chat_conn.css_class == "offline"
    assert chat_conn.label == "Offline"
    assert chat_conn.dot == "🔴"

    html = render_header_html(chat_conn)
    assert "status-badge offline" in html
    assert "status-dot offline" in html
    assert "Offline" in html
    assert "Online" not in html


# ==============================================================================
# Test Scenario C: Previously Connected Connector + Network Loss
# ==============================================================================

def test_scenario_c_connector_cannot_remain_falsely_connected_on_network_loss():
    """If connectors were previously connected, a network loss must immediately reflect as Offline."""
    real_runtime = AssistantRuntime()
    real_runtime._started = True

    # Simulate connectors were reporting healthy state
    real_runtime.whatsapp_connector = MagicMock()
    real_runtime.whatsapp_connector.get_status.return_value = {
        "running": True,
        "connected": True,
        "reconnecting": False,
        "qr": None,
    }
    real_runtime.telegram_connector = MagicMock()
    real_runtime.telegram_connector.get_status.return_value = {"connected": True}
    real_runtime.telegram_personal = MagicMock()
    real_runtime.telegram_personal.get_status.return_value = {"connected": True, "reconnecting": False}
    real_runtime.device_gateway = MagicMock()
    real_runtime.device_gateway.get_device_status.return_value = {
        "connected": True,
        "device_name": "Pixel 8",
        "reconnecting": False,
    }

    # Simulate network loss
    with patch.object(real_runtime, "is_network_available", return_value=False):
        wa_st = real_runtime.get_whatsapp_status(force_refresh=True)
        tg_st = real_runtime.get_telegram_status(force_refresh=True)
        and_st = real_runtime.get_android_status(force_refresh=True)
        chat_st = real_runtime.get_chat_status(force_refresh=True)

        # None of them must remain Connected!
        assert wa_st["connected"] is False
        assert wa_st["status"] == "offline"

        assert tg_st["connected"] is False
        assert tg_st["status"] == "offline"

        assert and_st["connected"] is False
        assert and_st["status"] == "offline"

        assert chat_st["connected"] is False
        assert chat_st["status"] == "offline"


# ==============================================================================
# Test Scenario D: Internet Restored -> Chat Returns Online Automatically
# ==============================================================================

def test_scenario_d_internet_restored_chat_returns_online():
    """When connectivity returns, Chat recovers to 'Online' automatically without permanent caching."""
    real_runtime = AssistantRuntime()
    real_runtime._started = True

    # First: network down
    with patch("assistant.runtime.check_network_connectivity", return_value=False):
        st1 = real_runtime.get_chat_status(force_refresh=True)
        conn1 = streamlit_app.resolve_connection_status("Chat", st1)
        assert conn1.label == "Offline"
        assert conn1.is_connected is False

    # Second: network restored
    with patch("assistant.runtime.check_network_connectivity", return_value=True):
        st2 = real_runtime.get_chat_status(force_refresh=True)
        conn2 = streamlit_app.resolve_connection_status("Chat", st2)
        assert conn2.label == "Online"
        assert conn2.is_connected is True


# ==============================================================================
# Test Scenario E: Connector Reconnecting
# ==============================================================================

def test_scenario_e_connector_reconnecting_shows_connecting_not_connected():
    """Show Connecting/Reconnecting rather than Connected prematurely when reconnecting."""
    real_runtime = AssistantRuntime()
    real_runtime._started = True

    # Network is available, but connectors are in reconnecting cycle
    with patch.object(real_runtime, "is_network_available", return_value=True):
        # 1. WhatsApp reconnecting
        real_runtime.whatsapp_connector = MagicMock()
        real_runtime.whatsapp_connector.get_status.return_value = {
            "running": True,
            "connected": False,
            "reconnecting": True,
            "qr": None,
        }
        wa_st = real_runtime.get_whatsapp_status()
        wa_conn = streamlit_app.resolve_connection_status("WhatsApp", wa_st)

        assert wa_st["connected"] is False
        assert wa_conn.is_connected is False
        assert wa_conn.css_class == "connecting"
        assert wa_conn.dot == "🟡"
        assert wa_conn.label == "Reconnecting"
        assert wa_conn.sidebar_label == "Reconnecting"

        # 2. Telegram user client reconnecting
        real_runtime.telegram_connector = MagicMock()
        real_runtime.telegram_connector.get_status.return_value = {"connected": False}
        real_runtime.telegram_personal = MagicMock()
        real_runtime.telegram_personal.get_status.return_value = {
            "connected": False,
            "reconnecting": True,
            "authorized": True,
        }
        tg_st = real_runtime.get_telegram_status()
        tg_conn = streamlit_app.resolve_connection_status("Telegram", tg_st)

        assert tg_st["connected"] is False
        assert tg_conn.is_connected is False
        assert tg_conn.css_class == "connecting"
        assert tg_conn.dot == "🟡"
        assert tg_conn.label == "Connecting"

        # 3. Android device reconnecting
        real_runtime.device_gateway = MagicMock()
        real_runtime.device_gateway.get_device_status.return_value = {
            "connected": False,
            "reconnecting": True,
            "device_name": None,
        }
        and_st = real_runtime.get_android_status()
        and_conn = streamlit_app.resolve_connection_status("Android", and_st)

        assert and_st["connected"] is False
        assert and_conn.is_connected is False
        assert and_conn.css_class == "connecting"
        assert and_conn.dot == "🟡"
        assert and_conn.label == "Reconnecting"


# ==============================================================================
# Test Scenario F: Connector Disconnected while Internet Available
# ==============================================================================

def test_scenario_f_connector_disconnected_while_internet_available():
    """Do not overwrite an individually disconnected connector with 'Connected' just because network is available."""
    real_runtime = AssistantRuntime()
    real_runtime._started = True

    real_runtime.whatsapp_connector = MagicMock()
    # WhatsApp bridge is stopped / disconnected
    real_runtime.whatsapp_connector.get_status.return_value = {
        "running": False,
        "connected": False,
        "reconnecting": False,
        "qr": None,
    }

    # Telegram is connected
    real_runtime.telegram_connector = MagicMock()
    real_runtime.telegram_connector.get_status.return_value = {"connected": True}
    real_runtime.telegram_personal = MagicMock()
    real_runtime.telegram_personal.get_status.return_value = {"connected": True, "reconnecting": False}

    # Android is connected
    real_runtime.device_gateway = MagicMock()
    real_runtime.device_gateway.get_device_status.return_value = {
        "connected": True,
        "device_name": "Galaxy S24",
        "reconnecting": False,
    }

    with patch.object(real_runtime, "is_network_available", return_value=True):
        chat_st = real_runtime.get_chat_status()
        wa_st = real_runtime.get_whatsapp_status()
        tg_st = real_runtime.get_telegram_status()
        and_st = real_runtime.get_android_status()

        chat_conn = streamlit_app.resolve_connection_status("Chat", chat_st)
        wa_conn = streamlit_app.resolve_connection_status("WhatsApp", wa_st)
        tg_conn = streamlit_app.resolve_connection_status("Telegram", tg_st)
        and_conn = streamlit_app.resolve_connection_status("Android", and_st)

        # Expected:
        # Chat       Online
        # WhatsApp   Disconnected / Not connected
        # Telegram   Connected
        # Android    Connected
        assert chat_conn.label == "Online"
        assert chat_conn.dot == "🟢"

        assert wa_conn.is_connected is False
        assert wa_conn.label == "Not connected"
        assert wa_conn.dot == "🔴"

        assert tg_conn.is_connected is True
        assert tg_conn.label == "Connected"
        assert tg_conn.dot == "🟢"

        assert and_conn.is_connected is True
        assert and_conn.label == "Connected"
        assert and_conn.dot == "🟢"


# ==============================================================================
# Test Scenario G: Streamlit Rerun Freshness (No Stale Resurrections)
# ==============================================================================

def test_scenario_g_streamlit_rerun_does_not_resurrect_stale_states():
    """Streamlit reruns must dynamically re-evaluate status and not restore cached online/connected."""
    mock_runtime = MagicMock()
    mock_runtime.is_started = True

    # Pass 1: Network is online, all connected
    mock_runtime.get_chat_status.return_value = {"connected": True, "status": "online", "label": "Online"}
    mock_runtime.get_whatsapp_status.return_value = {"connected": True, "status": "connected", "label": "Connected"}
    mock_runtime.get_telegram_status.return_value = {"connected": True, "status": "connected", "label": "Connected"}
    mock_runtime.get_android_status.return_value = {"connected": True, "status": "connected", "label": "Connected"}

    rendered_buttons_1 = []
    rendered_md_1 = []
    with patch("streamlit.sidebar"), \
         patch("streamlit.markdown", side_effect=lambda c, **kw: rendered_md_1.append(str(c))), \
         patch("streamlit.button", side_effect=lambda l, **kw: rendered_buttons_1.append(str(l)) and False):
        streamlit_app.render_sidebar(mock_runtime)

    assert any("WhatsApp   🟢 Connected" in b for b in rendered_buttons_1)
    assert any("Telegram   🟢 Connected" in b for b in rendered_buttons_1)
    assert any("Android   🟢 Connected" in b for b in rendered_buttons_1)
    assert any("status-badge online" in m for m in rendered_md_1)

    # Pass 2: Rerun triggered after laptop lost connectivity
    mock_runtime.get_chat_status.return_value = {"connected": False, "status": "offline", "label": "Offline"}
    mock_runtime.get_whatsapp_status.return_value = {"connected": False, "status": "offline", "label": "Offline"}
    mock_runtime.get_telegram_status.return_value = {"connected": False, "status": "offline", "label": "Offline"}
    mock_runtime.get_android_status.return_value = {"connected": False, "status": "offline", "label": "Offline"}

    rendered_buttons_2 = []
    rendered_md_2 = []
    with patch("streamlit.sidebar"), \
         patch("streamlit.markdown", side_effect=lambda c, **kw: rendered_md_2.append(str(c))), \
         patch("streamlit.button", side_effect=lambda l, **kw: rendered_buttons_2.append(str(l)) and False):
        streamlit_app.render_sidebar(mock_runtime)

    # Stale 'Connected' must NOT be resurrected on rerun
    assert any("WhatsApp   🔴 Offline" in b for b in rendered_buttons_2)
    assert any("Telegram   🔴 Offline" in b for b in rendered_buttons_2)
    assert any("Android   🔴 Offline" in b for b in rendered_buttons_2)
    assert any("status-badge offline" in m for m in rendered_md_2)
    assert not any("Connected" in b for b in rendered_buttons_2)


# ==============================================================================
# Test Scenario H: Android Strict Live-Socket Rule
# ==============================================================================

def test_scenario_h_android_strictly_requires_live_websocket_handle():
    """'Connected' strictly requires a live, authenticated WebSocket DeviceHandle.
    Configured URLs, stored tokens, or dead handles cannot produce Connected.
    """
    from assistant.device_gateway.device_gateway import DeviceGateway
    from assistant.device_gateway.transport.ws_transport import DeviceHandle

    validator = MagicMock()
    gateway = DeviceGateway(validator)

    # Case 1: No device attached
    assert gateway.is_device_connected() is False
    status1 = gateway.get_device_status()
    assert status1["connected"] is False

    # Case 2: Live device attached
    mock_ws = MagicMock()
    mock_ws.closed = False
    handle = DeviceHandle(mock_ws, device_name="Pixel 8 Pro")
    gateway.attach_device(handle)

    assert gateway.is_device_connected() is True
    status2 = gateway.get_device_status()
    assert status2["connected"] is True
    assert status2["device_name"] == "Pixel 8 Pro"

    # Case 3: WebSocket closes abruptly
    mock_ws.closed = True
    assert handle.is_alive() is False
    assert gateway.is_device_connected() is False

    status3 = gateway.get_device_status()
    assert status3["connected"] is False
    # Marked as reconnecting because it was previously connected
    assert status3["reconnecting"] is True


# ==============================================================================
# Test Scenario I: Shared Visual Component Styling
# ==============================================================================

def test_connection_status_shared_visual_styling():
    """All channels use the same ConnectionStatus HTML structure with dark theme classes."""
    chat = ConnectionStatus(name="Chat", status="online")
    wa = ConnectionStatus(name="WhatsApp", status="connected")
    tg = ConnectionStatus(name="Telegram", status="connecting")
    android = ConnectionStatus(name="Android", status="offline")

    chat_html = chat.to_html(include_name=True)
    wa_html = wa.to_html(include_name=True)
    tg_html = tg.to_html(include_name=True)
    android_html = android.to_html(include_name=True)

    # Shared structure: status-badge and status-dot
    for h in (chat_html, wa_html, tg_html, android_html):
        assert "status-badge" in h
        assert "status-dot" in h

    assert "online" in chat_html
    assert "connected" in wa_html
    assert "connecting" in tg_html
    assert "offline" in android_html
