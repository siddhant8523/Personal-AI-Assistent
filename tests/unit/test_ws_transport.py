"""Unit tests for WebSocket transport disconnect and lifecycle handling."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock
import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from assistant.device_gateway.device_gateway import DeviceGateway
from assistant.device_gateway.transport.protocol import DeviceCommand
from assistant.device_gateway.transport.ws_transport import DeviceHandle, _handler
from assistant.security.auth import DeviceAuth
from assistant.security.capability_validator import CapabilityValidator


@pytest.fixture
def auth():
    return DeviceAuth(expected_token="test-secret-token")


@pytest.fixture
def gateway():
    validator = CapabilityValidator(android_capabilities=["sms.read"], cloud_capabilities=[])
    gw = DeviceGateway(validator)
    gw.attach_device = MagicMock(wraps=gw.attach_device)
    gw.detach_device = MagicMock(wraps=gw.detach_device)
    gw.receive_event = MagicMock()
    gw.handle_device_result = MagicMock()
    return gw


class AsyncIterWS:
    """Mock WebSocket that supports async iteration and recv/send/close."""

    def __init__(self, messages=None, close_exc=None, remote_address=("192.168.1.50", 45678)):
        self.messages = list(messages or [])
        self.close_exc = close_exc
        self.remote_address = remote_address
        self.sent = []
        self.closed = []
        self.recv_call_count = 0

    async def recv(self):
        self.recv_call_count += 1
        if self.close_exc and not self.messages:
            raise self.close_exc
        if not self.messages:
            raise ConnectionClosedOK(Close(1000, "EOF"), None)
        return self.messages.pop(0)

    async def send(self, msg):
        if self.close_exc:
            raise self.close_exc
        self.sent.append(msg)

    async def close(self, code=1000, reason=""):
        self.closed.append((code, reason))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return self.messages.pop(0)
        if self.close_exc:
            raise self.close_exc
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_ws_transport_graceful_normal_disconnect(gateway, auth, caplog):
    ws = AsyncIterWS(
        messages=[
            json.dumps({"type": "AUTH", "token": "test-secret-token", "device_id": "Pixel-8-Pro"}),
            json.dumps({"type": "SMS_INBOUND", "sender": "+1234", "body": "test sms"}),
        ],
        close_exc=ConnectionClosedOK(Close(1000, "normal closure"), None),
        remote_address=("10.0.0.5", 55555),
    )

    with caplog.at_level(logging.INFO):
        await _handler(ws, gateway, auth)

    # Device should have been attached and then detached
    assert gateway.attach_device.call_count == 1
    assert gateway.detach_device.call_count == 1
    assert not gateway.is_device_connected()

    # Event processed
    gateway.receive_event.assert_called_once_with({"type": "SMS_INBOUND", "sender": "+1234", "body": "test sms"})

    # Check concise log with device identity and code
    assert "WebSocket client disconnected: Pixel-8-Pro (10.0.0.5:55555) (code=1000, reason='normal closure')" in caplog.text


@pytest.mark.asyncio
async def test_ws_transport_abrupt_disconnect(gateway, auth, caplog):
    ws = AsyncIterWS(
        messages=[
            json.dumps({"type": "AUTH", "token": "test-secret-token", "device_name": "Galaxy-S24"}),
            json.dumps({"type": "DEVICE_RESULT", "task_id": "task-999", "status": "ok"}),
        ],
        close_exc=ConnectionClosedError(Close(1006, "connection lost without close frame"), None),
        remote_address=("10.0.0.8", 44444),
    )

    with caplog.at_level(logging.INFO):
        await _handler(ws, gateway, auth)

    # Device detached cleanly
    assert gateway.attach_device.call_count == 1
    assert gateway.detach_device.call_count == 1
    assert not gateway.is_device_connected()

    gateway.handle_device_result.assert_called_once_with({"type": "DEVICE_RESULT", "task_id": "task-999", "status": "ok"})

    # Check concise log with code 1006 and device name
    assert "WebSocket client disconnected: Galaxy-S24 (10.0.0.8:44444) (code=1006, reason='connection lost without close frame')" in caplog.text


@pytest.mark.asyncio
async def test_ws_transport_disconnect_before_auth(gateway, auth, caplog):
    ws = AsyncIterWS(
        messages=[],
        close_exc=ConnectionClosedError(Close(1006, "closed immediately"), None),
        remote_address=("10.0.0.9", 33333),
    )

    with caplog.at_level(logging.INFO):
        # Handler must catch ConnectionClosed gracefully without raising
        await _handler(ws, gateway, auth)

    assert gateway.attach_device.call_count == 0
    assert gateway.detach_device.call_count == 0
    assert "WebSocket client disconnected: 10.0.0.9:33333 (code=1006, reason='closed immediately')" in caplog.text


@pytest.mark.asyncio
async def test_ws_transport_unauthorized_close(gateway, auth):
    ws = AsyncIterWS(
        messages=[json.dumps({"type": "AUTH", "token": "wrong-token"})],
    )

    await _handler(ws, gateway, auth)

    assert len(ws.closed) == 1
    assert ws.closed[0] == (4001, "unauthorized")
    assert gateway.attach_device.call_count == 0


@pytest.mark.asyncio
async def test_ws_transport_malformed_first_message(gateway, auth):
    ws = AsyncIterWS(
        messages=["not-valid-json"],
    )

    await _handler(ws, gateway, auth)

    assert len(ws.closed) == 1
    assert ws.closed[0] == (4002, "malformed payload")
    assert gateway.attach_device.call_count == 0


@pytest.mark.asyncio
async def test_device_handle_safe_send_closed_connection(caplog):
    mock_ws = AsyncMock()
    mock_ws.send.side_effect = ConnectionClosedError(Close(1006, "abrupt"), None)

    handle = DeviceHandle(mock_ws)
    cmd = DeviceCommand(task_id="test-task", capability="sms.read", params={})

    with caplog.at_level(logging.WARNING):
        res = handle.dispatch(cmd)
        assert res["status"] == "ok"
        # Allow async task to complete
        await asyncio.sleep(0.01)

    assert "Failed to send command to device, connection closed: code=1006" in caplog.text
