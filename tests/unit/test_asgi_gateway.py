"""Unit tests for the Unified ASGI Router & Render Gateway Proxy."""

import asyncio
import json
from unittest.mock import MagicMock
import httpx
import pytest
import uvicorn
import websockets
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.websockets import WebSocket

from assistant.device_gateway.device_gateway import DeviceGateway
from assistant.device_gateway.transport.protocol import DeviceCommand
from assistant.device_gateway.transport.ws_transport import _handler
from assistant.security.auth import DeviceAuth
from assistant.security.capability_validator import CapabilityValidator
from assistant.server import create_app, sanitize_close_code, start_streamlit_process


@pytest.fixture
def auth():
    return DeviceAuth(expected_token="render-secret-token-123")


@pytest.fixture
def gateway():
    validator = CapabilityValidator(android_capabilities=["sms.send", "device.info"], cloud_capabilities=[])
    gw = DeviceGateway(validator)
    gw.attach_device = MagicMock(wraps=gw.attach_device)
    gw.detach_device = MagicMock(wraps=gw.detach_device)
    gw.receive_event = MagicMock()
    gw.handle_device_result = MagicMock()
    return gw


def test_sanitize_close_code():
    # Reserved codes that must not appear in WebSocket Close control frames
    assert sanitize_close_code(1004) == 1000
    assert sanitize_close_code(1005) == 1000
    assert sanitize_close_code(1006) == 1000
    assert sanitize_close_code(1015) == 1000
    # None and non-int values
    assert sanitize_close_code(None) == 1000
    assert sanitize_close_code("1000") == 1000
    # Out of range codes
    assert sanitize_close_code(999) == 1000
    assert sanitize_close_code(5000) == 1000
    assert sanitize_close_code(-1) == 1000
    # Valid RFC 6455 codes
    assert sanitize_close_code(1000) == 1000
    assert sanitize_close_code(1001) == 1001
    assert sanitize_close_code(1008) == 1008
    assert sanitize_close_code(1011) == 1011
    assert sanitize_close_code(4000) == 4000


@pytest.mark.asyncio
async def test_health_check_endpoints():
    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        # 1. GET /healthz
        r = await client.get("/healthz")
        assert r.status_code == 200
        assert r.text == "OK"

        # 2. HEAD /healthz
        r = await client.head("/healthz")
        assert r.status_code == 200

        # 3. GET /_stcore/healthz
        r = await client.get("/_stcore/healthz")
        assert r.status_code == 200
        assert r.text == "OK"

        # 4. HEAD / (Render deployment probe)
        r = await client.head("/")
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_asgi_device_gateway_ws_auth_success_and_protocol(gateway, auth):
    async def real_gw_handler(ws):
        await _handler(ws, gateway, auth)

    gw_server = await websockets.serve(real_gw_handler, "127.0.0.1", 18765)

    app = create_app(
        streamlit_http_url="http://127.0.0.1:18501",
        streamlit_ws_url="ws://127.0.0.1:18501",
        gateway_ws_url="ws://127.0.0.1:18765",
    )

    proxy_config = uvicorn.Config(app, host="127.0.0.1", port=19001, log_level="error", lifespan="off")
    proxy_server = uvicorn.Server(proxy_config)
    proxy_task = asyncio.create_task(proxy_server.serve())

    await asyncio.sleep(0.3)

    try:
        async with websockets.connect("ws://127.0.0.1:19001/ws/device") as client_ws:
            auth_frame = {
                "type": "AUTH",
                "token": "render-secret-token-123",
                "device_id": "Pixel-8-Pro-UUID",
                "device_name": "Google Pixel 8 Pro",
            }
            await client_ws.send(json.dumps(auth_frame))
            await asyncio.sleep(0.1)

            assert gateway.is_device_connected()
            assert gateway.attach_device.call_count == 1

            sms_event = {
                "type": "SMS_INBOUND",
                "sender": "+15551234567",
                "body": "Your verification code is 492019",
            }
            await client_ws.send(json.dumps(sms_event))
            await asyncio.sleep(0.1)

            gateway.receive_event.assert_called_once_with(sms_event)

            result_payload = {
                "type": "DEVICE_RESULT",
                "task_id": "task_abc_123",
                "status": "ok",
                "result": {"battery_level": 94},
            }
            await client_ws.send(json.dumps(result_payload))
            await asyncio.sleep(0.1)

            assert gateway.handle_device_result.call_count == 1
            call_arg = gateway.handle_device_result.call_args[0][0]
            assert call_arg["task_id"] == "task_abc_123"

    finally:
        gw_server.close()
        await gw_server.wait_closed()
        proxy_server.should_exit = True
        await proxy_task


@pytest.mark.asyncio
async def test_asgi_device_gateway_ws_auth_failure(gateway, auth):
    async def real_gw_handler(ws):
        await _handler(ws, gateway, auth)

    gw_server = await websockets.serve(real_gw_handler, "127.0.0.1", 18766)

    app = create_app(
        streamlit_http_url="http://127.0.0.1:18501",
        streamlit_ws_url="ws://127.0.0.1:18501",
        gateway_ws_url="ws://127.0.0.1:18766",
    )

    proxy_config = uvicorn.Config(app, host="127.0.0.1", port=19002, log_level="error", lifespan="off")
    proxy_server = uvicorn.Server(proxy_config)
    proxy_task = asyncio.create_task(proxy_server.serve())

    await asyncio.sleep(0.3)

    try:
        async with websockets.connect("ws://127.0.0.1:19002/ws/device") as client_ws:
            bad_auth = {
                "type": "AUTH",
                "token": "wrong-token",
                "device_id": "Pixel-8",
            }
            await client_ws.send(json.dumps(bad_auth))
            
            try:
                await asyncio.wait_for(client_ws.recv(), timeout=1.0)
            except websockets.exceptions.ConnectionClosed:
                pass

            assert not gateway.is_device_connected()

    finally:
        gw_server.close()
        await gw_server.wait_closed()
        proxy_server.should_exit = True
        await proxy_task


@pytest.mark.asyncio
async def test_streamlit_http_and_websocket_forwarding():
    received_headers = {}

    async def mock_streamlit_backend(scope, receive, send):
        if scope["type"] == "http":
            req = Request(scope, receive)
            if scope["path"] == "/":
                resp = PlainTextResponse("<html>Streamlit Assistant UI</html>", headers={"content-type": "text/html"})
                await resp(scope, receive, send)
            elif scope["path"] == "/static/bundle.js":
                resp = PlainTextResponse("console.log('st');", headers={"content-type": "application/javascript"})
                await resp(scope, receive, send)
            else:
                resp = PlainTextResponse(f"Not Found: {scope['path']}", status_code=404)
                await resp(scope, receive, send)
        elif scope["type"] == "websocket":
            ws = WebSocket(scope, receive, send)
            for k, v in ws.headers.items():
                received_headers[k.lower()] = v
            subprotocols = ws.scope.get("subprotocols") or []
            selected_sub = subprotocols[0] if subprotocols else None
            await ws.accept(subprotocol=selected_sub)
            
            # 1. Test text frame echo
            text_msg = await ws.receive_text()
            await ws.send_text(f"streamlit_echo:{text_msg}")

            # 2. Test binary frame echo
            bytes_msg = await ws.receive_bytes()
            await ws.send_bytes(b"ST_BIN:" + bytes_msg)
            await ws.close()

    st_config = uvicorn.Config(mock_streamlit_backend, host="127.0.0.1", port=18502, log_level="error", lifespan="off")
    st_server = uvicorn.Server(st_config)
    st_task = asyncio.create_task(st_server.serve())

    app = create_app(
        streamlit_http_url="http://127.0.0.1:18502",
        streamlit_ws_url="ws://127.0.0.1:18502",
        gateway_ws_url="ws://127.0.0.1:18767",
    )

    proxy_config = uvicorn.Config(app, host="127.0.0.1", port=19003, log_level="error", lifespan="off")
    proxy_server = uvicorn.Server(proxy_config)
    proxy_task = asyncio.create_task(proxy_server.serve())

    await asyncio.sleep(0.3)

    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:19003") as client:
            r = await client.get("/")
            assert r.status_code == 200
            assert "Streamlit Assistant UI" in r.text

            r = await client.get("/static/bundle.js")
            assert r.status_code == 200
            assert "console.log" in r.text

        # Connect with subprotocols and headers
        async with websockets.connect(
            "ws://127.0.0.1:19003/_stcore/stream",
            subprotocols=["streamlit", "xsrf_token_value"],
            additional_headers={"Cookie": "_streamlit_xsrf=secret123", "User-Agent": "StreamlitTestBrowser/1.0"},
        ) as ws:
            # Check negotiated subprotocol forwarded to client
            assert ws.subprotocol == "streamlit"

            # Check text frame forwarding
            await ws.send("client_init")
            resp_text = await ws.recv()
            assert resp_text == "streamlit_echo:client_init"

            # Check binary frame forwarding
            await ws.send(b"\x00\x01\x02\x03")
            resp_bytes = await ws.recv()
            assert resp_bytes == b"ST_BIN:\x00\x01\x02\x03"

        assert "cookie" in received_headers
        assert "_streamlit_xsrf=secret123" in received_headers["cookie"]
        assert received_headers.get("user-agent") == "StreamlitTestBrowser/1.0"

    finally:
        st_server.should_exit = True
        proxy_server.should_exit = True
        await st_task
        await proxy_task


@pytest.mark.asyncio
async def test_real_streamlit_process_websocket_stream_regression():
    """Focused regression test for /_stcore/stream against real Streamlit child process."""
    proc = start_streamlit_process(port=18505, host="127.0.0.1")
    await asyncio.sleep(2.5)

    app = create_app(
        streamlit_http_url="http://127.0.0.1:18505",
        streamlit_ws_url="ws://127.0.0.1:18505",
        gateway_ws_url="ws://127.0.0.1:18768",
    )

    proxy_config = uvicorn.Config(app, host="127.0.0.1", port=19005, log_level="error", lifespan="off")
    proxy_server = uvicorn.Server(proxy_config)
    proxy_task = asyncio.create_task(proxy_server.serve())

    await asyncio.sleep(0.5)

    try:
        # 1. Test HTTP GET /
        async with httpx.AsyncClient(base_url="http://127.0.0.1:19005") as client:
            resp = await client.get("/")
            assert resp.status_code == 200

        # 2. Test WebSocket connection with external Render Origin and subprotocols
        async with websockets.connect(
            "ws://127.0.0.1:19005/_stcore/stream",
            subprotocols=["streamlit", "test-token-123"],
            additional_headers={"Origin": "https://personal-ai-assistent-raum.onrender.com"},
        ) as ws:
            assert ws.subprotocol == "streamlit"

            # Send Streamlit BackMsg rerun_script binary protobuf
            from streamlit.proto.BackMsg_pb2 import BackMsg
            from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

            back_msg = BackMsg()
            back_msg.rerun_script.query_string = ""
            await ws.send(back_msg.SerializeToString())

            # Receive binary ForwardMsg response from Streamlit
            reply_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            assert isinstance(reply_raw, bytes)
            assert len(reply_raw) > 0

            forward_msg = ForwardMsg()
            forward_msg.ParseFromString(reply_raw)
            # Response should be a valid ForwardMsg (e.g. new_session)
            assert forward_msg.WhichOneof("type") is not None

    finally:
        proxy_server.should_exit = True
        await proxy_task
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
