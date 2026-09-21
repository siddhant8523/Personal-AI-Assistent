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
from assistant.server import create_app


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
            await ws.accept()
            msg = await ws.receive_text()
            await ws.send_text(f"streamlit_echo:{msg}")
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

        async with websockets.connect("ws://127.0.0.1:19003/_stcore/stream") as ws:
            await ws.send("client_init")
            resp = await ws.recv()
            assert resp == "streamlit_echo:client_init"

    finally:
        st_server.should_exit = True
        proxy_server.should_exit = True
        await st_task
        await proxy_task


@pytest.mark.asyncio
async def test_asgi_lifespan_starts_device_gateway_and_proxies_auth():
    """Validates that ASGI lifespan starts AssistantRuntime and Device Gateway on 8765,

    verifies /ws/device proxies to it, registers device upon AUTH, cleanly stops
    on ASGI shutdown, and guarantees Streamlit child disables duplicate binding.
    """
    import os
    import socket
    from unittest.mock import patch
    from assistant.runtime import get_runtime, reset_runtime
    from assistant.server import start_streamlit_process

    # 1. Verify Streamlit child receives DEVICE_GATEWAY_ENABLED="false"
    with patch("subprocess.Popen") as mock_popen:
        start_streamlit_process(port=8501, host="127.0.0.1")
        assert mock_popen.called
        child_env = mock_popen.call_args[1].get("env", {})
        assert child_env.get("DEVICE_GATEWAY_ENABLED") == "false"

    # 2. Setup runtime environment
    reset_runtime()
    os.environ["DEVICE_GATEWAY_ENABLED"] = "true"
    os.environ["DEVICE_GATEWAY_PORT"] = "8765"
    os.environ["GMAIL_ENABLED"] = "false"
    os.environ["WHATSAPP_ENABLED"] = "false"
    os.environ["TELEGRAM_ENABLED"] = "false"
    os.environ["TELEGRAM_USER_ENABLED"] = "false"

    with patch("assistant.runtime.load_dotenv"):
        app = create_app(gateway_ws_url="ws://127.0.0.1:8765", warmup=False, manage_runtime=True)
        config = uvicorn.Config(app, host="127.0.0.1", port=19005, log_level="error", lifespan="on")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())

        for _ in range(50):
            if server.started:
                break
            await asyncio.sleep(0.1)
        assert server.started

        try:
            runtime = get_runtime()
            assert runtime.is_started
            assert runtime._device_gateway_thread is not None
            assert runtime._device_gateway_thread.is_alive()

            # Verify port 8765 accepts websocket connections
            async with websockets.connect("ws://127.0.0.1:8765") as direct_ws:
                assert direct_ws is not None

            # Verify duplicate startup calls are idempotent
            original_thread = runtime._device_gateway_thread
            runtime._start_device_gateway()
            assert runtime._device_gateway_thread is original_thread

            token = os.environ.get("DEVICE_GATEWAY_AUTH_TOKEN", "")

            async with websockets.connect("ws://127.0.0.1:19005/ws/device") as ws:
                auth_frame = {
                    "type": "AUTH",
                    "token": token,
                    "device_id": "Pixel-8-Lifespan-Test",
                    "device_name": "Google Pixel 8",
                }
                await ws.send(json.dumps(auth_frame))
                await asyncio.sleep(0.3)

                assert runtime.device_gateway.is_device_connected()
                dev_status = runtime.device_gateway.get_device_status()
                assert dev_status["connected"] is True
                assert dev_status["device_name"] == "Pixel-8-Lifespan-Test"

        finally:
            server.should_exit = True
            await server_task
            await asyncio.sleep(0.1)
            runtime = get_runtime()
            assert not runtime.is_started
            reset_runtime()

