"""
Unified Production ASGI Entrypoint (src/assistant/server.py)
============================================================
Exposes a unified port on Render ($PORT) that multiplexes:
1. Render HTTP Health Checks (`HEAD /`, `GET /healthz`, `GET /_stcore/healthz`) -> HTTP 200
2. Android Device Agent WebSocket (`/ws/device`, `/ws`) -> Device Gateway (ws://127.0.0.1:8765)
3. Streamlit Web UI & WebSocket (`/_stcore/stream`, `/*`) -> Internal Streamlit (http://127.0.0.1:8501)

Preserves local development workflow completely:
- Local Streamlit runs directly on 8501
- Local Device Gateway runs on 8765
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any

import httpx
import uvicorn
import websockets
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("assistant.server")

# Hop-by-hop headers that must not be forwarded by HTTP proxy
HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "content-encoding",
})


async def health_check_handler(request: Request) -> Response:
    """Immediate HTTP 200 responder for Render deployment health probes."""
    return PlainTextResponse("OK", status_code=200)


async def proxy_websocket(client_ws: WebSocket, upstream_url: str, connect_timeout: float = 5.0) -> None:
    """Transparently proxies WebSocket traffic bidirectionally between client and upstream server."""
    await client_ws.accept()
    subprotocols = client_ws.scope.get("subprotocols") or None

    # Retry connection briefly in case upstream service is finishing startup
    upstream_ws = None
    deadline = time.time() + connect_timeout
    while time.time() < deadline and upstream_ws is None:
        try:
            upstream_ws = await websockets.connect(upstream_url, subprotocols=subprotocols, ping_interval=None)
        except (OSError, websockets.exceptions.WebSocketException) as conn_err:
            await asyncio.sleep(0.2)

    if upstream_ws is None:
        logger.warning("[ASGI WS Proxy] Could not connect to upstream %s within timeout", upstream_url)
        try:
            await client_ws.close(code=1013)  # Try again later
        except Exception:
            pass
        return

    try:
        async def client_to_upstream() -> None:
            try:
                while True:
                    msg = await client_ws.receive()
                    msg_type = msg.get("type")
                    if msg_type == "websocket.receive":
                        if "text" in msg and msg["text"] is not None:
                            await upstream_ws.send(msg["text"])
                        elif "bytes" in msg and msg["bytes"] is not None:
                            await upstream_ws.send(msg["bytes"])
                    elif msg_type == "websocket.disconnect":
                        close_code = msg.get("code", 1000)
                        await upstream_ws.close(code=close_code)
                        break
            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                pass
            except Exception as exc:
                logger.warning("[ASGI WS Proxy] Client -> Upstream error: %s", exc)

        async def upstream_to_client() -> None:
            try:
                async for msg in upstream_ws:
                    if isinstance(msg, str):
                        await client_ws.send_text(msg)
                    else:
                        await client_ws.send_bytes(msg)
            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                pass
            except Exception as exc:
                logger.warning("[ASGI WS Proxy] Upstream -> Client error: %s", exc)

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    finally:
        try:
            await upstream_ws.close()
        except Exception:
            pass
        try:
            await client_ws.close()
        except Exception:
            pass


async def _warmup_streamlit(streamlit_http_url: str, timeout: float = 20.0) -> None:
    """Pings the internal Streamlit instance on boot to guarantee AssistantRuntime is initialized."""
    logger.info("[ASGI Proxy] Warming up Streamlit at %s...", streamlit_http_url)
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(f"{streamlit_http_url}/")
                if resp.status_code in (200, 302, 304):
                    logger.info("[ASGI Proxy] Streamlit warmed up successfully (status=%d).", resp.status_code)
                    return
            except Exception:
                await asyncio.sleep(0.5)
    logger.warning("[ASGI Proxy] Streamlit warmup completed timeout window.")


def create_app(
    streamlit_http_url: str = "http://127.0.0.1:8501",
    streamlit_ws_url: str = "ws://127.0.0.1:8501",
    gateway_ws_url: str = "ws://127.0.0.1:8765",
    warmup: bool = False,
) -> Starlette:
    """Creates the Starlette application with path-based routing."""
    http_client = httpx.AsyncClient(base_url=streamlit_http_url, timeout=120.0)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        if warmup:
            asyncio.create_task(_warmup_streamlit(streamlit_http_url))
        yield
        # Proper ASGI shutdown: close httpx client pool
        await http_client.aclose()
        # Clean up child process if attached
        proc = getattr(app.state, "streamlit_proc", None)
        if proc and proc.poll() is None:
            logger.info("Terminating Streamlit child process during ASGI shutdown...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    async def root_handler(request: Request) -> Response:
        # Render load balancers send HEAD / for zero-downtime health checking
        if request.method == "HEAD":
            return PlainTextResponse("OK", status_code=200)
        return await proxy_http_handler(request)

    async def proxy_http_handler(request: Request) -> Response:
        url_path = request.url.path
        if request.url.query:
            url_path = f"{url_path}?{request.url.query}"

        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }

        try:
            body = await request.body()
            upstream_resp = await http_client.request(
                method=request.method,
                url=url_path,
                content=body,
                headers=headers,
            )

            resp_headers = {
                k: v
                for k, v in upstream_resp.headers.items()
                if k.lower() not in HOP_BY_HOP_HEADERS
            }
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=resp_headers,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.debug("[ASGI Proxy] Streamlit booting up, request waiting: %s", exc)
            return PlainTextResponse(
                "Personal AI Assistant starting up, please refresh in a moment...",
                status_code=503,
                headers={"Retry-After": "2"},
            )

    async def device_gateway_ws_handler(ws: WebSocket) -> None:
        logger.info("[ASGI Proxy] Inbound Android Device Gateway connection from %s", ws.client)
        await proxy_websocket(ws, gateway_ws_url)

    async def streamlit_ws_handler(ws: WebSocket) -> None:
        target_ws_url = f"{streamlit_ws_url}{ws.url.path}"
        if ws.url.query:
            target_ws_url = f"{target_ws_url}?{ws.url.query}"
        await proxy_websocket(ws, target_ws_url)

    routes = [
        # Explicit Render & Streamlit health checks
        Route("/healthz", health_check_handler, methods=["GET", "HEAD"]),
        Route("/_stcore/healthz", health_check_handler, methods=["GET", "HEAD"]),
        # Android Device Gateway endpoints
        WebSocketRoute("/ws/device", device_gateway_ws_handler),
        WebSocketRoute("/ws", device_gateway_ws_handler),
        # Streamlit internal WebSockets
        WebSocketRoute("/_stcore/stream", streamlit_ws_handler),
        WebSocketRoute("/stream", streamlit_ws_handler),
        # Root and Catch-All HTTP proxy
        Route("/", root_handler, methods=["GET", "HEAD", "POST"]),
        Route("/{path:path}", proxy_http_handler, methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]),
    ]

    return Starlette(routes=routes, lifespan=lifespan)


def start_streamlit_process(
    script_path: str = "streamlit_app.py",
    port: int = 8501,
    host: str = "127.0.0.1",
) -> subprocess.Popen:
    """Launches Streamlit as an internal child process."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        script_path,
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--server.headless",
        "true",
        "--browser.serverAddress",
        "0.0.0.0",
        "--browser.gatherUsageStats",
        "false",
    ]

    env = os.environ.copy()
    # Ensure Device Gateway is enabled inside the Streamlit AssistantRuntime instance
    if "DEVICE_GATEWAY_ENABLED" not in env:
        env["DEVICE_GATEWAY_ENABLED"] = "true"

    logger.info("Starting Streamlit child process on %s:%d...", host, port)
    proc = subprocess.Popen(cmd, env=env)
    return proc


def main() -> None:
    """Main entrypoint for Render deployment."""
    public_port = int(os.environ.get("PORT", "10000"))
    public_host = os.environ.get("HOST", "0.0.0.0")

    streamlit_port = int(os.environ.get("STREAMLIT_INTERNAL_PORT", "8501"))
    streamlit_host = "127.0.0.1"

    gateway_port = int(os.environ.get("DEVICE_GATEWAY_PORT", "8765"))
    gateway_host = "127.0.0.1"

    logger.info("Starting Personal AI Assistant Unified Server on port %d...", public_port)

    # 1. Start Streamlit child process
    streamlit_proc = start_streamlit_process(port=streamlit_port, host=streamlit_host)

    # 2. Register termination signals for clean shutdown
    def handle_exit(signum: int, frame: Any) -> None:
        logger.info("Received signal %d, shutting down...", signum)
        if streamlit_proc.poll() is None:
            streamlit_proc.terminate()
            try:
                streamlit_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamlit_proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)

    # 3. Create and launch ASGI server with automatic warmup
    app = create_app(
        streamlit_http_url=f"http://{streamlit_host}:{streamlit_port}",
        streamlit_ws_url=f"ws://{streamlit_host}:{streamlit_port}",
        gateway_ws_url=f"ws://{gateway_host}:{gateway_port}",
        warmup=True,
    )
    app.state.streamlit_proc = streamlit_proc

    try:
        uvicorn.run(
            app,
            host=public_host,
            port=public_port,
            log_level="info",
            access_log=False,
        )
    finally:
        if streamlit_proc.poll() is None:
            streamlit_proc.terminate()


if __name__ == "__main__":
    main()
