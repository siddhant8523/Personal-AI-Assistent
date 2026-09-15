"""
WebSocket Transport
=======================
Real asyncio websocket server the Android Device Agent connects to. Every
connection must present DEVICE_GATEWAY_AUTH_TOKEN before any command is
accepted (Rule 18 -- device commands require authentication).

This module defines the server; wiring it into the app's event loop
happens in main.py (only started if DEVICE_GATEWAY_ENABLED=true).
"""

from __future__ import annotations

import asyncio
import json
import logging

import websockets
from websockets.exceptions import ConnectionClosed

from assistant.device_gateway.device_gateway import DeviceGateway
from assistant.security.auth import DeviceAuth

logger = logging.getLogger("assistant.device_gateway.ws")


class DeviceHandle:
    """Wraps a single active websocket connection to the Android app."""

    def __init__(self, ws, device_name: str | None = None):
        self._ws = ws
        self.device_name = device_name
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def is_alive(self) -> bool:
        """Verifies that the underlying WebSocket is open and not closed."""
        if self._ws is None:
            return False
        if getattr(self._ws, "closed", False):
            return False
        state = getattr(self._ws, "state", None)
        if state is not None and getattr(state, "name", "") in {"CLOSED", "CLOSING"}:
            return False
        return True

    async def _safe_send(self, msg_str: str) -> None:
        try:
            await self._ws.send(msg_str)
        except ConnectionClosed as exc:
            close_frame = getattr(exc, "rcvd", None) or getattr(exc, "sent", None)
            code = close_frame.code if close_frame else getattr(exc, "code", "unknown")
            logger.warning("Failed to send command to device, connection closed: code=%s", code)

    def dispatch(self, command) -> dict:
        msg_str = command.to_json()
        logger.info("[android] dispatching command capability=%s request_id=%s", command.capability, command.task_id)
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._safe_send(msg_str), self._loop)
        else:
            asyncio.create_task(self._safe_send(msg_str))
        return {"status": "ok", "detail": "queued for Android device", "device_task_id": command.task_id}


async def _handler(ws, gateway: DeviceGateway, auth: DeviceAuth):
    handle = None
    client_identity = None
    remote = getattr(ws, "remote_address", None)
    remote_str = f"{remote[0]}:{remote[1]}" if isinstance(remote, tuple) and len(remote) >= 2 else (str(remote) if remote else "")

    def get_identity() -> str:
        if client_identity and remote_str:
            return f"{client_identity} ({remote_str})"
        if client_identity:
            return str(client_identity)
        if remote_str:
            return remote_str
        return "unknown device"

    try:
        first_message = await ws.recv()
        try:
            payload = json.loads(first_message)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed AUTH frame from %s", get_identity())
            await ws.close(code=4002, reason="malformed payload")
            return

        if payload.get("type") != "AUTH" or not auth.verify(payload.get("token", "")):
            await ws.close(code=4001, reason="unauthorized")
            return

        client_identity = (
            payload.get("device_id")
            or payload.get("client_id")
            or payload.get("device_name")
            or payload.get("name")
        )

        handle = DeviceHandle(ws, device_name=client_identity)
        gateway.attach_device(handle)

        async for message in ws:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed device event from %s", get_identity())
                continue

            msg_type = payload.get("type")
            req_id = payload.get("task_id") or payload.get("device_task_id") or "n/a"
            logger.info("device -> laptop: type=%s request_id=%s length=%d", msg_type or "RESULT", req_id, len(message))

            if msg_type in {"SMS_INBOUND", "DEVICE_EVENT"}:
                gateway.receive_event(payload)
            elif msg_type in {"DEVICE_RESULT", "RESULT"} or "task_id" in payload:
                logger.info("[android] received capability result request_id=%s", req_id)
                gateway.handle_device_result(payload)
            else:
                logger.info("Unhandled device message type: %s", msg_type)

    except ConnectionClosed as exc:
        close_frame = getattr(exc, "rcvd", None) or getattr(exc, "sent", None)
        code = close_frame.code if close_frame else getattr(exc, "code", "unknown")
        reason = close_frame.reason if close_frame else getattr(exc, "reason", "")
        reason_str = f", reason={reason!r}" if reason else ""
        logger.info(
            "WebSocket client disconnected: %s (code=%s%s)",
            get_identity(),
            code,
            reason_str,
        )
    finally:
        if handle is not None:
            gateway.detach_device(handle)



async def serve(gateway: DeviceGateway, auth: DeviceAuth, host: str, port: int):
    async def handler(ws):
        await _handler(ws, gateway, auth)

    async with websockets.serve(handler, host, port):
        logger.info("Device Gateway listening on ws://%s:%s", host, port)
        await asyncio.Future()  # run forever
