"""
Device Gateway (Part 3, Section 12, 30-31)
==============================================
Laptop-side boundary to the Android Device Agent. The Agent Core never
knows how the phone connects, authenticates, or executes -- it only
requests a capability through this class.

Real transport is a websocket server (transport/ws_transport.py) that the
Android app connects to. This class exposes a synchronous-feeling
send_command() API on top of that; when no device is connected it fails
closed rather than pretending success (Rule 16 -- verify, don't assume).
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any

from assistant.device_gateway.transport.protocol import DeviceCommand, DeviceResult

if TYPE_CHECKING:
    from assistant.device_gateway.transport.ws_transport import DeviceHandle

from assistant.security.capability_validator import (
    CapabilityValidator,
    normalize_android_capability,
    ANDROID_CAPABILITY_ALIASES,
    REV_ANDROID_CAPABILITY_ALIASES,
)

logger = logging.getLogger("assistant.device_gateway")


class DeviceOfflineError(Exception):
    pass


class DeviceGateway:
    def __init__(self, capability_validator: CapabilityValidator, on_event=None):
        self._validator = capability_validator
        self._on_event = on_event
        self._connected_device = None   # set by the websocket transport on connect
        self._pending: dict[str, DeviceCommand] = {}
        self._pending_futures: dict[str, concurrent.futures.Future[DeviceResult]] = {}
        self.on_command_start: Any = None
        self.on_command_finish: Any = None


    def is_device_connected(self) -> bool:
        """Returns True ONLY if there is a live, authenticated WebSocket DeviceHandle."""
        if self._connected_device is None:
            return False
        if hasattr(self._connected_device, "is_alive") and not self._connected_device.is_alive():
            self._connected_device = None
            return False
        return True

    def get_device_status(self) -> dict[str, Any]:
        """Returns the real connection status of the Android device."""
        is_conn = self.is_device_connected()
        device_name = getattr(self._connected_device, "device_name", None) or "Android Device"
        reconnecting = bool(not is_conn and getattr(self, "_was_connected", False))
        return {
            "connected": is_conn,
            "device_name": device_name if is_conn else None,
            "reconnecting": reconnecting,
        }

    def attach_device(self, device: DeviceHandle) -> None:
        """Register an active device connection."""
        self._connected_device = device
        self._was_connected = True
        logger.info("[Device] Device connected")

    def detach_device(self, device: DeviceHandle | None = None) -> None:
        """Unregister the device connection."""
        if device is None or self._connected_device == device:
            self._connected_device = None
            logger.info("[Device] Device disconnected")
        else:
            logger.debug("[Device] Ignoring disconnect from stale Android device connection")

    def receive_event(self, event: dict) -> None:
        """Process an inbound event (SMS, battery, notification, etc.) from the device."""
        if self._on_event:
            self._on_event(event)

    def handle_device_result(self, raw_result: dict | DeviceResult) -> None:
        """Receive a command execution result from the Android device over WebSocket."""
        if isinstance(raw_result, DeviceResult):
            result = raw_result
        else:
            task_id = raw_result.get("task_id") or raw_result.get("device_task_id", "")
            status = raw_result.get("status", "ok" if raw_result.get("success", False) else "failed")
            detail = raw_result.get("detail", raw_result.get("result"))
            result = DeviceResult(task_id=task_id, status=status, detail=detail)

        logger.info("[Device] Command result received")
        logger.debug("[android] capability completed request_id=%s status=%s", result.task_id, result.status)
        fut = self._pending_futures.pop(result.task_id, None)
        if fut and not fut.done():
            logger.debug("[DEVICE] resolving future task_id=%s", result.task_id)
            fut.set_result(result)
            logger.debug("[DEVICE] future resolved task_id=%s", result.task_id)
        else:
            logger.warning("[Device] No pending future found for task_id=%s", result.task_id)

    def _candidate_capabilities(self, capability: str) -> list[str]:
        c_orig = capability.strip()
        c_lower = c_orig.lower()
        c_upper = c_orig.upper()
        c_norm = normalize_android_capability(c_orig)

        candidates = [c_norm, c_lower, c_orig, c_upper]
        if c_lower in ANDROID_CAPABILITY_ALIASES:
            candidates.append(ANDROID_CAPABILITY_ALIASES[c_lower])

        if c_lower in REV_ANDROID_CAPABILITY_ALIASES:
            candidates.append(REV_ANDROID_CAPABILITY_ALIASES[c_lower])
            candidates.append(REV_ANDROID_CAPABILITY_ALIASES[c_lower].upper())

        seen = set()
        res = []
        for cand in candidates:
            if cand not in seen:
                seen.add(cand)
                res.append(cand)
        return res

    def _send_single_command(self, capability: str, params: dict, timeout: float = 15.0) -> DeviceResult:
        logger.debug("[device] validating Android capability: %s", capability)
        if not self._validator.validate_android(capability):
            logger.warning("[device] capability not allowed: %s", capability)
            return DeviceResult(task_id="n/a", status="failed", detail=f"capability not allowed: {capability}")

        if not self.is_device_connected():
            logger.warning("[device] failed capability=%s: Android device offline", capability)
            raise DeviceOfflineError("Android device is not connected")

        command = DeviceCommand(capability=capability, params=params)
        logger.info("[Device] Executing device capability: %s", capability)
        logger.debug("[device] sending capability=%s request_id=%s", capability, command.task_id)
        self._pending[command.task_id] = command

        if callable(self.on_command_start):
            try:
                self.on_command_start(capability, params)
            except Exception as exc:
                logger.warning("[device] on_command_start callback failed: %s", exc)

        fut: concurrent.futures.Future[DeviceResult] = concurrent.futures.Future()
        self._pending_futures[command.task_id] = fut

        res: DeviceResult
        try:
            dispatch_info = self._connected_device.dispatch(command)
            if isinstance(dispatch_info, DeviceResult):
                self._pending_futures.pop(command.task_id, None)
                res = dispatch_info
            elif isinstance(dispatch_info, dict) and dispatch_info.get("status") == "failed":
                self._pending_futures.pop(command.task_id, None)
                res = DeviceResult(task_id=command.task_id, status="failed", detail=dispatch_info.get("detail"))
            else:
                logger.debug("[DEVICE] waiting for result task_id=%s", command.task_id)
                res = fut.result(timeout=timeout)
                logger.info("[Device] Device capability completed: status=%s", res.status)
        except concurrent.futures.TimeoutError:
            logger.warning("[device] command timed out request_id=%s capability=%s", command.task_id, capability)
            self._pending_futures.pop(command.task_id, None)
            res = DeviceResult(task_id=command.task_id, status="failed", detail="Android device request timed out")
        except Exception as exc:
            logger.exception("[device] exception sending capability=%s request_id=%s", capability, command.task_id)
            self._pending_futures.pop(command.task_id, None)
            res = DeviceResult(task_id=command.task_id, status="failed", detail=str(exc))
        finally:
            self._pending.pop(command.task_id, None)

        if callable(self.on_command_finish):
            try:
                self.on_command_finish(capability, res)
            except Exception as exc:
                logger.warning("[device] on_command_finish callback failed: %s", exc)

        return res



    def send_command(self, capability: str, params: dict, timeout: float = 15.0) -> DeviceResult:
        norm_cap = normalize_android_capability(capability)
        if norm_cap != capability.strip():
            logger.info("[device] capability normalized: %s -> %s", capability, norm_cap)

        candidates = self._candidate_capabilities(capability)
        last_result = None

        for idx, cand in enumerate(candidates):
            res = self._send_single_command(cand, params, timeout=timeout)
            if res.status == "ok":
                return res

            detail_str = str(res.detail or "")
            if "Unknown capability" in detail_str or "unknown capability" in detail_str.lower():
                logger.warning("[device] Android rejected capability '%s' (%s). Trying candidate fallback...", cand, detail_str)
                last_result = res
                continue
            
            # Non-capability error (e.g. timeout, offline, device execution error)
            return res

        return last_result or DeviceResult(task_id="n/a", status="failed", detail=f"Unknown capability: {capability}")
