"""
Presence Indicator System
==========================
Reusable, channel-agnostic presence and composing state abstraction.

Provides heartbeat-driven presence indication across the complete request
lifecycle without leaking platform-specific presence logic into the Agent Core.
"""

from __future__ import annotations

import logging
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger("assistant.channels.presence")


class PresenceIndicator(ABC):
    """Abstract generic presence indicator interface."""

    @abstractmethod
    def start(self) -> None:
        """Start indicating presence/composing state."""

    @abstractmethod
    def stop(self) -> None:
        """Stop indicating presence/composing state and clean up."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether presence is currently active for this indicator."""

    @property
    @abstractmethod
    def destination(self) -> str:
        """Target chat/recipient identifier."""

    @property
    @abstractmethod
    def request_id(self) -> str:
        """Identifier for the active request."""

    def __enter__(self) -> PresenceIndicator:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class NullPresenceIndicator(PresenceIndicator):
    """No-op presence indicator for channels that do not support presence."""

    def __init__(self, destination: str = "", request_id: str = ""):
        self._destination = destination
        self._request_id = request_id or str(uuid.uuid4())
        self._active = False

    def start(self) -> None:
        self._active = True
        logger.info("Presence started")

    def stop(self) -> None:
        if self._active:
            self._active = False
            logger.info("Presence stopped")

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def destination(self) -> str:
        return self._destination

    @property
    def request_id(self) -> str:
        return self._request_id


class _DestinationWorker:
    """Manages the background heartbeat thread and active requests for a single destination."""

    def __init__(
        self,
        key: tuple[str, str],
        interval_seconds: float,
        send_pulse: Callable[[], Any],
        send_cleanup: Callable[[], Any] | None = None,
    ):
        self.key = key
        self.interval_seconds = max(0.01, interval_seconds)
        self.send_pulse = send_pulse
        self.send_cleanup = send_cleanup

        self.lock = threading.Lock()
        self.active_request_ids: set[str] = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def add_request(self, request_id: str) -> bool:
        """Registers a request ID. Starts background heartbeat if this is the first request.
        Returns True if heartbeat was newly started, False if already running."""
        with self.lock:
            first_request = len(self.active_request_ids) == 0
            self.active_request_ids.add(request_id)
            if first_request:
                self._stop_event.clear()
                # Send immediate pulse on start
                try:
                    self.send_pulse()
                except Exception as exc:
                    logger.debug("Initial presence pulse failed for %s: %s", self.key, exc)

                self._thread = threading.Thread(
                    target=self._run_heartbeat,
                    name=f"presence-worker-{self.key[0]}-{self.key[1]}",
                    daemon=True,
                )
                self._thread.start()
                return True
            return False

    def remove_request(self, request_id: str) -> bool:
        """Unregisters a request ID. If no requests remain, stops heartbeat and sends cleanup.
        Returns True if heartbeat was stopped, False if other requests are still active."""
        with self.lock:
            self.active_request_ids.discard(request_id)
            if self.active_request_ids:
                # Other requests are still active on this destination
                return False

            # No remaining requests; terminate heartbeat
            self._stop_event.set()

        # Join the thread outside the lock to prevent deadlocks
        if self._thread is not None and self._thread.is_alive():
            if threading.current_thread() != self._thread:
                self._thread.join(timeout=2.0)
        self._thread = None

        # Final cleanup state (e.g. WhatsApp paused)
        if self.send_cleanup is not None:
            try:
                self.send_cleanup()
            except Exception as exc:
                logger.debug("Presence cleanup failed for %s (ignored): %s", self.key, exc)

        return True

    def _run_heartbeat(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval_seconds):
                break
            if self._stop_event.is_set():
                break

            with self.lock:
                if not self.active_request_ids:
                    break

            try:
                self.send_pulse()
                logger.debug("Presence heartbeat pulse sent for %s", self.key)
            except Exception as exc:
                logger.debug("Presence heartbeat tick failed for %s: %s", self.key, exc)


class PresenceCoordinator:
    """Thread-safe coordinator for active presence workers across destinations."""

    def __init__(self):
        self._lock = threading.Lock()
        self._workers: dict[tuple[str, str], _DestinationWorker] = {}

    def register(
        self,
        channel_type: str,
        destination: str,
        request_id: str,
        interval_seconds: float,
        send_pulse: Callable[[], Any],
        send_cleanup: Callable[[], Any] | None = None,
    ) -> bool:
        key = (channel_type, destination)
        with self._lock:
            if key not in self._workers:
                self._workers[key] = _DestinationWorker(
                    key=key,
                    interval_seconds=interval_seconds,
                    send_pulse=send_pulse,
                    send_cleanup=send_cleanup,
                )
            worker = self._workers[key]

        return worker.add_request(request_id)

    def unregister(self, channel_type: str, destination: str, request_id: str) -> bool:
        key = (channel_type, destination)
        with self._lock:
            worker = self._workers.get(key)

        if worker is None:
            return False

        stopped = worker.remove_request(request_id)
        if stopped:
            with self._lock:
                # If still empty, clean up worker entry
                if key in self._workers and not self._workers[key].active_request_ids:
                    self._workers.pop(key, None)
        return stopped

    def is_active(self, channel_type: str, destination: str, request_id: str | None = None) -> bool:
        key = (channel_type, destination)
        with self._lock:
            worker = self._workers.get(key)
            if not worker:
                return False
            with worker.lock:
                if request_id is not None:
                    return request_id in worker.active_request_ids
                return len(worker.active_request_ids) > 0

    def get_worker(self, channel_type: str, destination: str) -> _DestinationWorker | None:
        key = (channel_type, destination)
        with self._lock:
            return self._workers.get(key)


# Global default coordinator instance
_default_coordinator = PresenceCoordinator()


class TelegramPresenceIndicator(PresenceIndicator):
    """Periodic typing presence indicator for Telegram Bot API."""

    def __init__(
        self,
        connector: Any,
        chat_id: str,
        request_id: str | None = None,
        interval_seconds: float = 4.0,
        coordinator: PresenceCoordinator | None = None,
    ):
        self.connector = connector
        self._chat_id = str(chat_id)
        self._request_id = request_id or str(uuid.uuid4())
        self._interval_seconds = interval_seconds
        self._coordinator = coordinator or _default_coordinator
        self._active = False

    def _send_typing(self) -> None:
        if hasattr(self.connector, "send_chat_action"):
            self.connector.send_chat_action(self._chat_id, "typing")
        elif callable(self.connector):
            self.connector(self._chat_id, "typing")

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        logger.info("Presence started")
        self._coordinator.register(
            channel_type="telegram",
            destination=self._chat_id,
            request_id=self._request_id,
            interval_seconds=self._interval_seconds,
            send_pulse=self._send_typing,
            send_cleanup=None,  # Telegram has no "stop typing" action
        )

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        self._coordinator.unregister(
            channel_type="telegram",
            destination=self._chat_id,
            request_id=self._request_id,
        )
        logger.info("Presence stopped")

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def destination(self) -> str:
        return self._chat_id

    @property
    def request_id(self) -> str:
        return self._request_id


class WhatsAppPresenceIndicator(PresenceIndicator):
    """Periodic composing presence indicator for WhatsApp (Baileys)."""

    def __init__(
        self,
        connector: Any,
        jid: str,
        request_id: str | None = None,
        interval_seconds: float = 4.0,
        coordinator: PresenceCoordinator | None = None,
    ):
        self.connector = connector
        self._jid = str(jid)
        self._request_id = request_id or str(uuid.uuid4())
        self._interval_seconds = interval_seconds
        self._coordinator = coordinator or _default_coordinator
        self._active = False

    def _send_composing(self) -> None:
        if hasattr(self.connector, "send_presence_update"):
            self.connector.send_presence_update(self._jid, "composing")
        elif hasattr(self.connector, "sendPresenceUpdate"):
            import inspect
            res = self.connector.sendPresenceUpdate("composing", self._jid)
            if inspect.isawaitable(res):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(res, loop)
                    else:
                        loop.run_until_complete(res)
                except RuntimeError:
                    asyncio.run(res)
        elif callable(self.connector):
            self.connector(self._jid, "composing")

    def _send_paused(self) -> None:
        try:
            if hasattr(self.connector, "send_presence_update"):
                self.connector.send_presence_update(self._jid, "paused")
            elif hasattr(self.connector, "sendPresenceUpdate"):
                import inspect
                res = self.connector.sendPresenceUpdate("paused", self._jid)
                if inspect.isawaitable(res):
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.run_coroutine_threadsafe(res, loop)
                        else:
                            loop.run_until_complete(res)
                    except RuntimeError:
                        asyncio.run(res)
            elif callable(self.connector):
                self.connector(self._jid, "paused")
        except Exception as exc:
            logger.debug("[WhatsApp Presence] Failed to send paused presence: %s", exc)

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        logger.info("Presence started")
        self._coordinator.register(
            channel_type="whatsapp",
            destination=self._jid,
            request_id=self._request_id,
            interval_seconds=self._interval_seconds,
            send_pulse=self._send_composing,
            send_cleanup=self._send_paused,
        )

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        self._coordinator.unregister(
            channel_type="whatsapp",
            destination=self._jid,
            request_id=self._request_id,
        )
        logger.info("Presence stopped")

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def destination(self) -> str:
        return self._jid

    @property
    def request_id(self) -> str:
        return self._request_id
