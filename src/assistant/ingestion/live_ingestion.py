"""Small, stoppable polling workers for connector-to-router delivery.

Polling is intentionally isolated here: connectors return only a bounded batch
of recent metadata, and every item is normalised before it enters the router.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from assistant.ingestion.unified_message import Origin, UnifiedMessage

logger = logging.getLogger("assistant.live_ingestion")


class PollingWorker:
    def __init__(self, name: str, interval_seconds: float, fetch: Callable[[], list[dict]], adapt: Callable[[dict], UnifiedMessage], deliver: Callable[[UnifiedMessage], object]):
        self.name = name
        self.interval_seconds = max(1.0, interval_seconds)
        self._fetch, self._adapt, self._deliver = fetch, adapt, deliver
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name=f"ingestion-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for payload in self._fetch():
                    message = self._adapt(payload)
                    # Source echoes must remain visible to EchoFilter, which
                    # is the single policy point for dropping them.
                    if payload.get("from_me"):
                        message.origin = Origin.AGENT
                    self._deliver(message)
            except Exception as exc:  # connector failures must not stop the agent
                logger.warning("%s ingestion cycle temporary network error: %s", self.name, exc)
            self._stop.wait(self.interval_seconds)

