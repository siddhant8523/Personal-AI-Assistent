"""
Device Gateway Protocol
===========================
JSON envelope exchanged with the Android Device Agent over the websocket
transport. Kept intentionally tiny -- the Agent Core only ever asks for
a capability + params (Part 3, Section 12: "It simply requests a
capability.").
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DeviceCommand:
    capability: str                 # e.g. "SEND_SMS"
    params: dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: f"DEV-{uuid.uuid4().hex[:8]}")
    issued_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class DeviceResult:
    task_id: str
    status: str          # "ok" | "failed"
    detail: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "status": self.status, "detail": self.detail}

    @classmethod
    def from_json(cls, raw: str) -> "DeviceResult":
        data = json.loads(raw)
        return cls(task_id=data["task_id"], status=data["status"], detail=data.get("detail"))

