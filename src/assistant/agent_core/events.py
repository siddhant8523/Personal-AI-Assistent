"""
Agent Stream Event Model (src/assistant/agent_core/events.py)
=============================================================
Structured, strongly typed, serializable event model emitted by Agent Core
during streaming execution.

Event types:
  - status: Lifecycle and stage indicators ("Thinking...", "Checking context & memory...")
  - token: Real-time incremental LLM content tokens
  - tool_start: A tool is about to be executed (with sanitized parameters)
  - tool_complete: A tool has finished executing (with sanitized summary)
  - approval_required: Execution paused for human-in-the-loop approval
  - error: Execution or LLM resilience failure
  - complete: Final response from completed execution
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal[
    "status",
    "token",
    "tool_start",
    "tool_complete",
    "approval_required",
    "error",
    "complete",
]

_SENSITIVE_KEY_PATTERNS = {
    "api_key",
    "apikey",
    "token",
    "auth",
    "authorization",
    "secret",
    "password",
    "passwd",
    "credential",
    "credentials",
    "session_file",
    "private_key",
    "access_token",
    "bearer",
}


def sanitize_value(val: Any, depth: int = 0, max_depth: int = 5) -> Any:
    """Recursively sanitize data to prevent leaking secrets in event payloads."""
    if depth > max_depth:
        return "<truncated>"

    if isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            k_str = str(k).lower()
            if any(pattern in k_str for pattern in _SENSITIVE_KEY_PATTERNS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_value(v, depth + 1, max_depth)
        return sanitized
    elif isinstance(val, (list, tuple)):
        return [sanitize_value(item, depth + 1, max_depth) for item in val]
    elif isinstance(val, (str, int, float, bool)) or val is None:
        if isinstance(val, str):
            # Check for inline bearer tokens or long hex/base64 patterns that look like keys
            if len(val) > 40 and re.search(r"(?:bearer\s+[a-zA-Z0-9_\-\.]{20,}|gsk_[a-zA-Z0-9]{20,})", val, re.IGNORECASE):
                return "[REDACTED_SECRET]"
        return val
    else:
        # Fallback for complex objects: string representation
        return str(val)


def sanitize_tool_result(result_text: str | None, max_len: int = 300) -> str:
    """Produce a concise, sanitized summary of a tool execution result."""
    if not result_text:
        return ""
    text = str(result_text).strip()
    # Check for secret patterns
    text = re.sub(r"(gsk_[a-zA-Z0-9_\-]{20,})", "[REDACTED_KEY]", text)
    text = re.sub(r"(Bearer\s+[a-zA-Z0-9_\-\.]{20,})", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    return text


@dataclass
class AgentStreamEvent:
    """Strongly typed structured execution event emitted during agent command streaming."""

    type: EventType
    content: str | None = None
    message: str | None = None
    final_response: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    task_id: str | None = None
    task_type: str | None = None
    target: str | None = None
    draft: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------------------
    # Factory Constructors
    # --------------------------------------------------------------------------

    @classmethod
    def status(cls, message: str, **metadata: Any) -> AgentStreamEvent:
        meta = metadata.pop("metadata", None)
        merged = dict(meta) if isinstance(meta, dict) else {}
        merged.update(metadata)
        return cls(type="status", message=message, metadata=merged)

    @classmethod
    def token(cls, content: str, **metadata: Any) -> AgentStreamEvent:
        meta = metadata.pop("metadata", None)
        merged = dict(meta) if isinstance(meta, dict) else {}
        merged.update(metadata)
        return cls(type="token", content=content, metadata=merged)

    @classmethod
    def tool_start(
        cls,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        task_id: str | None = None,
        **metadata: Any,
    ) -> AgentStreamEvent:
        sanitized_args = sanitize_value(tool_args) if tool_args else {}
        meta = metadata.pop("metadata", None)
        merged = dict(meta) if isinstance(meta, dict) else {}
        merged.update(metadata)
        return cls(
            type="tool_start",
            tool_name=tool_name,
            tool_args=sanitized_args,
            task_id=task_id,
            metadata=merged,
        )

    @classmethod
    def tool_complete(
        cls,
        tool_name: str,
        tool_result: str | None = None,
        task_id: str | None = None,
        success: bool = True,
        **metadata: Any,
    ) -> AgentStreamEvent:
        sanitized_res = sanitize_tool_result(tool_result)
        meta = metadata.pop("metadata", None)
        merged = dict(meta) if isinstance(meta, dict) else {}
        merged.update(metadata)
        merged["success"] = success
        return cls(
            type="tool_complete",
            tool_name=tool_name,
            tool_result=sanitized_res,
            task_id=task_id,
            metadata=merged,
        )

    @classmethod
    def approval_required(
        cls,
        task_id: str,
        task_type: str,
        target: str,
        draft: str,
        message: str | None = None,
        **metadata: Any,
    ) -> AgentStreamEvent:
        meta = metadata.pop("metadata", None)
        merged = dict(meta) if isinstance(meta, dict) else {}
        merged.update(metadata)
        return cls(
            type="approval_required",
            task_id=task_id,
            task_type=task_type,
            target=target,
            draft=draft,
            message=message,
            metadata=merged,
        )

    @classmethod
    def error(cls, message: str, **metadata: Any) -> AgentStreamEvent:
        meta = metadata.pop("metadata", None)
        merged = dict(meta) if isinstance(meta, dict) else {}
        merged.update(metadata)
        return cls(type="error", message=message, metadata=merged)

    @classmethod
    def complete(cls, final_response: str, **metadata: Any) -> AgentStreamEvent:
        meta = metadata.pop("metadata", None)
        merged = dict(meta) if isinstance(meta, dict) else {}
        merged.update(metadata)
        return cls(type="complete", final_response=final_response, metadata=merged)

    # --------------------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert event to a JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "type": self.type,
            "timestamp": self.timestamp,
        }
        if self.content is not None:
            d["content"] = self.content
        if self.message is not None:
            d["message"] = self.message
        if self.final_response is not None:
            d["final_response"] = self.final_response
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
        if self.tool_args is not None:
            d["tool_args"] = self.tool_args
        if self.tool_result is not None:
            d["tool_result"] = self.tool_result
        if self.task_id is not None:
            d["task_id"] = self.task_id
        if self.task_type is not None:
            d["task_type"] = self.task_type
        if self.target is not None:
            d["target"] = self.target
        if self.draft is not None:
            d["draft"] = self.draft
        if self.metadata:
            d["metadata"] = sanitize_value(self.metadata)
        return d
