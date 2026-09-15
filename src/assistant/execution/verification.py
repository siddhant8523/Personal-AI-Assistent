"""
Verification (Rule 16)
==========================
"A requested action is not equivalent to a successful action." The agent
must base success on execution result, never on "I sent the request".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VerificationResult:
    success: bool
    detail: str = ""


def verify_connector_result(raw_result: Any) -> VerificationResult:
    if isinstance(raw_result, dict) and raw_result.get("status") == "ok":
        return VerificationResult(success=True, detail=str(raw_result.get("detail", "")))
    return VerificationResult(success=False, detail=str(raw_result))
