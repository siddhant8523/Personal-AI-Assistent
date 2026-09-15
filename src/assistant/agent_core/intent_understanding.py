"""
Intent Understanding (Part 1, Section 7.2)
==============================================
Cheap rule-based intent parsing first; falls back to LLM only for
ambiguous free text. Keeps demo/offline mode fully functional and keeps
LLM calls scoped to where they're actually needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from assistant.llm.llm_client import LLMClient
from assistant.llm.prompts import intent_prompt


@dataclass
class Intent:
    intent: str                       # SEND_MESSAGE | SEND_FILE | SEND_EMAIL | MAKE_CALL | QUERY_PRIORITY | GENERAL_CHAT
    platform: Optional[str] = None    # whatsapp | telegram | sms | gmail | none
    recipient: Optional[str] = None
    content: Optional[str] = None


_SEND_PATTERN = re.compile(
    r"(?:send|text|message)\s+(?P<recipient>[\w ]+?)\s+(?:on\s+)?(?P<platform>whatsapp|telegram|sms|gmail|email)?"
    r"[:,]?\s*(?:that\s+)?(?P<content>.+)",
    re.IGNORECASE,
)
_PRIORITY_PATTERN = re.compile(r"(important|priority|urgent)\s+(messages?|emails?|inbox)", re.IGNORECASE)


def parse_intent(text: str, llm: LLMClient | None = None) -> Intent:
    if _PRIORITY_PATTERN.search(text):
        return Intent(intent="QUERY_PRIORITY")

    match = _SEND_PATTERN.search(text)
    if match:
        platform = (match.group("platform") or "whatsapp").lower()
        if platform == "email":
            platform = "gmail"
        return Intent(
            intent="SEND_EMAIL" if platform == "gmail" else "SEND_MESSAGE",
            platform=platform,
            recipient=match.group("recipient").strip(),
            content=match.group("content").strip(),
        )

    # Fall back to LLM for anything that doesn't match a simple pattern.
    if llm is not None:
        raw = llm.generate(intent_prompt.SYSTEM, intent_prompt.build_user_prompt(text, ""))
        return Intent(intent="GENERAL_CHAT", content=raw)

    return Intent(intent="GENERAL_CHAT", content=text)
