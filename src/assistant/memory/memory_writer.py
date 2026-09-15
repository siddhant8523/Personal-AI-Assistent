"""
Memory Writer
================
Controlled write path (Part 2, Section 22) — the agent should not blindly
persist every conversation turn. A cheap heuristic policy decides the
destination store; swap for an LLM-classified policy later if needed.
"""

from __future__ import annotations

from assistant.memory.memory_service import MemoryService

_PREFERENCE_MARKERS = ("i prefer", "always", "never", "remember that", "from now on")


def write_interaction(memory: MemoryService, conversation_id: str, text: str) -> str:
    lowered = text.lower()

    if any(marker in lowered for marker in _PREFERENCE_MARKERS):
        memory.long_term.set(key=f"pref::{conversation_id}::{hash(text) & 0xffff}", value=text)
        memory.semantic.add(text, tags="preference")
        return "long_term"

    memory.daily.add(text)
    return "daily"
