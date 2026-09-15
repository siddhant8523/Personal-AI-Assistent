"""
Context Builder (Part 2, Section 23-24)
===========================================
Assembles exactly what the LLM needs -- request + relevant memory + task
state -- and nothing more. Prevents unnecessary context from bloating
every LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant.memory.memory_service import MemoryBundle, MemoryService


@dataclass
class BuiltContext:
    request: str
    memory: MemoryBundle
    summary: str


class ContextBuilder:
    def __init__(self, memory_service: MemoryService):
        self._memory = memory_service

    def build(self, conversation_id: str, request: str) -> BuiltContext:
        bundle = self._memory.retrieve_context(conversation_id, query=request)
        summary_lines = [
            f"User preferences: {bundle.user.strip()[:300]}",
        ]
        if bundle.daily:
            summary_lines.append("Today so far: " + "; ".join(bundle.daily[-5:]))
        if bundle.semantic:
            summary_lines.append("Relevant memory: " + "; ".join(bundle.semantic))
        if bundle.session:
            summary_lines.append(f"Session state: {bundle.session}")

        return BuiltContext(request=request, memory=bundle, summary="\n".join(summary_lines))
