"""
LangGraph State
==================
Single typed state object threaded through every node. Kept intentionally
flat -- each node reads what it needs and writes its own outputs; nothing
here replaces the EXISTING Task / UnifiedMessage models, it just carries
references/results between graph nodes for one Agent Chat turn.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    conversation_id: str
    input_text: str
    context_summary: str              # from the EXISTING ContextBuilder
    messages: list[BaseMessage]       # LLM conversation turn (system + human [+ ai/tool])
    tool_calls: list[dict[str, Any]]
    tool_results: list[str]
    executed_tools: list[str]
    reply: Optional[str]
    error: Optional[str]

    # NOTE: approve/reject is no longer routed *inside* the graph via a
    # regex-detected pseudo-command. Consequential tools (send_*, etc.)
    # call LangGraph's own `interrupt()` from within tool_exec_node, which
    # actually pauses graph execution (persisted by the checkpointer) until
    # AgentOrchestrator resumes it with `Command(resume=...)`. See
    # graph/tools.py and agent_core/orchestrator.py.
