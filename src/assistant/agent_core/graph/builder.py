"""
Graph Builder
================
Assembles the LangGraph StateGraph.

    START
      v
    load_context_node        (existing ContextBuilder + MemoryService)
      v
    agent_node                (LangChain tool-calling, LLM online, or
      |                        offline regex fallback -- same behavior
      |                        as before)
      v
    (conditional) ----------+
      v                     v
    tool_exec_node    (direct reply)
      v                     v
      +--------> respond_node
                      v
                     END

`tool_exec_node` runs whatever tools agent_node selected (tools.py).
Consequential tools (propose_send_message, send_*_file, ...) call
LangGraph's own `interrupt()` when the EXISTING PolicyEngine says the
capability requires approval -- that actually suspends this graph run
(state is preserved by the checkpointer) until AgentOrchestrator resumes
it with `Command(resume="approve"/"reject")`. There is no separate
"approval_node" pretending to be a pause; the pause is real.

A MemorySaver checkpointer persists state per conversation_id (used as
the LangGraph thread_id), giving each Agent Chat conversation resumable
graph history -- this is also what makes `interrupt()`/`Command(resume=)`
possible across separate `handle_agent_command()` calls (i.e. across
separate inbound chat turns): the graph is genuinely paused in the
checkpointer between the proposal turn and the approve/reject turn.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.context_builder import ContextBuilder
from assistant.agent_core.graph.nodes import (
    make_agent_node,
    make_load_context_node,
    make_tool_exec_node,
    respond_node,
)
from assistant.agent_core.graph.state import AgentState
from assistant.agent_core.graph.tools import build_tools
from assistant.agent_core.task_planner import TaskPlanner
from assistant.execution.tool_router import ToolRouter
from assistant.files.attachment_manager import AttachmentManager
from assistant.files.file_resolver import FileResolver
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.llm.llm_client import LLMClient
from assistant.memory.memory_service import MemoryService
from assistant.tools.web_information import WebInformationService


DEVICE_TOOL_NAMES = {
    "set_alarm",
    "list_alarms",
    "cancel_alarm",
    "set_timer",
    "execute_android_intent",
    "open_mobile_app",
    "make_call",
    "read_sms",
    "list_mobile_files",
    "find_mobile_files",
    "send_sms",
}


def _route_after_agent(state: AgentState) -> str:
    return "tool_exec_node" if state.get("tool_calls") else "respond_node"


def _route_after_tool_exec(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "respond_node"

    executed_tools = state.get("executed_tools", [])
    if any(t in DEVICE_TOOL_NAMES for t in executed_tools):
        return "respond_node"

    last_msg = messages[-1]
    from langchain_core.messages import ToolMessage
    if isinstance(last_msg, ToolMessage):
        if getattr(last_msg, "name", None) in DEVICE_TOOL_NAMES:
            return "respond_node"

        content = str(last_msg.content or "")
        content_lower = content.lower()
        if (
            "Approved and executed" in content
            or "Approved, but failed to execute" in content
            or "Rejected" in content
            or "TODAY'S PRIORITY" in content
            or "no pending actions" in content_lower
            or "name was not changed" in content_lower
            or "assistant name changed" in content_lower
            or "alarm set for" in content_lower
            or "timer set for" in content_lower
            or "active alarms on android" in content_lower
            or "no active alarms" in content_lower
            or "alarm successfully cancelled" in content_lower
            or "opened application" in content_lower
            or "intent " in content_lower
            or "android device is offline" in content_lower
            or "failed to set alarm" in content_lower
            or "failed to set timer" in content_lower
            or "could not parse alarm time" in content_lower
            or "calling " in content_lower
            or "couldn't find a contact" in content_lower
            or "found multiple contacts" in content_lower
            or "sms from" in content_lower
            or "no sms messages found" in content_lower
            or "files in " in content_lower
            or "found files matching" in content_lower
        ):
            return "respond_node"
    return "agent_node"


def build_agent_graph(
    llm: LLMClient,
    memory: MemoryService,
    context_builder: ContextBuilder,
    task_planner: TaskPlanner,
    approval_manager: ApprovalManager,
    tool_router: ToolRouter,
    priority_inbox: PriorityInbox,
    telegram_connector=None,
    telegram_personal=None,
    whatsapp_connector=None,
    gmail_connector=None,
    file_resolver: FileResolver | None = None,
    attachment_manager: AttachmentManager | None = None,
    web_information: WebInformationService | None = None,
    device_gateway=None,
):
    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        priority_inbox=priority_inbox,
        tool_router=tool_router,
        telegram_connector=telegram_connector,
        telegram_personal=telegram_personal,
        whatsapp_connector=whatsapp_connector,
        gmail_connector=gmail_connector,
        file_resolver=file_resolver,
        attachment_manager=attachment_manager,
        identity_manager=memory.identity,
        user_profile=memory.user_profile,
        web_information=web_information,
        device_gateway=device_gateway,
    )



    graph = StateGraph(AgentState)

    graph.add_node("load_context_node", make_load_context_node(context_builder, memory))
    graph.add_node("agent_node", make_agent_node(llm, tools, memory.soul.text))
    graph.add_node("tool_exec_node", make_tool_exec_node(tools))
    graph.add_node("respond_node", respond_node)

    graph.add_edge(START, "load_context_node")
    graph.add_edge("load_context_node", "agent_node")
    graph.add_conditional_edges("agent_node", _route_after_agent, ["tool_exec_node", "respond_node"])
    graph.add_conditional_edges("tool_exec_node", _route_after_tool_exec, ["agent_node", "respond_node"])
    graph.add_edge("respond_node", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
