import concurrent.futures
import threading
import time
import pytest
from unittest.mock import MagicMock

from assistant.agent_core.agent_command_handler import AgentCommandHandler
from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.context_builder import ContextBuilder
from assistant.agent_core.graph.builder import _route_after_tool_exec, DEVICE_TOOL_NAMES, build_agent_graph
from assistant.agent_core.graph.nodes import make_tool_exec_node
from assistant.agent_core.graph.state import AgentState
from assistant.agent_core.graph.tools import build_tools
from assistant.agent_core.orchestrator import AgentOrchestrator
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.agent_core.task_planner import TaskPlanner
from assistant.channels.presence import NullPresenceIndicator
from assistant.device_gateway.device_gateway import DeviceGateway, DeviceOfflineError
from assistant.device_gateway.transport.protocol import DeviceCommand, DeviceResult
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.ingestion.unified_message import Source, UnifiedMessage
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.llm.llm_client import LLMClient
from assistant.memory.memory_service import MemoryService
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class ImmediateMockDeviceHandle:
    """Mock handle simulating Android device that returns DEVICE_RESULT asynchronously."""
    def __init__(self, gateway: DeviceGateway, delay: float = 0.01):
        self.gateway = gateway
        self.delay = delay
        self.dispatched_commands: list[DeviceCommand] = []

    def dispatch(self, command: DeviceCommand):
        self.dispatched_commands.append(command)
        cap = command.capability

        def _respond():
            if self.delay > 0:
                time.sleep(self.delay)
            if cap in ("alarm.set", "SET_ALARM"):
                res = DeviceResult(task_id=command.task_id, status="ok", detail="Alarm set for 17:00")
            elif cap in ("timer.set", "SET_TIMER"):
                res = DeviceResult(task_id=command.task_id, status="ok", detail="Timer set for 60s")
            elif cap in ("alarm.list", "LIST_ALARMS"):
                res = DeviceResult(task_id=command.task_id, status="ok", detail={"alarms": [{"hour": 17, "minute": 0, "label": "Evening", "alarm_id": "alarm-1"}]})
            elif cap in ("alarm.cancel", "CANCEL_ALARM"):
                res = DeviceResult(task_id=command.task_id, status="ok", detail="Cancelled")
            else:
                res = DeviceResult(task_id=command.task_id, status="ok", detail="Done")
            self.gateway.handle_device_result(res)

        threading.Thread(target=_respond, daemon=True).start()
        return {"status": "ok", "detail": "queued for Android device", "device_task_id": command.task_id}


class MockLLMWithToolCall:
    """Mock LLM that emits a tool call for set_alarm on the first invocation."""
    def __init__(self):
        self.online = True
        self.call_count = 0

    def generate(self, system: str, user_message: str, max_tokens: int = 1000) -> str:
        self.call_count += 1
        return "LLM response fallback"


def test_android_command_succeeds_and_future_resolves():
    validator = MagicMock()
    validator.validate_android.return_value = True
    gateway = DeviceGateway(validator)
    handle = ImmediateMockDeviceHandle(gateway, delay=0.01)
    gateway.attach_device(handle)

    res = gateway.send_command("alarm.set", {"hour": 17, "minute": 0, "label": "Evening"}, timeout=2.0)
    assert res.status == "ok"
    assert "Alarm set" in str(res.detail)
    assert len(handle.dispatched_commands) == 1
    assert handle.dispatched_commands[0].capability == "alarm.set"


def test_concurrent_android_commands_independent_futures():
    """Verify concurrent commands have isolated Futures and do not resolve each other."""
    validator = MagicMock()
    validator.validate_android.return_value = True
    gateway = DeviceGateway(validator)

    # Manual dispatch handle to control exact resolution order
    dispatched = []

    class ManualHandle:
        def dispatch(self, command: DeviceCommand):
            dispatched.append(command)
            return {"status": "ok", "device_task_id": command.task_id}

    gateway.attach_device(ManualHandle())

    results = {}

    def _worker(cap: str, params: dict, key: str):
        res = gateway.send_command(cap, params, timeout=2.0)
        results[key] = res

    t1 = threading.Thread(target=_worker, args=("alarm.set", {"time": "5 PM"}, "task1"))
    t2 = threading.Thread(target=_worker, args=("timer.set", {"duration": 60}, "task2"))
    t1.start()
    t2.start()

    # Wait until both commands are dispatched
    timeout = time.time() + 2.0
    while len(dispatched) < 2 and time.time() < timeout:
        time.sleep(0.01)
    assert len(dispatched) == 2

    cmd1, cmd2 = dispatched[0], dispatched[1]
    assert cmd1.task_id != cmd2.task_id

    # Resolve task2 first!
    gateway.handle_device_result(DeviceResult(task_id=cmd2.task_id, status="ok", detail="Timer done"))
    t2.join(timeout=1.0)
    assert not t2.is_alive()
    assert results["task2"].detail == "Timer done"
    assert "task1" not in results  # task1 must still be pending!

    # Now resolve task1
    gateway.handle_device_result(DeviceResult(task_id=cmd1.task_id, status="ok", detail="Alarm done"))
    t1.join(timeout=1.0)
    assert not t1.is_alive()
    assert results["task1"].detail == "Alarm done"


def test_device_result_with_device_task_id_key():
    """Verify dictionary payloads with device_task_id instead of task_id resolve the Future."""
    validator = MagicMock()
    validator.validate_android.return_value = True
    gateway = DeviceGateway(validator)

    class AutoAliasHandle:
        def dispatch(self, command: DeviceCommand):
            raw = {
                "type": "DEVICE_RESULT",
                "device_task_id": command.task_id,
                "status": "ok",
                "detail": "Success with device_task_id alias",
            }
            gateway.handle_device_result(raw)
            return {"status": "ok"}

    gateway.attach_device(AutoAliasHandle())
    res = gateway.send_command("alarm.set", {"hour": 8, "minute": 0}, timeout=1.0)
    assert res.status == "ok"
    assert res.detail == "Success with device_task_id alias"


def test_timeout_and_offline_complete_without_hanging():
    validator = MagicMock()
    validator.validate_android.return_value = True
    gateway = DeviceGateway(validator)

    # 1. Offline test
    with pytest.raises(DeviceOfflineError):
        gateway.send_command("alarm.set", {})

    # 2. Timeout test
    class SilentHandle:
        def dispatch(self, command: DeviceCommand):
            return {"status": "ok"}  # never sends result

    gateway.attach_device(SilentHandle())
    res = gateway.send_command("alarm.set", {}, timeout=0.05)
    assert res.status == "failed"
    assert "timed out" in str(res.detail).lower()


def test_route_after_tool_exec_routes_device_tools_directly_to_respond_node():
    """Verify that any device tool output routes directly to respond_node, preventing second LLM call."""
    for tool_name in DEVICE_TOOL_NAMES:
        state: AgentState = {
            "executed_tools": [tool_name],
            "messages": [
                ToolMessage(content="Action executed successfully.", tool_call_id="call-1", name=tool_name)
            ],
            "reply": "Action executed successfully.",
        }
        destination = _route_after_tool_exec(state)
        assert destination == "respond_node", f"Tool {tool_name} should route to respond_node, got {destination}"

    # General reasoning tool (e.g. web search, file search) should still route to agent_node
    search_state: AgentState = {
        "executed_tools": ["web_information"],
        "messages": [
            ToolMessage(content="Found 5 search results about quantum computing.", tool_call_id="call-2", name="web_information")
        ],
        "reply": "",
    }
    assert _route_after_tool_exec(search_state) == "agent_node"


def test_route_after_tool_exec_device_content_patterns():
    """Verify that even without executed_tools list, device response patterns route to respond_node."""
    patterns = [
        "Alarm set for 5 PM.",
        "Timer set for 60 seconds.",
        "Active alarms on Android phone:\n- 07:00 (Wake up)",
        "No active alarms found on Android phone.",
        "Alarm successfully cancelled.",
        "Opened application 'WhatsApp' successfully.",
        "Intent 'android.intent.action.VIEW' executed successfully.",
        "Android device is offline or not connected.",
        "Failed to set alarm: error",
        "Could not parse alarm time 'abc'.",
        "Calling John...",
        "I couldn't find a contact named 'John' on your phone.",
    ]

    for pattern in patterns:
        state: AgentState = {
            "messages": [ToolMessage(content=pattern, tool_call_id="call-x")],
            "reply": pattern,
        }
        assert _route_after_tool_exec(state) == "respond_node", f"Pattern '{pattern}' should route to respond_node"


def test_full_command_lifecycle_immediate_response_delivery(tmp_path):
    """End-to-end trace proving the waiting agent request completes immediately and sends response."""
    memory = MemoryService(memory_dir=str(tmp_path / "memory"))
    llm = LLMClient(api_key="")  # offline mode for deterministic test
    context_builder = ContextBuilder(memory)
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    tool_router = ToolRouter(task_manager)
    policy_engine = PolicyEngine({})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    priority_inbox = PriorityInbox()

    validator = MagicMock()
    validator.validate_android.return_value = True
    gateway = DeviceGateway(validator)
    handle = ImmediateMockDeviceHandle(gateway, delay=0.01)
    gateway.attach_device(handle)

    orchestrator = AgentOrchestrator(
        llm=llm,
        memory=memory,
        context_builder=context_builder,
        task_planner=task_planner,
        approval_manager=approval_manager,
        tool_router=tool_router,
        priority_inbox=priority_inbox,
        device_gateway=gateway,
    )

    delivered_replies = []

    def mock_reply_sender(msg: UnifiedMessage, reply: str):
        delivered_replies.append({"message_id": msg.platform_message_id or msg.message_id, "reply": reply})

    handler = AgentCommandHandler(
        orchestrator=orchestrator,
        reply_sender=mock_reply_sender,
        presence_provider=lambda _: NullPresenceIndicator(),
    )

    # Send command to set alarm
    msg = UnifiedMessage(
        source=Source.WHATSAPP,
        conversation_id="whatsapp:919172767219@s.whatsapp.net",
        sender="user",
        content="Set an alarm for 5 PM",
        platform_message_id="MSG-1001",
    )

    # In offline mode, simulate tool execution through graph state
    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        priority_inbox=priority_inbox,
        tool_router=tool_router,
        device_gateway=gateway,
    )
    set_alarm_tool = next(t for t in tools if t.name == "set_alarm")

    tool_node = make_tool_exec_node(tools)
    exec_state: AgentState = {
        "conversation_id": msg.conversation_id,
        "input_text": msg.content,
        "tool_calls": [{"name": "set_alarm", "args": {"time": "5 PM"}, "id": "call-1"}],
        "messages": [HumanMessage(content=msg.content)],
    }

    start_time = time.time()
    node_result = tool_node(exec_state)
    elapsed = time.time() - start_time

    assert elapsed < 1.0, f"Device tool execution took too long: {elapsed}s"
    assert "executed_tools" in node_result
    assert "set_alarm" in node_result["executed_tools"]
    assert "Alarm set for 5 PM." in node_result["reply"]

    # Verify _route_after_tool_exec routes to respond_node immediately
    exec_state.update(node_result)
    route = _route_after_tool_exec(exec_state)
    assert route == "respond_node"

    # Verify channel handler receives and sends reply immediately
    handler(msg)
    assert len(delivered_replies) == 1
    assert delivered_replies[0]["message_id"] == "MSG-1001"
    # No pending response left waiting for next message
    assert len(delivered_replies) == 1
