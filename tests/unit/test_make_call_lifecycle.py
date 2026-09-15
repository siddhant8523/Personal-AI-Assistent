"""
Regression tests for make_call and SMS action lifecycle (Scenarios A - F).
========================================================================

A. Android disconnected + make_call
   -> no device dispatch
   -> no success/"phone call placed" message
   -> no task created or approval requested

B. Android connected + make_call
   -> normal approval flow
   -> task created in WAITING_FOR_APPROVAL
   -> device command dispatched upon approval

C. Android disconnects after approval but before dispatch
   -> safe failure
   -> no false success ("Approved and executed")
   -> clearly reports approval succeeded but execution failed

D. Android execution succeeds
   -> genuine success message ("Approved and executed")

E. Android execution fails
   -> failure message, not success ("Approved, but failed to execute")

F. Android disconnected + direct-phone SMS
   -> no task created or approval requested
   -> clear unavailable-device response
"""

import pytest
from unittest.mock import MagicMock
from langgraph.types import interrupt

from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.graph.tools import build_tools
from assistant.agent_core.task_planner import TaskPlanner
from assistant.device_gateway.device_gateway import DeviceGateway
from assistant.device_gateway.transport.protocol import DeviceResult
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task import TaskState
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.memory.user_profile import UserProfile
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.security.capability_validator import CapabilityValidator
from assistant.ui.styles import render_tool_html, get_tool_friendly_name


class MockDeviceHandle:
    def __init__(self, gateway: DeviceGateway, device_id="test_android"):
        self.gateway = gateway
        self.device_id = device_id
        self.alive = True
        self.last_command = None
        self.mock_result = None

    def is_alive(self):
        return self.alive

    def set_alive(self, state: bool):
        self.alive = state

    def dispatch(self, command):
        self.last_command = command
        cap = command.capability.upper().replace(".", "_")
        if self.mock_result is not None:
            res = DeviceResult(
                task_id=command.task_id,
                status=self.mock_result.status,
                detail=self.mock_result.detail,
            )
        elif cap in ("MAKE_CALL", "CALL_MAKE"):
            res = DeviceResult(task_id=command.task_id, status="ok", detail=f"Phone call initiated to {command.params.get('recipient')}")
        elif cap in ("CONTACT_FIND", "CONTACTS_SEARCH"):
            query = command.params.get("query", "")
            if query == "mummy":
                res = DeviceResult(task_id=command.task_id, status="ok", detail={"query": query, "matches": [{"name": "Mummy", "phone_number": "+917798298569"}]})
            else:
                res = DeviceResult(task_id=command.task_id, status="ok", detail={"query": query, "matches": []})
        else:
            res = DeviceResult(task_id=command.task_id, status="ok", detail="Command executed")

        self.gateway.handle_device_result(res.to_dict())
        return {"status": "ok", "detail": "queued", "device_task_id": command.task_id}


@pytest.fixture
def mock_setup(tmp_path):
    validator = CapabilityValidator(
        android_capabilities=["call.make", "sms.send", "contact.find"],
        cloud_capabilities=[],
    )
    gateway = DeviceGateway(validator)
    handle = MockDeviceHandle(gateway)
    gateway.attach_device(handle)

    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    policy_engine = PolicyEngine({"call.make": {"requires_approval": True}, "sms.send": {"requires_approval": True}})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)
    priority_inbox = PriorityInbox()
    user_profile = UserProfile(str(tmp_path / "memory"))

    def _make_call_handler(task):
        phone_num = (
            task.parameters.get("phone_number")
            or task.parameters.get("number")
            or (task.target if (task.target.startswith("+") or task.target.replace("-", "").replace(" ", "").isdigit()) else "")
        )
        params = {
            "recipient": task.target,
            "phone_number": phone_num,
            "number": phone_num,
            **task.parameters,
        }
        return gateway.send_command("call.make", params).to_dict()

    def _send_sms_handler(task):
        params = {
            "recipient": task.target,
            "phone_number": task.parameters.get("phone_number") or task.target,
            "content": task.parameters.get("content", ""),
            **task.parameters,
        }
        return gateway.send_command("sms.send", params).to_dict()

    tool_router.register("make_call", _make_call_handler)
    tool_router.register("call.make", _make_call_handler)
    tool_router.register("send_sms", _send_sms_handler)
    tool_router.register("sms.send", _send_sms_handler)

    return {
        "gateway": gateway,
        "handle": handle,
        "task_manager": task_manager,
        "task_planner": task_planner,
        "approval_manager": approval_manager,
        "tool_router": tool_router,
        "priority_inbox": priority_inbox,
        "user_profile": user_profile,
    }


def _get_tools(setup):
    return build_tools(
        task_planner=setup["task_planner"],
        approval_manager=setup["approval_manager"],
        priority_inbox=setup["priority_inbox"],
        tool_router=setup["tool_router"],
        user_profile=setup["user_profile"],
        device_gateway=setup["gateway"],
    )


# ==============================================================================
# Scenario A: Disconnected Android + make_call
# ==============================================================================

def test_scenario_a_disconnected_make_call(mock_setup):
    """Scenario A: Android disconnected + make_call
    -> no device dispatch
    -> no success/"phone call placed" message
    -> no task created or approval requested
    -> returns clear unavailable message
    """
    setup = mock_setup
    # Detach device so gateway reports disconnected
    setup["gateway"].detach_device(setup["handle"])
    assert not setup["gateway"].is_device_connected()

    tools = _get_tools(setup)
    make_call_tool = next(t for t in tools if t.name == "make_call")

    # Direct phone number
    res = make_call_tool.invoke({"recipient": "+917798298569"})
    assert res == "Android device is offline or not connected."
    assert "Phone call placed" not in res
    assert "Approved" not in res

    # Verify no tasks created or waiting approval
    assert len(setup["task_manager"]._tasks) == 0
    assert setup["handle"].last_command is None

    # Also verify with contact name when disconnected
    res_contact = make_call_tool.invoke({"recipient": "mummy"})
    assert res_contact == "Android device is offline or not connected."
    assert len(setup["task_manager"]._tasks) == 0
    assert setup["handle"].last_command is None


# ==============================================================================
# Scenario B: Connected Android + make_call
# ==============================================================================

def test_scenario_b_connected_make_call_triggers_approval(mock_setup):
    """Scenario B: Android connected + make_call
    -> normal approval flow
    -> task created in WAITING_FOR_APPROVAL
    -> interrupt is triggered with draft
    """
    setup = mock_setup
    assert setup["gateway"].is_device_connected()

    tools = _get_tools(setup)
    make_call_tool = next(t for t in tools if t.name == "make_call")

    # make_call pauses via interrupt()
    with pytest.raises(Exception):
        make_call_tool.invoke({"recipient": "+917798298569"})

    # Verify task was created in WAITING_FOR_APPROVAL
    waiting_task = setup["task_manager"].find_waiting_approval(
        "make_call", "+917798298569", "Make phone call to +917798298569 (+91******8569)"
    )
    assert waiting_task is not None
    assert waiting_task.execution_state == TaskState.WAITING_FOR_APPROVAL
    assert waiting_task.parameters["phone_number"] == "+917798298569"

    # Command is NOT dispatched yet because approval is pending
    assert setup["handle"].last_command is None


# ==============================================================================
# Scenario C: Disconnect after approval but before dispatch
# ==============================================================================

def test_scenario_c_disconnect_after_approval_before_dispatch(mock_setup):
    """Scenario C: Android disconnects after approval but before dispatch
    -> safe failure
    -> no false success ("Approved and executed")
    -> reports approval succeeded but execution failed
    """
    setup = mock_setup
    tools = _get_tools(setup)
    make_call_tool = next(t for t in tools if t.name == "make_call")

    # 1. Trigger approval while connected
    with pytest.raises(Exception):
        make_call_tool.invoke({"recipient": "+917798298569"})

    waiting_task = setup["task_manager"].find_waiting_approval(
        "make_call", "+917798298569", "Make phone call to +917798298569 (+91******8569)"
    )
    assert waiting_task is not None

    # 2. Device disconnects during approval review
    setup["handle"].set_alive(False)
    assert not setup["gateway"].is_device_connected()

    # 3. User approves -> tool resumes execution
    approved = setup["approval_manager"].approve(waiting_task.task_id)
    result = setup["tool_router"].dispatch(approved, source="android", conversation_id=approved.target)

    # Tool router returns failed result with DeviceOfflineError detail
    assert result.get("status") == "failed"
    assert "Android device is not connected" in result.get("detail", "")

    # Formatting logic in _make_call
    status = result.get("status")
    detail = result.get("detail")
    if status in ("ok", "success", "sent"):
        reply = f"Approved and executed {waiting_task.task_id}: {detail or 'Phone call placed.'}"
    else:
        reply = f"Approved, but failed to execute {waiting_task.task_id}: {detail or 'Execution failed.'}"

    assert "Approved and executed" not in reply
    assert f"Approved, but failed to execute {waiting_task.task_id}: Android device is not connected" == reply


# ==============================================================================
# Scenario D: Android execution succeeds
# ==============================================================================

def test_scenario_d_android_execution_succeeds(mock_setup):
    """Scenario D: Android execution succeeds
    -> genuine success message
    -> device command dispatched and completed
    """
    setup = mock_setup
    setup["handle"].mock_result = DeviceResult(
        task_id="cmd_call_1",
        status="ok",
        detail="Phone call initiated to +917798298569",
    )

    tools = _get_tools(setup)
    make_call_tool = next(t for t in tools if t.name == "make_call")

    with pytest.raises(Exception):
        make_call_tool.invoke({"recipient": "+917798298569"})

    waiting_task = setup["task_manager"].find_waiting_approval(
        "make_call", "+917798298569", "Make phone call to +917798298569 (+91******8569)"
    )
    assert waiting_task is not None

    # Approve and dispatch
    approved = setup["approval_manager"].approve(waiting_task.task_id)
    result = setup["tool_router"].dispatch(approved, source="android", conversation_id=approved.target)

    assert result.get("status") == "ok"
    status = result.get("status")
    detail = result.get("detail")
    if status in ("ok", "success", "sent"):
        reply = f"Approved and executed {waiting_task.task_id}: {detail or 'Phone call placed.'}"
    else:
        reply = f"Approved, but failed to execute {waiting_task.task_id}: {detail or 'Execution failed.'}"

    assert reply == f"Approved and executed {waiting_task.task_id}: Phone call initiated to +917798298569"
    assert setup["handle"].last_command.capability == "call.make"
    assert setup["handle"].last_command.params["phone_number"] == "+917798298569"


# ==============================================================================
# Scenario E: Android execution fails
# ==============================================================================

def test_scenario_e_android_execution_fails(mock_setup):
    """Scenario E: Android execution fails
    -> failure message, not success
    -> 'Approved and executed' is NOT claimed
    """
    setup = mock_setup
    setup["handle"].mock_result = DeviceResult(
        task_id="cmd_call_2",
        status="failed",
        detail="CALL_PHONE permission denied by user",
    )

    tools = _get_tools(setup)
    make_call_tool = next(t for t in tools if t.name == "make_call")

    with pytest.raises(Exception):
        make_call_tool.invoke({"recipient": "+917798298569"})

    waiting_task = setup["task_manager"].find_waiting_approval(
        "make_call", "+917798298569", "Make phone call to +917798298569 (+91******8569)"
    )

    approved = setup["approval_manager"].approve(waiting_task.task_id)
    result = setup["tool_router"].dispatch(approved, source="android", conversation_id=approved.target)

    assert result.get("status") == "failed"
    status = result.get("status")
    detail = result.get("detail")
    if status in ("ok", "success", "sent"):
        reply = f"Approved and executed {waiting_task.task_id}: {detail or 'Phone call placed.'}"
    else:
        reply = f"Approved, but failed to execute {waiting_task.task_id}: {detail or 'Execution failed.'}"

    assert "Approved and executed" not in reply
    assert reply == f"Approved, but failed to execute {waiting_task.task_id}: CALL_PHONE permission denied by user"


# ==============================================================================
# Scenario F: Disconnected Android + direct-phone SMS
# ==============================================================================

def test_scenario_f_disconnected_sms_direct_phone(mock_setup):
    """Scenario F: Android disconnected + direct-phone SMS
    -> no task created
    -> no approval requested
    -> clear unavailable-device response
    """
    setup = mock_setup
    setup["gateway"].detach_device(setup["handle"])
    assert not setup["gateway"].is_device_connected()

    tools = _get_tools(setup)
    propose_tool = next(t for t in tools if t.name == "propose_send_message")

    # Send SMS to a direct phone number while disconnected
    res = propose_tool.invoke({
        "platform": "sms",
        "recipient": "+917798298569",
        "content": "Meeting at 5pm",
    })

    assert res == "Android device is offline or not connected."
    # Verify no tasks created or dispatched
    assert len(setup["task_manager"]._tasks) == 0
    assert setup["handle"].last_command is None


# ==============================================================================
# Streamlit & UI Rendering Tests
# ==============================================================================

def test_streamlit_tool_render_states():
    """Verify tool rendering states for running, pending approval, completed, and failed."""
    # 1. Running state
    running_html = render_tool_html("make_call", completed=False)
    assert "tool-card running" in running_html
    assert "◌" in running_html
    assert "Placing phone call..." in running_html

    # 2. Pending approval state (MUST NOT say 'Phone call placed' or have checkmark)
    pending_html = render_tool_html("make_call", completed=False, pending=True)
    assert "tool-card pending" in pending_html
    assert "⏳" in pending_html
    assert "Placing phone call... (Awaiting approval)" in pending_html
    assert "✓" not in pending_html
    assert "Phone call placed" not in pending_html

    # 3. Completed state (genuinely confirmed)
    completed_html = render_tool_html("make_call", completed=True)
    assert "tool-card completed" in completed_html
    assert "✓" in completed_html
    assert "Phone call placed" in completed_html

    # 4. Failed state
    failed_html = render_tool_html("make_call", failed=True)
    assert "tool-card failed" in failed_html
    assert "✕" in failed_html
