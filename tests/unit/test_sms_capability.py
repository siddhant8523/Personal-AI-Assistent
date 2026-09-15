import pytest
from unittest.mock import MagicMock

from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.graph.tools import build_tools
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.agent_core.task_planner import TaskPlanner
from assistant.device_gateway.device_gateway import DeviceGateway
from assistant.device_gateway.transport.protocol import DeviceCommand, DeviceResult
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task import TaskState
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.security.capability_validator import CapabilityValidator


class MockSmsDeviceHandle:
    def __init__(self, gateway: DeviceGateway):
        self.gateway = gateway
        self.last_command = None

    def dispatch(self, command: DeviceCommand):
        self.last_command = command
        cap = command.capability
        if cap in ("contact.find", "CONTACT_FIND"):
            query = command.params.get("query", "")
            if query == "Rahul":
                res = DeviceResult(
                    task_id=command.task_id,
                    status="ok",
                    detail={"query": query, "matches": [{"name": "Rahul", "phone_number": "+919876543210"}]},
                )
            elif query == "Multi":
                res = DeviceResult(
                    task_id=command.task_id,
                    status="ok",
                    detail={
                        "query": query,
                        "matches": [
                            {"name": "Multi 1", "phone_number": "+111"},
                            {"name": "Multi 2", "phone_number": "+222"},
                        ],
                    },
                )
            else:
                res = DeviceResult(task_id=command.task_id, status="ok", detail={"query": query, "matches": []})
        elif cap in ("sms.send", "SEND_SMS"):
            res = DeviceResult(task_id=command.task_id, status="ok", detail="sent")
        else:
            res = DeviceResult(task_id=command.task_id, status="ok", detail="ok")

        self.gateway.handle_device_result(res.to_dict())
        return {"status": "ok", "detail": "queued", "device_task_id": command.task_id}


@pytest.fixture
def setup_sms_gateway():
    validator = CapabilityValidator(
        android_capabilities=["sms.send", "sms.read", "contact.find"],
        cloud_capabilities=[],
    )
    gateway = DeviceGateway(validator)
    handle = MockSmsDeviceHandle(gateway)
    gateway.attach_device(handle)
    return gateway, handle


def test_sms_direct_phone_number_proposal(setup_sms_gateway):
    gateway, handle = setup_sms_gateway
    task_manager = TaskManager(OutboundRegistry())
    policy_engine = PolicyEngine({"send_sms": {"requires_approval": True}})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)
    priority_inbox = PriorityInbox()

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        priority_inbox=priority_inbox,
        tool_router=tool_router,
        device_gateway=gateway,
    )

    send_tool = next(t for t in tools if t.name == "propose_send_message")

    # Invoking with direct phone number triggers LangGraph interrupt for approval
    with pytest.raises(Exception):
        send_tool.invoke({"platform": "sms", "recipient": "+919876543210", "content": "Hello Direct"})

    # Task created in WAITING_FOR_APPROVAL
    waiting_tasks = [t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]

    assert len(waiting_tasks) == 1
    task = waiting_tasks[0]
    assert task.target == "+919876543210"
    assert task.parameters["phone_number"] == "+919876543210"
    assert task.parameters["content"] == "Hello Direct"


def test_sms_contact_name_resolution(setup_sms_gateway):
    gateway, handle = setup_sms_gateway
    task_manager = TaskManager(OutboundRegistry())
    policy_engine = PolicyEngine({"send_sms": {"requires_approval": True}})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)
    priority_inbox = PriorityInbox()

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        priority_inbox=priority_inbox,
        tool_router=tool_router,
        device_gateway=gateway,
    )

    send_tool = next(t for t in tools if t.name == "propose_send_message")

    # Invoking with contact name 'Rahul' resolves to phone number and triggers interrupt
    with pytest.raises(Exception):
        send_tool.invoke({"platform": "sms", "recipient": "Rahul", "content": "Hello Rahul"})

    waiting_tasks = [t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]
    assert len(waiting_tasks) == 1
    task = waiting_tasks[0]
    assert task.target == "Rahul (+919876543210)"
    assert task.parameters["phone_number"] == "+919876543210"
    assert task.parameters["content"] == "Hello Rahul"


def test_sms_multiple_matching_contacts_ambiguity(setup_sms_gateway):
    gateway, _ = setup_sms_gateway
    task_manager = TaskManager(OutboundRegistry())
    policy_engine = PolicyEngine({"send_sms": {"requires_approval": True}})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        priority_inbox=PriorityInbox(),
        tool_router=tool_router,
        device_gateway=gateway,
    )

    send_tool = next(t for t in tools if t.name == "propose_send_message")
    res = send_tool.invoke({"platform": "sms", "recipient": "Multi", "content": "Hello Multi"})

    assert "Found multiple contacts matching 'Multi'" in res
    assert "- Multi 1: +111" in res
    assert "- Multi 2: +222" in res
    # No task created
    assert len([t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]) == 0


def test_sms_contact_not_found(setup_sms_gateway):
    gateway, _ = setup_sms_gateway
    task_manager = TaskManager(OutboundRegistry())
    policy_engine = PolicyEngine({"send_sms": {"requires_approval": True}})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        priority_inbox=PriorityInbox(),
        tool_router=tool_router,
        device_gateway=gateway,
    )

    send_tool = next(t for t in tools if t.name == "propose_send_message")
    res = send_tool.invoke({"platform": "sms", "recipient": "UnknownContact", "content": "Hello"})

    assert "I couldn't find a contact named 'UnknownContact' on your phone" in res
    assert len([t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]) == 0


def test_sms_approval_and_final_capability_execution(setup_sms_gateway):
    gateway, handle = setup_sms_gateway
    task_manager = TaskManager(OutboundRegistry())
    policy_engine = PolicyEngine({"send_sms": {"requires_approval": True}})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)

    # Register handler in tool router
    def _send_sms_handler(task):
        phone_num = task.parameters.get("phone_number") or task.parameters.get("number")
        if not phone_num:
            import re
            m = re.search(r"\(([\+\d\s\-]+)\)", str(task.target or ""))
            if m:
                phone_num = m.group(1).strip()

        content = task.parameters.get("text") or task.parameters.get("content") or ""
        params = {
            "recipient": task.target,
            "phone_number": phone_num or "",
            "number": phone_num or "",
            "text": content,
            **task.parameters,
        }
        return gateway.send_command("sms.send", params).to_dict()

    tool_router.register("send_sms", _send_sms_handler)

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        priority_inbox=PriorityInbox(),
        tool_router=tool_router,
        device_gateway=gateway,
    )

    send_tool = next(t for t in tools if t.name == "propose_send_message")

    with pytest.raises(Exception):
        send_tool.invoke({"platform": "sms", "recipient": "Rahul", "content": "Test SMS"})

    waiting = [t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]
    assert len(waiting) == 1
    task_id = waiting[0].task_id


    # Approve task
    approved_task = approval_manager.approve(task_id)
    res = tool_router.dispatch(approved_task, source="sms", conversation_id=approved_task.target)

    assert res["status"] == "ok"

    assert handle.last_command.capability == "sms.send"
    assert handle.last_command.params["phone_number"] == "+919876543210"
    assert handle.last_command.params["text"] == "Test SMS"
