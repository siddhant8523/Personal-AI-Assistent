import pytest
from unittest.mock import MagicMock

from assistant.device_gateway.device_gateway import DeviceGateway, DeviceOfflineError
from assistant.device_gateway.transport.protocol import DeviceCommand, DeviceResult
from assistant.security.capability_validator import CapabilityValidator
from assistant.agent_core.graph.tools import build_tools
from assistant.agent_core.task_planner import TaskPlanner
from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.intelligence.priority_inbox import PriorityInbox


class MockDeviceHandle:
    def __init__(self, gateway: DeviceGateway):
        self.gateway = gateway
        self.last_command = None

    def dispatch(self, command: DeviceCommand):
        self.last_command = command
        cap = command.capability.upper().replace(".", "_")
        # Simulate asynchronous Android response
        if cap in ("READ_SMS", "SMS_READ"):
            res = DeviceResult(
                task_id=command.task_id,
                status="ok",
                detail=[
                    {"sender": "+123456789", "body": "Hello from Android test!"},
                    {"sender": "Bank", "body": "Your OTP is 998877"},
                ],
            )
        elif cap in ("MAKE_CALL", "CALL_MAKE"):
            res = DeviceResult(task_id=command.task_id, status="ok", detail="Calling Maosi...")
        elif cap in ("SET_ALARM", "ALARM_SET"):
            res = DeviceResult(task_id=command.task_id, status="ok", detail="Alarm set for 7 AM")
        elif cap in ("CONTACT_FIND", "CONTACTS_SEARCH"):
            query = command.params.get("query", "")
            if query == "mummy":
                res = DeviceResult(task_id=command.task_id, status="ok", detail={"query": query, "matches": [{"name": "Mummy", "phone_number": "+917798298569"}]})
            elif query == "multi":
                res = DeviceResult(task_id=command.task_id, status="ok", detail={"query": query, "matches": [{"name": "Multi 1", "phone_number": "+111"}, {"name": "Multi 2", "phone_number": "+222"}]})
            else:
                res = DeviceResult(task_id=command.task_id, status="ok", detail={"query": query, "matches": []})
        else:
            res = DeviceResult(task_id=command.task_id, status="ok", detail="Command executed")
        
        # Deliver result asynchronously to gateway
        self.gateway.handle_device_result(res.to_dict())
        return {"status": "ok", "detail": "queued", "device_task_id": command.task_id}


@pytest.fixture
def setup_gateway():
    validator = CapabilityValidator(
        android_capabilities=[
            "sms.send",
            "sms.read",
            "call.make",
            "file.read",
            "file.upload",
            "file.list",
            "file.find",
            "intent.execute",
            "intent.open_app",
            "alarm.set",
            "alarm.list",
            "alarm.cancel",
            "timer.set",
            "contact.find",
            "contacts.search",
        ],
        cloud_capabilities=["send_email", "send_telegram_message", "send_whatsapp_message"],
    )

    gateway = DeviceGateway(validator)
    handle = MockDeviceHandle(gateway)
    gateway.attach_device(handle)
    return gateway, handle


def test_device_gateway_send_command_correlation(setup_gateway):
    gateway, handle = setup_gateway
    res = gateway.send_command("READ_SMS", {"limit": 5})

    assert res.status == "ok"
    assert len(res.detail) == 2
    assert res.detail[0]["body"] == "Hello from Android test!"
    assert handle.last_command.capability == "sms.read"


def test_device_offline_raises_exception():
    validator = CapabilityValidator(android_capabilities=["sms.read"], cloud_capabilities=[])
    offline_gateway = DeviceGateway(validator)

    with pytest.raises(DeviceOfflineError):
        offline_gateway.send_command("READ_SMS", {})


def test_reconnect_race_condition_stale_detach():
    validator = CapabilityValidator(android_capabilities=["sms.read"], cloud_capabilities=[])
    gateway = DeviceGateway(validator)

    handle_a = MagicMock()
    handle_b = MagicMock()

    # Device A connects
    gateway.attach_device(handle_a)
    assert gateway.is_device_connected() is True
    assert gateway._connected_device is handle_a

    # Device B reconnects and attaches
    gateway.attach_device(handle_b)
    assert gateway.is_device_connected() is True
    assert gateway._connected_device is handle_b

    # Stale connection A closes and attempts to detach
    gateway.detach_device(handle_a)
    # Device B must remain active
    assert gateway.is_device_connected() is True
    assert gateway._connected_device is handle_b

    # Active connection B closes and detaches
    gateway.detach_device(handle_b)
    assert gateway.is_device_connected() is False
    assert gateway._connected_device is None


def test_unallowed_capability():
    validator = CapabilityValidator(android_capabilities=["sms.read"], cloud_capabilities=[])
    gateway = DeviceGateway(validator)
    gateway.attach_device(MagicMock())

    res = gateway.send_command("UNALLOWED_CAPABILITY", {})
    assert res.status == "failed"
    assert "capability not allowed" in res.detail


def test_capability_normalization_and_canonical_dispatch(setup_gateway):
    gateway, handle = setup_gateway

    test_cases = [
        ("make_call", "call.make", {"recipient": "vector"}),
        ("send_sms", "sms.send", {"recipient": "+123", "content": "hi"}),
        ("read_sms", "sms.read", {"limit": 1}),
        ("read_file", "file.read", {"path": "/sdcard/test.txt"}),
        ("upload_file", "file.upload", {"path": "/sdcard/test.txt"}),
        ("run_intent", "intent.execute", {"action": "android.settings.SETTINGS"}),
        ("set_alarm", "alarm.set", {"time": "07:00"}),
        ("set_timer", "timer.set", {"seconds": 60}),
    ]

    for legacy_name, canonical_name, params in test_cases:
        res = gateway.send_command(legacy_name, params)
        assert res.status == "ok"
        assert handle.last_command.capability == canonical_name, f"Expected {canonical_name}, got {handle.last_command.capability}"


def test_capability_validator_rules():
    validator = CapabilityValidator(
        android_capabilities=["call.make", "alarm.set", "sms.send"],
        cloud_capabilities=["send_email"],
    )

    # Allowed canonical and legacy Android capabilities
    assert validator.validate_android("call.make") is True
    assert validator.validate_android("make_call") is True
    assert validator.validate_android("alarm.set") is True
    assert validator.validate_android("set_alarm") is True
    assert validator.validate_android("sms.send") is True
    assert validator.validate_android("send_sms") is True

    # Unallowed capability must be rejected
    assert validator.validate_android("fake.capability") is False
    assert validator.validate_android("hack_device") is False

    # Cloud capabilities unaffected
    assert validator.validate_cloud("send_email") is True
    assert validator.validate_cloud("make_call") is False


def test_read_sms_tool_execution(setup_gateway):
    gateway, _ = setup_gateway
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    policy_engine = PolicyEngine({"make_call": {"requires_approval": True}})
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

    read_sms_tool = next((t for t in tools if t.name == "read_sms"), None)
    assert read_sms_tool is not None

    output = read_sms_tool.invoke({"limit": 5})
    assert "From +123456789: Hello from Android test!" in output
    assert "From Bank: Your OTP is 998877" in output


def test_set_alarm_tool_execution(setup_gateway):
    gateway, handle = setup_gateway
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    policy_engine = PolicyEngine({})
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

    set_alarm_tool = next((t for t in tools if t.name == "set_alarm"), None)
    assert set_alarm_tool is not None

    output = set_alarm_tool.invoke({"time": "7:57 PM", "label": "Evening"})
    assert "Alarm set for 7:57 PM (Evening)." in output
    assert handle.last_command.capability == "alarm.set"
    assert handle.last_command.params["hour"] == 19
    assert handle.last_command.params["minute"] == 57


def test_alarm_time_parser():
    from assistant.tools.alarm_parser import parse_alarm_time

    res_12_pm = parse_alarm_time("7:57 pm", "Test")
    assert res_12_pm["hour"] == 19
    assert res_12_pm["minute"] == 57
    assert res_12_pm["label"] == "Test"

    res_12_am = parse_alarm_time("8:30 am")
    assert res_12_am["hour"] == 8
    assert res_12_am["minute"] == 30

    res_24 = parse_alarm_time("22:15")
    assert res_24["hour"] == 22
    assert res_24["minute"] == 15

    res_rel = parse_alarm_time("in 10 minutes")
    assert res_rel["delay_ms"] == 600000


def test_mobile_path_resolver():
    from assistant.files.mobile_path_resolver import canonical_mobile_path, is_mobile_path

    assert is_mobile_path("/Books/cptopic.pdf") is True
    assert is_mobile_path("Books/cptopic.pdf") is True
    assert is_mobile_path("/storage/emulated/0/Download/file.pdf") is True
    assert is_mobile_path("~/Documents/report.pdf") is False

    assert canonical_mobile_path("/Books/cptopic.pdf") == "/storage/emulated/0/Books/cptopic.pdf"
    assert canonical_mobile_path("Books/cptopic.pdf") == "/storage/emulated/0/Books/cptopic.pdf"


def test_mobile_file_upload_routing(setup_gateway):
    gateway, handle = setup_gateway
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    policy_engine = PolicyEngine({})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)
    priority_inbox = PriorityInbox()

    from assistant.files.file_policy import FilePolicy
    from assistant.files.file_resolver import FileResolver
    from assistant.files.attachment_manager import AttachmentManager

    file_policy = FilePolicy(allowed_roots=["./data/outbound_files"])
    file_resolver = FileResolver(file_policy)
    attachment_manager = AttachmentManager(file_policy)

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        priority_inbox=priority_inbox,
        tool_router=tool_router,
        file_resolver=file_resolver,
        attachment_manager=attachment_manager,
        device_gateway=gateway,
    )

    send_file_tool = next((t for t in tools if t.name == "send_file"), None)
    assert send_file_tool is not None

    # Simulate mobile file send (raises interrupt exception for human approval after fetching file)
    with pytest.raises(Exception):
        send_file_tool.invoke({
            "platform": "telegram",
            "recipient": "opfrenzygamer",
            "file_query": "/Books/cptopic.pdf",
            "message": "Here is the PDF"
        })
    assert handle.last_command.capability == "file.upload"
    assert handle.last_command.params["path"] == "/storage/emulated/0/Books/cptopic.pdf"



def test_open_mobile_app_tool(setup_gateway):
    gateway, handle = setup_gateway
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    policy_engine = PolicyEngine({})
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

    open_app_tool = next((t for t in tools if t.name == "open_mobile_app"), None)
    assert open_app_tool is not None

    output = open_app_tool.invoke({"app_name": "WhatsApp"})
    assert "Opened application 'WhatsApp'" in output
    assert handle.last_command.capability in ("intent.open_app", "intent.execute")
    assert handle.last_command.params["package"] == "com.whatsapp"


def test_make_call_contact_resolution(setup_gateway, tmp_path):
    gateway, handle = setup_gateway
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    policy_engine = PolicyEngine({"call.make": {"requires_approval": True}})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)
    priority_inbox = PriorityInbox()

    from assistant.memory.user_profile import UserProfile
    user_profile = UserProfile(str(tmp_path / "memory"))

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        priority_inbox=priority_inbox,
        tool_router=tool_router,
        user_profile=user_profile,
        device_gateway=gateway,
    )

    make_call_tool = next((t for t in tools if t.name == "make_call"), None)
    assert make_call_tool is not None

    # Invoking make_call with contact 'mummy' (resolves to +917798298569, then triggers interrupt)
    with pytest.raises(Exception):
        make_call_tool.invoke({"recipient": "mummy"})

    # Inspect created task
    waiting_task = task_manager.find_waiting_approval("make_call", "mummy", "Make phone call to mummy (+91******8569)")
    assert waiting_task is not None
    assert waiting_task.parameters["phone_number"] == "+917798298569"
    assert waiting_task.parameters["number"] == "+917798298569"



def test_make_call_unresolved_contact(setup_gateway, tmp_path):
    gateway, handle = setup_gateway
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    policy_engine = PolicyEngine({})
    task_planner = TaskPlanner(task_manager, policy_engine)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)
    priority_inbox = PriorityInbox()

    from assistant.memory.user_profile import UserProfile
    user_profile = UserProfile(str(tmp_path / "memory"))

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        priority_inbox=priority_inbox,
        tool_router=tool_router,
        user_profile=user_profile,
        device_gateway=gateway,
    )

    make_call_tool = next((t for t in tools if t.name == "make_call"), None)
    assert make_call_tool is not None

    res = make_call_tool.invoke({"recipient": "unknown_person"})
    assert "I couldn't find a contact named 'unknown_person' on your phone" in res



def test_list_and_cancel_alarms_tools(setup_gateway):
    gateway, handle = setup_gateway
    outbound_registry = OutboundRegistry()
    task_manager = TaskManager(outbound_registry)
    policy_engine = PolicyEngine({})
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

    list_alarms_tool = next((t for t in tools if t.name == "list_alarms"), None)
    cancel_alarm_tool = next((t for t in tools if t.name == "cancel_alarm"), None)

    assert list_alarms_tool is not None
    assert cancel_alarm_tool is not None

    res_list = list_alarms_tool.invoke({})
    assert handle.last_command.capability == "alarm.list"

    res_cancel = cancel_alarm_tool.invoke({"alarm_id": "alarm-12345678"})
    assert handle.last_command.capability == "alarm.cancel"
    assert handle.last_command.params["alarm_id"] == "alarm-12345678"




