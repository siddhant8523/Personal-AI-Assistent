import time
from unittest.mock import MagicMock
import pytest

from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.graph.tools import build_tools
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.agent_core.task_planner import TaskPlanner
from assistant.connectors.whatsapp.whatsapp_connector import WhatsAppConnector
from assistant.contacts.contact_cache import ContactCache, mask_phone, phone_to_jid
from assistant.device_gateway.device_gateway import DeviceGateway
from assistant.device_gateway.transport.protocol import DeviceCommand, DeviceResult
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task import TaskState
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.security.capability_validator import CapabilityValidator
from assistant.storage.db import get_connection, reset_connection


class MockContactDeviceHandle:
    def __init__(self, gateway: DeviceGateway):
        self.gateway = gateway
        self.call_count = 0
        self.last_query = None
        self.contacts = {
            "vector": [{"name": "Vector", "phone_number": "+917264064119"}],
            "multi": [
                {"name": "Vector Work", "phone_number": "+917264064119"},
                {"name": "Vector Home", "phone_number": "+919876543210"},
            ],
        }
        self.permission_denied = False

    def dispatch(self, command: DeviceCommand):
        self.call_count += 1
        cap = command.capability
        if cap == "contact.find":
            if self.permission_denied:
                res = DeviceResult(
                    task_id=command.task_id,
                    status="error",
                    detail="CONTACTS_PERMISSION_DENIED: User denied READ_CONTACTS permission.",
                )
            else:
                query = command.params.get("query", "")
                self.last_query = query
                q_lower = query.strip().lower()
                matches = self.contacts.get(q_lower, [])
                res = DeviceResult(
                    task_id=command.task_id,
                    status="ok",
                    detail={"query": query, "matches": matches},
                )
        else:
            res = DeviceResult(task_id=command.task_id, status="ok", detail="ok")

        self.gateway.handle_device_result(res.to_dict())
        return {"status": "ok", "detail": "queued", "device_task_id": command.task_id}


@pytest.fixture
def contact_env(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_assistant.db")
    monkeypatch.setenv("DATABASE_PATH", db_file)
    monkeypatch.setenv("CONTACT_CACHE_TTL_DAYS", "10")
    reset_connection()

    validator = CapabilityValidator(
        android_capabilities=["contact.find", "sms.send"],
        cloud_capabilities=[],
    )
    gateway = DeviceGateway(validator)
    handle = MockContactDeviceHandle(gateway)
    gateway.attach_device(handle)

    cache = ContactCache(ttl_days=10)
    connector = WhatsAppConnector(
        bridge_url="http://localhost:3000",
        enabled=False,
        contact_cache=cache,
        device_gateway=gateway,
    )

    yield {
        "gateway": gateway,
        "handle": handle,
        "cache": cache,
        "connector": connector,
        "db_file": db_file,
    }

    reset_connection()


# 1. First lookup queries Android Contacts.
def test_1_first_lookup_queries_android_contacts(contact_env):
    conn = contact_env["connector"]
    handle = contact_env["handle"]

    assert handle.call_count == 0
    jid, err = conn.resolve_recipient("Vector")

    assert err is None
    assert jid == "917264064119@s.whatsapp.net"
    assert handle.call_count == 1
    assert handle.last_query == "Vector"


# 2. Successful lookup creates cache entry.
def test_2_successful_lookup_creates_cache_entry(contact_env):
    conn = contact_env["connector"]
    cache = contact_env["cache"]

    jid, err = conn.resolve_recipient("Vector")
    assert err is None

    entry = cache.get("Vector")
    assert entry is not None
    assert entry.normalized_name == "vector"
    assert entry.display_name == "Vector"
    assert entry.phone_number == "+917264064119"
    assert entry.resolved_jid == "917264064119@s.whatsapp.net"
    assert entry.cached_at > 0
    assert entry.expires_at > entry.cached_at


# 3. Second lookup within TTL does NOT query Android Contacts.
def test_3_second_lookup_within_ttl_does_not_query_android_contacts(contact_env):
    conn = contact_env["connector"]
    handle = contact_env["handle"]

    # First lookup
    jid1, _ = conn.resolve_recipient("Vector")
    assert handle.call_count == 1

    # Second lookup within TTL
    jid2, err2 = conn.resolve_recipient("Vector")
    assert err2 is None
    assert jid2 == jid1
    # Android Contacts should NOT be called again
    assert handle.call_count == 1


# 4. Cache lookup is case-insensitive.
def test_4_cache_lookup_is_case_insensitive(contact_env):
    conn = contact_env["connector"]
    handle = contact_env["handle"]

    # First lookup with mixed case
    conn.resolve_recipient("Vector")
    assert handle.call_count == 1

    # Lookups with uppercase, lowercase, and spaces
    jid_lower, _ = conn.resolve_recipient("vector")
    jid_upper, _ = conn.resolve_recipient("VECTOR")
    jid_spaces, _ = conn.resolve_recipient("  Vector  ")

    assert jid_lower == "917264064119@s.whatsapp.net"
    assert jid_upper == "917264064119@s.whatsapp.net"
    assert jid_spaces == "917264064119@s.whatsapp.net"
    # None of these should have queried Android Contacts
    assert handle.call_count == 1


# 5. Cache expiration triggers Android Contacts lookup.
def test_5_cache_expiration_triggers_android_contacts_lookup(contact_env):
    conn = contact_env["connector"]
    cache = contact_env["cache"]
    handle = contact_env["handle"]

    # Pre-populate an expired entry
    past_time = time.time() - 1000
    cache.set(
        name="Vector",
        display_name="Vector",
        phone_number="+917264064119",
        resolved_jid="917264064119@s.whatsapp.net",
        now=past_time - (11 * 86400),  # expired
    )

    # Resolution should detect expiration, purge, and query Android Contacts
    assert handle.call_count == 0
    jid, err = conn.resolve_recipient("Vector")

    assert err is None
    assert jid == "917264064119@s.whatsapp.net"
    assert handle.call_count == 1


# 6. Refresh updates phone number.
def test_6_refresh_updates_phone_number(contact_env):
    conn = contact_env["connector"]
    cache = contact_env["cache"]
    handle = contact_env["handle"]

    # First lookup caches +917264064119
    conn.resolve_recipient("Vector")
    entry = cache.get("Vector")
    assert entry.phone_number == "+917264064119"

    # Android contacts updates Vector's number to +919876543210
    handle.contacts["vector"] = [{"name": "Vector", "phone_number": "+919876543210"}]

    # Force refresh / refresh_contact
    new_jid, err = conn.refresh_contact("Vector")
    assert err is None
    assert new_jid == "919876543210@s.whatsapp.net"

    refreshed_entry = cache.get("Vector")
    assert refreshed_entry.phone_number == "+919876543210"
    assert refreshed_entry.resolved_jid == "919876543210@s.whatsapp.net"


# 7. Expired cache does not get used.
def test_7_expired_cache_does_not_get_used(contact_env):
    conn = contact_env["connector"]
    cache = contact_env["cache"]
    handle = contact_env["handle"]

    # Cache an entry with old number and expired timestamp
    past_time = time.time() - 2000
    cache.set(
        name="Vector",
        display_name="Vector",
        phone_number="+911111111111",
        resolved_jid="911111111111@s.whatsapp.net",
        now=past_time - (15 * 86400),
    )

    # Android contacts has the real current number
    handle.contacts["vector"] = [{"name": "Vector", "phone_number": "+917264064119"}]

    jid, err = conn.resolve_recipient("Vector")
    assert err is None
    # Old expired number +911111111111 must NOT be used
    assert jid == "917264064119@s.whatsapp.net"
    assert handle.call_count == 1


# 8. Multiple matching contacts remain ambiguous.
def test_8_multiple_matching_contacts_remain_ambiguous(contact_env):
    conn = contact_env["connector"]
    cache = contact_env["cache"]
    handle = contact_env["handle"]

    # "multi" matches two contacts in handle
    jid, err = conn.resolve_recipient("multi")

    assert jid is None
    assert "Found multiple contacts matching 'multi'" in err
    assert "Vector Work: +917264064119" in err
    assert "Vector Home: +919876543210" in err

    # Ambiguous result must NOT be cached
    assert cache.get("multi") is None


# 9. Direct phone number bypasses contact lookup.
def test_9_direct_phone_number_bypasses_contact_lookup(contact_env):
    conn = contact_env["connector"]
    cache = contact_env["cache"]
    handle = contact_env["handle"]

    jid1, err1 = conn.resolve_recipient("+917264064119")
    assert err1 is None
    assert jid1 == "917264064119@s.whatsapp.net"
    assert handle.call_count == 0
    assert len(cache.all_entries()) == 0

    jid2, err2 = conn.resolve_recipient("917264064119")
    assert err2 is None
    assert jid2 == "917264064119@s.whatsapp.net"
    assert handle.call_count == 0
    assert len(cache.all_entries()) == 0


# 10. Valid cache works while Android device is disconnected.
def test_10_valid_cache_works_while_android_device_is_disconnected(contact_env):
    conn = contact_env["connector"]
    gateway = contact_env["gateway"]

    # Populate cache when online
    conn.resolve_recipient("Vector")

    # Disconnect Android device
    gateway.detach_device()
    assert not gateway.is_device_connected()

    # Cached lookup succeeds offline
    jid, err = conn.resolve_recipient("Vector")
    assert err is None
    assert jid == "917264064119@s.whatsapp.net"


# 11. Expired/missing cache with disconnected Android gives proper response.
def test_11_expired_or_missing_cache_disconnected_android_proper_response(contact_env):
    conn = contact_env["connector"]
    gateway = contact_env["gateway"]

    # Disconnect Android device
    gateway.detach_device()
    assert not gateway.is_device_connected()

    jid, err = conn.resolve_recipient("NonExistentContact")
    assert jid is None
    assert "Android device is disconnected" in err
    assert "provide their phone number directly" in err


# 12. Permission-required behavior.
def test_12_permission_required_behavior(contact_env):
    conn = contact_env["connector"]
    handle = contact_env["handle"]

    handle.permission_denied = True

    jid, err = conn.resolve_recipient("Vector")
    assert jid is None
    assert "Contacts permission is denied" in err
    assert "grant Contacts permission in Android Settings" in err


# 13. Invalid WhatsApp recipient invalidates/refetches the cache.
def test_13_invalid_whatsapp_recipient_invalidates_refetches_cache(contact_env):
    conn = contact_env["connector"]
    cache = contact_env["cache"]
    handle = contact_env["handle"]

    # Cache an invalid recipient
    cache.set(
        name="Vector",
        display_name="Vector",
        phone_number="+910000000000",
        resolved_jid="910000000000@s.whatsapp.net",
    )

    # Invalidate and refresh
    handle.contacts["vector"] = [{"name": "Vector", "phone_number": "+917264064119"}]
    new_jid, err = conn.refresh_contact("Vector")

    assert err is None
    assert new_jid == "917264064119@s.whatsapp.net"
    assert cache.get("Vector").phone_number == "+917264064119"


# 14. Existing approval is still required.
def test_14_existing_approval_is_still_required(contact_env):
    conn = contact_env["connector"]
    gateway = contact_env["gateway"]

    # Populate cache
    conn.resolve_recipient("Vector")

    task_manager = TaskManager(OutboundRegistry())
    policy_engine = PolicyEngine({"send_whatsapp_message": {"requires_approval": True}})
    task_planner = TaskPlanner(task_manager, policy_engine, whatsapp_connector=conn)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)

    from assistant.intelligence.priority_inbox import PriorityInbox
    priority_inbox = PriorityInbox()

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        tool_router=tool_router,
        priority_inbox=priority_inbox,
        whatsapp_connector=conn,
        device_gateway=gateway,
    )

    propose_tool = next(t for t in tools if t.name == "propose_send_message")

    # Proposing message to cached recipient must pause for approval (LangGraph interrupt)
    with pytest.raises(Exception):
        propose_tool.invoke({
            "platform": "whatsapp",
            "recipient": "Vector",
            "content": "Hello Vector from cache",
        })

    waiting = [t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]
    assert len(waiting) == 1
    assert waiting[0].target == "917264064119@s.whatsapp.net"
    assert waiting[0].parameters["content"] == "Hello Vector from cache"


# 15. Entire contact book is never cached.
def test_15_entire_contact_book_is_never_cached(contact_env):
    conn = contact_env["connector"]
    cache = contact_env["cache"]
    handle = contact_env["handle"]

    # Gateway handle has multiple contacts available
    handle.contacts["alice"] = [{"name": "Alice", "phone_number": "+1111111111"}]
    handle.contacts["bob"] = [{"name": "Bob", "phone_number": "+2222222222"}]
    handle.contacts["vector"] = [{"name": "Vector", "phone_number": "+917264064119"}]

    # Only resolve Vector
    conn.resolve_recipient("Vector")

    entries = cache.all_entries()
    assert len(entries) == 1
    assert entries[0].normalized_name == "vector"
    # Neither Alice nor Bob should be in the cache
    assert cache.get("alice") is None
    assert cache.get("bob") is None

    # Inspect SQLite database schema columns directly
    db_conn = get_connection()
    cols = [col[1] for col in db_conn.execute("PRAGMA table_info(contact_cache)").fetchall()]
    # Ensure only the 6 required minimal fields exist
    expected_cols = ["normalized_name", "display_name", "phone_number", "resolved_jid", "cached_at", "expires_at"]
    assert set(cols) == set(expected_cols)
    assert "email" not in cols
    assert "address" not in cols
    assert "notes" not in cols
    assert "photo" not in cols


def test_mask_phone_privacy():
    assert mask_phone("+917264064119") == "+91******4119"
    assert mask_phone("12345") == "***"
    assert mask_phone("") == "***"


def test_phone_number_is_durable_identity_and_jid_rederived(contact_env):
    conn = contact_env["connector"]
    cache = contact_env["cache"]

    # Manually store entry with phone number and empty resolved_jid
    cache.set(
        name="Vector",
        display_name="Vector",
        phone_number="+917264064119",
        resolved_jid=None,
    )

    jid, err = conn.resolve_recipient("Vector")
    assert err is None
    # JID is safely derived from durable phone number
    assert jid == "917264064119@s.whatsapp.net"


def test_direct_phone_number_sending_works_without_contacts_permission(contact_env):
    conn = contact_env["connector"]
    handle = contact_env["handle"]

    # Deny Contacts permission
    handle.permission_denied = True

    # Direct phone number resolution must work completely without Contacts permission
    jid, err = conn.resolve_recipient("+917264064119")
    assert err is None
    assert jid == "917264064119@s.whatsapp.net"
    assert handle.call_count == 0


def test_recipient_change_after_refresh_requires_approval_again(contact_env):
    conn = contact_env["connector"]
    cache = contact_env["cache"]
    gateway = contact_env["gateway"]
    handle = contact_env["handle"]

    # Initial resolve and approval setup
    conn.resolve_recipient("Vector")

    from assistant.intelligence.priority_inbox import PriorityInbox
    task_manager = TaskManager(OutboundRegistry())
    policy_engine = PolicyEngine({"send_whatsapp_message": {"requires_approval": True}})
    task_planner = TaskPlanner(task_manager, policy_engine, whatsapp_connector=conn)
    approval_manager = ApprovalManager(task_manager)
    tool_router = ToolRouter(task_manager)

    tools = build_tools(
        task_planner=task_planner,
        approval_manager=approval_manager,
        tool_router=tool_router,
        priority_inbox=PriorityInbox(),
        whatsapp_connector=conn,
        device_gateway=gateway,
    )
    propose_tool = next(t for t in tools if t.name == "propose_send_message")

    # Propose 1: targeting initial number
    with pytest.raises(Exception):
        propose_tool.invoke({"platform": "whatsapp", "recipient": "Vector", "content": "Msg 1"})

    tasks1 = [t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]
    assert len(tasks1) == 1
    assert tasks1[0].target == "917264064119@s.whatsapp.net"

    # Number changes in Android Contacts
    handle.contacts["vector"] = [{"name": "Vector", "phone_number": "+919876543210"}]
    conn.refresh_contact("Vector")

    # Propose 2: must pause and require approval again for the new recipient
    with pytest.raises(Exception):
        propose_tool.invoke({"platform": "whatsapp", "recipient": "Vector", "content": "Msg 2"})

    tasks2 = [t for t in task_manager._tasks.values() if t.execution_state == TaskState.WAITING_FOR_APPROVAL]
    assert len(tasks2) == 2
    # The new task targets the new recipient and is waiting for approval
    assert tasks2[1].target == "919876543210@s.whatsapp.net"
    assert tasks2[1].execution_state == TaskState.WAITING_FOR_APPROVAL

