"""
Unit tests for Message Intelligence / Priority Analyzer background batch process.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest

from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage
from assistant.intelligence.classifier import Category
from assistant.intelligence.message_intelligence_models import MessageAnalysisItem
from assistant.intelligence.priority_analyzer import (
    PriorityAnalyzer,
    PriorityAnalyzerWorker,
)
from assistant.intelligence.priority_engine import PriorityEngine, PriorityLevel, PriorityResult
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.intelligence.priority_rules import PriorityRulesManager
from assistant.llm.llm_client import (
    LLMAuthenticationError,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
)
from assistant.memory.user_profile import UserProfile
from assistant.storage.db import get_connection, init_db, reset_connection


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    db_file = str(tmp_path / "test_assistant.db")
    os.environ["DATABASE_PATH"] = db_file
    reset_connection()
    init_db()
    yield
    reset_connection()


@pytest.fixture
def mock_user_profile(tmp_path):
    mem_dir = str(tmp_path / "memory")
    profile = UserProfile(mem_dir)
    return profile


def _create_message(
    msg_id: str,
    source: Source = Source.TELEGRAM,
    sender: str = "Alice",
    content: str = "Hello",
    ts: float | None = None,
) -> UnifiedMessage:
    return UnifiedMessage(
        message_id=msg_id,
        conversation_id=f"{source.value}:{sender}",
        source=source,
        sender=sender,
        content=content,
        timestamp=ts or time.time(),
        origin=Origin.USER,
    )


def _add_pending_message(
    inbox: PriorityInbox,
    msg_id: str,
    source: Source = Source.TELEGRAM,
    sender: str = "Alice",
    content: str = "Hello",
    level: PriorityLevel = PriorityLevel.LOW,
    score: int = 1,
) -> None:

    msg = _create_message(msg_id, source=source, sender=sender, content=content)
    result = PriorityResult(
        message=msg,
        category=Category.OTHER,
        score=score,
        level=level,
    )
    inbox.add(result)


# ==============================================================================
# 1. Scheduling, Batching & Non-Blocking Tests
# ==============================================================================

def test_no_pending_messages_does_not_call_llm(mock_user_profile):
    """When pending_count == 0, analyzer MUST NOT call the LLM."""
    inbox = PriorityInbox()
    mock_llm = MagicMock()
    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)

    res = analyzer.process_cycle()

    assert res["status"] == "idle"
    assert res["processed"] == 0
    mock_llm.model.invoke.assert_not_called()
    mock_llm.generate.assert_not_called()


def test_scheduler_runs_at_configured_interval():
    inbox = PriorityInbox()
    analyzer = PriorityAnalyzer(priority_inbox=inbox)
    worker = PriorityAnalyzerWorker(analyzer, interval_minutes=15.0)

    assert worker.interval_seconds == 15.0 * 60.0


def test_default_interval_is_30_minutes():
    inbox = PriorityInbox()
    analyzer = PriorityAnalyzer(priority_inbox=inbox)
    worker = PriorityAnalyzerWorker(analyzer)

    assert worker.interval_seconds == 30.0 * 60.0


def test_interval_can_be_configured():
    inbox = PriorityInbox()
    analyzer = PriorityAnalyzer(priority_inbox=inbox)
    worker = PriorityAnalyzerWorker(analyzer, interval_minutes=5.0)

    assert worker.interval_seconds == 300.0


def test_batch_size_is_respected(mock_user_profile):
    inbox = PriorityInbox(max_per_level=200)
    for i in range(120):
        _add_pending_message(inbox, f"msg-{i}", content=f"Content {i}")

    mock_llm = MagicMock()
    # Return valid mock JSON response for 50 items
    def fake_invoke(messages):
        # Extract payload from messages
        items = [
            {
                "message_id": str(i + 1),
                "category": "WORK",
                "intent": "TASK",
                "importance": "MEDIUM",
                "urgency": "LOW",
                "requires_action": False,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": None,
                "reason": "Test item",
            }
            for i in range(50)
        ]
        resp = MagicMock()
        resp.content = json.dumps({"items": items})
        return resp

    mock_llm.model.invoke.side_effect = fake_invoke
    analyzer = PriorityAnalyzer(
        priority_inbox=inbox,
        user_profile=mock_user_profile,
        llm_client=mock_llm,
        batch_size=50,
        max_batches_per_cycle=1,
    )

    res = analyzer.process_cycle()
    assert res["processed"] == 50


def test_max_batches_per_cycle_is_respected(mock_user_profile):
    inbox = PriorityInbox(max_per_level=200)
    for i in range(100):
        _add_pending_message(inbox, f"msg-batch-{i}", content=f"Content {i}")


    mock_llm = MagicMock()
    def fake_invoke(messages):
        resp = MagicMock()
        # Parse payload IDs from user prompt
        text = messages[1].content
        match = text.split("Messages:\n")[-1]
        ids = [m["message_id"] for m in json.loads(match)]
        items = [
            {
                "message_id": mid,
                "category": "WORK",
                "intent": "TASK",
                "importance": "LOW",
                "urgency": "LOW",
                "requires_action": False,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": None,
                "reason": "Batch test",
            }
            for mid in ids
        ]
        resp.content = json.dumps({"items": items})
        return resp

    mock_llm.model.invoke.side_effect = fake_invoke
    analyzer = PriorityAnalyzer(
        priority_inbox=inbox,
        user_profile=mock_user_profile,
        llm_client=mock_llm,
        batch_size=20,
        max_batches_per_cycle=2,
    )

    res = analyzer.process_cycle()
    # With batch_size=20 and max_batches=2, exactly 40 messages processed
    assert res["processed"] == 40
    assert res["completed"] == 40
    # Remaining 60 still pending
    assert inbox.count_pending_messages() == 60


def test_analyzer_remains_non_blocking():
    inbox = PriorityInbox()
    analyzer = PriorityAnalyzer(priority_inbox=inbox)
    worker = PriorityAnalyzerWorker(analyzer, interval_minutes=30.0)

    worker.start()
    assert worker._thread is not None
    assert worker._thread.is_alive()
    assert worker._thread.daemon is True

    worker.stop()
    assert not worker._thread.is_alive()


def test_startup_does_not_call_llm_by_default(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "startup-msg-1")

    mock_llm = MagicMock()
    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    worker = PriorityAnalyzerWorker(analyzer, run_on_startup=False)

    worker.start()
    time.sleep(0.1)
    worker.stop()

    mock_llm.model.invoke.assert_not_called()
    assert inbox.count_pending_messages() == 1


# ==============================================================================
# 2. Multi-Source Ingestion & Batching Tests
# ==============================================================================

def test_pending_gmail_message_processed(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "gmail-101", source=Source.GMAIL, sender="recruiter@tech.com", content="Interview invite")

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "WORK",
            "intent": "INTERVIEW",
            "importance": "HIGH",
            "urgency": "HIGH",
            "requires_action": True,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": "2026-09-15T11:00:00",
            "reason": "Interview confirmation",
        }]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    assert res["completed"] == 1
    items = inbox.today()
    assert len(items) == 1
    assert items[0]["level"] == "HIGH"
    assert items[0]["system_category"] == "WORK"
    assert items[0]["deadline"] == "2026-09-15T11:00:00"


def test_pending_telegram_message_processed(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "tg-101", source=Source.TELEGRAM, sender="Rahul", content="Hey check this out")

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "PERSONAL",
            "intent": "CHAT",
            "importance": "LOW",
            "urgency": "LOW",
            "requires_action": False,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": None,
            "reason": "Casual greeting",
        }]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    assert res["completed"] == 1
    items = inbox.today()
    assert items[0]["level"] == "LOW"


def test_pending_whatsapp_message_processed(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "wa-101", source=Source.WHATSAPP, sender="Boss", content="Need the report ASAP")

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "WORK",
            "intent": "TASK",
            "importance": "HIGH",
            "urgency": "HIGH",
            "requires_action": True,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": None,
            "reason": "Urgent report request",
        }]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    assert res["completed"] == 1
    items = inbox.today()
    assert items[0]["level"] == "HIGH"


def test_pending_sms_message_processed(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "sms-101", source=Source.SMS, sender="+919876543210", content="Your OTP is 491021")

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "TRANSACTIONAL",
            "intent": "NOTIFICATION",
            "importance": "HIGH",
            "urgency": "HIGH",
            "requires_action": True,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": None,
            "reason": "OTP code",
        }]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    assert res["completed"] == 1
    items = inbox.today()
    assert items[0]["level"] == "HIGH"


def test_multiple_sources_combined_into_batch(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "m1", source=Source.GMAIL, sender="hr@job.com", content="Offer letter")
    _add_pending_message(inbox, "m2", source=Source.TELEGRAM, sender="Rahul", content="Hi")
    _add_pending_message(inbox, "m3", source=Source.WHATSAPP, sender="Manager", content="Meeting at 2")
    _add_pending_message(inbox, "m4", source=Source.SMS, sender="Bank", content="Debit of Rs 500")

    mock_llm = MagicMock()
    captured_payload = []
    def fake_invoke(messages):
        nonlocal captured_payload
        text = messages[1].content
        match = text.split("Messages:\n")[-1]
        captured_payload = json.loads(match)
        items = [
            {
                "message_id": m["message_id"],
                "category": "WORK",
                "intent": "TASK",
                "importance": "MEDIUM",
                "urgency": "LOW",
                "requires_action": False,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": None,
                "reason": "Combined batch test",
            }
            for m in captured_payload
        ]
        resp = MagicMock()
        resp.content = json.dumps({"items": items})
        return resp

    mock_llm.model.invoke.side_effect = fake_invoke
    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)

    res = analyzer.process_cycle()
    assert res["completed"] == 4
    # Ensure all 4 sources were present in the single LLM invocation
    sources = {m["source"] for m in captured_payload}
    assert sources == {"gmail", "telegram", "whatsapp", "sms"}


def test_pending_messages_are_selected(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "p1", content="Hello first message")
    _add_pending_message(inbox, "p2", content="Hello second message")

    pending = inbox.get_pending_messages(limit=10)
    assert len(pending) == 2



def test_completed_messages_are_never_reprocessed(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "done-1")

    # Manually mark as COMPLETED
    inbox.update_analysis_success(1, {"category": "WORK"}, "HIGH")
    assert inbox.count_pending_messages() == 0

    mock_llm = MagicMock()
    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    assert res["processed"] == 0
    mock_llm.model.invoke.assert_not_called()


def test_duplicate_messages_are_not_analyzed_twice(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "dup-1", content="Duplicate message text")
    _add_pending_message(inbox, "dup-1", content="Duplicate message text")

    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) as cnt FROM priority_inbox").fetchone()["cnt"]
    assert count == 1


# ==============================================================================
# 3. Locking, Concurrency & Crash Recovery Tests
# ==============================================================================

def test_processing_lock_prevents_duplicate_processing():
    inbox = PriorityInbox()
    _add_pending_message(inbox, "lock-1")

    locked1 = inbox.mark_messages_processing([1])
    assert locked1 == [1]

    # Second worker attempts to lock same message
    locked2 = inbox.mark_messages_processing([1])
    assert locked2 == []


def test_stale_processing_messages_recover_to_pending(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "stale-1")

    # Mark PROCESSING with timestamp 20 minutes ago
    past_time = time.time() - 1200
    inbox.mark_messages_processing([1], started_at=past_time)

    analyzer = PriorityAnalyzer(priority_inbox=inbox, stale_timeout_seconds=900)
    recovered = analyzer.recover_stale_processing()

    assert recovered == 1
    assert inbox.count_pending_messages() == 1


# ==============================================================================
# 4. LLM Resilience, Bounded Retries & Deferral Tests
# ==============================================================================

def test_network_failure_retries_once_and_defers(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "net-1")

    mock_llm = MagicMock()
    mock_llm.model.invoke.side_effect = LLMNetworkError("DNS failure")

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm, max_retries=1)

    with patch("time.sleep") as mock_sleep:
        res = analyzer.process_cycle()

    # Attempted once + 1 retry = 2 calls
    assert mock_llm.model.invoke.call_count == 2
    assert res["deferred"] == 1
    # Message deferred back to PENDING for next cycle
    assert inbox.count_pending_messages() == 1


def test_rate_limit_with_short_retry_delay_retries(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "rate-1")

    mock_llm = MagicMock()
    success_resp = MagicMock()
    success_resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "WORK",
            "intent": "TASK",
            "importance": "LOW",
            "urgency": "LOW",
            "requires_action": False,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": None,
            "reason": "OK",
        }]
    })
    # First call rate limited (2.0s), second succeeds
    mock_llm.model.invoke.side_effect = [
        LLMRateLimitError("Rate limit", retry_after=2.0),
        success_resp,
    ]

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm, max_retries=1)

    with patch("time.sleep") as mock_sleep:
        res = analyzer.process_cycle()
        mock_sleep.assert_called_with(2.0)

    assert res["completed"] == 1


def test_long_rate_limit_delay_defers_to_next_cycle(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "rate-long")

    mock_llm = MagicMock()
    mock_llm.model.invoke.side_effect = LLMRateLimitError("Rate limit", retry_after=120.0)

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)

    with patch("time.sleep") as mock_sleep:
        res = analyzer.process_cycle()
        mock_sleep.assert_not_called()  # Must not sleep for 120s!

    assert res["deferred"] == 1
    assert inbox.count_pending_messages() == 1


def test_authentication_failure_does_not_repeatedly_retry(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "auth-1")

    mock_llm = MagicMock()
    mock_llm.model.invoke.side_effect = LLMAuthenticationError("Bad API key")

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm, max_retries=3)

    res = analyzer.process_cycle()

    # Auth error should fail without retrying 3 times
    assert mock_llm.model.invoke.call_count == 1
    assert res["deferred"] == 1


def test_malformed_structured_output_is_rejected(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "bad-json")

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = "This is definitely not JSON!"
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)

    res = analyzer.process_cycle()
    assert res["deferred"] == 1
    assert inbox.count_pending_messages() == 1


def test_valid_messages_from_partially_successful_batch_are_persisted(mock_user_profile):
    """If msg 1 and 2 validate but msg 3 has invalid schema:

    msg 1 & 2 -> COMPLETED
    msg 3 -> deferred back to PENDING for retry.
    """
    inbox = PriorityInbox()
    _add_pending_message(inbox, "part-1", content="Valid 1")
    _add_pending_message(inbox, "part-2", content="Valid 2")
    _add_pending_message(inbox, "part-3", content="Bad schema")

    mock_llm = MagicMock()
    resp = MagicMock()
    # Message 3 has invalid importance value ("SUPER_HIGH" is not in HIGH/MEDIUM/LOW)
    resp.content = json.dumps({
        "items": [
            {
                "message_id": "1",
                "category": "WORK",
                "intent": "TASK",
                "importance": "HIGH",
                "urgency": "HIGH",
                "requires_action": True,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": None,
                "reason": "OK 1",
            },
            {
                "message_id": "2",
                "category": "PERSONAL",
                "intent": "CHAT",
                "importance": "LOW",
                "urgency": "LOW",
                "requires_action": False,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": None,
                "reason": "OK 2",
            },
            {
                "message_id": "3",
                "category": "WORK",
                "intent": "TASK",
                "importance": "SUPER_HIGH",  # INVALID SCHEMA!
                "urgency": "LOW",
                "requires_action": False,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": None,
                "reason": "Invalid",
            },
        ]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    assert res["completed"] == 2
    assert res["deferred"] == 1

    # Check status in DB
    conn = get_connection()
    row1 = conn.execute("SELECT analysis_status FROM priority_inbox WHERE id = 1").fetchone()
    row2 = conn.execute("SELECT analysis_status FROM priority_inbox WHERE id = 2").fetchone()
    row3 = conn.execute("SELECT analysis_status FROM priority_inbox WHERE id = 3").fetchone()

    assert row1["analysis_status"] == "COMPLETED"
    assert row2["analysis_status"] == "COMPLETED"
    assert row3["analysis_status"] == "PENDING"


def test_failed_messages_remain_retryable(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "retry-1")

    inbox.mark_messages_processing([1])
    inbox.defer_processing_messages([1], error="timeout")

    assert inbox.count_pending_messages() == 1


def test_next_analyzer_cycle_retries_deferred_messages(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "cycle-retry-1")

    mock_llm = MagicMock()
    # First cycle fails with network error
    mock_llm.model.invoke.side_effect = LLMNetworkError("Network down")
    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm, max_retries=0)

    res1 = analyzer.process_cycle()
    assert res1["deferred"] == 1
    assert inbox.count_pending_messages() == 1

    # Next cycle recovers and succeeds
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "WORK",
            "intent": "TASK",
            "importance": "MEDIUM",
            "urgency": "LOW",
            "requires_action": False,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": None,
            "reason": "Success on second cycle",
        }]
    })
    mock_llm.model.invoke.side_effect = None
    mock_llm.model.invoke.return_value = resp

    res2 = analyzer.process_cycle()
    assert res2["completed"] == 1
    assert inbox.count_pending_messages() == 0


def test_llm_analyzer_failure_does_not_break_normal_chat(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "chat-isolation-1")

    mock_llm = MagicMock()
    mock_llm.model.invoke.side_effect = Exception("Catastrophic analyzer failure")

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    # Must not raise exception, but return deferred
    assert res["deferred"] == 1


# ==============================================================================
# 5. Rules, Preferences, Precedence & Privacy Tests
# ==============================================================================

def test_user_md_preference_affects_final_priority(tmp_path):
    mem_dir = str(tmp_path / "memory")
    profile = UserProfile(mem_dir)
    # Important contacts: "manager", "mom", "dad"
    inbox = PriorityInbox()
    _add_pending_message(inbox, "pref-1", sender="Mom", content="Call me when free")

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "PERSONAL",
            "intent": "CHAT",
            "importance": "LOW",  # LLM classified as LOW
            "urgency": "LOW",
            "requires_action": False,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": None,
            "reason": "Casual family chat",
        }]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    assert res["completed"] == 1
    items = inbox.today()
    # Because Mom is in important_contacts, final level is HIGH!
    assert items[0]["level"] == "HIGH"
    # System importance still preserved as LOW for auditing!
    assert items[0]["system_importance"] == "LOW"


def test_persistent_user_rule_overrides_system_classification(mock_user_profile):
    rules_mgr = PriorityRulesManager()
    rules_mgr.add_rule(rule_type="SENDER", pattern="Rahul", target_level="HIGH", reason="Always prioritize Rahul")

    inbox = PriorityInbox()
    _add_pending_message(inbox, "rule-1", sender="Rahul Sharma", content="Check this meme")

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "SOCIAL",
            "intent": "CHAT",
            "importance": "LOW",
            "urgency": "LOW",
            "requires_action": False,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": None,
            "reason": "Casual meme",
        }]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, rules_manager=rules_mgr, user_profile=mock_user_profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    assert res["completed"] == 1
    items = inbox.today()
    assert items[0]["level"] == "HIGH"
    assert items[0]["system_importance"] == "LOW"


def test_deterministic_conflict_handling_multiple_rules(mock_user_profile):
    """When multiple rules match, deterministic precedence:

    SENDER > DOMAIN > KEYWORD > CATEGORY.
    """
    rules_mgr = PriorityRulesManager()
    # Add rules in arbitrary order: CATEGORY rule says LOW, SENDER rule says HIGH
    rules_mgr.add_rule(rule_type="CATEGORY", pattern="WORK", target_level="LOW", reason="Work category low")
    rules_mgr.add_rule(rule_type="KEYWORD", pattern="interview", target_level="MEDIUM", reason="Interview keyword")
    rules_mgr.add_rule(rule_type="SENDER", pattern="recruiter@tech.com", target_level="HIGH", reason="Recruiter sender")

    matched = rules_mgr.match_rule(
        sender="recruiter@tech.com",
        content="Technical interview confirmation",
        category="WORK",
    )

    assert matched is not None
    assert matched["rule_type"] == "SENDER"
    assert matched["target_level"] == "HIGH"


def test_one_time_message_override_does_not_become_permanent_rule(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "msg-override-1", sender="Alex", content="First message")
    _add_pending_message(inbox, "msg-override-2", sender="Alex", content="Second message")

    # User manually overrides message 1 to HIGH
    inbox.set_message_override(1, level="HIGH", reason="User manual override")

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [
            {
                "message_id": "1",
                "category": "PERSONAL",
                "intent": "CHAT",
                "importance": "LOW",
                "urgency": "LOW",
                "requires_action": False,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": None,
                "reason": "Chat",
            },
            {
                "message_id": "2",
                "category": "PERSONAL",
                "intent": "CHAT",
                "importance": "LOW",
                "urgency": "LOW",
                "requires_action": False,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": None,
                "reason": "Chat",
            },
        ]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    analyzer.process_cycle()

    conn = get_connection()
    row1 = conn.execute("SELECT final_priority FROM priority_inbox WHERE id = 1").fetchone()
    row2 = conn.execute("SELECT final_priority FROM priority_inbox WHERE id = 2").fetchone()

    assert row1["final_priority"] == "HIGH"
    assert row2["final_priority"] == "LOW"  # Alex did NOT permanently become HIGH!


def test_spam_scam_risk_stored_separately_from_priority(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "scam-1", sender="SuspiciousBank", content="Your account is locked. Click here to unlock.")

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "TRANSACTIONAL",
            "intent": "NOTIFICATION",
            "importance": "HIGH",
            "urgency": "HIGH",
            "requires_action": False,
            "spam": False,
            "scam": True,
            "risk_score": 0.95,
            "deadline": None,
            "reason": "Suspicious phishing attempt",
        }]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    assert res["completed"] == 1
    items = inbox.today()
    assert items[0]["level"] == "HIGH"  # Elevated to HIGH so user is alerted!
    assert items[0]["is_scam"] == 1      # Risk clearly preserved separately!
    assert items[0]["risk_score"] == 0.95


def test_cross_platform_messages_use_unified_message_correctly():
    inbox = PriorityInbox()
    for src in (Source.GMAIL, Source.TELEGRAM, Source.WHATSAPP, Source.SMS):
        _add_pending_message(inbox, f"cross-{src.value}", source=src, sender="Tester", content=f"From {src.value}")

    items = inbox.get_pending_messages(limit=10)
    sources = {item["source"] for item in items}
    assert sources == {"gmail", "telegram", "whatsapp", "sms"}


def test_deadline_extraction_does_not_invent_times():
    """MessageAnalysisItem validation ensures null deadline remains null and does not fabricate hours/minutes."""
    item = MessageAnalysisItem.model_validate({
        "message_id": "1",
        "category": "WORK",
        "intent": "MEETING",
        "importance": "MEDIUM",
        "urgency": "MEDIUM",
        "requires_action": True,
        "spam": False,
        "scam": False,
        "risk_score": 0.0,
        "deadline": None,
        "reason": "Meeting tomorrow",
    })
    assert item.deadline is None


def test_no_sensitive_message_content_appears_in_analyzer_logs(caplog, mock_user_profile):
    caplog.set_level(logging.INFO)
    inbox = PriorityInbox()
    sensitive_content = "SECRET_PIN_987654_CONFIDENTIAL"
    sensitive_phone = "+919999888877"
    _add_pending_message(inbox, "sens-1", sender=sensitive_phone, content=sensitive_content)

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "WORK",
            "intent": "TASK",
            "importance": "LOW",
            "urgency": "LOW",
            "requires_action": False,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": None,
            "reason": "Standard task",
        }]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    analyzer.process_cycle()

    all_logs = " ".join(record.getMessage() for record in caplog.records)
    assert sensitive_content not in all_logs
    assert sensitive_phone not in all_logs


# ==============================================================================
# 6. Priority LLM Separation & Configuration Tests
# ==============================================================================

def test_priority_llm_separate_key_and_model(monkeypatch):
    from assistant.llm.llm_client import LLMClient, get_priority_llm_client

    monkeypatch.setenv("GROQ_API_KEY", "chat-key-123")
    monkeypatch.setenv("GROQ_MODEL", "chat-model-abc")
    monkeypatch.setenv("PRIORITY_GROQ_API_KEY", "priority-key-xyz")
    monkeypatch.setenv("PRIORITY_GROQ_MODEL", "priority-model-custom")

    chat_llm = LLMClient()
    priority_llm = get_priority_llm_client()

    assert priority_llm is not None
    assert priority_llm.api_key == "priority-key-xyz"
    assert priority_llm.model == "priority-model-custom"

    # Chat LLM is strictly separate
    assert chat_llm.api_key == "chat-key-123"
    assert chat_llm.model == "chat-model-abc"


def test_priority_llm_missing_key_does_not_fall_back_to_chat_key(monkeypatch):
    from assistant.llm.llm_client import get_priority_llm_client

    monkeypatch.setenv("GROQ_API_KEY", "chat-key-123")
    monkeypatch.delenv("PRIORITY_GROQ_API_KEY", raising=False)

    priority_llm = get_priority_llm_client()
    # Must NOT silently fall back to chat key!
    assert priority_llm is None


def test_analyzer_defers_cleanly_when_priority_llm_missing(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(inbox, "no-key-1", content="Meeting tomorrow")

    # PriorityAnalyzer initialized with llm_client=None (when key is missing)
    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=None)
    res = analyzer.process_cycle()

    assert res["deferred"] == 1
    # Message remained retryable
    assert inbox.count_pending_messages() == 1


# ==============================================================================
# 7. Temporal Intelligence & Agenda Planning Tests
# ==============================================================================

def test_temporal_intelligence_tomorrow_meeting(mock_user_profile):
    inbox = PriorityInbox()
    _add_pending_message(
        inbox,
        "wa-meet-1",
        source=Source.WHATSAPP,
        sender="44199676252415@lid",
        content="Hi can we meet tomorrow at 9am morning at rana cafe meeting with the boss",
    )

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({
        "items": [{
            "message_id": "1",
            "category": "WORK",
            "intent": "MEETING",
            "importance": "HIGH",
            "urgency": "HIGH",
            "requires_action": True,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": "2026-09-15 09:00",
            "reason": "Confirmed morning meeting at Rana Cafe",
        }]
    })
    mock_llm.model.invoke.return_value = resp

    analyzer = PriorityAnalyzer(priority_inbox=inbox, user_profile=mock_user_profile, llm_client=mock_llm)
    res = analyzer.process_cycle()

    assert res["completed"] == 1
    items = inbox.today()
    assert len(items) == 1
    assert items[0]["level"] == "HIGH"
    assert items[0]["system_intent"] == "MEETING"
    assert items[0]["requires_action"] == 1
    assert items[0]["deadline"] == "2026-09-15 09:00"


def test_get_agenda_tomorrow_retrieves_tomorrow_commitments(mock_user_profile):
    inbox = PriorityInbox()
    # Message 1: Tomorrow meeting with deadline
    _add_pending_message(inbox, "m-tmrw", sender="Boss", content="Meeting tomorrow at 9am at Cafe")
    inbox.update_analysis_success(
        id=1,
        analysis={
            "category": "WORK",
            "intent": "MEETING",
            "importance": "HIGH",
            "urgency": "HIGH",
            "requires_action": True,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": "2026-09-15 09:00",
            "reason": "Boss meeting",
        },
        final_level="HIGH",
    )

    # Message 2: Today message without tomorrow deadline
    _add_pending_message(inbox, "m-today", sender="Colleague", content="Check this document today")
    inbox.update_analysis_success(
        id=2,
        analysis={
            "category": "WORK",
            "intent": "TASK",
            "importance": "LOW",
            "urgency": "LOW",
            "requires_action": False,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": "2026-09-14 18:00",
            "reason": "Doc check",
        },
        final_level="LOW",
    )

    agenda_tmrw = inbox.get_agenda(target_date="tomorrow")
    # Must retrieve the tomorrow meeting
    assert len(agenda_tmrw) >= 1
    assert any("Meeting tomorrow at 9am" in a["content"] for a in agenda_tmrw)
    # Today's document check should not pollute tomorrow's agenda
    assert not any("Check this document today" in a["content"] for a in agenda_tmrw)


def test_render_agenda_formatting_and_commitments_first():
    from assistant.agent_core.graph.tools import render_agenda

    inbox = PriorityInbox()
    _add_pending_message(inbox, "meet-1", source=Source.WHATSAPP, sender="John", content="Coffee at 10am tomorrow")
    inbox.update_analysis_success(
        id=1,
        analysis={
            "category": "WORK",
            "intent": "MEETING",
            "importance": "HIGH",
            "urgency": "HIGH",
            "requires_action": True,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": "2026-09-15 10:00",
            "reason": "Coffee meeting",
        },
        final_level="HIGH",
    )

    output = render_agenda(inbox, target_date="tomorrow")

    assert "SCHEDULE / AGENDA FOR TOMORROW:" in output
    assert "whatsapp" in output.lower()
    assert "John" in output
    assert "John" in output
    assert "Coffee at 10am tomorrow" in output
    assert "Deadline: 2026-09-15 10:00" in output
    assert "Intent: MEETING" in output
    assert "Action Required" in output


def test_query_agenda_tool_integration():
    from assistant.agent_core.graph.tools import build_tools

    inbox = PriorityInbox()
    _add_pending_message(inbox, "tool-meet", source=Source.WHATSAPP, sender="Team", content="Standup tomorrow 9am")
    inbox.update_analysis_success(
        id=1,
        analysis={
            "category": "WORK",
            "intent": "MEETING",
            "importance": "HIGH",
            "urgency": "HIGH",
            "requires_action": True,
            "spam": False,
            "scam": False,
            "risk_score": 0.0,
            "deadline": "2026-09-15 09:00",
            "reason": "Standup",
        },
        final_level="HIGH",
    )

    mock_router = MagicMock()
    mock_planner = MagicMock()
    mock_approval = MagicMock()
    mock_outbound = MagicMock()

    tools = build_tools(
        task_planner=mock_planner,
        approval_manager=mock_approval,
        priority_inbox=inbox,
        tool_router=mock_router,
    )

    agenda_tool = next((t for t in tools if t.name == "query_agenda"), None)
    assert agenda_tool is not None

    result = agenda_tool.invoke({"target_date": "tomorrow", "query": "what is scheduled tomorrow?"})
    assert "Standup tomorrow 9am" in result
    assert "Deadline: 2026-09-15 09:00" in result

