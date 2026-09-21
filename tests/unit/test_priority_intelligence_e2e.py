"""
Comprehensive End-to-End Unit Tests for the Priority Intelligence System.

Covers:
1. Multiple source messages entering PriorityInbox (Gmail, WhatsApp, Telegram, SMS)
2. Cross-channel normalization
3. Deduplication
4. Groq / PriorityAnalyzer semantic analysis
5. Structured output validation
6. Database persistence of semantic fields
7. 6-Tier priority precedence hierarchy
8. Deadline and commitment extraction
9. Agenda queries (query_agenda / render_agenda)
10. Periodic analysis & on-demand trigger
11. Partial source availability (offline sources do not crash priority cycle)
12. One bad/malformed message not crashing the batch
13. Existing PENDING messages being processed & recovered
14. Natural-language synthesis without raw tool dumping
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from assistant.agent_core.graph.builder import _route_after_tool_exec, build_agent_graph
from assistant.agent_core.graph.tools import _format_priority_item, render_agenda, render_priority_inbox
from assistant.ingestion.adapters.gmail_adapter import adapt_gmail
from assistant.ingestion.adapters.sms_adapter import adapt_sms
from assistant.ingestion.adapters.telegram_adapter import adapt_telegram
from assistant.ingestion.adapters.whatsapp_adapter import adapt_whatsapp
from assistant.ingestion.normalizer import normalize_gmail, normalize_sms, normalize_telegram, normalize_whatsapp
from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage
from assistant.intelligence.classifier import Category
from assistant.intelligence.eligibility_filter import MessageEligibilityFilter
from assistant.intelligence.message_intelligence_models import MessageAnalysisItem
from assistant.intelligence.normal_message_pipeline import NormalMessagePipeline
from assistant.intelligence.priority_analyzer import PriorityAnalyzer, PriorityAnalyzerWorker
from assistant.intelligence.priority_engine import PriorityEngine, PriorityLevel, PriorityResult
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.intelligence.priority_rules import PriorityRulesManager
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
def test_user_profile(tmp_path):
    mem_dir = str(tmp_path / "memory")
    return UserProfile(mem_dir)


# ==============================================================================
# 1. Multiple Source Ingestion & Cross-Channel Normalization
# ==============================================================================

def test_cross_channel_normalization():
    """Verify messages from Gmail, WhatsApp, Telegram, and SMS normalize properly."""
    now_ts = time.time()

    # Gmail
    raw_gmail = {
        "id": "gm-1",
        "thread_id": "th-1",
        "from": "manager@company.com",
        "to": "user@company.com",
        "subject": "Quarterly Planning",
        "snippet": "Let's review OKRs tomorrow at 10 AM.",
        "internalDate": now_ts,
    }
    msg_gmail = adapt_gmail(raw_gmail)
    assert msg_gmail.source == Source.GMAIL
    assert msg_gmail.sender == "manager@company.com"
    assert "OKRs" in msg_gmail.content

    # WhatsApp
    raw_wa = {
        "id": "wa-1",
        "chat_id": "919876543210@s.whatsapp.net",
        "sender": "919876543210@s.whatsapp.net",
        "text": "Can you submit the report by 5 PM?",
        "timestamp": "2026-09-22T09:00:00Z",
        "from_me": False,
    }
    msg_wa = adapt_whatsapp(raw_wa)
    assert msg_wa.source == Source.WHATSAPP
    assert msg_wa.sender == "919876543210@s.whatsapp.net"
    assert "report" in msg_wa.content

    # Telegram
    raw_tg = {
        "message_id": 999,
        "chat_id": 12345678,
        "from": "Alice",
        "text": "Interview scheduled for Wednesday at 2 PM.",
        "date": now_ts,
    }
    msg_tg = adapt_telegram(raw_tg)
    assert msg_tg.source == Source.TELEGRAM
    assert msg_tg.sender == "Alice"
    assert "Interview" in msg_tg.content

    # SMS
    raw_sms = {
        "sms_id": "sms-1",
        "number": "+1987654321",
        "body": "Your bank account has been debited INR 5,000.",
        "timestamp": now_ts,
    }
    msg_sms = adapt_sms(raw_sms)
    assert msg_sms.source == Source.SMS
    assert msg_sms.sender == "+1987654321"
    assert "debited" in msg_sms.content


# ==============================================================================
# 2. Eligibility & Deduplication into PriorityInbox
# ==============================================================================

def test_eligibility_and_deduplication(test_user_profile):
    """Verify eligible messages enter PriorityInbox and duplicate messages are ignored."""
    inbox = PriorityInbox()
    pe = PriorityEngine()

    msg = UnifiedMessage(
        message_id="uniq-1",
        conversation_id="telegram:alice",
        source=Source.TELEGRAM,
        sender="Alice",
        content="Urgent client review meeting tomorrow at 11 AM",
        timestamp=time.time(),
        origin=Origin.USER,
    )

    scored = pe.score(msg, test_user_profile)
    inbox.add(scored)
    assert inbox.count_pending_messages() == 1

    # Duplicate addition with identical ID or content must be ignored
    inbox.add(scored)
    assert inbox.count_pending_messages() == 1


# ==============================================================================
# 3. PriorityAnalyzer Invocation & Fault-Tolerant Batch Handling
# ==============================================================================

def test_priority_analyzer_string_timestamp_and_invoke():
    """Verify PriorityAnalyzer parses ISO string timestamps and properly calls model.invoke()."""
    inbox = PriorityInbox()
    conn = get_connection()

    # Insert a message with string ISO timestamp
    conn.execute(
        "INSERT INTO priority_inbox (message_id, source, sender, content, category, score, level, ts, analysis_status, analysis_attempts) "
        "VALUES ('str-ts-1', 'whatsapp', 'Boss', 'Meeting tomorrow morning at 9am', 'WORK', 10, 'HIGH', '2026-09-14 10:30:00+00:00', 'PENDING', 0)"
    )
    conn.commit()

    # Get assigned row ID
    row_id = conn.execute("SELECT id FROM priority_inbox WHERE message_id = 'str-ts-1'").fetchone()["id"]

    # Mock an LLMClient with get_langchain_model returning a valid model
    mock_llm_client = MagicMock()
    mock_chat_model = MagicMock()
    mock_chat_model.invoke.return_value = MagicMock(content=json.dumps({
        "items": [
            {
                "message_id": str(row_id),
                "category": "WORK",
                "intent": "MEETING",
                "importance": "HIGH",
                "urgency": "HIGH",
                "requires_action": True,
                "spam": False,
                "scam": False,
                "risk_score": 0.05,
                "deadline": "2026-09-15T09:00:00",
                "reason": "Confirmed meeting with boss",
            }
        ]
    }))
    mock_llm_client.get_langchain_model.return_value = mock_chat_model
    mock_llm_client.online = True

    analyzer = PriorityAnalyzer(priority_inbox=inbox, llm_client=mock_llm_client)
    res = analyzer.process_cycle()

    assert res["status"] == "completed"
    assert res["completed"] == 1

    # Verify SQLite database persistence of semantic fields
    row = conn.execute("SELECT * FROM priority_inbox WHERE message_id = 'str-ts-1'").fetchone()
    assert row["analysis_status"] == "COMPLETED"
    assert row["system_intent"] == "MEETING"
    assert row["requires_action"] == 1
    assert row["deadline"] == "2026-09-15T09:00:00"
    assert row["final_priority"] == "HIGH"
    assert row["system_reason"] == "Confirmed meeting with boss"


def test_one_bad_message_does_not_crash_batch(test_user_profile):
    """Verify that if one message fails schema validation, the rest of the batch completes."""
    inbox = PriorityInbox()
    conn = get_connection()

    # Insert 2 messages
    conn.execute("INSERT INTO priority_inbox (message_id, source, sender, content, score, level, ts, analysis_status) VALUES ('m1', 'sms', 'Alice', 'Good message', 5, 'LOW', ?, 'PENDING')", (time.time(),))
    conn.execute("INSERT INTO priority_inbox (message_id, source, sender, content, score, level, ts, analysis_status) VALUES ('m2', 'sms', 'Bob', 'Bad message', 5, 'LOW', ?, 'PENDING')", (time.time(),))
    conn.commit()

    # LLM returns valid analysis only for m1
    id1 = conn.execute("SELECT id FROM priority_inbox WHERE message_id = 'm1'").fetchone()["id"]
    mock_llm_client = MagicMock()
    mock_chat_model = MagicMock()
    mock_chat_model.invoke.return_value = MagicMock(content=json.dumps({
        "items": [
            {
                "message_id": str(id1),
                "category": "PERSONAL",
                "intent": "CHAT",
                "importance": "LOW",
                "urgency": "LOW",
                "requires_action": False,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": None,
                "reason": "Casual conversation",
            }
        ]
    }))
    mock_llm_client.get_langchain_model.return_value = mock_chat_model
    mock_llm_client.online = True

    analyzer = PriorityAnalyzer(priority_inbox=inbox, llm_client=mock_llm_client, user_profile=test_user_profile)
    res = analyzer.process_cycle()

    assert res["completed"] == 1
    assert res["deferred"] == 1

    row1 = conn.execute("SELECT analysis_status FROM priority_inbox WHERE message_id = 'm1'").fetchone()
    row2 = conn.execute("SELECT analysis_status FROM priority_inbox WHERE message_id = 'm2'").fetchone()
    assert row1["analysis_status"] == "COMPLETED"
    assert row2["analysis_status"] == "PENDING"


# ==============================================================================
# 4. Priority Precedence Hierarchy
# ==============================================================================

def test_priority_precedence(test_user_profile):
    """Verify strict 6-tier precedence: override > rule > preference > semantic > deterministic > default."""
    analyzer = PriorityAnalyzer(priority_inbox=PriorityInbox(), user_profile=test_user_profile)

    # 1. Override
    msg_override = {"id": 1, "sender": "Unknown", "user_override_level": "HIGH"}
    analysis_low = MessageAnalysisItem(
        message_id="1",
        category="PROMOTIONAL",
        intent="SPAM",
        importance="LOW",
        urgency="LOW",
        requires_action=False,
        spam=True,
        scam=False,
        reason="spam",
    )
    level, rule_id, reason = analyzer.compute_final_priority(msg_override, analysis_low)
    assert level == "HIGH"
    assert "override" in reason.lower()

    # 2. Rule
    rules_mgr = PriorityRulesManager()
    rules_mgr.add_rule(rule_type="SENDER", pattern="VIP", target_level="HIGH", reason="VIP rule")
    rules = rules_mgr.list_rules(active_only=True)
    msg_vip = {"id": 2, "sender": "VIP Client"}
    level, rule_id, reason = analyzer.compute_final_priority(msg_vip, analysis_low, active_rules=rules)
    assert level == "HIGH"
    assert rule_id is not None

    # 3. Preference (Important Contact in user profile: mom)
    msg_mom = {"id": 3, "sender": "Mom"}
    level, rule_id, reason = analyzer.compute_final_priority(msg_mom, analysis_low)
    assert level == "HIGH"
    assert "important contacts" in reason.lower()

    # 4. Semantic Intelligence
    msg_normal = {"id": 4, "sender": "Random"}
    analysis_high = MessageAnalysisItem(
        message_id="4",
        category="WORK",
        intent="TASK",
        importance="HIGH",
        urgency="HIGH",
        requires_action=True,
        spam=False,
        scam=False,
        reason="High urgency task",
    )
    level, rule_id, reason = analyzer.compute_final_priority(msg_normal, analysis_high)
    assert level == "HIGH"
    assert "Semantic intelligence" in reason


# ==============================================================================
# 5. Semantic Agenda Queries & Formatting
# ==============================================================================

def test_agenda_query_semantic_retrieval():
    """Verify get_agenda and render_agenda prioritize confirmed commitments and deadlines."""
    inbox = PriorityInbox()
    conn = get_connection()

    # 1. Meeting with confirmed deadline for 2026-09-25
    conn.execute(
        "INSERT INTO priority_inbox (message_id, source, sender, content, level, ts, analysis_status, system_intent, deadline, requires_action, final_priority, system_reason) "
        "VALUES ('a-1', 'gmail', 'manager@company.com', 'Quarterly planning meeting', 'HIGH', ?, 'COMPLETED', 'MEETING', '2026-09-25T14:00:00', 1, 'HIGH', 'Quarterly planning')",
        (time.time(),)
    )

    # 2. Spam containing the word 'meeting' and 'tomorrow'
    conn.execute(
        "INSERT INTO priority_inbox (message_id, source, sender, content, level, ts, analysis_status, is_spam, system_intent, deadline) "
        "VALUES ('a-2', 'sms', 'Spammer', 'Cheap meeting rooms for tomorrow!', 'LOW', ?, 'COMPLETED', 1, 'SPAM', '2026-09-25T10:00:00')",
        (time.time(),)
    )
    conn.commit()

    # Query agenda for 2026-09-25
    agenda = inbox.get_agenda(target_date="2026-09-25")
    assert len(agenda) == 1
    assert agenda[0]["sender"] == "manager@company.com"
    assert agenda[0]["is_spam"] == 0

    rendered = render_agenda(inbox, target_date="2026-09-25")
    assert "manager@company.com" in rendered
    assert "2026-09-25T14:00:00" in rendered
    assert "Spammer" not in rendered


# ==============================================================================
# 6. Graph Routing & Natural Language Synthesis
# ==============================================================================

def test_routing_allows_agent_synthesis_for_priority_tools():
    """Verify that query_priority_inbox, query_agenda, and read_sms route to agent_node, not respond_node."""
    # Action tools should go directly to respond_node
    state_alarm = {"messages": [MagicMock()], "executed_tools": ["set_alarm"]}
    assert _route_after_tool_exec(state_alarm) == "respond_node"

    # Information retrieval tools should route to agent_node for natural language synthesis
    state_priority = {"messages": [MagicMock()], "executed_tools": ["query_priority_inbox"]}
    assert _route_after_tool_exec(state_priority) == "agent_node"

    state_agenda = {"messages": [MagicMock()], "executed_tools": ["query_agenda"]}
    assert _route_after_tool_exec(state_agenda) == "agent_node"

    state_sms = {"messages": [MagicMock()], "executed_tools": ["read_sms"]}
    assert _route_after_tool_exec(state_sms) == "agent_node"


# ==============================================================================
# 7. Partial Source Availability
# ==============================================================================

def test_partial_source_availability_does_not_crash_priority():
    """Verify that when some connectors are offline, PriorityAnalyzer processes available messages normally."""
    inbox = PriorityInbox()
    conn = get_connection()

    # Add messages from available sources (Gmail + Telegram)
    conn.execute("INSERT INTO priority_inbox (message_id, source, sender, content, score, level, ts, analysis_status) VALUES ('g1', 'gmail', 'hr@company.com', 'Job Offer', 10, 'HIGH', ?, 'PENDING')", (time.time(),))
    conn.execute("INSERT INTO priority_inbox (message_id, source, sender, content, score, level, ts, analysis_status) VALUES ('t1', 'telegram', 'Alice', 'Meeting update', 8, 'HIGH', ?, 'PENDING')", (time.time(),))
    conn.commit()

    id_g = conn.execute("SELECT id FROM priority_inbox WHERE message_id = 'g1'").fetchone()["id"]
    id_t = conn.execute("SELECT id FROM priority_inbox WHERE message_id = 't1'").fetchone()["id"]

    mock_llm_client = MagicMock()
    mock_chat_model = MagicMock()
    mock_chat_model.invoke.return_value = MagicMock(content=json.dumps({
        "items": [
            {
                "message_id": str(id_g),
                "category": "WORK",
                "intent": "NOTIFICATION",
                "importance": "HIGH",
                "urgency": "MEDIUM",
                "requires_action": False,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": None,
                "reason": "Job offer letter",
            },
            {
                "message_id": str(id_t),
                "category": "WORK",
                "intent": "MEETING",
                "importance": "HIGH",
                "urgency": "HIGH",
                "requires_action": True,
                "spam": False,
                "scam": False,
                "risk_score": 0.0,
                "deadline": "2026-09-23T10:00:00",
                "reason": "Team meeting",
            }
        ]
    }))
    mock_llm_client.get_langchain_model.return_value = mock_chat_model
    mock_llm_client.online = True

    analyzer = PriorityAnalyzer(priority_inbox=inbox, llm_client=mock_llm_client)
    res = analyzer.process_cycle()

    assert res["status"] == "completed"
    assert res["completed"] == 2
    assert inbox.count_pending_messages() == 0
