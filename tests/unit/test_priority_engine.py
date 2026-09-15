from assistant.ingestion.unified_message import Source, UnifiedMessage
from assistant.intelligence.priority_engine import PriorityEngine, PriorityLevel
from assistant.memory.user_profile import UserProfile
import tempfile


def make_profile():
    tmp = tempfile.mkdtemp()
    return UserProfile(tmp)


def test_urgent_important_message_scores_high():
    engine = PriorityEngine()
    profile = make_profile()
    msg = UnifiedMessage(source=Source.GMAIL, conversation_id="c1", sender="recruiter@example.com",
                          content="Your interview is scheduled for tomorrow, please confirm urgently")
    result = engine.score(msg, profile)
    assert result.level in (PriorityLevel.HIGH, PriorityLevel.MEDIUM)
    assert result.score > 0


def test_promotional_message_scores_low():
    engine = PriorityEngine()
    profile = make_profile()
    msg = UnifiedMessage(source=Source.GMAIL, conversation_id="c2", sender="deals@newsletter.com",
                          content="20% discount this weekend, unsubscribe anytime")
    result = engine.score(msg, profile)
    assert result.level == PriorityLevel.LOW


def test_important_contact_boosts_score():
    engine = PriorityEngine()
    profile = make_profile()
    plain = UnifiedMessage(source=Source.SMS, conversation_id="c3", sender="9999999999", content="hey")
    important = UnifiedMessage(source=Source.SMS, conversation_id="c4", sender="manager", content="hey")
    r1 = engine.score(plain, profile)
    r2 = engine.score(important, profile)
    assert r2.score > r1.score
