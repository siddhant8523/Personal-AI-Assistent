"""
Memory Service
=================
Facade the Agent Core talks to instead of touching each memory store
directly (Part 2, Section 21). Decides what's relevant and hands the
Context Builder a compact bundle rather than the entire memory repository.

Skills (``memory/skills/*.md``) are initialized on startup via
``initialize_default_skills()`` but are **not** injected into every LLM
call — use ``load_skill()`` on demand when a request needs domain guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from assistant.memory.daily_memory import DailyMemory
from assistant.memory.long_term_memory import LongTermMemory
from assistant.memory.semantic_memory import SemanticMemory
from assistant.memory.session_memory import SessionMemory
from assistant.memory.skill_initializer import initialize_default_skills
from assistant.memory.skills import Skills
from assistant.memory.soul import Soul
from assistant.memory.identity import IdentityManager
from assistant.memory.user_profile import UserProfile


@dataclass
class MemoryBundle:
    soul: str
    user: str
    session: dict = field(default_factory=dict)
    daily: list[str] = field(default_factory=list)
    long_term: dict = field(default_factory=dict)
    semantic: list[str] = field(default_factory=list)


class MemoryService:
    def __init__(self, memory_dir: str, session_ttl_minutes: int = 60):
        initialize_default_skills(memory_dir)
        self.soul = Soul(memory_dir)
        self.identity = IdentityManager(self.soul)
        self.user_profile = UserProfile(memory_dir)
        self.skills = Skills(memory_dir)
        self.session = SessionMemory(ttl_minutes=session_ttl_minutes)
        self.daily = DailyMemory()
        self.long_term = LongTermMemory()
        self.semantic = SemanticMemory()

    def list_skills(self) -> list[str]:
        """Return available skill stems (on-demand — not auto-injected into LLM context)."""
        return self.skills.list_skills()

    def load_skill(self, name: str) -> str | None:
        """Load one skill Markdown file by stem or filename. Returns None if missing."""
        return self.skills.load_skill(name)

    def retrieve_context(self, conversation_id: str, query: str) -> MemoryBundle:
        return MemoryBundle(
            soul=self.soul.text(),
            user=self.user_profile.text(),
            session=self.session.get(conversation_id),
            daily=self.daily.for_day(),
            long_term=self.long_term.all(),
            semantic=self.semantic.search(query),
        )
