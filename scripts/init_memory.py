"""
One-time / idempotent setup: creates SOUL.md, USER.md and the sqlite
schema if they don't already exist. Safe to re-run.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

from assistant.memory.memory_service import MemoryService
from assistant.memory.skill_initializer import initialize_default_skills
from assistant.storage.db import init_db

if __name__ == "__main__":
    load_dotenv()
    init_db()
    memory_dir = os.environ.get("MEMORY_DIR", "./data/memory")
    skill_result = initialize_default_skills(memory_dir)
    MemoryService(memory_dir=memory_dir)
    created = ", ".join(skill_result["created"]) or "(none)"
    print(f"Initialized database and memory files under {memory_dir}")
    print(f"Skills created: {created}")
