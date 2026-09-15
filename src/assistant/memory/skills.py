"""
Skills loader (Part 2, Section 20)
====================================
Skills are Markdown instruction files that teach the Agent how to use a
domain (Gmail, Telegram, etc.). They are NOT executable tools.

The Agent loads a skill on demand when a request requires that capability.
Skill contents are never injected into every LLM request automatically.
"""

from __future__ import annotations

import os


class Skills:
    SKILLS_SUBDIR = "skills"

    def __init__(self, memory_dir: str):
        self.skills_dir = os.path.join(memory_dir, self.SKILLS_SUBDIR)
        os.makedirs(self.skills_dir, exist_ok=True)

    def list_skills(self) -> list[str]:
        """Return sorted skill stems (e.g. ``gmail_skill``) for every ``.md`` file."""
        if not os.path.isdir(self.skills_dir):
            return []
        return sorted(
            filename[:-3]
            for filename in os.listdir(self.skills_dir)
            if filename.endswith(".md") and os.path.isfile(os.path.join(self.skills_dir, filename))
        )

    def load_skill(self, name: str) -> str | None:
        """Load one skill by stem (``gmail`` or ``gmail_skill``) or full filename."""
        path = self._resolve_path(name)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def skill_path(self, name: str) -> str:
        return self._resolve_path(name)

    def _resolve_path(self, name: str) -> str:
        stem = name.strip()
        if stem.endswith(".md"):
            filename = stem
        else:
            if not stem.endswith("_skill"):
                stem = f"{stem}_skill"
            filename = f"{stem}.md"
        return os.path.join(self.skills_dir, filename)
