"""
Credential Store (Rule 20)
=============================
Credentials NEVER go into LLM prompts, memory files, USER.md, or SOUL.md.
This is the only place that reads secrets, and it reads them from the
environment (.env), not from any file the LLM or memory system touches.
"""

from __future__ import annotations

import os


class CredentialStore:
    def get(self, name: str, default: str = "") -> str:
        return os.environ.get(name, default)

    def require(self, name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Missing required credential: {name} (check your .env)")
        return value

    def has(self, name: str) -> bool:
        return bool(os.environ.get(name))
