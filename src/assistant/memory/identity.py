"""Persistent, user-controlled assistant identity stored in SOUL.md."""

from __future__ import annotations

import re

from assistant.memory.soul import Soul


class IdentityManager:
    _NAME_RE = re.compile(r"^[\w][\w .'-]{0,48}$", re.UNICODE)
    _IDENTITY_RE = re.compile(r"\n## Assistant identity\n- Name: .*?(?=\n## |\Z)", re.DOTALL)

    def __init__(self, soul: Soul):
        self._soul = soul

    def name(self) -> str:
        match = re.search(r"^## Assistant identity\n- Name: (.+)$", self._soul.text(), re.MULTILINE)
        return match.group(1).strip() if match else "SOUL"

    def set_name(self, name: str) -> str:
        name = name.strip()
        if not self._NAME_RE.fullmatch(name):
            raise ValueError("Name must be 1–49 letters, numbers, spaces, apostrophes, periods, or hyphens.")
        text = self._soul.text()
        section = f"\n## Assistant identity\n- Name: {name}\n"
        updated = self._IDENTITY_RE.sub(section.rstrip(), text) if self._IDENTITY_RE.search(text) else text.rstrip() + section
        with open(self._soul.path, "w") as handle:
            handle.write(updated.rstrip() + "\n")
        return name
