"""
USER.md + user_preferences.yaml loader (Part 2, Section 15).

USER.md holds freeform durable context for the LLM. user_preferences.yaml
holds the same information in a machine-readable form (e.g. important
contacts) so rule-based components like sender_analysis.py don't need an
LLM call just to check "is this person important".
"""

from __future__ import annotations

import os

import yaml

_DEFAULT_USER_MD = """\
# USER.md

## Preferred communication style
Concise. No unnecessary pleasantries.

## Important contacts
- Manager
- Family

## Recurring instructions
- Always ask before sending anything on my behalf.
"""

_DEFAULT_PREFS = {
    "important_contacts": ["manager", "mom", "dad", "mummy", "maosi"],
    "quiet_hours": {"start": "23:00", "end": "07:00"},
}



class UserProfile:
    def __init__(self, memory_dir: str):
        self.md_path = os.path.join(memory_dir, "USER.md")
        self.prefs_path = os.path.join(memory_dir, "user_preferences.yaml")
        os.makedirs(memory_dir, exist_ok=True)

        if not os.path.exists(self.md_path):
            with open(self.md_path, "w") as f:
                f.write(_DEFAULT_USER_MD)

        if not os.path.exists(self.prefs_path):
            with open(self.prefs_path, "w") as f:
                yaml.safe_dump(_DEFAULT_PREFS, f)

    def text(self) -> str:
        with open(self.md_path, "r") as f:
            return f.read()

    def preferences(self) -> dict:
        with open(self.prefs_path, "r") as f:
            return yaml.safe_load(f) or {}

    def important_contacts(self) -> list[str]:
        return self.preferences().get("important_contacts", [])

    def resolve_contact_number(self, recipient: str) -> str | None:
        rec = (recipient or "").strip()
        if not rec:
            return None
        if rec.startswith("+") or rec.replace("-", "").replace(" ", "").isdigit():
            return rec

        contacts = self.preferences().get("contacts", {})
        rec_lower = rec.lower()
        for k, v in contacts.items():
            if k.lower() == rec_lower or rec_lower in k.lower():
                return str(v)
        return None

