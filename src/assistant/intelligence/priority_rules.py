"""
Priority Rules Manager
=======================
Persists machine-enforceable priority rules in SQLite and applies
deterministic conflict resolution across multiple matching rules.
"""

from __future__ import annotations

import re
import time
from typing import Any

from assistant.storage.db import get_connection

# Deterministic rule type precedence: SENDER > DOMAIN > KEYWORD > CATEGORY
RULE_TYPE_PRECEDENCE = {
    "SENDER": 4,
    "DOMAIN": 3,
    "KEYWORD": 2,
    "CATEGORY": 1,
}

# Deterministic target level rank when rule types match: HIGH > MEDIUM > LOW > IGNORE
TARGET_LEVEL_RANK = {
    "HIGH": 4,
    "MEDIUM": 3,
    "LOW": 2,
    "IGNORE": 1,
}


class PriorityRulesManager:
    def __init__(self, conn=None):
        self._conn = conn

    @property
    def conn(self):
        return self._conn or get_connection()

    def add_rule(
        self,
        rule_type: str,
        pattern: str,
        target_level: str,
        reason: str = "",
        is_active: int = 1,
    ) -> int:
        r_type = rule_type.upper().strip()
        t_level = target_level.upper().strip()
        pat = pattern.strip()
        now = time.time()

        cursor = self.conn.execute(
            "INSERT INTO priority_rules (rule_type, pattern, target_level, reason, created_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (r_type, pat, t_level, reason, now, is_active),
        )
        self.conn.commit()
        return cursor.lastrowid

    def list_rules(self, active_only: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT id, rule_type, pattern, target_level, reason, created_at, is_active FROM priority_rules"
        if active_only:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY id ASC"
        rows = self.conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def delete_rule(self, rule_id: int) -> None:
        self.conn.execute("DELETE FROM priority_rules WHERE id = ?", (rule_id,))
        self.conn.commit()

    def match_rule(
        self,
        sender: str = "",
        content: str = "",
        category: str = "",
        rules: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Find the matching rule with deterministic conflict resolution.

        Precedence when multiple rules match:
        1. Rule type specificity: SENDER > DOMAIN > KEYWORD > CATEGORY
        2. Target level rank: HIGH > MEDIUM > LOW > IGNORE
        3. Pattern length: longer pattern is more specific
        4. Lower rule ID: deterministic tie-breaker
        """
        active_rules = rules if rules is not None else self.list_rules(active_only=True)
        if not active_rules:
            return None

        sender_norm = (sender or "").lower().strip()
        content_norm = (content or "").lower().strip()
        cat_norm = (category or "").upper().strip()

        matched: list[dict[str, Any]] = []

        for rule in active_rules:
            r_type = rule.get("rule_type", "").upper()
            pat = rule.get("pattern", "").strip().lower()
            if not pat:
                continue

            is_match = False
            if r_type == "SENDER":
                is_match = (pat in sender_norm) or (sender_norm in pat)
            elif r_type == "DOMAIN":
                # Matches email domain or URL domain in sender/content
                is_match = f"@{pat}" in sender_norm or pat in sender_norm or pat in content_norm
            elif r_type == "KEYWORD":
                pattern_re = r"\b" + re.escape(pat) + r"\b"
                is_match = bool(re.search(pattern_re, content_norm)) or (pat in content_norm)
            elif r_type == "CATEGORY":
                is_match = pat.upper() == cat_norm

            if is_match:
                matched.append(rule)

        if not matched:
            return None

        # Sort matches deterministically
        def sort_key(r: dict[str, Any]) -> tuple[int, int, int, int]:
            type_score = RULE_TYPE_PRECEDENCE.get(r.get("rule_type", "").upper(), 0)
            level_score = TARGET_LEVEL_RANK.get(r.get("target_level", "").upper(), 0)
            pattern_len = len(r.get("pattern", ""))
            rule_id = r.get("id", 0)
            return (type_score, level_score, pattern_len, -rule_id)

        matched.sort(key=sort_key, reverse=True)
        return matched[0]
