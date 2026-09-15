"""
Priority Inbox
=================
Unified, cross-platform view of scored messages (Part 2, Section 11).
Backed by SQLite via storage/db.py so it survives restarts.
"""

from __future__ import annotations

import json
import os
import time


from assistant.intelligence.priority_engine import PriorityResult
from assistant.storage.db import get_connection


class PriorityInbox:
    def __init__(self, max_per_level: int | None = None):
        if max_per_level is not None:
            self.max_per_level = max_per_level
        else:
            self.max_per_level = int(os.environ.get("PRIORITY_INBOX_MAX_PER_LEVEL", "20"))

    def add(self, result: PriorityResult) -> None:
        conn = get_connection()
        msg_id = result.message.message_id
        src = result.message.source.value if hasattr(result.message.source, "value") else str(result.message.source)
        sender = result.message.sender
        content = result.message.content
        lvl = result.level.value if hasattr(result.level, "value") else str(result.level)
        cat = result.category.value if hasattr(result.category, "value") else str(result.category)

        # 1. Deduplication check
        existing = conn.execute(
            "SELECT id FROM priority_inbox WHERE message_id = ? OR (source = ? AND sender = ? AND content = ?)",
            (msg_id, src, sender, content),
        ).fetchone()
        if existing:
            return

        # 2. Insert new record with PENDING analysis_status
        conn.execute(
            "INSERT INTO priority_inbox (message_id, source, sender, content, category, score, level, ts, analysis_status, analysis_attempts) "
            "VALUES (?,?,?,?,?,?,?,?, 'PENDING', 0)",
            (msg_id, src, sender, content, cat, result.score, lvl, result.message.timestamp),
        )

        # 3. Enforce hard limit of MAX per priority level
        rows = conn.execute(
            "SELECT id FROM priority_inbox WHERE level = ? ORDER BY score DESC, ts DESC",
            (lvl,),
        ).fetchall()
        if len(rows) > self.max_per_level:
            excess_ids = [r["id"] for r in rows[self.max_per_level:]]
            placeholders = ",".join("?" * len(excess_ids))
            conn.execute(f"DELETE FROM priority_inbox WHERE id IN ({placeholders})", excess_ids)

        conn.commit()


    def count_pending_messages(self) -> int:
        conn = get_connection()
        row = conn.execute("SELECT COUNT(*) as cnt FROM priority_inbox WHERE analysis_status = 'PENDING'").fetchone()
        return row["cnt"] if row else 0

    def get_pending_messages(self, limit: int = 50) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, message_id, source, sender, content, category, score, level, ts, analysis_attempts, user_override_level "
            "FROM priority_inbox "
            "WHERE analysis_status = 'PENDING' "
            "ORDER BY ts ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


    def mark_messages_processing(self, ids: list[int], started_at: float | None = None) -> list[int]:
        if not ids:
            return []
        conn = get_connection()
        now = started_at or time.time()
        placeholders = ",".join("?" * len(ids))
        # Select only those that are currently PENDING
        valid_rows = conn.execute(
            f"SELECT id FROM priority_inbox WHERE id IN ({placeholders}) AND analysis_status = 'PENDING'",
            ids,
        ).fetchall()
        locked_ids = [r["id"] for r in valid_rows]
        if locked_ids:
            locked_placeholders = ",".join("?" * len(locked_ids))
            conn.execute(
                f"UPDATE priority_inbox SET analysis_status = 'PROCESSING', analysis_started_at = ?, analysis_attempts = analysis_attempts + 1 "
                f"WHERE id IN ({locked_placeholders})",
                [now] + locked_ids,
            )
            conn.commit()
        return locked_ids

    def defer_processing_messages(self, ids: list[int], error: str = "") -> None:
        if not ids:
            return
        conn = get_connection()
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"UPDATE priority_inbox SET analysis_status = 'PENDING', analysis_error = ? "
            f"WHERE id IN ({placeholders})",
            [error] + ids,
        )
        conn.commit()

    def update_analysis_success(
        self,
        id: int,
        analysis: dict,
        final_level: str,
        user_rule_id: int | None = None,
        override_reason: str = "",
        model: str = "",
        version: str = "1.0",
        completed_at: float | None = None,
    ) -> None:
        conn = get_connection()
        now = completed_at or time.time()
        conn.execute(
            "UPDATE priority_inbox SET "
            "analysis_status = 'COMPLETED', analysis_completed_at = ?, analysis_error = NULL, "
            "analysis_model = ?, analysis_version = ?, "
            "system_category = ?, system_intent = ?, system_urgency = ?, system_importance = ?, "
            "requires_action = ?, is_spam = ?, is_scam = ?, risk_score = ?, deadline = ?, system_reason = ?, "
            "final_priority = ?, user_rule_id = ?, override_reason = ? "
            "WHERE id = ?",
            (
                now, str(model or ""), str(version or "1.0"),
                analysis.get("category"), analysis.get("intent"),
                analysis.get("urgency"), analysis.get("importance"),
                1 if analysis.get("requires_action") else 0,
                1 if analysis.get("spam") else 0,
                1 if analysis.get("scam") else 0,
                float(analysis.get("risk_score", 0.0)),
                analysis.get("deadline"),
                analysis.get("reason", ""),
                final_level, user_rule_id, override_reason,
                id,
            ),
        )
        conn.commit()

    def mark_message_failed(self, id: int, error: str) -> None:
        conn = get_connection()
        conn.execute(
            "UPDATE priority_inbox SET analysis_status = 'FAILED', analysis_error = ? WHERE id = ?",
            (error, id),
        )
        conn.commit()

    def recover_stale_processing(self, stale_before: float) -> int:
        conn = get_connection()
        cursor = conn.execute(
            "UPDATE priority_inbox SET analysis_status = 'PENDING', analysis_error = 'stale_processing_timeout' "
            "WHERE analysis_status = 'PROCESSING' AND analysis_started_at < ?",
            (stale_before,),
        )
        conn.commit()
        return cursor.rowcount

    def set_message_override(self, id: int, level: str, reason: str = "") -> None:
        conn = get_connection()
        conn.execute(
            "UPDATE priority_inbox SET user_override_level = ?, final_priority = ?, override_reason = ? WHERE id = ?",
            (level.upper(), level.upper(), reason, id),
        )
        conn.commit()

    def today(self) -> list[dict]:
        conn = get_connection()
        cutoff = time.time() - 24 * 3600
        rows = conn.execute(
            "SELECT source, sender, content, category, score, COALESCE(final_priority, level) as level, ts, "
            "system_category, system_intent, system_urgency, system_importance, requires_action, is_spam, is_scam, risk_score, deadline, system_reason "
            "FROM priority_inbox "
            "WHERE ts >= ? ORDER BY score DESC, ts DESC", (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_messages(
        self,
        source: str | None = None,
        limit: int = 10,
        query: str = "",
    ) -> list[dict]:
        """Retrieve recent persisted messages ordered newest first (ts DESC).

        Unlike today(), this does not apply a 24-hour cutoff, does not filter by priority score,
        and does not restrict to unread status.
        """
        conn = get_connection()
        conditions = []
        params: list[object] = []
        if source:
            conditions.append("source = ?")
            params.append(source.lower().strip())
        if query:
            conditions.append("(content LIKE ? OR sender LIKE ?)")
            q_pat = f"%{query.strip()}%"
            params.extend([q_pat, q_pat])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = (
            f"SELECT id, message_id, source, sender, content, category, score, COALESCE(final_priority, level) as level, ts, "
            f"system_category, system_intent, system_urgency, system_importance, requires_action, is_spam, is_scam, risk_score, deadline, system_reason "
            f"FROM priority_inbox {where_clause} "
            f"ORDER BY ts DESC LIMIT ?"
        )
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_agenda(
        self,
        target_date: str | None = "today",
        limit: int = 20,
    ) -> list[dict]:
        """Retrieve agenda items (meetings, commitments, action items, deadlines)
        for a target date ('today', 'tomorrow', 'upcoming', or YYYY-MM-DD).

        Orders items chronologically by deadline when known, and prioritizes
        confirmed commitments/actions over generic messages.
        """
        import datetime
        import re

        now = datetime.datetime.now()
        target_clean = (target_date or "today").strip().lower()

        date_str = ""
        relative_word = ""
        if "tomorrow" in target_clean:
            target_dt = now + datetime.timedelta(days=1)
            date_str = target_dt.strftime("%Y-%m-%d")
            relative_word = "tomorrow"
        elif "today" in target_clean:
            date_str = now.strftime("%Y-%m-%d")
            relative_word = "today"
        elif re.match(r"^\d{4}-\d{2}-\d{2}", target_clean):
            date_str = target_clean[:10]
        else:
            date_str = now.strftime("%Y-%m-%d")

        conn = get_connection()
        recent_cutoff = time.time() - 72 * 3600  # last 3 days
        sql = (
            "SELECT id, message_id, source, sender, content, category, score, "
            "COALESCE(final_priority, level) as level, ts, "
            "system_category, system_intent, system_urgency, system_importance, "
            "requires_action, is_spam, is_scam, risk_score, deadline, system_reason "
            "FROM priority_inbox "
            "WHERE (COALESCE(final_priority, level) != 'IGNORE') AND ("
            "  (deadline IS NOT NULL AND deadline LIKE ?)"
            "  OR (ts >= ? AND (content LIKE ? OR content LIKE ?))"
            "  OR (ts >= ? AND requires_action = 1 AND ? = 'today')"
            ") "
            "ORDER BY "
            "  CASE WHEN deadline IS NOT NULL AND deadline != '' THEN 0 ELSE 1 END ASC, "
            "  CASE WHEN requires_action = 1 OR system_intent IN ('MEETING', 'INTERVIEW', 'TASK') THEN 0 ELSE 1 END ASC, "
            "  deadline ASC, score DESC, ts DESC "
            "LIMIT ?"
        )
        date_pattern = f"%{date_str}%"
        rel_pattern = f"%{relative_word}%" if relative_word else "%__none__%"
        rel_pattern_cap = f"%{relative_word.capitalize()}%" if relative_word else "%__none__%"
        rows = conn.execute(
            sql,
            (date_pattern, recent_cutoff, rel_pattern, rel_pattern_cap, recent_cutoff, relative_word, limit),
        ).fetchall()
        return [dict(r) for r in rows]



