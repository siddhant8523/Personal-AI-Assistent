"""Long-Term Memory — durable key/value facts, sqlite-backed (Part 2, Sec 18)."""

from __future__ import annotations

import time

from assistant.storage.db import get_connection


class LongTermMemory:
    def set(self, key: str, value: str) -> None:
        conn = get_connection()
        conn.execute(
            "INSERT INTO long_term_memory (key, value, ts) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
            (key, value, time.time()),
        )
        conn.commit()

    def get(self, key: str) -> str | None:
        conn = get_connection()
        row = conn.execute("SELECT value FROM long_term_memory WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def all(self) -> dict[str, str]:
        conn = get_connection()
        rows = conn.execute("SELECT key, value FROM long_term_memory").fetchall()
        return {r["key"]: r["value"] for r in rows}
