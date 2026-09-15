"""Daily Memory — recent events, sqlite-backed (Part 2, Section 17)."""

from __future__ import annotations

import datetime as dt
import time

from assistant.storage.db import get_connection


class DailyMemory:
    def add(self, entry: str, day: str | None = None) -> None:
        day = day or dt.date.today().isoformat()
        conn = get_connection()
        conn.execute("INSERT INTO daily_memory (day, entry, ts) VALUES (?,?,?)", (day, entry, time.time()))
        conn.commit()

    def for_day(self, day: str | None = None) -> list[str]:
        day = day or dt.date.today().isoformat()
        conn = get_connection()
        rows = conn.execute("SELECT entry FROM daily_memory WHERE day = ? ORDER BY ts", (day,)).fetchall()
        return [r["entry"] for r in rows]
