"""
Storage
==========
Single SQLite database for everything that needs to persist:
priority inbox, daily memory, long-term memory, semantic memory entries,
tasks, and drafts. Keeps the project dependency-light (stdlib sqlite3)
while giving every plane a durable backing store.
"""

from __future__ import annotations

import os
import sqlite3
import threading

_local = threading.local()


def _db_path() -> str:
    # Read lazily (not at import time) so tests can point this at a temp
    # file via DATABASE_PATH before the first connection is opened.
    return os.environ.get("DATABASE_PATH", "./data/assistant.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS priority_inbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT, source TEXT, sender TEXT, content TEXT,
    category TEXT, score INTEGER, level TEXT, ts REAL
);

CREATE TABLE IF NOT EXISTS priority_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL,
    pattern TEXT NOT NULL,
    target_level TEXT NOT NULL,
    reason TEXT,
    created_at REAL NOT NULL,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS daily_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT, entry TEXT, ts REAL
);

CREATE TABLE IF NOT EXISTS long_term_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE, value TEXT, ts REAL
);

CREATE TABLE IF NOT EXISTS semantic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT, tags TEXT, ts REAL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT, target TEXT, parameters TEXT,
    state TEXT, draft TEXT, created_at REAL, updated_at REAL,
    result TEXT, error TEXT
);

CREATE TABLE IF NOT EXISTS drafts (
    task_id TEXT PRIMARY KEY,
    draft TEXT, version INTEGER, created_at REAL
);

CREATE TABLE IF NOT EXISTS contact_cache (
    normalized_name TEXT PRIMARY KEY,
    display_name TEXT,
    phone_number TEXT,
    resolved_jid TEXT,
    cached_at REAL,
    expires_at REAL
);
"""

_PRIORITY_INBOX_COLUMNS = [
    ("analysis_status", "TEXT DEFAULT 'PENDING'"),
    ("analysis_attempts", "INTEGER DEFAULT 0"),
    ("analysis_started_at", "REAL"),
    ("analysis_completed_at", "REAL"),
    ("analysis_error", "TEXT"),
    ("analysis_model", "TEXT"),
    ("analysis_version", "TEXT"),
    ("system_category", "TEXT"),
    ("system_intent", "TEXT"),
    ("system_urgency", "TEXT"),
    ("system_importance", "TEXT"),
    ("requires_action", "INTEGER DEFAULT 0"),
    ("is_spam", "INTEGER DEFAULT 0"),
    ("is_scam", "INTEGER DEFAULT 0"),
    ("risk_score", "REAL DEFAULT 0.0"),
    ("deadline", "TEXT"),
    ("system_reason", "TEXT"),
    ("final_priority", "TEXT"),
    ("user_rule_id", "INTEGER"),
    ("user_override_level", "TEXT"),
    ("override_reason", "TEXT"),
]


_db_lock = threading.Lock()


def _migrate_priority_inbox(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("PRAGMA table_info(priority_inbox)")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    for col_name, col_def in _PRIORITY_INBOX_COLUMNS:
        if col_name not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE priority_inbox ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise


def get_connection() -> sqlite3.Connection:
    path = _db_path()
    cached_path = getattr(_local, "path", None)
    if not hasattr(_local, "conn") or cached_path != path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
        _local.path = path
        with _db_lock:
            conn.executescript(SCHEMA)
            _migrate_priority_inbox(conn)
            conn.commit()
    return _local.conn


def reset_connection() -> None:
    """Mainly for tests: drop the cached connection so the next
    get_connection() call re-opens against the current DATABASE_PATH."""
    if hasattr(_local, "conn"):
        _local.conn.close()
        del _local.conn
        del _local.path


def init_db() -> None:
    conn = get_connection()
    conn.executescript(SCHEMA)
    _migrate_priority_inbox(conn)
    conn.commit()


