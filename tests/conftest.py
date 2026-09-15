"""
Pytest configuration: isolates each test run against a throwaway SQLite
file so tests never depend on (or pollute) the real ./data/assistant.db.
"""

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test_assistant.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    from assistant.storage import db
    db.reset_connection()
    yield
    db.reset_connection()
