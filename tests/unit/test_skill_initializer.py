"""
Unit tests for default Skill Markdown initialization.
"""

from __future__ import annotations

import os

import pytest

from assistant.memory.memory_service import MemoryService
from assistant.memory.skill_initializer import DEFAULT_SKILL_FILENAMES, initialize_default_skills
from assistant.memory.skills import Skills

REQUIRED_SECTIONS = (
    "## Purpose",
    "## Capabilities",
    "## Available Operations",
    "## When to Use",
    "## When NOT to Use",
    "## Required Information",
    "## Recipient Resolution",
    "## File/Attachment Handling",
    "## Approval Requirements",
    "## Safety Rules",
    "## Failure Handling",
    "## Examples",
    "## Tool/Connector Mapping",
)

SKILL_NOT_TOOL_MARKERS = (
    "Skills Are NOT Tools",
    "Skill MD",
    "Tool Router",
)


@pytest.fixture
def memory_dir(tmp_path):
    return str(tmp_path / "memory")


def test_initialize_creates_all_five_default_skills(memory_dir):
    result = initialize_default_skills(memory_dir)

    assert set(result["created"]) == set(DEFAULT_SKILL_FILENAMES)
    assert result["skipped"] == []

    skills_dir = os.path.join(memory_dir, "skills")
    for filename in DEFAULT_SKILL_FILENAMES:
        path = os.path.join(skills_dir, filename)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 500


def test_initialize_does_not_overwrite_existing_skills(memory_dir):
    skills_dir = os.path.join(memory_dir, "skills")
    os.makedirs(skills_dir, exist_ok=True)

    custom_path = os.path.join(skills_dir, "gmail_skill.md")
    custom_content = "# Custom Gmail Skill\n\nUser-edited content must persist.\n"
    with open(custom_path, "w", encoding="utf-8") as f:
        f.write(custom_content)

    result = initialize_default_skills(memory_dir)

    assert "gmail_skill.md" in result["skipped"]
    assert "gmail_skill.md" not in result["created"]
    assert len(result["created"]) == len(DEFAULT_SKILL_FILENAMES) - 1

    with open(custom_path, encoding="utf-8") as f:
        assert f.read() == custom_content


def test_initialize_is_idempotent(memory_dir):
    first = initialize_default_skills(memory_dir)
    second = initialize_default_skills(memory_dir)

    assert set(first["created"]) == set(DEFAULT_SKILL_FILENAMES)
    assert second["created"] == []
    assert set(second["skipped"]) == set(DEFAULT_SKILL_FILENAMES)


def test_skill_files_contain_useful_sections(memory_dir):
    initialize_default_skills(memory_dir)
    skills = Skills(memory_dir)

    for stem in skills.list_skills():
        content = skills.load_skill(stem)
        assert content is not None
        assert len(content.strip()) > 800
        for section in REQUIRED_SECTIONS:
            assert section in content, f"{stem} missing section {section!r}"
        for marker in SKILL_NOT_TOOL_MARKERS:
            assert marker in content, f"{stem} missing marker {marker!r}"


def test_gmail_skill_covers_project_capabilities(memory_dir):
    initialize_default_skills(memory_dir)
    content = Skills(memory_dir).load_skill("gmail") or ""

    for topic in (
        "search_gmail",
        "read_gmail",
        "send_email",
        "10 MB",
        "approval",
        "Priority Inbox",
        "Gmail Connector",
        "thread",
    ):
        assert topic.lower() in content.lower(), f"gmail skill should mention {topic!r}"


def test_telegram_skill_distinguishes_bot_and_personal(memory_dir):
    initialize_default_skills(memory_dir)
    content = Skills(memory_dir).load_skill("telegram") or ""

    assert "Telegram Bot" in content
    assert "personal account" in content.lower() or "Personal Account" in content
    assert "TELEGRAM_BOT_TOKEN" in content
    assert "TELEGRAM_API_ID" in content
    assert "Telethon" in content
    assert "separate" in content.lower()


def test_whatsapp_skill_describes_baileys_architecture(memory_dir):
    initialize_default_skills(memory_dir)
    content = Skills(memory_dir).load_skill("whatsapp") or ""

    for topic in ("Baileys", "bridge", "Echo Filter", "10 MB", "AGENT_CHAT", "WHATSAPP_ENABLED"):
        assert topic in content, f"whatsapp skill should mention {topic!r}"


def test_android_skill_includes_cross_platform_file_flow(memory_dir):
    initialize_default_skills(memory_dir)
    content = Skills(memory_dir).load_skill("android") or ""

    assert "report.pdf" in content
    assert "data/outbound_files" in content
    assert "File Resolver" in content
    assert "Device Gateway" in content
    assert "UPLOAD_FILE" in content


def test_file_skill_covers_policy_and_limits(memory_dir):
    initialize_default_skills(memory_dir)
    content = Skills(memory_dir).load_skill("file") or ""

    for topic in (
        "FilePolicy",
        "FileResolver",
        "AttachmentManager",
        "10 MB",
        "allowed",
        "AmbiguousFileError",
    ):
        assert topic in content, f"file skill should mention {topic!r}"


def test_skills_contain_no_secrets(memory_dir):
    initialize_default_skills(memory_dir)
    forbidden = ("sk-proj-", "sk_live_", "sk_test_", "api_key=", "password=", "Bearer ", "BEGIN PRIVATE KEY")

    for stem in Skills(memory_dir).list_skills():
        content = (Skills(memory_dir).load_skill(stem) or "").lower()
        for token in forbidden:
            assert token.lower() not in content, f"{stem} must not contain secret pattern {token!r}"



def test_memory_service_list_and_load_skills(memory_dir):
    memory = MemoryService(memory_dir=memory_dir)

    listed = memory.list_skills()
    assert set(listed) == {
        "gmail_skill",
        "telegram_skill",
        "whatsapp_skill",
        "file_skill",
        "android_skill",
    }

    gmail = memory.load_skill("gmail")
    assert gmail is not None
    assert "# Gmail Skill" in gmail

    assert memory.load_skill("nonexistent_skill") is None


def test_memory_service_startup_initializes_skills_once(memory_dir):
    memory = MemoryService(memory_dir=memory_dir)
    paths = {memory.skills.skill_path(name) for name in memory.list_skills()}

    first_sizes = {p: os.path.getsize(p) for p in paths}
    MemoryService(memory_dir=memory_dir)
    second_sizes = {p: os.path.getsize(p) for p in paths}

    assert first_sizes == second_sizes
