"""
Unit tests for Safe Logging Sanitization and Privacy Filters
=============================================================
Verifies:
1. Strict 5-step sanitization order:
   - Step 1: Detect/remove sensitive fields (passwords, tokens, phone numbers, JIDs, LIDs).
   - Step 2: Detect/remove known huge protocol/binary fields (Baileys payloads, media keys, file hashes).
   - Step 3: Prevent recursive/nested object expansion.
   - Step 4: Apply maximum serialized/log-line size.
   - Step 5: Only then emit the final log message.
2. Never truncates an unsanitized object.
3. Verification that huge Baileys protocol objects never reach the logger.
"""

import logging
import pytest

from assistant.logging_config import (
    SafeLoggingFilter,
    sanitize_text,
    shallow_sanitize,
    configure_logging,
    MAX_LOG_LINE_LENGTH,
)


def test_sanitize_text_redactions():
    # WhatsApp PN JID
    assert sanitize_text("Incoming from 1234567890@s.whatsapp.net now") == "Incoming from [REDACTED_JID] now"

    # WhatsApp LID
    assert sanitize_text("LID mapping: 123456789012345@lid target") == "LID mapping: [REDACTED_LID] target"

    # Phone numbers (10-15 digits)
    assert sanitize_text("Call +14155552671 please") == "Call [REDACTED_NUMBER] please"
    assert sanitize_text("Number 9876543210123 here") == "Number [REDACTED_NUMBER] here"

    # Bearer tokens
    assert sanitize_text("Authorization: Bearer abcdef1234567890xyz") == "Authorization: Bearer [REDACTED]"

    # Key-value secrets
    assert sanitize_text("config: api_key='sk-123456789'") == "config: api_key: [REDACTED]"
    assert sanitize_text('config: password="my-secret-pass"') == "config: password: [REDACTED]"


def test_shallow_sanitize_order():
    # Step 1: Sensitive fields removed
    data = {
        "user": "Alice",
        "api_key": "secret-123",
        "phone_number": "+14155552671",
        # Step 2: Huge fields removed
        "histNotification": {"big": "payload"},
        "initialHistBootstrapInlinePayload": "data" * 1000,
        "mediaKey": "bytes-key",
        # Nested dict to test Step 3 (no deep recursion)
        "nested": {
            "level2": {
                "level3": "deep",
            }
        }
    }

    sanitized = shallow_sanitize(data)

    # Sensitive fields redacted
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["phone_number"] == "[REDACTED]"
    assert sanitized["user"] == "Alice"

    # Huge fields completely stripped
    assert "histNotification" not in sanitized
    assert "initialHistBootstrapInlinePayload" not in sanitized
    assert "mediaKey" not in sanitized

    # Nested depth bounded (Step 3)
    assert "nested" in sanitized
    assert isinstance(sanitized["nested"], dict)
    assert sanitized["nested"]["level2"] == "<dict len=1>"


def test_safe_logging_filter_5_step_order(caplog):
    logger = logging.getLogger("test.sanitization.filter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.filters.clear()

    safe_filter = SafeLoggingFilter()
    logger.addFilter(safe_filter)

    caplog.handler.addFilter(safe_filter)

    with caplog.at_level(logging.INFO):
        # 1. Test sensitive fields and JIDs redacted in formatted messages
        logger.info("User %s sent message to %s", "+14155552671", "1234567890@s.whatsapp.net")
        assert "[REDACTED_NUMBER]" in caplog.text
        assert "[REDACTED_JID]" in caplog.text
        assert "+14155552671" not in caplog.text
        assert "1234567890@s.whatsapp.net" not in caplog.text

        caplog.clear()

        # 2. Test huge protocol objects discarded
        huge_baileys_event = {
            "msg": "Incoming message",
            "histNotification": {"initialHistBootstrapInlinePayload": "A" * 10000},
            "mediaKey": "secretmedia",
            "safe_field": "ok_value",
        }
        logger.info(huge_baileys_event)
        # Verify huge fields were stripped before rendering
        assert "histNotification" not in caplog.text
        assert "mediaKey" not in caplog.text
        assert "ok_value" in caplog.text

        caplog.clear()

        # 3. Test maximum length truncation happens ONLY on already-sanitized content
        long_prefix = "Sensitive number +14155552671 " + ("SafeContent " * 500)
        logger.info(long_prefix)
        # Must be sanitized FIRST (no phone number)
        assert "+14155552671" not in caplog.text
        assert "[REDACTED_NUMBER]" in caplog.text
        # And must be truncated to MAX_LOG_LINE_LENGTH
        assert "[TRUNCATED]" in caplog.text


def test_safe_logging_filter_drops_raw_history_sync():
    logger = logging.getLogger("test.sanitization.drop")
    logger.setLevel(logging.INFO)
    safe_filter = SafeLoggingFilter()

    # Create record with history sync attribute
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="History sync notification received",
        args=(),
        exc_info=None,
    )
    record.histNotification = {"data": "huge"}

    # Must be dropped completely
    assert safe_filter.filter(record) is False


def test_third_party_loggers_silenced():
    configure_logging(enable_console=False)

    for name in [
        "httpx",
        "telethon",
        "urllib3",
        "httpcore",
        "websockets.server",
        "websockets.protocol",
        "assistant.ingestion",
        "assistant.normal_pipeline",
        "assistant.whatsapp",
        "assistant.telegram",
        "assistant.channels.telegram",
    ]:
        assert logging.getLogger(name).level >= logging.WARNING

    assert logging.getLogger("googleapiclient.discovery_cache").level >= logging.ERROR
