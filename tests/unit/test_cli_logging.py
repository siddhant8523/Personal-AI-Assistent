"""
Unit tests for CLI logging isolation.
=======================================
Verifies:
1. Background/personal Telegram error logs do not bleed into CLI stdout/stderr.
2. All application and background error/info logs are written to the log file.
3. Telethon and asyncio errors do not pollute CLI stdout/stderr.
4. Console logging can be optionally enabled via CONSOLE_LOGGING / enable_console=True.
5. Normal CLI assistant responses and UI rendering remain clean and unaffected.
"""

import logging
from assistant.channels.cli_ui_renderer import render_assistant_reply
from assistant.main import configure_logging


def test_background_logger_error_not_in_console_but_written_to_file(tmp_path, capsys):
    log_file = str(tmp_path / "test_assistant.log")
    configure_logging(log_file=log_file, enable_console=False, log_level="INFO")

    # Clear captured output before logging
    capsys.readouterr()

    logger = logging.getLogger("assistant.telegram.personal")
    error_msg = "Error fetching recent personal Telegram messages: Cannot send requests while disconnected"
    logger.error(error_msg)

    captured = capsys.readouterr()
    # Must NOT appear on console stdout or stderr
    assert error_msg not in captured.out
    assert error_msg not in captured.err

    # MUST be written to the log file
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert error_msg in content
    assert "assistant.telegram.personal" in content
    assert "ERROR" in content


def test_telethon_and_asyncio_errors_not_in_console_but_in_file(tmp_path, capsys):
    log_file = str(tmp_path / "test_assistant.log")
    configure_logging(log_file=log_file, enable_console=False, log_level="INFO")

    capsys.readouterr()

    telethon_logger = logging.getLogger("telethon.network.mtprotosender")
    telethon_msg = "Automatic reconnection failed 5 time(s)"
    telethon_logger.error(telethon_msg)

    asyncio_logger = logging.getLogger("asyncio")
    asyncio_msg = "ConnectionError exception in shielded future"
    asyncio_logger.error(asyncio_msg)

    captured = capsys.readouterr()
    assert telethon_msg not in captured.out
    assert telethon_msg not in captured.err
    assert asyncio_msg not in captured.out
    assert asyncio_msg not in captured.err

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert telethon_msg in content
    assert asyncio_msg in content


def test_console_logging_opt_in(tmp_path, capsys):
    log_file = str(tmp_path / "test_assistant.log")
    configure_logging(log_file=log_file, enable_console=True, log_level="INFO")

    capsys.readouterr()

    logger = logging.getLogger("assistant.telegram.personal")
    error_msg = "Explicit console dev error"
    logger.error(error_msg)

    captured = capsys.readouterr()
    # When console logging is opted into, ERROR appears in stderr
    assert error_msg in captured.err

    # And still written to file
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert error_msg in content


def test_cli_assistant_responses_unaffected(capsys):
    capsys.readouterr()
    render_assistant_reply("Hello! How can I help you today?")
    captured = capsys.readouterr()
    assert "Joe" in captured.out
    assert "Hello! How can I help you today?" in captured.out
