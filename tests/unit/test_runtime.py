"""
Unit tests for AssistantRuntime (Phase 1).
==========================================
Verifies:
1. Runtime initializes successfully with all expected planes.
2. Runtime exposes required UI service properties (orchestrator, memory, priority, gateway, etc.).
3. Runtime start() is idempotent and does not spawn duplicate workers.
4. Runtime stop() is idempotent and handles partial/unstarted state safely.
5. Dict-like subscripting remains backward-compatible with legacy main:build_app().
6. Context manager (__enter__ / __exit__) functions correctly.
7. Background services are managed cleanly and shutdown safely.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from assistant.runtime import AssistantRuntime, get_runtime, reset_runtime
from assistant.storage.db import init_db, reset_connection


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    db_file = str(tmp_path / "test_assistant.db")
    os.environ["DATABASE_PATH"] = db_file
    reset_connection()
    init_db()
    reset_runtime()
    yield
    reset_runtime()
    reset_connection()


def test_runtime_initialization_and_exposed_services():
    """Verify runtime initializes all subsystems and exposes expected public properties."""
    runtime = AssistantRuntime()

    # Core and execution planes
    assert runtime.orchestrator is not None
    assert runtime.agent_orchestrator is runtime.orchestrator
    assert runtime.memory is not None
    assert runtime.memory_service is runtime.memory
    assert runtime.priority_inbox is not None
    assert runtime.priority_analyzer is not None
    assert runtime.device_gateway is not None
    assert runtime.tool_router is not None
    assert runtime.task_manager is not None
    assert runtime.outbound_registry is not None
    assert runtime.llm is not None
    assert runtime.normal_llm is runtime.llm
    assert runtime.priority_llm is not None

    # Connectors
    assert "whatsapp" in runtime.connectors
    assert "telegram" in runtime.connectors
    assert "telegram_personal" in runtime.connectors
    assert "gmail" in runtime.connectors

    # Channels and Router
    assert runtime.router is not None
    assert runtime.cli_channel is not None
    assert runtime.telegram_channel is not None
    assert runtime.whatsapp_channel is not None

    # Status
    assert runtime.is_started is False


def test_runtime_backward_compatible_dict_access():
    """Verify legacy dictionary-style access used by earlier tests/callers works as expected."""
    runtime = AssistantRuntime()

    assert runtime["orchestrator"] is runtime.orchestrator
    assert runtime["cli_channel"] is runtime.cli_channel
    assert runtime["priority_inbox"] is runtime.priority_inbox
    assert runtime["device_gateway"] is runtime.device_gateway
    assert "task_manager" in runtime

    with pytest.raises(KeyError):
        _ = runtime["non_existent_service"]


def test_runtime_start_is_idempotent():
    """Verify start() does not start workers multiple times on repeated invocations."""
    runtime = AssistantRuntime()

    # Mock workers
    mock_gmail_worker = MagicMock()
    mock_tp_worker = MagicMock()
    mock_wa_worker = MagicMock()
    mock_analyzer_worker = MagicMock()
    mock_telegram_channel = MagicMock()

    runtime.gmail_worker = mock_gmail_worker
    runtime.telegram_personal_worker = mock_tp_worker
    runtime.whatsapp_worker = mock_wa_worker
    runtime.priority_analyzer_worker = mock_analyzer_worker
    runtime.telegram_channel = mock_telegram_channel

    with patch.dict(os.environ, {
        "GMAIL_ENABLED": "true",
        "TELEGRAM_USER_ENABLED": "true",
        "WHATSAPP_ENABLED": "false",
        "PRIORITY_ANALYZER_ENABLED": "true",
        "DEVICE_GATEWAY_ENABLED": "false",
    }):
        # First start
        runtime.start()
        assert runtime.is_started is True
        assert mock_gmail_worker.start.call_count == 1
        assert mock_tp_worker.start.call_count == 1
        assert mock_analyzer_worker.start.call_count == 1
        assert mock_telegram_channel.start.call_count == 1

        # Second start (should be a no-op, no duplicate starts)
        runtime.start()
        assert mock_gmail_worker.start.call_count == 1
        assert mock_tp_worker.start.call_count == 1
        assert mock_analyzer_worker.start.call_count == 1
        assert mock_telegram_channel.start.call_count == 1


def test_runtime_stop_is_idempotent():
    """Verify stop() cleanly stops all running workers and does not crash when called multiple times."""
    runtime = AssistantRuntime()

    mock_gmail_worker = MagicMock()
    mock_tp_worker = MagicMock()
    mock_analyzer_worker = MagicMock()
    mock_telegram_channel = MagicMock()
    mock_wa_connector = MagicMock()

    runtime.ingestion_workers = [mock_gmail_worker, mock_tp_worker]
    runtime.priority_analyzer_worker = mock_analyzer_worker
    runtime.telegram_channel = mock_telegram_channel
    runtime.whatsapp_connector = mock_wa_connector
    runtime._started = True

    # First stop
    runtime.stop()
    assert runtime.is_started is False
    assert mock_gmail_worker.stop.call_count == 1
    assert mock_tp_worker.stop.call_count == 1
    assert mock_analyzer_worker.stop.call_count == 1
    assert mock_telegram_channel.stop.call_count == 1
    assert mock_wa_connector.stop_bridge.call_count == 1

    # Second stop (idempotent, safe)
    runtime.stop()
    assert runtime.is_started is False


def test_partially_initialized_runtime_shutdown_safe():
    """Verify that if an unstarted or partially initialized runtime is stopped, it doesn't raise errors."""
    runtime = AssistantRuntime.__new__(AssistantRuntime)
    runtime._lock = __import__("threading").RLock()
    runtime._started = False

    # Should not raise AttributeError or crash
    runtime.stop()
    assert runtime.is_started is False


def test_runtime_context_manager():
    """Verify runtime works cleanly as a context manager (__enter__ / __exit__)."""
    runtime = AssistantRuntime()
    with patch.object(runtime, "start") as mock_start, patch.object(runtime, "stop") as mock_stop:
        with runtime as r:
            assert r is runtime
            mock_start.assert_called_once()
        mock_stop.assert_called_once()


def test_main_build_app_returns_runtime():
    """Verify assistant.main.build_app() returns the runtime instance."""
    from assistant.main import build_app

    app = build_app()
    assert isinstance(app, AssistantRuntime)
    assert app.cli_channel is not None
    assert app["orchestrator"] is not None


def test_global_runtime_singleton():
    """Verify get_runtime() provides a singleton instance across calls."""
    r1 = get_runtime()
    r2 = get_runtime()
    assert r1 is r2
    reset_runtime()
    r3 = get_runtime()
    assert r3 is not r1
