"""
Unit Tests for Log Rotation and Retention
==========================================
Verifies:
A. New log file is created.
B. Log file below 5 MB does not rotate.
C. Log file reaching/exceeding 5 MB rotates.
D. assistant.log becomes assistant2.log.
E. Existing rotation chain shifts correctly (2->3, 3->4, 4->5).
F. assistant5.log is deleted when maximum rotation count is reached.
G. Same behavior works for whatsapp_bridge.log.
H. Files older than 30 minutes are removed.
I. Current active assistant.log is NOT removed by retention cleanup.
J. Current active whatsapp_bridge.log is NOT removed by retention cleanup.
K. Missing data/logs directory is created safely.
L. Logger initialization does not duplicate handlers.
M. Streamlit reruns do not create duplicate handlers.
N. Subprocess stdout writes continue in active log after copytruncate rotation.
O. Partial rotation chains shift safely without error.
"""

import logging
import os
import subprocess
import sys
import time

import pytest

from assistant.logging_config import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_FILES,
    DEFAULT_RETENTION_SECONDS,
    ProductionRotatingFileHandler,
    cleanup_old_rotated_logs,
    configure_logging,
    get_rotated_log_path,
    rotate_log_file,
)


@pytest.fixture(autouse=True)
def reset_logging():
    """Cleanly reset root logger handlers before and after each test."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def test_a_new_log_file_is_created(tmp_path):
    """Test A: New log file is created when logging is configured."""
    log_file = str(tmp_path / "assistant.log")
    configure_logging(log_file=log_file, enable_console=False, log_level="INFO")

    logger = logging.getLogger("test.create")
    logger.info("Test message 1")

    assert os.path.isfile(log_file)
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Test message 1" in content


def test_b_log_file_below_max_does_not_rotate(tmp_path):
    """Test B: Log file below maxBytes does not rotate."""
    log_file = str(tmp_path / "assistant.log")
    handler = ProductionRotatingFileHandler(
        log_file,
        maxBytes=1000,
        maxFiles=5,
        maxAgeSeconds=1800,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    test_logger = logging.getLogger("test.below_limit")
    test_logger.handlers = [handler]
    test_logger.setLevel(logging.INFO)

    # Write 300 bytes (well below 1000 bytes)
    test_logger.info("A" * 300)
    handler.flush()

    rot2 = get_rotated_log_path(log_file, 2)
    assert not os.path.exists(rot2)
    assert os.path.exists(log_file)
    assert os.path.getsize(log_file) >= 300

    # Also test rotate_log_file function directly
    rotated = rotate_log_file(log_file, max_bytes=1000)
    assert not rotated
    assert not os.path.exists(rot2)


def test_c_and_d_log_file_exceeding_max_rotates_and_becomes_assistant2(tmp_path):
    """Test C & D: Log file reaching/exceeding maxBytes rotates, assistant.log becomes assistant2.log."""
    log_file = str(tmp_path / "assistant.log")
    handler = ProductionRotatingFileHandler(
        log_file,
        maxBytes=500,
        maxFiles=5,
        maxAgeSeconds=1800,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    test_logger = logging.getLogger("test.rotate_d")
    test_logger.handlers = [handler]
    test_logger.setLevel(logging.INFO)

    # Initial write of 400 bytes
    test_logger.info("InitialData" + "X" * 350)
    handler.flush()

    rot2 = get_rotated_log_path(log_file, 2)
    assert not os.path.exists(rot2)

    # Second write pushes total above 500 bytes -> triggers rotation
    test_logger.info("SecondData" + "Y" * 300)
    handler.flush()

    # assistant2.log must now exist and contain InitialData
    assert os.path.exists(rot2)
    with open(rot2, "r", encoding="utf-8") as f:
        rot_content = f.read()
    assert "InitialData" in rot_content

    # Active assistant.log must contain the new SecondData
    with open(log_file, "r", encoding="utf-8") as f:
        active_content = f.read()
    assert "SecondData" in active_content
    assert "InitialData" not in active_content


def test_e_existing_rotation_chain_shifts_correctly(tmp_path):
    """Test E: Existing rotation chain shifts correctly (2->3, 3->4, 4->5)."""
    log_file = str(tmp_path / "assistant.log")

    # Cycle 1: assistant.log -> assistant2.log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Batch1")
    rotate_log_file(log_file, max_bytes=5)
    assert os.path.exists(get_rotated_log_path(log_file, 2))
    with open(get_rotated_log_path(log_file, 2), "r", encoding="utf-8") as f:
        assert f.read() == "Batch1"

    # Cycle 2: Batch1 -> assistant3.log, Batch2 -> assistant2.log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Batch2")
    rotate_log_file(log_file, max_bytes=5)
    with open(get_rotated_log_path(log_file, 3), "r", encoding="utf-8") as f:
        assert f.read() == "Batch1"
    with open(get_rotated_log_path(log_file, 2), "r", encoding="utf-8") as f:
        assert f.read() == "Batch2"

    # Cycle 3: Batch1 -> 4, Batch2 -> 3, Batch3 -> 2
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Batch3")
    rotate_log_file(log_file, max_bytes=5)
    with open(get_rotated_log_path(log_file, 4), "r", encoding="utf-8") as f:
        assert f.read() == "Batch1"
    with open(get_rotated_log_path(log_file, 3), "r", encoding="utf-8") as f:
        assert f.read() == "Batch2"
    with open(get_rotated_log_path(log_file, 2), "r", encoding="utf-8") as f:
        assert f.read() == "Batch3"

    # Cycle 4: Batch1 -> 5, Batch2 -> 4, Batch3 -> 3, Batch4 -> 2
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Batch4")
    rotate_log_file(log_file, max_bytes=5)
    with open(get_rotated_log_path(log_file, 5), "r", encoding="utf-8") as f:
        assert f.read() == "Batch1"
    with open(get_rotated_log_path(log_file, 4), "r", encoding="utf-8") as f:
        assert f.read() == "Batch2"
    with open(get_rotated_log_path(log_file, 3), "r", encoding="utf-8") as f:
        assert f.read() == "Batch3"
    with open(get_rotated_log_path(log_file, 2), "r", encoding="utf-8") as f:
        assert f.read() == "Batch4"


def test_f_assistant5_deleted_when_max_count_reached(tmp_path):
    """Test F: assistant5.log is deleted when the maximum rotation count is reached."""
    log_file = str(tmp_path / "assistant.log")

    # 4 rotations fill slots 2, 3, 4, 5
    for i in range(1, 5):
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Batch{i}")
        rotate_log_file(log_file, max_bytes=5, max_files=5)

    # At this point (4 rotations):
    # assistant5.log has Batch1
    # assistant4.log has Batch2
    # assistant3.log has Batch3
    # assistant2.log has Batch4
    assert open(get_rotated_log_path(log_file, 5)).read() == "Batch1"
    assert open(get_rotated_log_path(log_file, 4)).read() == "Batch2"
    assert open(get_rotated_log_path(log_file, 3)).read() == "Batch3"
    assert open(get_rotated_log_path(log_file, 2)).read() == "Batch4"

    # Now 5th rotation with Batch5:
    # assistant5.log (Batch1) must be DELETED
    # assistant5.log receives Batch2
    # assistant4.log receives Batch3
    # assistant3.log receives Batch4
    # assistant2.log receives Batch5
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Batch5")
    rotate_log_file(log_file, max_bytes=5, max_files=5)

    assert open(get_rotated_log_path(log_file, 5)).read() == "Batch2"
    assert open(get_rotated_log_path(log_file, 4)).read() == "Batch3"
    assert open(get_rotated_log_path(log_file, 3)).read() == "Batch4"
    assert open(get_rotated_log_path(log_file, 2)).read() == "Batch5"

    # Ensure no assistant6.log exists
    rot6 = get_rotated_log_path(log_file, 6)
    assert not os.path.exists(rot6)


def test_g_same_behavior_works_for_whatsapp_bridge_log(tmp_path):
    """Test G: Same behavior works independently for whatsapp_bridge.log."""
    bridge_log = str(tmp_path / "whatsapp_bridge.log")

    for i in range(1, 4):
        with open(bridge_log, "w", encoding="utf-8") as f:
            f.write(f"BridgeData{i}")
        rotate_log_file(bridge_log, max_bytes=5, max_files=5, use_copytruncate=True)

    rot2 = get_rotated_log_path(bridge_log, 2)
    rot3 = get_rotated_log_path(bridge_log, 3)
    rot4 = get_rotated_log_path(bridge_log, 4)

    assert os.path.exists(rot2)
    assert os.path.exists(rot3)
    assert os.path.exists(rot4)
    # Most recent rotated file is rot2 (BridgeData3)
    assert open(rot2).read() == "BridgeData3"
    assert open(rot3).read() == "BridgeData2"
    assert open(rot4).read() == "BridgeData1"
    # Active file exists and was truncated
    assert os.path.exists(bridge_log)
    assert os.path.getsize(bridge_log) == 0


def test_h_files_older_than_30_minutes_are_removed(tmp_path):
    """Test H: Rotated files older than 30 minutes (1800s) are removed."""
    log_file = str(tmp_path / "assistant.log")
    with open(log_file, "w") as f:
        f.write("active")

    rot2 = get_rotated_log_path(log_file, 2)
    rot3 = get_rotated_log_path(log_file, 3)
    rot4 = get_rotated_log_path(log_file, 4)

    with open(rot2, "w") as f:
        f.write("recent 10 min old")
    with open(rot3, "w") as f:
        f.write("old 35 min old")
    with open(rot4, "w") as f:
        f.write("ancient 60 min old")

    now = time.time()
    # rot2: 600s old (<1800s)
    os.utime(rot2, (now - 600, now - 600))
    # rot3: 2100s old (>1800s)
    os.utime(rot3, (now - 2100, now - 2100))
    # rot4: 3600s old (>1800s)
    os.utime(rot4, (now - 3600, now - 3600))

    deleted = cleanup_old_rotated_logs(log_file, max_age_seconds=1800, now=now)

    assert rot3 in deleted
    assert rot4 in deleted
    assert not os.path.exists(rot3)
    assert not os.path.exists(rot4)
    # rot2 must remain
    assert os.path.exists(rot2)


def test_i_current_active_assistant_log_is_not_removed_by_retention(tmp_path):
    """Test I: Current active assistant.log is NEVER removed by retention cleanup."""
    log_file = str(tmp_path / "assistant.log")
    with open(log_file, "w") as f:
        f.write("active app log content")

    # Set mtime to 3 hours ago (10800s)
    now = time.time()
    os.utime(log_file, (now - 10800, now - 10800))

    deleted = cleanup_old_rotated_logs(log_file, max_age_seconds=1800, now=now)

    assert log_file not in deleted
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        assert f.read() == "active app log content"


def test_j_current_active_whatsapp_bridge_log_is_not_removed_by_retention(tmp_path):
    """Test J: Current active whatsapp_bridge.log is NEVER removed by retention cleanup."""
    bridge_log = str(tmp_path / "whatsapp_bridge.log")
    with open(bridge_log, "w") as f:
        f.write("active bridge log content")

    now = time.time()
    os.utime(bridge_log, (now - 10800, now - 10800))

    deleted = cleanup_old_rotated_logs(bridge_log, max_age_seconds=1800, now=now)

    assert bridge_log not in deleted
    assert os.path.exists(bridge_log)
    with open(bridge_log, "r") as f:
        assert f.read() == "active bridge log content"


def test_k_missing_data_logs_directory_created_safely(tmp_path):
    """Test K: Missing log directory is created safely on startup/configuration."""
    nested_log = str(tmp_path / "nested" / "deep" / "logs" / "assistant.log")
    assert not os.path.exists(os.path.dirname(nested_log))

    configure_logging(log_file=nested_log, enable_console=False)
    logging.getLogger("test.nested").info("nested dir message")

    assert os.path.exists(os.path.dirname(nested_log))
    assert os.path.exists(nested_log)


def test_l_logger_initialization_does_not_duplicate_handlers(tmp_path):
    """Test L: Multiple configure_logging calls do not duplicate handlers."""
    log_file = str(tmp_path / "assistant.log")

    # Initialize 5 times
    for _ in range(5):
        configure_logging(log_file=log_file, enable_console=False, log_level="INFO")

    root = logging.getLogger()
    rot_handlers = [h for h in root.handlers if isinstance(h, ProductionRotatingFileHandler)]
    assert len(rot_handlers) == 1

    test_logger = logging.getLogger("test.idempotent")
    test_logger.info("UniqueMessageOnce")

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert content.count("UniqueMessageOnce") == 1


def test_m_streamlit_reruns_do_not_create_duplicate_handlers(tmp_path):
    """Test M: Simulated Streamlit reruns do not create duplicate handlers or duplicate log lines."""
    log_file = str(tmp_path / "assistant.log")

    # Simulate 10 Streamlit reruns
    for rerun in range(10):
        configure_logging(log_file=log_file, enable_console=False)

    root = logging.getLogger()
    assert len([h for h in root.handlers if isinstance(h, ProductionRotatingFileHandler)]) == 1

    logger = logging.getLogger("assistant.streamlit_test")
    logger.info("StreamlitActionLogged")

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert content.count("StreamlitActionLogged") == 1


def test_n_whatsapp_bridge_copytruncate_preserves_active_descriptor(tmp_path):
    """Test N: Subprocess stdout writes continue in active log after copytruncate rotation.

    Proves that an external process holding an open append file descriptor (such as Node.js)
    continues appending to whatsapp_bridge.log after rotation, and its new writes do not
    remain stuck in whatsapp_bridge2.log.
    """
    bridge_log = str(tmp_path / "whatsapp_bridge.log")
    rot2 = get_rotated_log_path(bridge_log, 2)

    # Open log file in append mode as done by WhatsAppConnector
    f_handle = open(bridge_log, "a", encoding="utf-8")

    # Launch subprocess writing to f_handle
    script = (
        "import time, sys\n"
        "for i in range(8):\n"
        "    print(f'BRIDGE_TICK_{i}', flush=True)\n"
        "    time.sleep(0.15)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=f_handle,
        stderr=f_handle,
    )

    # Let subprocess write ticks 0, 1, 2
    time.sleep(0.4)

    # Rotate with copytruncate=True
    rotated = rotate_log_file(bridge_log, max_bytes=10, max_files=5, use_copytruncate=True)
    assert rotated is True
    assert os.path.exists(rot2)

    # Wait for subprocess to complete writing remaining ticks
    proc.wait(timeout=5)
    f_handle.close()

    with open(rot2, "r", encoding="utf-8") as f:
        rot_content = f.read()
    with open(bridge_log, "r", encoding="utf-8") as f:
        active_content = f.read()

    # Pre-rotation ticks are in rotated file
    assert "BRIDGE_TICK_0" in rot_content
    # Post-rotation ticks MUST be in active file
    assert "BRIDGE_TICK_7" in active_content
    # Post-rotation ticks must NOT be in rotated file
    assert "BRIDGE_TICK_7" not in rot_content


def test_o_partial_rotation_chains_shift_safely(tmp_path):
    """Test O: Partially existing rotation chains shift without errors."""
    log_file = str(tmp_path / "assistant.log")
    rot4 = get_rotated_log_path(log_file, 4)
    rot5 = get_rotated_log_path(log_file, 5)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("ActiveBatch")
    with open(rot4, "w", encoding="utf-8") as f:
        f.write("Orphaned4")

    # Rotate when files 2 and 3 do not exist
    rotated = rotate_log_file(log_file, max_bytes=5, max_files=5)
    assert rotated is True

    # rot4 should have shifted to rot5
    assert os.path.exists(rot5)
    with open(rot5, "r", encoding="utf-8") as f:
        assert f.read() == "Orphaned4"

    # Active should have moved to rot2
    rot2 = get_rotated_log_path(log_file, 2)
    assert os.path.exists(rot2)
    with open(rot2, "r", encoding="utf-8") as f:
        assert f.read() == "ActiveBatch"
