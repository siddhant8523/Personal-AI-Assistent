"""
Production Logging Configuration & Log Rotation
================================================
Implements size-based rotation and age-based retention for application logs:
- Central active log: data/logs/assistant.log
- WhatsApp bridge log: data/logs/whatsapp_bridge.log
- Max file size: 5 MB (5 * 1024 * 1024 bytes)
- Rotation chain: assistant.log -> assistant2.log .. assistant5.log (max 5 files total)
- Retention: Rotated files older than 30 minutes are pruned based on mtime.
- Active logs (assistant.log and whatsapp_bridge.log) are NEVER deleted.
- Streamlit rerun safe: Idempotent initialization prevents duplicate handlers.

Deployment Safety & Limitations:
Streamlit Community Cloud operates a single application process per deployment.
For thread safety (e.g. concurrent Streamlit reruns/sessions), threading locks protect
initialization and rotation. For multi-process safety, non-blocking file locking (fcntl.flock)
prevents rotation chain corruption. True multi-worker distributed log aggregation is
outside single-node scope; Streamlit Cloud Logs remain available through the cloud dashboard.
"""

from __future__ import annotations

import fcntl
import logging
import logging.handlers
import os
import re
import shutil
import threading
import time
from typing import Any

# ============================================================
# CONSTANTS & DEFAULTS
# ============================================================

DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_MAX_FILES = 5  # 1 active + 4 rotated (indices 2, 3, 4, 5)
DEFAULT_RETENTION_SECONDS = 1800  # 30 minutes

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "data", "logs")
DEFAULT_ASSISTANT_LOG = os.path.join(DEFAULT_LOGS_DIR, "assistant.log")
DEFAULT_WHATSAPP_LOG = os.path.join(DEFAULT_LOGS_DIR, "whatsapp_bridge.log")

LOG_FORMAT = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

_CONFIG_LOCK = threading.Lock()


# ============================================================
# PATH & ROTATION HELPERS
# ============================================================

def get_rotated_log_path(log_path: str, index: int) -> str:
    """Returns the rotated log path for a given rotation index (2..5).

    Example:
        assistant.log, index 2 -> assistant2.log
        whatsapp_bridge.log, index 3 -> whatsapp_bridge3.log
    """
    dirname, basename = os.path.split(log_path)
    stem, ext = os.path.splitext(basename)
    return os.path.join(dirname, f"{stem}{index}{ext}")


def cleanup_old_rotated_logs(
    log_path: str,
    max_age_seconds: float = DEFAULT_RETENTION_SECONDS,
    max_files: int = DEFAULT_MAX_FILES,
    now: float | None = None,
) -> list[str]:
    """Deletes rotated log files older than max_age_seconds or with index > max_files.

    CRITICAL SAFETY RULES:
    1. NEVER deletes the active log file (log_path).
    2. Only inspects rotated files matching '{stem}{number}{ext}' (e.g. assistant2.log).
    3. Never touches unrelated files or directories (e.g. auth_state/).
    """
    dirname, basename = os.path.split(os.path.abspath(log_path))
    if not os.path.isdir(dirname):
        return []

    stem, ext = os.path.splitext(basename)
    # Matches e.g. assistant2.log, whatsapp_bridge4.log
    pattern = re.compile(rf"^{re.escape(stem)}(\d+){re.escape(ext)}$")
    current_time = now if now is not None else time.time()
    deleted_files: list[str] = []

    try:
        entries = os.listdir(dirname)
    except OSError:
        return []

    for filename in entries:
        # ABSOLUTE RULE: Never delete the active file
        if filename == basename:
            continue

        match = pattern.match(filename)
        if not match:
            continue

        try:
            rot_index = int(match.group(1))
        except ValueError:
            continue

        # Active file is index 1/un-indexed; rotated files start at 2
        if rot_index < 2:
            continue

        full_path = os.path.join(dirname, filename)
        try:
            mtime = os.path.getmtime(full_path)
            age = current_time - mtime

            # Delete if older than retention limit or exceeds max rotation count
            if age > max_age_seconds or rot_index > max_files:
                os.remove(full_path)
                deleted_files.append(full_path)
        except OSError:
            # Continue safely if file disappeared or permissions error
            pass

    return deleted_files


def rotate_log_file(
    log_path: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
    max_age_seconds: float = DEFAULT_RETENTION_SECONDS,
    use_copytruncate: bool = False,
    now: float | None = None,
) -> bool:
    """Performs size-based rotation and age retention on a log file.

    Rotation chain:
        {stem}5.log -> delete
        {stem}4.log -> {stem}5.log
        {stem}3.log -> {stem}4.log
        {stem}2.log -> {stem}3.log
        {stem}.log  -> {stem}2.log
        new active {stem}.log

    When use_copytruncate is True:
        Essential for external subprocesses (e.g. Node Baileys bridge) holding
        an open file descriptor. The file is copied to {stem}2.log and truncated
        to 0 bytes in-place so the child process's append-mode descriptor
        continues writing to the active log file rather than an orphaned inode.

    Returns True if rotation occurred, False otherwise.
    """
    log_path = os.path.abspath(log_path)
    if not os.path.isfile(log_path):
        return False

    try:
        file_size = os.path.getsize(log_path)
    except OSError:
        return False

    if file_size < max_bytes:
        return False

    dirname = os.path.dirname(log_path)
    os.makedirs(dirname, exist_ok=True)

    # Lightweight file lock to protect rotation chain against concurrent processes
    lock_path = f"{log_path}.rotlock"
    lock_fd = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another process is currently rotating; gracefully skip
            return False
    except OSError:
        # If lock file creation fails, proceed with best-effort rotation
        pass

    try:
        # 1. Clean up old rotated files prior to shifting
        cleanup_old_rotated_logs(log_path, max_age_seconds=max_age_seconds, max_files=max_files, now=now)

        # 2. Shift existing rotated files from max_files down to 2
        # Max file (e.g. assistant5.log) is deleted if it exists
        oldest_file = get_rotated_log_path(log_path, max_files)
        if os.path.exists(oldest_file):
            try:
                os.remove(oldest_file)
            except OSError:
                pass

        for i in range(max_files - 1, 1, -1):
            src = get_rotated_log_path(log_path, i)
            dst = get_rotated_log_path(log_path, i + 1)
            if os.path.exists(src):
                try:
                    os.replace(src, dst)
                except OSError:
                    pass

        # 3. Rotate active log to index 2
        dst2 = get_rotated_log_path(log_path, 2)

        if use_copytruncate:
            # Read and truncate in-place preserving inode for open subprocess descriptors
            try:
                with open(log_path, "rb+") as f:
                    content = f.read()
                    f.seek(0)
                    f.truncate(0)
                    f.flush()
                with open(dst2, "wb") as f_rot:
                    f_rot.write(content)
            except OSError:
                # Fallback to copy and truncate
                try:
                    shutil.copyfile(log_path, dst2)
                    with open(log_path, "r+", encoding="utf-8") as f:
                        f.truncate(0)
                except OSError:
                    return False
        else:
            try:
                os.replace(log_path, dst2)
            except OSError:
                return False

            # Touch fresh active file
            try:
                with open(log_path, "a", encoding="utf-8"):
                    pass
            except OSError:
                pass

        # 4. Final cleanup pass
        cleanup_old_rotated_logs(log_path, max_age_seconds=max_age_seconds, max_files=max_files, now=now)
        return True

    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except OSError:
                pass


# ============================================================
# CUSTOM LOGGING HANDLER
# ============================================================

class ProductionRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """Size-based rotating file handler with custom naming and age-based retention.

    - Max size: 5 MB
    - Rotated files: base2.log .. base5.log (max 5 files total)
    - Age retention: Rotated files older than 30 minutes are pruned.
    - Active log is never deleted.
    - Process and thread safe.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        maxBytes: int = DEFAULT_MAX_BYTES,
        maxFiles: int = DEFAULT_MAX_FILES,
        maxAgeSeconds: float = DEFAULT_RETENTION_SECONDS,
        encoding: str = "utf-8",
        delay: bool = False,
    ) -> None:
        super().__init__(filename, mode, encoding=encoding, delay=delay)
        self.maxBytes = maxBytes
        self.maxFiles = maxFiles
        self.maxAgeSeconds = maxAgeSeconds
        self._last_cleanup_time: float = 0.0

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        """Determines if the record will cause the file to exceed maxBytes."""
        if self.stream is None:
            self.stream = self._open()
        if self.maxBytes > 0:
            try:
                msg = "%s\n" % self.format(record)
                self.stream.seek(0, 2)  # Seek to end
                if self.stream.tell() + len(msg.encode(self.encoding or "utf-8")) >= self.maxBytes:
                    return True
            except (OSError, ValueError):
                pass
        return False

    def doRollover(self) -> None:
        """Performs rotation shift and age cleanup."""
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        try:
            # Shift files
            cleanup_old_rotated_logs(
                self.baseFilename,
                max_age_seconds=self.maxAgeSeconds,
                max_files=self.maxFiles,
            )

            oldest = get_rotated_log_path(self.baseFilename, self.maxFiles)
            if os.path.exists(oldest):
                try:
                    os.remove(oldest)
                except OSError:
                    pass

            for i in range(self.maxFiles - 1, 1, -1):
                src = get_rotated_log_path(self.baseFilename, i)
                dst = get_rotated_log_path(self.baseFilename, i + 1)
                if os.path.exists(src):
                    try:
                        os.replace(src, dst)
                    except OSError:
                        pass

            dst2 = get_rotated_log_path(self.baseFilename, 2)
            if os.path.exists(self.baseFilename):
                try:
                    os.replace(self.baseFilename, dst2)
                except OSError:
                    pass

            cleanup_old_rotated_logs(
                self.baseFilename,
                max_age_seconds=self.maxAgeSeconds,
                max_files=self.maxFiles,
            )
        except OSError:
            pass

        if not self.delay:
            self.stream = self._open()

    def emit(self, record: logging.LogRecord) -> None:
        """Emits a record with throttled periodic retention cleanup."""
        now = time.time()
        if now - self._last_cleanup_time > 60.0:
            self._last_cleanup_time = now
            try:
                cleanup_old_rotated_logs(
                    self.baseFilename,
                    max_age_seconds=self.maxAgeSeconds,
                    max_files=self.maxFiles,
                    now=now,
                )
            except OSError:
                pass

        super().emit(record)


# ============================================================
# SAFE LOGGING FILTER & SANITIZATION
# ============================================================

MAX_LOG_LINE_LENGTH = 4096

SENSITIVE_KEY_SUBSTRINGS = (
    "password",
    "secret",
    "token",
    "auth",
    "api_key",
    "apikey",
    "credentials",
    "phone",
    "phone_number",
)

HUGE_FIELDS = frozenset({
    "histNotification",
    "initialHistBootstrapInlinePayload",
    "mediaKey",
    "fileSha256",
    "fileEncSha256",
    "directPath",
    "encHandle",
    "clientHello",
    "helloMsg",
    "node",
    "ephemeral",
    "signal",
    "buffer",
    "raw_bytes",
    "payload",
})

RE_JID = re.compile(r"\b[0-9]{8,15}@s\.whatsapp\.net\b")
RE_LID = re.compile(r"\b[0-9]{12,18}@lid\b")
RE_PHONE = re.compile(r"(?<![A-Za-z0-9_])(?<!timestamp=)(?<!time=)(?<!ts=)\+?[0-9]{10,15}(?![A-Za-z0-9_])(?!\.\d)")
RE_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{15,}")
RE_SECRET_KV = re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"]+['\"]")


def sanitize_text(text: str) -> str:
    """Sanitizes text strings by redacting JIDs, LIDs, phone numbers, and secrets."""
    if not text:
        return ""
    text = RE_BEARER.sub("Bearer [REDACTED]", text)
    text = RE_SECRET_KV.sub(r"\1: [REDACTED]", text)
    text = RE_JID.sub("[REDACTED_JID]", text)
    text = RE_LID.sub("[REDACTED_LID]", text)
    text = RE_PHONE.sub("[REDACTED_NUMBER]", text)
    return text


def shallow_sanitize(val: Any, depth: int = 0) -> Any:
    """Sanitizes objects using strict order:
    1. Detect/remove known sensitive fields.
    2. Detect/remove known huge protocol/binary fields.
    3. Prevent recursive/nested object expansion (shallow inspection only).
    """
    if depth > 1:
        if isinstance(val, dict):
            return f"<dict len={len(val)}>"
        elif isinstance(val, (list, tuple, set)):
            return f"<{type(val).__name__} len={len(val)}>"
        return str(val)[:100]

    if isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            k_str = str(k).lower()
            # Step 1: Detect/remove known sensitive fields
            if any(s in k_str for s in SENSITIVE_KEY_SUBSTRINGS):
                sanitized[k] = "[REDACTED]"
            # Step 2: Detect/remove known huge protocol/binary fields
            elif str(k) in HUGE_FIELDS:
                continue
            else:
                sanitized[k] = shallow_sanitize(v, depth + 1)
        return sanitized

    if isinstance(val, (list, tuple)):
        # Limit collection size to prevent unbounded serialization
        items = val[:10]
        sanitized_items = [shallow_sanitize(x, depth + 1) for x in items]
        if len(val) > 10:
            sanitized_items.append(f"...<{len(val) - 10} more items>")
        return tuple(sanitized_items) if isinstance(val, tuple) else sanitized_items

    if isinstance(val, str):
        return sanitize_text(val)

    return val


class SafeLoggingFilter(logging.Filter):
    """Sanitizing logging filter enforcing strict 5-step sanitization order:
    1. Detect/remove known sensitive fields.
    2. Detect/remove known huge protocol/binary fields.
    3. Prevent recursive/nested object expansion.
    4. Apply maximum serialized/log-line size.
    5. Only then emit the final log message.

    Never truncates an unsanitized object.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Step 2 quick-check: Drop raw Baileys history sync payloads entirely
        if hasattr(record, "histNotification") or hasattr(record, "initialHistBootstrapInlinePayload"):
            return False

        try:
            # 1, 2, 3: Sanitize args and msg shallowly without deep recursion
            if record.args:
                if isinstance(record.args, dict):
                    # Check for huge protocol fields to discard
                    if any(k in HUGE_FIELDS for k in record.args):
                        record.args = {k: v for k, v in record.args.items() if k not in HUGE_FIELDS}
                    sanitized_args = shallow_sanitize(record.args)
                elif isinstance(record.args, (list, tuple)):
                    sanitized_args = tuple(shallow_sanitize(a) for a in record.args)
                else:
                    sanitized_args = shallow_sanitize(record.args)
            else:
                sanitized_args = ()

            if isinstance(record.msg, dict):
                # Step 2: Discard huge fields directly
                if any(k in HUGE_FIELDS for k in record.msg):
                    record.msg = {k: v for k, v in record.msg.items() if k not in HUGE_FIELDS}
                sanitized_msg = str(shallow_sanitize(record.msg))
            elif isinstance(record.msg, str):
                sanitized_msg = record.msg
            else:
                sanitized_msg = str(record.msg)

            # Format safely
            if sanitized_args:
                try:
                    formatted = sanitized_msg % sanitized_args
                except Exception:
                    formatted = f"{sanitized_msg} {sanitized_args}"
            else:
                formatted = sanitized_msg

            # Step 1: Detect/remove known sensitive strings in final rendered text
            formatted = sanitize_text(formatted)

            # Step 4: Apply maximum serialized/log-line size ONLY AFTER full sanitization
            if len(formatted) > MAX_LOG_LINE_LENGTH:
                formatted = formatted[:MAX_LOG_LINE_LENGTH] + " ... [TRUNCATED]"

            # Step 5: Only then emit the final log message
            record.msg = formatted
            record.args = ()
            return True

        except Exception:
            # Failsafe: Never crash the application due to logging filter
            record.msg = "[LOGGING SANITIZATION ERROR]"
            record.args = ()
            return True


# ============================================================
# CONFIGURATION & STREAMLIT RERUN SAFETY
# ============================================================

def configure_logging(
    log_file: str | None = None,
    log_level: str | int | None = None,
    enable_console: bool | None = None,
) -> None:
    """Configures application logging with rotation and retention.

    IDEMPOTENT & STREAMLIT RERUN SAFE:
    Inspects root logger handlers. If a ProductionRotatingFileHandler already exists
    for the target log_file, it is reused and updated in-place. Duplicate handlers
    are never added on Streamlit reruns.
    """
    with _CONFIG_LOCK:
        path = os.path.abspath(log_file or DEFAULT_ASSISTANT_LOG)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Initial retention cleanup on logger configuration
        cleanup_old_rotated_logs(path)

        level_name = log_level or os.environ.get("LOG_LEVEL", "INFO")
        if isinstance(level_name, str):
            level = getattr(logging, level_name.upper(), logging.INFO)
        else:
            level = level_name

        if enable_console is None:
            enable_console = os.environ.get("CONSOLE_LOGGING", "false").lower() in ("true", "1", "yes")

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Ensure SafeLoggingFilter is attached
        safe_filter = None
        for f in root_logger.filters:
            if isinstance(f, SafeLoggingFilter):
                safe_filter = f
                break
        if safe_filter is None:
            safe_filter = SafeLoggingFilter()
            root_logger.addFilter(safe_filter)

        # Check existing handlers to ensure no duplicates
        existing_rot_handler: ProductionRotatingFileHandler | None = None
        existing_console_handler: logging.StreamHandler | None = None
        handlers_to_remove: list[logging.Handler] = []

        for handler in root_logger.handlers:
            if isinstance(handler, ProductionRotatingFileHandler):
                if os.path.abspath(handler.baseFilename) == path:
                    existing_rot_handler = handler
                else:
                    handlers_to_remove.append(handler)
            elif type(handler) is logging.FileHandler:
                handlers_to_remove.append(handler)
            elif type(handler) is logging.StreamHandler:
                if existing_console_handler is None:
                    existing_console_handler = handler
                else:
                    handlers_to_remove.append(handler)

        for handler in handlers_to_remove:
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        # Configure or reuse rotating file handler
        if existing_rot_handler is None:
            existing_rot_handler = ProductionRotatingFileHandler(path, encoding="utf-8")
            existing_rot_handler.setFormatter(LOG_FORMAT)
            existing_rot_handler.setLevel(level)
            root_logger.addHandler(existing_rot_handler)
        else:
            existing_rot_handler.setLevel(level)
            existing_rot_handler.setFormatter(LOG_FORMAT)

        # Configure or reuse console handler
        if enable_console:
            import sys
            if existing_console_handler is None:
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setLevel(logging.ERROR)
                console_handler.setFormatter(LOG_FORMAT)
                root_logger.addHandler(console_handler)
            else:
                existing_console_handler.stream = sys.stderr
                existing_console_handler.setLevel(logging.ERROR)
                existing_console_handler.setFormatter(LOG_FORMAT)
        else:
            if existing_console_handler is not None:
                root_logger.removeHandler(existing_console_handler)

        # Silence verbose third-party loggers
        for noisy in [
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
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Specifically silence googleapiclient file_cache warning
        logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
