"""
Groq Speech-to-Text (STT) Service.
==================================
Reusable transcription service using Groq's Whisper API (`whisper-large-v3-turbo`).
Transcribes audio into text supporting multilingual input (English, Hindi, Hinglish, etc.).
"""

from __future__ import annotations

import logging
import mimetypes
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger("assistant.stt")

# Groq maximum upload file size is 25 MB
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024

# Audio formats officially supported by Groq Whisper API
SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".flac",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
})

# Known MIME types mapped directly to Groq-supported audio extensions
MIME_TO_EXTENSION: dict[str, str] = {
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/webm": ".webm",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mpga": ".mpga",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/ogg": ".ogg",
}

# Aliases and variants that cleanly map to a supported format container
EXTENSION_ALIASES: dict[str, str] = {
    ".oga": ".ogg",
    ".spx": ".ogg",
}


def resolve_supported_audio_extension(
    file_path_or_name: str | None = None,
    mime_type: str | None = None,
    default: str | None = None,
) -> str | None:
    """Resolve a safe, Groq-supported audio extension from a file path/name or MIME type.

    Returns the extension including leading dot (e.g. '.ogg') or None if it cannot be resolved.
    """
    # 1. Check file path or name extension
    if file_path_or_name:
        ext = Path(file_path_or_name).suffix.lower()
        if ext in EXTENSION_ALIASES:
            return EXTENSION_ALIASES[ext]
        if ext in SUPPORTED_AUDIO_EXTENSIONS:
            return ext

    # 2. Check MIME type (strip parameters like '; codecs=opus')
    if mime_type:
        clean_mime = mime_type.split(";")[0].strip().lower()
        if clean_mime in MIME_TO_EXTENSION:
            return MIME_TO_EXTENSION[clean_mime]
        for known_mime, target_ext in MIME_TO_EXTENSION.items():
            if known_mime in clean_mime:
                return target_ext

    # 3. Fallback to default if supplied
    if default:
        def_ext = default.lower() if default.startswith(".") else f".{default.lower()}"
        if def_ext in EXTENSION_ALIASES:
            return EXTENSION_ALIASES[def_ext]
        if def_ext in SUPPORTED_AUDIO_EXTENSIONS:
            return def_ext

    return None


class STTError(Exception):
    """Controlled error raised when speech-to-text processing fails."""


@contextmanager
def temp_audio_file(
    data: bytes,
    suffix: str = ".ogg",
    prefix: str = "audio_",
) -> Generator[str, None, None]:
    """Secure temporary audio file context manager with guaranteed cleanup.

    Writes the audio bytes to a temporary file, yields the file path, and guarantees
    unlinking the file in the finally block.
    """
    if not suffix.startswith("."):
        suffix = f".{suffix}"

    # Normalize aliases like .oga -> .ogg
    norm_suffix = resolve_supported_audio_extension(suffix, default=suffix) or suffix
    if not norm_suffix.startswith("."):
        norm_suffix = f".{norm_suffix}"

    tmp = tempfile.NamedTemporaryFile(suffix=norm_suffix, prefix=prefix, delete=False)
    tmp_path = tmp.name
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        yield tmp_path
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                logger.debug("Cleaned up temporary audio file: %s", tmp_path)
        except OSError as exc:
            logger.warning("Failed to clean up temporary audio file %s: %s", tmp_path, exc)


class GroqSTTService:
    """Reusable Speech-to-Text service backed by Groq Whisper."""

    DEFAULT_MODEL = "whisper-large-v3-turbo"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any = None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "").strip()
        self.model = model or os.getenv("GROQ_STT_MODEL", self.DEFAULT_MODEL).strip()
        self._client = client

        if self._client is None and self.api_key:
            try:
                from groq import Groq

                self._client = Groq(api_key=self.api_key)
            except ImportError:
                logger.warning("Groq SDK not installed; GroqSTTService running offline.")
                self._client = None
            except Exception as exc:
                logger.warning("Failed to initialize Groq client: %s", exc)
                self._client = None

    @property
    def is_available(self) -> bool:
        """Whether the STT service has an initialized client and API key."""
        return self._client is not None and bool(self.api_key)

    def transcribe_audio(
        self,
        audio_path: str,
        mime_type: str | None = None,
    ) -> str:
        """Transcribe an audio file at `audio_path` using Groq Whisper.

        Args:
            audio_path: Absolute or relative path to the audio file.
            mime_type: Optional MIME type of the audio file.

        Returns:
            The transcribed text string.

        Raises:
            STTError: If the file is invalid, missing, oversized, unsupported, or transcription fails.
        """
        if not self.is_available:
            raise STTError("Groq STT is not configured or GROQ_API_KEY is missing.")

        path = Path(audio_path)
        if not path.exists():
            raise STTError(f"Audio file does not exist: {audio_path}")

        if not path.is_file():
            raise STTError(f"Audio path is not a regular file: {audio_path}")

        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise STTError(f"Could not read audio file size: {exc}") from exc

        if file_size == 0:
            raise STTError("Audio file is empty (0 bytes).")

        if file_size > MAX_AUDIO_SIZE_BYTES:
            max_mb = MAX_AUDIO_SIZE_BYTES / (1024 * 1024)
            raise STTError(f"Audio file size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed size ({max_mb:.0f} MB).")

        file_ext = path.suffix.lower()
        supported_ext = resolve_supported_audio_extension(file_path_or_name=path.name, mime_type=mime_type)

        inferred_mime = mime_type
        if not inferred_mime:
            inferred_mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        if not supported_ext:
            raise STTError(
                f"Unsupported audio format '{file_ext or 'unknown'}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
            )

        # Ensure filename sent to Groq has a supported extension
        if file_ext == supported_ext:
            upload_filename = path.name
        else:
            upload_filename = f"{path.stem}{supported_ext}"

        # Requirement 2: SAFE diagnostic logging immediately before the Groq API request
        logger.info(
            "[STT] Diagnostic metadata before Groq request: "
            "temp_file_path=%s, filename=%s, file_extension=%s, file_size_bytes=%d, mime_type=%s, upload_filename=%s, model=%s",
            str(path),
            path.name,
            file_ext,
            file_size,
            inferred_mime,
            upload_filename,
            self.model,
        )

        try:
            with open(path, "rb") as f:
                transcription = self._client.audio.transcriptions.create(
                    file=(upload_filename, f.read()),
                    model=self.model,
                )

            text = getattr(transcription, "text", "") or ""
            text = text.strip()

            if not text:
                logger.warning("[STT] Transcription returned empty text for %s", upload_filename)
                raise STTError("Audio transcription produced an empty result.")

            logger.info("[STT] Transcription succeeded for %s (length=%d characters)", upload_filename, len(text))
            return text

        except STTError:
            raise
        except Exception as exc:
            logger.error("[STT] Groq transcription API failure: %s", exc)
            raise STTError(f"Speech-to-text failed: {exc}") from exc
