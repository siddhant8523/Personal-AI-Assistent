"""Unit tests for Groq Speech-to-Text service."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from assistant.stt.groq_stt import (
    GroqSTTService,
    STTError,
    resolve_supported_audio_extension,
    temp_audio_file,
    SUPPORTED_AUDIO_EXTENSIONS,
)


def test_resolve_supported_audio_extension():
    # Direct extensions
    assert resolve_supported_audio_extension(file_path_or_name="voice.oga") == ".ogg"
    assert resolve_supported_audio_extension(file_path_or_name="voice.spx") == ".ogg"
    assert resolve_supported_audio_extension(file_path_or_name="audio.mp3") == ".mp3"
    assert resolve_supported_audio_extension(file_path_or_name="recording.m4a") == ".m4a"
    assert resolve_supported_audio_extension(file_path_or_name="note.opus") == ".opus"
    assert resolve_supported_audio_extension(file_path_or_name="track.flac") == ".flac"
    assert resolve_supported_audio_extension(file_path_or_name="sound.wav") == ".wav"
    assert resolve_supported_audio_extension(file_path_or_name="clip.webm") == ".webm"

    # MIME types
    assert resolve_supported_audio_extension(mime_type="audio/ogg; codecs=opus") == ".ogg"
    assert resolve_supported_audio_extension(mime_type="audio/opus") == ".opus"
    assert resolve_supported_audio_extension(mime_type="audio/mp4") == ".m4a"
    assert resolve_supported_audio_extension(mime_type="audio/x-m4a") == ".m4a"
    assert resolve_supported_audio_extension(mime_type="audio/mpeg") == ".mp3"
    assert resolve_supported_audio_extension(mime_type="audio/wav") == ".wav"
    assert resolve_supported_audio_extension(mime_type="audio/flac") == ".flac"
    assert resolve_supported_audio_extension(mime_type="audio/webm") == ".webm"

    # Unknown format without default returns None
    assert resolve_supported_audio_extension(file_path_or_name="file.xyz") is None
    assert resolve_supported_audio_extension(mime_type="application/octet-stream") is None

    # Fallback to default
    assert resolve_supported_audio_extension(file_path_or_name="unknown_file", default=".ogg") == ".ogg"


def test_groq_stt_not_available_without_key():
    service = GroqSTTService(api_key="")
    assert not service.is_available
    with pytest.raises(STTError, match="Groq STT is not configured or GROQ_API_KEY is missing"):
        service.transcribe_audio("any_file.ogg")


def test_groq_stt_transcribe_success(tmp_path, caplog):
    audio_file = tmp_path / "test.ogg"
    audio_file.write_bytes(b"fake audio data")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Hello this is a test audio transcription"
    mock_client.audio.transcriptions.create.return_value = mock_resp

    service = GroqSTTService(api_key="gsk-test-key", model="whisper-large-v3-turbo", client=mock_client)

    with caplog.at_level(logging.INFO):
        result = service.transcribe_audio(str(audio_file))

    assert result == "Hello this is a test audio transcription"
    mock_client.audio.transcriptions.create.assert_called_once()
    call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == "whisper-large-v3-turbo"
    assert call_kwargs["file"][0] == "test.ogg"
    assert call_kwargs["file"][1] == b"fake audio data"

    # Requirement 2: Check safe diagnostic log
    assert "Diagnostic metadata before Groq request" in caplog.text
    assert "filename=test.ogg" in caplog.text
    assert "file_extension=.ogg" in caplog.text
    assert "file_size_bytes=15" in caplog.text


def test_groq_stt_normalizes_oga_to_ogg_for_groq(tmp_path):
    # Telegram voice file with .oga extension
    audio_file = tmp_path / "file_0.oga"
    audio_file.write_bytes(b"telegram-opus-audio")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Transcribed OGA voice note"
    mock_client.audio.transcriptions.create.return_value = mock_resp

    service = GroqSTTService(api_key="gsk-test-key", client=mock_client)
    result = service.transcribe_audio(str(audio_file), mime_type="audio/ogg")

    assert result == "Transcribed OGA voice note"
    mock_client.audio.transcriptions.create.assert_called_once()
    upload_filename = mock_client.audio.transcriptions.create.call_args.kwargs["file"][0]
    # Must be mapped to .ogg so Groq accepts it
    assert upload_filename == "file_0.ogg"


def test_groq_stt_unsupported_extension(tmp_path):
    audio_file = tmp_path / "test.xyz"
    audio_file.write_bytes(b"random-bytes")

    service = GroqSTTService(api_key="gsk-test-key", client=MagicMock())
    with pytest.raises(STTError, match="Unsupported audio format"):
        service.transcribe_audio(str(audio_file))


def test_groq_stt_file_not_found():
    mock_client = MagicMock()
    service = GroqSTTService(api_key="gsk-test-key", client=mock_client)
    with pytest.raises(STTError, match="Audio file does not exist"):
        service.transcribe_audio("/non/existent/path/audio.ogg")


def test_groq_stt_empty_audio_file(tmp_path):
    audio_file = tmp_path / "empty.ogg"
    audio_file.write_bytes(b"")

    mock_client = MagicMock()
    service = GroqSTTService(api_key="gsk-test-key", client=mock_client)
    with pytest.raises(STTError, match="Audio file is empty"):
        service.transcribe_audio(str(audio_file))


def test_groq_stt_file_oversized(tmp_path, monkeypatch):
    audio_file = tmp_path / "oversized.ogg"
    audio_file.write_bytes(b"x" * 100)

    import assistant.stt.groq_stt as groq_stt_mod
    monkeypatch.setattr(groq_stt_mod, "MAX_AUDIO_SIZE_BYTES", 50)

    mock_client = MagicMock()
    service = GroqSTTService(api_key="gsk-test-key", client=mock_client)
    with pytest.raises(STTError, match="exceeds maximum allowed size"):
        service.transcribe_audio(str(audio_file))


def test_groq_stt_empty_transcript_from_api(tmp_path):
    audio_file = tmp_path / "test.ogg"
    audio_file.write_bytes(b"valid audio bytes")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "   "
    mock_client.audio.transcriptions.create.return_value = mock_resp

    service = GroqSTTService(api_key="gsk-test-key", client=mock_client)
    with pytest.raises(STTError, match="Audio transcription produced an empty result"):
        service.transcribe_audio(str(audio_file))


def test_groq_stt_api_error_wrapped(tmp_path):
    audio_file = tmp_path / "test.ogg"
    audio_file.write_bytes(b"valid audio bytes")

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.side_effect = RuntimeError("API Connection timeout")

    service = GroqSTTService(api_key="gsk-test-key", client=mock_client)
    with pytest.raises(STTError, match="Speech-to-text failed: API Connection timeout"):
        service.transcribe_audio(str(audio_file))


def test_temp_audio_file_cleanup_on_success():
    created_path = None
    with temp_audio_file(b"test audio content", suffix=".ogg") as path:
        created_path = path
        assert os.path.exists(created_path)
        assert os.path.isfile(created_path)
        assert path.endswith(".ogg")
        with open(created_path, "rb") as f:
            assert f.read() == b"test audio content"

    assert not os.path.exists(created_path)


def test_temp_audio_file_normalizes_oga_suffix():
    created_path = None
    with temp_audio_file(b"test audio content", suffix=".oga") as path:
        created_path = path
        assert path.endswith(".ogg")

    assert not os.path.exists(created_path)


def test_temp_audio_file_cleanup_on_exception():
    created_path = None
    with pytest.raises(ValueError, match="Intentional failure"):
        with temp_audio_file(b"test audio content", suffix=".m4a") as path:
            created_path = path
            assert os.path.exists(created_path)
            assert path.endswith(".m4a")
            raise ValueError("Intentional failure")

    assert not os.path.exists(created_path)
