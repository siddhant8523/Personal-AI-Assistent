"""Unit tests for Telegram voice and audio message handling with STT."""

from unittest.mock import MagicMock
import pytest

from assistant.channels.telegram_channel import TelegramChannel
from assistant.ingestion.unified_message import Source
from assistant.stt.groq_stt import STTError


def test_telegram_channel_transcribes_voice_message():
    connector = MagicMock()
    connector.enabled = True
    router = MagicMock()
    stt_service = MagicMock()

    stt_service.transcribe_audio.return_value = "What is the weather today?"

    channel = TelegramChannel(
        connector=connector,
        router=router,
        agent_chat_id="123456",
        poll_seconds=0.001,
        stt_service=stt_service,
    )

    def mock_get_updates(offset):
        channel.stop()
        return [
            {
                "update_id": 101,
                "message": {
                    "chat": {"id": 123456},
                    "from": {"id": 123456},
                    "voice": {"file_id": "voice_file_abc123", "mime_type": "audio/ogg"},
                },
            }
        ]

    connector.get_updates.side_effect = mock_get_updates
    connector.get_file.return_value = {"file_path": "voice/file_0.oga"}
    connector.download_file.return_value = b"fake-opus-voice-bytes"

    mock_indicator = MagicMock()
    channel.get_presence_indicator = MagicMock(return_value=mock_indicator)

    thread = channel.start()
    thread.join(timeout=2)

    # Verify presence indicator was started and stopped
    mock_indicator.start.assert_called_once()
    mock_indicator.stop.assert_called_once()

    # Verify connector methods were called
    connector.get_file.assert_called_once_with("voice_file_abc123")
    connector.download_file.assert_called_once_with("voice/file_0.oga")
    stt_service.transcribe_audio.assert_called_once()

    # Verify audio file sent to transcribe_audio has .ogg extension (resolved from .oga)
    called_path = stt_service.transcribe_audio.call_args[0][0]
    assert called_path.endswith(".ogg"), f"Expected .ogg extension, got {called_path}"
    assert stt_service.transcribe_audio.call_args.kwargs.get("mime_type") == "audio/ogg"

    # Verify message was routed to router
    assert router.route.call_count == 1
    routed_msg = router.route.call_args[0][0]
    assert routed_msg.source == Source.TELEGRAM
    assert routed_msg.conversation_id == "telegram_bot:123456"
    assert routed_msg.content == "What is the weather today?"
    assert routed_msg.metadata.get("input_type") == "audio"
    assert routed_msg.metadata.get("message_type") == "voice"


def test_telegram_channel_transcribes_audio_file_with_mime():
    connector = MagicMock()
    connector.enabled = True
    router = MagicMock()
    stt_service = MagicMock()

    stt_service.transcribe_audio.return_value = "This is a podcast audio"

    channel = TelegramChannel(
        connector=connector,
        router=router,
        agent_chat_id="123456",
        poll_seconds=0.001,
        stt_service=stt_service,
    )

    def mock_get_updates(offset):
        channel.stop()
        return [
            {
                "update_id": 103,
                "message": {
                    "chat": {"id": 123456},
                    "from": {"id": 123456},
                    "audio": {
                        "file_id": "audio_file_m4a",
                        "file_name": "interview.m4a",
                        "mime_type": "audio/mp4",
                    },
                },
            }
        ]

    connector.get_updates.side_effect = mock_get_updates
    connector.get_file.return_value = {"file_path": "music/file_1.m4a"}
    connector.download_file.return_value = b"fake-m4a-bytes"

    mock_indicator = MagicMock()
    channel.get_presence_indicator = MagicMock(return_value=mock_indicator)

    thread = channel.start()
    thread.join(timeout=2)

    stt_service.transcribe_audio.assert_called_once()
    called_path = stt_service.transcribe_audio.call_args[0][0]
    assert called_path.endswith(".m4a"), f"Expected .m4a extension, got {called_path}"
    assert stt_service.transcribe_audio.call_args.kwargs.get("mime_type") == "audio/mp4"

    assert router.route.call_count == 1
    routed_msg = router.route.call_args[0][0]
    assert routed_msg.content == "This is a podcast audio"
    assert routed_msg.metadata.get("message_type") == "audio"


def test_telegram_channel_handles_stt_error_gracefully():
    connector = MagicMock()
    connector.enabled = True
    router = MagicMock()
    stt_service = MagicMock()

    stt_service.transcribe_audio.side_effect = STTError("API rate limit exceeded")

    channel = TelegramChannel(
        connector=connector,
        router=router,
        agent_chat_id="123456",
        poll_seconds=0.001,
        stt_service=stt_service,
    )

    def mock_get_updates(offset):
        channel.stop()
        return [
            {
                "update_id": 102,
                "message": {
                    "chat": {"id": 123456},
                    "from": {"id": 123456},
                    "voice": {"file_id": "voice_file_xyz", "mime_type": "audio/ogg"},
                },
            }
        ]

    connector.get_updates.side_effect = mock_get_updates
    connector.get_file.return_value = {"file_path": "voice/file_1.oga"}
    connector.download_file.return_value = b"fake-voice-bytes"

    mock_indicator = MagicMock()
    channel.get_presence_indicator = MagicMock(return_value=mock_indicator)

    thread = channel.start()
    thread.join(timeout=2)

    # Indicator must be stopped in finally block
    mock_indicator.start.assert_called_once()
    mock_indicator.stop.assert_called_once()

    # User should receive polite notice
    connector.send_message.assert_called_once()
    call_args = connector.send_message.call_args[0]
    assert call_args[0] == "123456"
    assert "could not transcribe your voice message" in call_args[1]

    # Nothing should be routed to agent pipeline
    router.route.assert_not_called()
