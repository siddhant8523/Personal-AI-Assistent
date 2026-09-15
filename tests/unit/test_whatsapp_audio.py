"""Unit tests for WhatsApp voice and audio message handling with STT."""

from unittest.mock import MagicMock, patch
import pytest

from assistant.connectors.whatsapp.whatsapp_connector import WhatsAppConnector
from assistant.ingestion.adapters.whatsapp_adapter import adapt_whatsapp
from assistant.ingestion.unified_message import Source
from assistant.stt.groq_stt import STTError


def test_download_media_disabled():
    connector = WhatsAppConnector(bridge_url="http://localhost:3000", enabled=False)
    assert connector.download_media("msg_123") == b""


def test_download_media_success():
    connector = WhatsAppConnector(bridge_url="http://localhost:3000", enabled=True)
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.content = b"sample-audio-bytes"
        mock_resp.ok = True
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = connector.download_media("msg_123")
        assert result == b"sample-audio-bytes"
        mock_get.assert_called_once_with("http://localhost:3000/media/msg_123", timeout=30)


def test_download_media_failure():
    connector = WhatsAppConnector(bridge_url="http://localhost:3000", enabled=True)
    with patch("requests.get", side_effect=Exception("Connection refused")):
        result = connector.download_media("msg_123")
        assert result == b""


def test_whatsapp_fetch_recent_transcribes_audio_ogg_opus():
    stt_service = MagicMock()
    stt_service.transcribe_audio.return_value = "Turn off the lights please"

    connector = WhatsAppConnector(
        bridge_url="http://localhost:3000",
        enabled=True,
        stt_service=stt_service,
    )

    recent_payload = [
        {
            "id": "3EB0VOICE123",
            "chat_id": "52909752496163@lid",
            "sender": "52909752496163@lid",
            "from_me": False,
            "text": "",
            "media_type": "audio",
            "mimetype": "audio/ogg; codecs=opus",
        }
    ]

    with patch("requests.get") as mock_get, patch.object(connector, "download_media", return_value=b"fake-audio") as mock_dl:
        mock_resp = MagicMock()
        mock_resp.json.return_value = recent_payload
        mock_resp.ok = True
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        items = connector.fetch_recent()

        assert len(items) == 1
        m = items[0]
        assert m["id"] == "3EB0VOICE123"
        assert m["text"] == "Turn off the lights please"
        assert m["input_type"] == "audio"
        assert m["message_type"] == "voice"

        mock_dl.assert_called_once_with("3EB0VOICE123")
        stt_service.transcribe_audio.assert_called_once()
        called_path = stt_service.transcribe_audio.call_args[0][0]
        assert called_path.endswith(".ogg"), f"Expected .ogg extension, got {called_path}"
        assert stt_service.transcribe_audio.call_args.kwargs.get("mime_type") == "audio/ogg; codecs=opus"

        # Check normalization
        unified = adapt_whatsapp(m)
        assert unified.source == Source.WHATSAPP
        assert unified.conversation_id == "whatsapp:52909752496163@lid"
        assert unified.content == "Turn off the lights please"
        assert unified.metadata.get("input_type") == "audio"
        assert unified.metadata.get("message_type") == "voice"


def test_whatsapp_fetch_recent_transcribes_audio_opus_format():
    stt_service = MagicMock()
    stt_service.transcribe_audio.return_value = "This is a direct opus voice note"

    connector = WhatsAppConnector(
        bridge_url="http://localhost:3000",
        enabled=True,
        stt_service=stt_service,
    )

    recent_payload = [
        {
            "id": "3EB0OPUS123",
            "chat_id": "52909752496163@lid",
            "sender": "52909752496163@lid",
            "from_me": False,
            "text": "",
            "media_type": "audio",
            "mimetype": "audio/opus",
        }
    ]

    with patch("requests.get") as mock_get, patch.object(connector, "download_media", return_value=b"fake-opus-data"):
        mock_resp = MagicMock()
        mock_resp.json.return_value = recent_payload
        mock_resp.ok = True
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        items = connector.fetch_recent()

        assert len(items) == 1
        stt_service.transcribe_audio.assert_called_once()
        called_path = stt_service.transcribe_audio.call_args[0][0]
        assert called_path.endswith(".opus"), f"Expected .opus extension, got {called_path}"


def test_whatsapp_fetch_recent_transcribes_audio_mp4_format():
    stt_service = MagicMock()
    stt_service.transcribe_audio.return_value = "Audio recording in mp4"

    connector = WhatsAppConnector(
        bridge_url="http://localhost:3000",
        enabled=True,
        stt_service=stt_service,
    )

    recent_payload = [
        {
            "id": "3EB0MP4AUDIO",
            "chat_id": "52909752496163@lid",
            "sender": "52909752496163@lid",
            "from_me": False,
            "text": "",
            "media_type": "audio",
            "mimetype": "audio/mp4",
        }
    ]

    with patch("requests.get") as mock_get, patch.object(connector, "download_media", return_value=b"fake-mp4-data"):
        mock_resp = MagicMock()
        mock_resp.json.return_value = recent_payload
        mock_resp.ok = True
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        items = connector.fetch_recent()

        assert len(items) == 1
        stt_service.transcribe_audio.assert_called_once()
        called_path = stt_service.transcribe_audio.call_args[0][0]
        assert called_path.endswith(".m4a"), f"Expected .m4a extension, got {called_path}"


def test_whatsapp_fetch_recent_handles_stt_failure():
    stt_service = MagicMock()
    stt_service.transcribe_audio.side_effect = STTError("Transcription failed")

    connector = WhatsAppConnector(
        bridge_url="http://localhost:3000",
        enabled=True,
        stt_service=stt_service,
    )

    recent_payload = [
        {
            "id": "3EB0FAILAUDIO",
            "chat_id": "52909752496163@lid",
            "sender": "52909752496163@lid",
            "from_me": False,
            "text": "",
            "media_type": "audio",
            "mimetype": "audio/ogg; codecs=opus",
        }
    ]

    with patch("requests.get") as mock_get, \
         patch.object(connector, "download_media", return_value=b"fake-audio"), \
         patch.object(connector, "send_message") as mock_send:

        mock_resp = MagicMock()
        mock_resp.json.return_value = recent_payload
        mock_resp.ok = True
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        items = connector.fetch_recent()

        # Since text remained empty, items should be empty
        assert len(items) == 0
        # Agent user should receive notice
        mock_send.assert_called_once()
        call_args = mock_send.call_args[0]
        assert call_args[0] == "52909752496163@lid"
        assert "could not transcribe your voice message" in call_args[1]
