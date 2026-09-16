"""Telegram Bot control channel. It never handles personal-account chats."""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from assistant.channels.presence import PresenceIndicator, TelegramPresenceIndicator
from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage
from assistant.stt.groq_stt import STTError, resolve_supported_audio_extension, temp_audio_file

logger = logging.getLogger("assistant.channels.telegram")


class TelegramChannel:
    def __init__(
        self,
        connector,
        router,
        agent_chat_id: str = "",
        poll_seconds: float = 2,
        stt_service=None,
    ):
        self.connector, self.router = connector, router
        self.agent_chat_id, self.poll_seconds = str(agent_chat_id), poll_seconds
        self.stt_service = stt_service
        self._offset: int | None = None
        self._stop = threading.Event()

    def get_presence_indicator(
        self, chat_id: str | None = None, request_id: str | None = None, interval_seconds: float = 4.0
    ) -> PresenceIndicator:
        target = str(chat_id or self.agent_chat_id)
        return TelegramPresenceIndicator(
            connector=self.connector,
            chat_id=target,
            request_id=request_id,
            interval_seconds=interval_seconds,
        )

    def start(self) -> threading.Thread | None:
        if not self.connector.enabled:
            logger.info("Telegram Bot channel is disabled (TELEGRAM_ENABLED=false or no bot token).")
            return None
        thread = threading.Thread(target=self._run, name="telegram-bot-channel", daemon=True)
        thread.start()
        logger.info("Telegram Bot channel listener started (agent_chat_id=%s).", self.agent_chat_id or "all")
        return thread

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                updates = self.connector.get_updates(self._offset)
                for update in updates:
                    self._offset = max(self._offset or 0, update.get("update_id", 0) + 1)
                    raw = update.get("message", {})
                    if not raw:
                        continue
                    chat_id = str(raw.get("chat", {}).get("id", ""))
                    if self.agent_chat_id and chat_id != self.agent_chat_id:
                        continue

                    text = raw.get("text", "")
                    metadata = {}
                    voice = raw.get("voice") or raw.get("audio")

                    if not text and voice:
                        file_id = voice.get("file_id")
                        if file_id and self.stt_service:
                            indicator = self.get_presence_indicator(chat_id=chat_id)
                            indicator.start()
                            try:
                                file_info = self.connector.get_file(file_id)
                                file_path = file_info.get("file_path", "")
                                if file_path:
                                    mime_type = voice.get("mime_type") or ("audio/ogg" if raw.get("voice") else "")
                                    ext = resolve_supported_audio_extension(
                                        file_path_or_name=file_path or voice.get("file_name"),
                                        mime_type=mime_type,
                                        default=".ogg" if raw.get("voice") else ".mp3",
                                    ) or ".ogg"
                                    audio_bytes = self.connector.download_file(file_path)
                                    with temp_audio_file(audio_bytes, suffix=ext) as tmp_audio:
                                        text = self.stt_service.transcribe_audio(tmp_audio, mime_type=mime_type)
                                        metadata = {
                                            "input_type": "audio",
                                            "message_type": "voice" if raw.get("voice") else "audio",
                                        }
                            except STTError as exc:
                                print(f"[STT ERROR] {type(exc).__name__}: {exc}", flush=True)
                                traceback.print_exc()
                                logger.warning("Telegram STT transcription error: %s", exc)
                                self.connector.send_message(
                                    chat_id,
                                    "Sorry, I could not transcribe your voice message. Please try again or send text.",
                                )
                                continue
                            except Exception as exc:
                                logger.error("Failed downloading/transcribing Telegram audio: %s", exc)
                                self.connector.send_message(
                                    chat_id,
                                    "Sorry, I encountered an issue processing your audio message.",
                                )
                                continue
                            finally:
                                indicator.stop()

                    if not text:
                        continue

                    sender = str(raw.get("from", {}).get("id", chat_id))
                    msg = UnifiedMessage(
                        source=Source.TELEGRAM,
                        conversation_id=f"telegram_bot:{chat_id}",
                        sender=sender,
                        content=text,
                        metadata=metadata,
                        origin=Origin.USER,
                    )
                    self.router.route(msg)
            except (ConnectionResetError, ConnectionError, OSError) as exc:
                logger.debug("Telegram Bot listener connection lost (%s). Retrying in %ss...", exc, self.poll_seconds)
                time.sleep(3)
            except Exception as exc:
                exc_str = str(exc)
                if "Connection reset" in exc_str or "Connection aborted" in exc_str or "Read timed out" in exc_str:
                    logger.debug("Telegram Bot listener network glitch (%s). Retrying...", exc_str)
                    time.sleep(3)
                else:
                    logger.info("Error in Telegram Bot listener loop: %s", exc)

    def reply(self, original: UnifiedMessage, text: str) -> None:
        target = original.conversation_id.removeprefix("telegram_bot:")
        if len(text) > 4096:
            for i in range(0, len(text), 4096):
                self.connector.send_message(target, text[i:i + 4096])
        else:
            self.connector.send_message(target, text)

