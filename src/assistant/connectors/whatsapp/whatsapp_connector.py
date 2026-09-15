"""
WhatsApp Connector (Part 1, Section 8; Part 3, Section 9)
==============================================================
Baileys itself is a Node.js library (unofficial WhatsApp Web protocol),
so the Python side talks to a small local Node bridge process over HTTP
(see baileys_bridge/index.js) rather than embedding Baileys directly.
This isolation is deliberate -- Part 1, Section 8: "the connector must
be isolated behind a WhatsApp-specific interface so the rest of the
architecture does not depend directly on Baileys internals."

In mock mode (bridge not running / WHATSAPP_ENABLED=false), send() logs
instead of calling out, and fetch_recent() returns nothing -- CLI-driven
/sim commands are used to exercise the pipeline in that case.
"""

from __future__ import annotations
from typing import Any

import atexit
import logging
import os
import subprocess
import time
import requests

from assistant.logging_config import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_FILES,
    DEFAULT_RETENTION_SECONDS,
    DEFAULT_WHATSAPP_LOG,
    cleanup_old_rotated_logs,
    rotate_log_file,
)

logger = logging.getLogger("assistant.whatsapp")


class WhatsAppConnector:
    def __init__(
        self,
        bridge_url: str,
        enabled: bool = False,
        stt_service=None,
        contact_cache=None,
        device_gateway=None,
    ):
        self.bridge_url = bridge_url.rstrip("/")
        self.enabled = enabled
        self.stt_service = stt_service
        if contact_cache is None:
            try:
                from assistant.contacts.contact_cache import ContactCache
                self.contact_cache = ContactCache()
            except Exception:
                self.contact_cache = None
        else:
            self.contact_cache = contact_cache
        self.device_gateway = device_gateway
        self._process: subprocess.Popen | None = None
        self._log_file_handle = None
        self._recent_messages_history: list[dict] = []
        self._last_rotation_check: float = 0.0
        self._log_path = DEFAULT_WHATSAPP_LOG

    def set_device_gateway(self, device_gateway) -> None:
        """Attach or update the DeviceGateway instance for Android Contacts lookups."""
        self.device_gateway = device_gateway

    def set_contact_cache(self, contact_cache) -> None:
        """Attach or update the ContactCache instance."""
        self.contact_cache = contact_cache

    def invalidate_recipient(self, target: str) -> None:
        """Invalidate a cached contact entry by name, phone number, or JID."""
        if self.contact_cache is not None:
            from assistant.contacts.contact_cache import ContactCache
            norm = ContactCache.normalize_name(target)
            self.contact_cache.delete(norm)
            for entry in self.contact_cache.all_entries():
                if (
                    entry.resolved_jid == target
                    or entry.phone_number == target
                    or entry.display_name.lower() == target.lower()
                ):
                    self.contact_cache.delete(entry.normalized_name)

    def refresh_contact(self, name: str) -> tuple[str | None, str | None]:
        """Invalidates cached entry, re-queries Android Contacts, and returns new resolved JID."""
        self.invalidate_recipient(name)
        return self.resolve_recipient(name, force_refresh=True)

    def check_bridge_log_rotation(
        self,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_files: int = DEFAULT_MAX_FILES,
        max_age_seconds: float = DEFAULT_RETENTION_SECONDS,
        force: bool = False,
    ) -> bool:
        """Rotates whatsapp_bridge.log if size >= max_bytes and prunes rotated files older than 30m.

        Uses copytruncate so the Node child process's open file descriptor continues
        appending to the active whatsapp_bridge.log without needing a bridge restart.
        """
        now = time.time()
        if not force and (now - getattr(self, "_last_rotation_check", 0.0)) < 30.0:
            return False
        self._last_rotation_check = now

        log_path = getattr(self, "_log_path", DEFAULT_WHATSAPP_LOG)
        try:
            return rotate_log_file(
                log_path,
                max_bytes=max_bytes,
                max_files=max_files,
                max_age_seconds=max_age_seconds,
                use_copytruncate=True,
            )
        except Exception as exc:
            logger.debug("[WhatsApp Connector] Error checking bridge log rotation: %s", exc)
            return False

    def get_status(self) -> dict[str, Any]:
        """Returns real status of the Baileys bridge and WhatsApp connection."""
        self.check_bridge_log_rotation()
        is_process_alive = bool(self._process and self._process.poll() is None)
        try:
            resp = requests.get(f"{self.bridge_url}/health", timeout=1.5)
            if resp.ok:
                data = resp.json()
                return {
                    "running": True,
                    "connected": bool(data.get("connected")),
                    "reconnecting": bool(data.get("reconnecting")),
                    "qr": data.get("qr"),
                    "agent_chat_id": data.get("agent_chat_id"),
                }
        except Exception:
            pass
        return {
            "running": is_process_alive,
            "connected": False,
            "reconnecting": False,
            "qr": None,
            "agent_chat_id": None,
        }

    def start_bridge(self, force: bool = False) -> bool:
        """Starts the Baileys bridge process idempotently.

        Safe across repeated calls and Streamlit reruns.
        """
        if force:
            self.enabled = True

        if not self.enabled:
            return False

        # Guard: check if process handle is already alive
        if self._process is not None and self._process.poll() is None:
            logger.info("[WhatsApp Connector] Baileys bridge process is already running (pid=%s)", self._process.pid)
            return True

        # Guard: check if an existing bridge instance is already responding on port
        try:
            resp = requests.get(f"{self.bridge_url}/health", timeout=1)
            if resp.ok:
                logger.info("[WhatsApp Connector] Baileys bridge is already running at %s", self.bridge_url)
                return True
        except Exception as exc:
            logger.debug("[WhatsApp Connector] Initial bridge health check failed (will attempt startup): %s", exc)

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        bridge_dir = os.path.join(root_dir, "src", "assistant", "connectors", "whatsapp", "baileys_bridge")
        log_path = getattr(self, "_log_path", DEFAULT_WHATSAPP_LOG)
        logs_dir = os.path.dirname(log_path)
        os.makedirs(logs_dir, exist_ok=True)

        # Pre-start rotation & retention check
        try:
            rotate_log_file(log_path, use_copytruncate=True)
            cleanup_old_rotated_logs(log_path)
        except Exception as exc:
            logger.debug("[WhatsApp Connector] Pre-start log rotation error: %s", exc)

        logger.info("[WhatsApp Connector] Starting Baileys Node bridge process (logs -> data/logs/whatsapp_bridge.log)...")

        self._log_file_handle = open(log_path, "a", encoding="utf-8")

        self._process = subprocess.Popen(
            ["node", "index.js"],
            cwd=bridge_dir,
            stdout=self._log_file_handle,
            stderr=self._log_file_handle,
            env=os.environ.copy(),
        )

        atexit.register(self.stop_bridge)

        start_time = time.time()
        max_wait = 20.0
        while time.time() - start_time < max_wait:
            if self._process.poll() is not None:
                logger.error("[WhatsApp Connector] Baileys bridge process exited prematurely (exit code: %s)", self._process.returncode)
                return False
            try:
                resp = requests.get(f"{self.bridge_url}/health", timeout=1)
                if resp.ok:
                    logger.info("[WhatsApp Connector] Baileys bridge started and healthy")
                    return True
            except Exception as exc:
                logger.debug("[WhatsApp Connector] Bridge health check poll failed: %s", exc)
            time.sleep(0.5)

        logger.warning("[WhatsApp Connector] Timeout waiting for Baileys bridge health check")
        return False

    def stop_bridge(self) -> None:
        if self._process:
            logger.info("[WhatsApp Connector] Stopping Baileys bridge process...")
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception as exc:
                logger.debug("[WhatsApp Connector] Terminate bridge failed, trying kill: %s", exc)
                try:
                    self._process.kill()
                except Exception as exc_k:
                    logger.debug("[WhatsApp Connector] Kill bridge process failed: %s", exc_k)
            self._process = None

        if self._log_file_handle:
            try:
                self._log_file_handle.close()
            except Exception as exc:
                logger.debug("[WhatsApp Connector] Closing bridge log file handle failed: %s", exc)
            self._log_file_handle = None


    def send_message(self, chat_id: str, text: str) -> dict:
        logger.debug("[WhatsApp] Sending message")
        if chat_id and not (chat_id.endswith("@s.whatsapp.net") or chat_id.endswith("@g.us") or chat_id.endswith("@lid")):
            res_jid, err = self.resolve_recipient(chat_id)
            if res_jid:
                chat_id = res_jid
            else:
                return {"status": "failed", "detail": "Invalid WhatsApp recipient"}

        if not self.enabled:
            logger.debug("[WhatsApp Connector] MOCK send")
            return {"status": "ok", "detail": f"[MOCK] would send WhatsApp"}

        try:
            logger.debug("[WhatsApp Connector] sending response to bridge")
            resp = requests.post(f"{self.bridge_url}/send", json={"chat_id": chat_id, "text": text}, timeout=10)
            if resp.ok:
                return {"status": "ok", "detail": resp.json()}
            if resp.status_code == 400 or "not found" in resp.text.lower() or "invalid recipient" in resp.text.lower():
                self.invalidate_recipient(chat_id)
            return {"status": "failed", "detail": resp.text}
        except Exception as exc:
            logger.error("[WhatsApp Connector] Failed to send message to bridge: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def send_presence_update(self, chat_id: str, state: str = "composing") -> dict:
        if chat_id and not (chat_id.endswith("@s.whatsapp.net") or chat_id.endswith("@g.us") or chat_id.endswith("@lid")):
            res_jid, err = self.resolve_recipient(chat_id)
            if res_jid:
                chat_id = res_jid
            else:
                return {"status": "failed", "detail": "Invalid WhatsApp recipient"}

        if not self.enabled:
            logger.debug("[WhatsApp Connector] MOCK presence update to %s: %s", chat_id, state)
            return {"status": "ok", "detail": f"[MOCK] would send WhatsApp presence {state} to {chat_id}"}

        try:
            resp = requests.post(f"{self.bridge_url}/presence", json={"chat_id": chat_id, "state": state}, timeout=10)
            if resp.ok:
                return {"status": "ok", "detail": resp.json()}
            return {"status": "failed", "detail": resp.text}
        except Exception as exc:
            logger.debug("[WhatsApp Connector] Failed to send presence update to bridge: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def is_agent_user_identity(self, jid: str, sender: str = "") -> bool:
        agent_setting = os.environ.get("WHATSAPP_AGENT_CHAT_ID", "919172767219@s.whatsapp.net").strip()
        agent_digits = "".join(filter(str.isdigit, agent_setting)) or "919172767219"
        agent_pn = f"{agent_digits}@s.whatsapp.net"

        jids_to_check = [str(jid or "").strip().lower(), str(sender or "").strip().lower()]
        for j in jids_to_check:
            if not j:
                continue
            if j == agent_setting.lower() or j == agent_pn.lower() or j == agent_digits:
                return True
            if j == "52909752496163@lid" or j == "52909752496163":
                return True
            if j.endswith("@s.whatsapp.net"):
                j_digits = "".join(filter(str.isdigit, j.split("@")[0]))
                if j_digits and j_digits == agent_digits:
                    return True
            if j.endswith("@lid"):
                lid_num = j.split("@")[0]
                if lid_num == "52909752496163":
                    return True
        return False

    def _get_msg_id(self, m: dict) -> str:
        msg_id = m.get("id") or m.get("message_id") or m.get("key_id")
        if msg_id:
            return str(msg_id)
        chat_id = str(m.get("chat_id", ""))
        sender = str(m.get("sender", ""))
        timestamp = str(m.get("timestamp", ""))
        text = str(m.get("text", ""))
        import hashlib
        return hashlib.sha256(f"{chat_id}:{sender}:{timestamp}:{text}".encode()).hexdigest()

    def download_media(self, msg_id: str) -> bytes:
        """Download decrypted media buffer for a message from the Baileys bridge."""
        if not self.enabled:
            return b""
        try:
            resp = requests.get(f"{self.bridge_url}/media/{msg_id}", timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            logger.error("[WhatsApp Connector] Failed to download media %s from bridge: %s", msg_id, exc)
            return b""

    def fetch_recent(self) -> list[dict]:
        if not self.enabled:
            return []
        try:
            resp = requests.get(f"{self.bridge_url}/recent", timeout=10)
            resp.raise_for_status()
            items = resp.json()

            if not hasattr(self, "_processed_message_ids"):
                self._processed_message_ids = set()

            new_items = []
            for m in items:
                msg_id = self._get_msg_id(m)
                chat_id = str(m.get("chat_id", ""))
                sender = str(m.get("sender", ""))
                from_me = bool(m.get("from_me", False))
                text = str(m.get("text", ""))

                is_dup = msg_id in self._processed_message_ids
                logger.debug(
                    "[WhatsApp] message_id=%s from_me=%s duplicate=%s media_type=%s",
                    msg_id, str(from_me).lower(), str(is_dup).lower(), m.get("media_type"),
                )

                if is_dup:
                    logger.debug("[WhatsApp] Ignoring already processed message")
                    continue

                self._processed_message_ids.add(msg_id)

                is_agent = self.is_agent_user_identity(chat_id, sender)

                # Process audio message if STT service is configured
                if m.get("media_type") == "audio" and not from_me:
                    logger.info("[WhatsApp] Processing inbound audio message")
                    if self.stt_service:
                        indicator = None
                        try:
                            from assistant.channels.presence import WhatsAppPresenceIndicator
                            indicator = WhatsAppPresenceIndicator(connector=self, jid=chat_id)
                            indicator.start()
                        except Exception as p_err:
                            logger.debug("[WhatsApp] Could not start presence indicator: %s", p_err)

                        try:
                            raw_id = m.get("id") or msg_id
                            audio_bytes = self.download_media(raw_id)
                            if audio_bytes:
                                from assistant.stt.groq_stt import (
                                    resolve_supported_audio_extension,
                                    temp_audio_file,
                                )
                                mimetype = m.get("mimetype", "")
                                ext = resolve_supported_audio_extension(
                                    mime_type=mimetype,
                                    default=".ogg" if not mimetype or "ogg" in mimetype or "opus" in mimetype else None,
                                )
                                if not ext:
                                    logger.warning("[WhatsApp] Unrecognized audio mimetype %r, defaulting to .ogg", mimetype)
                                    ext = ".ogg"

                                with temp_audio_file(audio_bytes, suffix=ext) as tmp_audio:
                                    transcript = self.stt_service.transcribe_audio(tmp_audio, mime_type=mimetype)
                                    m["text"] = transcript
                                    m["input_type"] = "audio"
                                    m["message_type"] = "voice"
                                    logger.info("[WhatsApp] Audio message transcribed successfully")
                            else:
                                logger.warning("[WhatsApp] Downloaded empty audio")
                        except Exception as exc:
                            logger.error("[WhatsApp] STT processing failed for audio: %s", exc)
                            if is_agent:
                                self.send_message(chat_id, "Sorry, I could not transcribe your voice message. Please try again or send text.")
                        finally:
                            if indicator:
                                try:
                                    indicator.stop()
                                except Exception:
                                    pass

                # If message has no text (e.g. unsupported media without caption or failed STT), skip adding to new_items
                if not m.get("text"):
                    logger.debug("[WhatsApp] Skipping message with empty text")
                    continue

                new_items.append(m)

                if not from_me and not is_agent:
                    self._recent_messages_history.append(m)

            if len(self._processed_message_ids) > 1000:
                self._processed_message_ids = set(list(self._processed_message_ids)[-1000:])

            if len(self._recent_messages_history) > 200:
                self._recent_messages_history = self._recent_messages_history[-200:]
            return new_items
        except Exception as exc:
            logger.warning("[WhatsApp Connector] Error fetching recent messages from bridge: %s", exc)
            return []

    def get_recent_history(self, query: str = "", limit: int = 10) -> list[dict]:
        items = list(self._recent_messages_history)
        if not items and self.enabled:
            try:
                resp = requests.get(f"{self.bridge_url}/recent/peek", timeout=5)
                if resp.ok:
                    filtered = []
                    for m in resp.json():
                        chat_id = str(m.get("chat_id", ""))
                        sender = str(m.get("sender", ""))
                        from_me = bool(m.get("from_me", False))
                        is_agent = self.is_agent_user_identity(chat_id, sender)
                        if not from_me and not is_agent:
                            filtered.append(m)
                    items = filtered
            except Exception:
                pass

        if not items:
            try:
                from assistant.storage.db import get_connection
                conn = get_connection()
                q_sql = (
                    "SELECT message_id, source, sender, content, ts "
                    "FROM priority_inbox WHERE source = 'whatsapp' "
                    "ORDER BY ts DESC LIMIT ?"
                )
                rows = conn.execute(q_sql, (limit,)).fetchall()
                for r in reversed(rows):
                    items.append({
                        "id": r["message_id"],
                        "sender": r["sender"],
                        "chat_id": r["sender"],
                        "text": r["content"],
                        "timestamp": r["ts"],
                    })
            except Exception as exc:
                logger.debug("[WhatsApp Connector] DB fallback error: %s", exc)

        if query:
            q = query.lower()
            items = [
                m for m in items
                if q in str(m.get("text", "")).lower()
                or q in str(m.get("sender", "")).lower()
                or q in str(m.get("chat_id", "")).lower()
            ]
        return items[-limit:]

    def resolve_recipient(self, recipient: str, force_refresh: bool = False) -> tuple[str | None, str | None]:
        target = recipient.strip()
        logger.debug("[WhatsApp] Resolving recipient")
        if not target:
            return None, 'Recipient name cannot be empty.'

        # 0. Already a WhatsApp JID
        if target.endswith("@s.whatsapp.net") or target.endswith("@g.us") or target.endswith("@lid"):
            logger.info("[WhatsApp] Recipient resolved")
            return target, None

        # 1. Direct phone-number path (bypasses contact-cache lookup, does not cache)
        clean_target = target.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
        is_direct_phone = False
        phone_digits = ""
        if target.startswith("+") and clean_target[1:].isdigit():
            is_direct_phone = True
            phone_digits = clean_target[1:]
        elif clean_target.isdigit() and len(clean_target) >= 10:
            is_direct_phone = True
            phone_digits = clean_target

        if is_direct_phone:
            if len(phone_digits) == 10:
                phone_digits = f"91{phone_digits}"
            res_jid = f"{phone_digits}@s.whatsapp.net"
            logger.info("[WhatsApp] Recipient resolved")
            return res_jid, None

        # 2. Existing known WhatsApp mapping
        matches = []
        seen = set()
        q = target.lower()
        for m in getattr(self, "_recent_messages_history", []):
            sender = (m.get("sender") or "").lower()
            chat = (m.get("chat_id") or "").lower()
            if q in sender or q in chat:
                jid = m.get("chat_id") or m.get("sender")
                if jid and jid not in seen and "@" in jid:
                    seen.add(jid)
                    matches.append({"name": m.get("sender") or m.get("chat_id"), "jid": jid})

        if self.enabled:
            try:
                resp = requests.get(f"{self.bridge_url}/contacts/resolve", params={"name": target}, timeout=5)
                if resp.ok:
                    data = resp.json()
                    bridge_matches = data.get("matches", [])
                    for bm in bridge_matches:
                        if bm["jid"] not in seen and "@" in bm["jid"]:
                            seen.add(bm["jid"])
                            matches.append(bm)
            except Exception as exc:
                logger.warning("[WhatsApp Connector] Error resolving contact via bridge: %s", exc)

        if len(matches) == 1:
            res_jid = matches[0]["jid"]
            logger.info("[WhatsApp] Recipient resolved")
            return res_jid, None
        elif len(matches) > 1:
            names = ", ".join(f"{m['name']} ({m['jid']})" for m in matches)
            return None, f'I found multiple WhatsApp contacts matching "{target}": {names}. Which one should I use?'

        # 3. Contact Cache lookup (if not force_refresh)
        if self.contact_cache is not None and not force_refresh:
            cached = self.contact_cache.get(target)
            if cached is not None:
                from assistant.contacts.contact_cache import phone_to_jid
                res_jid = cached.resolved_jid or phone_to_jid(cached.phone_number)
                logger.info("[WhatsApp] Recipient resolved")
                return res_jid, None

        # 4. Android Contacts lookup via DeviceGateway
        if self.device_gateway is None or not self.device_gateway.is_device_connected():
            logger.info("[WhatsApp] Android device offline/disconnected, cannot search contacts")
            return None, (
                f"Android device is disconnected. Please connect your device to search contacts for '{target}', "
                f"or provide their phone number directly."
            )

        try:
            res = self.device_gateway.send_command("contact.find", {"query": target})
        except Exception as exc:
            logger.error("[WhatsApp] Error querying Android contacts: %s", exc)
            return None, f"Error searching contacts on Android phone: {exc}"

        if getattr(res, "status", "") != "ok":
            detail_str = str(getattr(res, "detail", "") or "")
            if "PERMISSION_DENIED" in detail_str or "permission" in detail_str.lower():
                return None, "Contacts permission is denied on your Android phone. Please grant Contacts permission in Android Settings, or provide their phone number directly."
            return None, f"Failed to search contacts on phone: {res.detail}"

        android_matches = []
        if hasattr(res, "detail") and isinstance(res.detail, dict):
            android_matches = res.detail.get("matches", [])

        if len(android_matches) > 1:
            options = [f"{m.get('name')}: {m.get('phone_number')}" for m in android_matches if isinstance(m, dict)]
            return None, (
                f"Found multiple contacts matching '{target}':\n"
                + "\n".join(f"- {opt}" for opt in options)
                + "\nPlease specify which number to message."
            )

        if len(android_matches) == 1 and isinstance(android_matches[0], dict):
            c_name = android_matches[0].get("name", target)
            phone_num = str(android_matches[0].get("phone_number", "")).strip()
            if phone_num:
                from assistant.contacts.contact_cache import phone_to_jid
                res_jid = phone_to_jid(phone_num)
                if self.contact_cache is not None:
                    self.contact_cache.set(
                        name=target,
                        display_name=c_name,
                        phone_number=phone_num,
                        resolved_jid=res_jid,
                    )
                logger.info("[WhatsApp] Recipient resolved")
                return res_jid, None

        # 5. Direct phone-number fallback
        if clean_target.isdigit() and len(clean_target) >= 10:
            final_digits = f"91{clean_target}" if len(clean_target) == 10 else clean_target
            res_jid = f"{final_digits}@s.whatsapp.net"
            logger.info("[WhatsApp] Recipient resolved")
            return res_jid, None

        return None, f'I couldn\'t find a WhatsApp contact named "{target}". Please provide the phone number.'

    def send_file(self, chat_id: str, path: str, caption: str = "") -> dict:
        if not self.enabled:
            return {"status": "ok", "detail": f"[MOCK] would send WhatsApp file {path} to {chat_id}"}
        try:
            with open(path, "rb") as handle:
                resp = requests.post(
                    f"{self.bridge_url}/send-file",
                    data={"chat_id": chat_id, "caption": caption},
                    files={"file": (path.rsplit("/", 1)[-1], handle)}, timeout=60,
                )
            if resp.ok:
                return {"status": "ok", "detail": resp.json()}
            return {"status": "failed", "detail": resp.text}
        except Exception as exc:
            logger.error("[WhatsApp Connector] Failed to send file to bridge: %s", exc)
            return {"status": "failed", "detail": str(exc)}
