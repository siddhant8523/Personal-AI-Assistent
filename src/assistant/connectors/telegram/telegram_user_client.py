"""Personal Telegram (MTProto/Telethon) connector, separate from Bot API."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

from collections import OrderedDict

logger = logging.getLogger("assistant.telegram.personal")


class TelegramUserClient:
    def __init__(
        self,
        api_id: str,
        api_hash: str,
        session_file: str,
        enabled: bool = False,
        max_seen: int = 5000,
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_file = session_file
        self.enabled = enabled and bool(api_id and api_hash and session_file)
        self.max_seen = max_seen
        self._client = None
        self._loop = None
        self._seen_messages: OrderedDict[tuple[str, str], bool] = OrderedDict()

        if self.enabled:
            try:
                self._loop = asyncio.new_event_loop()
                t = threading.Thread(target=self._run_loop, name="telethon-loop", daemon=True)
                t.start()

                asyncio.set_event_loop(self._loop)
                from telethon import TelegramClient

                self._client = TelegramClient(session_file, int(api_id), api_hash, loop=self._loop)
                
                connect_future = asyncio.run_coroutine_threadsafe(self._client.connect(), self._loop)
                connect_future.result(timeout=10)

                auth_future = asyncio.run_coroutine_threadsafe(self._client.is_user_authorized(), self._loop)
                is_auth = auth_future.result(timeout=10)

                if not is_auth:
                    disc_future = asyncio.run_coroutine_threadsafe(self._client.disconnect(), self._loop)
                    disc_future.result(timeout=5)
                    self._client = None
                    self.enabled = False
                    logger.warning("Personal Telegram needs an interactive Telethon login before it can run")
                else:
                    logger.info("Personal Telegram account connected and authorized.")
            except Exception as exc:
                logger.error("Failed to initialize Personal Telegram client: %s", exc)
                self._client = None
                self.enabled = False

    def get_status(self) -> dict[str, Any]:
        """Returns the real connection and authorization status of the Telegram User Client."""
        has_creds = bool(self.api_id and self.api_hash and self.session_file)
        is_auth = False
        is_conn = False
        if self._client and self._loop and self._loop.is_running():
            try:
                auth_future = asyncio.run_coroutine_threadsafe(self._client.is_user_authorized(), self._loop)
                is_auth = bool(auth_future.result(timeout=2))
            except Exception:
                pass
            try:
                is_conn = bool(self._client.is_connected())
            except Exception:
                pass

        is_actually_connected = bool(is_auth and is_conn)
        is_reconnecting = bool(self.enabled and is_auth and not is_conn)
        return {
            "has_credentials": has_creds,
            "enabled": self.enabled,
            "authorized": is_auth,
            "connected": is_actually_connected,
            "reconnecting": is_reconnecting,
            "session_file": self.session_file,
        }

    def _is_seen(self, key: tuple[str, str]) -> bool:
        return key in self._seen_messages

    def _mark_seen(self, key: tuple[str, str]) -> None:
        if key in self._seen_messages:
            self._seen_messages.move_to_end(key)
        else:
            self._seen_messages[key] = True
            if len(self._seen_messages) > self.max_seen:
                self._seen_messages.popitem(last=False)


    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _resolve_target_async(self, recipient: str) -> Any:
        target = recipient.strip()
        if target.lower() in ("me", "myself", "saved", "saved messages"):
            return "me"
        if target.replace("-", "").isdigit():
            return int(target)
        if target.startswith("@") or target.startswith("+"):
            return target

        my_id = int(os.environ.get("TELEGRAM_AGENT_CHAT_ID", "0"))
        target_lower = target.lower()

        # 1. Exact match among dialogs (excluding self)
        try:
            async for dialog in self._client.iter_dialogs(limit=100):
                if dialog.name and target_lower == dialog.name.lower() and dialog.id != my_id:
                    return dialog.entity
            # 2. Partial match among dialogs (excluding self)
            async for dialog in self._client.iter_dialogs(limit=100):
                if dialog.name and target_lower in dialog.name.lower() and dialog.id != my_id:
                    return dialog.entity
        except Exception as exc:
            logger.warning("Error searching Telegram dialogs: %s", exc)

        return target

    def get_me(self) -> dict:
        """Retrieve profile information for the authenticated personal Telegram account."""
        if not self.enabled or not self._client or not self._loop:
            return {"status": "failed", "detail": "Personal Telegram is not authenticated"}

        async def _get():
            me = await self._client.get_me()
            return {
                "status": "ok",
                "name": f"{me.first_name or ''} {me.last_name or ''}".strip(),
                "username": me.username or "",
                "user_id": me.id,
                "phone": me.phone or "",
                "is_bot": me.bot,
            }

        try:
            future = asyncio.run_coroutine_threadsafe(_get(), self._loop)
            return future.result(timeout=15)
        except Exception as exc:
            logger.error("Error getting personal Telegram account info: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def find_contact(self, query: str) -> dict:
        """Search for a contact or user on Telegram by name, username, or phone."""
        if not self.enabled or not self._client or not self._loop:
            return {"status": "failed", "detail": "Personal Telegram is not authenticated"}

        async def _find():
            q = query.strip().lower()
            matches = []
            seen_ids = set()

            # Search dialogs
            async for dialog in self._client.iter_dialogs(limit=200):
                name = dialog.name or ""
                entity = dialog.entity
                username = getattr(entity, "username", "") or ""
                phone = getattr(entity, "phone", "") or ""
                entity_id = dialog.id

                if (q in name.lower() or (username and q in username.lower()) or (phone and q in phone)) and entity_id not in seen_ids:
                    seen_ids.add(entity_id)
                    entity_type = "user" if dialog.is_user else ("group" if dialog.is_group else "channel")
                    matches.append({
                        "name": name,
                        "username": username,
                        "user_id": entity_id,
                        "phone": phone,
                        "entity_type": entity_type,
                    })

            if not matches:
                return {
                    "status": "not_found",
                    "message": f"I couldn't find {query} on Telegram.",
                    "matches": [],
                }
            elif len(matches) == 1:
                return {
                    "status": "ok",
                    "count": 1,
                    "matches": matches,
                    "detail": f"Found {matches[0]['name']} (@{matches[0]['username'] or matches[0]['user_id']})",
                }
            else:
                formatted_names = ", ".join(f"{m['name']} (@{m['username'] or m['user_id']})" for m in matches)
                return {
                    "status": "ambiguous",
                    "count": len(matches),
                    "matches": matches,
                    "message": f"I found multiple Telegram users matching '{query}': {formatted_names}. Which one do you mean?",
                }

        try:
            future = asyncio.run_coroutine_threadsafe(_find(), self._loop)
            return future.result(timeout=15)
        except Exception as exc:
            logger.error("Error finding contact on Telegram: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def list_dialogs(self, limit: int = 50) -> dict:
        """List active Telegram dialogs (private chats, groups, channels)."""
        if not self.enabled or not self._client or not self._loop:
            return {"status": "failed", "detail": "Personal Telegram is not authenticated"}

        async def _list():
            items = []
            async for dialog in self._client.iter_dialogs(limit=limit):
                chat_type = "user" if dialog.is_user else ("group" if dialog.is_group else "channel")
                entity = dialog.entity
                last_msg = dialog.message.message if dialog.message else ""
                date_ts = dialog.message.date.timestamp() if dialog.message and dialog.message.date else None
                date_iso = dialog.message.date.isoformat() if dialog.message and dialog.message.date else ""
                is_out = bool(dialog.message.out) if dialog.message else False
                items.append({
                    "name": dialog.name or str(dialog.id),
                    "username": getattr(entity, "username", "") or "",
                    "chat_id": dialog.id,
                    "chat_type": chat_type,
                    "unread_count": dialog.unread_count,
                    "last_message": last_msg[:100] if last_msg else "",
                    "timestamp": date_ts,
                    "date_iso": date_iso,
                    "is_outgoing": is_out,
                })
            return {"status": "ok", "count": len(items), "dialogs": items}

        try:
            future = asyncio.run_coroutine_threadsafe(_list(), self._loop)
            return future.result(timeout=15)
        except Exception as exc:
            logger.error("Error listing Telegram dialogs: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def count_summary(self) -> dict:
        """Retrieve count breakdown of Telegram contacts, dialogs, groups, and channels."""
        if not self.enabled or not self._client or not self._loop:
            return {"status": "failed", "detail": "Personal Telegram is not authenticated"}

        async def _count():
            total_dialogs = 0
            private_chats = 0
            groups = 0
            channels = 0

            async for dialog in self._client.iter_dialogs(limit=500):
                total_dialogs += 1
                if dialog.is_user:
                    private_chats += 1
                elif dialog.is_group:
                    groups += 1
                elif dialog.is_channel:
                    channels += 1

            # Count saved contacts
            contacts_count = 0
            try:
                from telethon.tl.functions.contacts import GetContactsRequest
                res = await self._client(GetContactsRequest(hash=0))
                contacts_count = len(getattr(res, "users", []))
            except Exception:
                contacts_count = private_chats

            return {
                "status": "ok",
                "contacts_count": contacts_count,
                "total_dialogs": total_dialogs,
                "private_chats": private_chats,
                "groups_count": groups,
                "channels_count": channels,
                "detail": (
                    f"Telegram Account Breakdown:\n"
                    f"- Total Active Dialogs: {total_dialogs}\n"
                    f"- Telegram Contacts: {contacts_count}\n"
                    f"- Private User Chats: {private_chats}\n"
                    f"- Groups: {groups}\n"
                    f"- Channels: {channels}"
                ),
            }

        try:
            future = asyncio.run_coroutine_threadsafe(_count(), self._loop)
            return future.result(timeout=20)
        except Exception as exc:
            logger.error("Error summarizing Telegram counts: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def read_messages(self, recipient: str, limit: int = 20) -> dict:
        """Read recent messages from a specific Telegram person, group, or chat."""
        if not self.enabled or not self._client or not self._loop:
            return {"status": "failed", "detail": "Personal Telegram is not authenticated"}

        async def _read():
            target = await self._resolve_target_async(recipient)
            messages = []
            async for msg in self._client.iter_messages(target, limit=limit):
                media_type = None
                if msg.photo:
                    media_type = "photo"
                elif msg.document:
                    media_type = "document"
                elif msg.video:
                    media_type = "video"
                elif msg.audio:
                    media_type = "audio"
                elif msg.voice:
                    media_type = "voice"

                sender_name = "You" if msg.out else str(msg.sender_id or recipient)
                if msg.sender and hasattr(msg.sender, "first_name"):
                    sender_name = f"{msg.sender.first_name or ''} {msg.sender.last_name or ''}".strip() or sender_name

                messages.append({
                    "message_id": str(msg.id),
                    "sender": sender_name,
                    "timestamp": msg.date.isoformat() if msg.date else None,
                    "text": msg.message or "",
                    "is_outgoing": msg.out,
                    "media_type": media_type,
                    "reply_to_msg_id": str(msg.reply_to_msg_id) if msg.reply_to_msg_id else None,
                })
            return {"status": "ok", "recipient": recipient, "count": len(messages), "messages": messages}

        try:
            future = asyncio.run_coroutine_threadsafe(_read(), self._loop)
            return future.result(timeout=15)
        except Exception as exc:
            logger.error("Error reading Telegram messages: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def search_messages(self, query: str, recipient: str = "") -> dict:
        """Search Telegram messages for specific text/keywords."""
        if not self.enabled or not self._client or not self._loop:
            return {"status": "failed", "detail": "Personal Telegram is not authenticated"}

        async def _search():
            target = None
            if recipient.strip():
                target = await self._resolve_target_async(recipient)

            results = []
            async for msg in self._client.iter_messages(target, search=query, limit=50):
                sender_name = "You" if msg.out else str(msg.sender_id or "Other")
                if msg.sender and hasattr(msg.sender, "first_name"):
                    sender_name = f"{msg.sender.first_name or ''} {msg.sender.last_name or ''}".strip() or sender_name

                chat_title = ""
                if msg.chat and hasattr(msg.chat, "title"):
                    chat_title = msg.chat.title
                elif msg.chat and hasattr(msg.chat, "first_name"):
                    chat_title = f"{msg.chat.first_name or ''} {msg.chat.last_name or ''}".strip()

                results.append({
                    "message_id": str(msg.id),
                    "sender": sender_name,
                    "chat": chat_title or str(msg.chat_id),
                    "timestamp": msg.date.isoformat() if msg.date else None,
                    "text": msg.message or "",
                })
            return {"status": "ok", "query": query, "count": len(results), "matches": results}

        try:
            future = asyncio.run_coroutine_threadsafe(_search(), self._loop)
            return future.result(timeout=15)
        except Exception as exc:
            logger.error("Error searching Telegram messages: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def list_groups(self, limit: int = 50) -> dict:
        """List Telegram groups and supergroups accessible by the user."""
        if not self.enabled or not self._client or not self._loop:
            return {"status": "failed", "detail": "Personal Telegram is not authenticated"}

        async def _list_g():
            groups = []
            async for dialog in self._client.iter_dialogs(limit=limit):
                if dialog.is_group:
                    groups.append({
                        "name": dialog.name,
                        "group_id": dialog.id,
                        "unread_count": dialog.unread_count,
                    })
            return {"status": "ok", "count": len(groups), "groups": groups}

        try:
            future = asyncio.run_coroutine_threadsafe(_list_g(), self._loop)
            return future.result(timeout=15)
        except Exception as exc:
            logger.error("Error listing Telegram groups: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def list_channels(self, limit: int = 50) -> dict:
        """List Telegram channels accessible by the user."""
        if not self.enabled or not self._client or not self._loop:
            return {"status": "failed", "detail": "Personal Telegram is not authenticated"}

        async def _list_c():
            channels = []
            async for dialog in self._client.iter_dialogs(limit=limit):
                if dialog.is_channel:
                    channels.append({
                        "name": dialog.name,
                        "channel_id": dialog.id,
                        "unread_count": dialog.unread_count,
                    })
            return {"status": "ok", "count": len(channels), "channels": channels}

        try:
            future = asyncio.run_coroutine_threadsafe(_list_c(), self._loop)
            return future.result(timeout=15)
        except Exception as exc:
            logger.error("Error listing Telegram channels: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def fetch_recent(self, limit: int = 50) -> list[dict]:
        if not self.enabled or not self._client or not self._loop:
            return []

        async def _fetch():
            items = []
            async for dialog in self._client.iter_dialogs(limit=limit):
                message = dialog.message
                if not message or message.out:
                    continue
                key = (str(dialog.id), str(message.id))
                if self._is_seen(key):
                    continue
                self._mark_seen(key)
                items.append({

                    "message_id": str(message.id),
                    "chat_id": str(dialog.id),
                    "from_id": dialog.name or str(dialog.id),
                    "text": message.message or "",
                    "timestamp": message.date.timestamp() if message.date else None,
                })
            return items

        try:
            future = asyncio.run_coroutine_threadsafe(_fetch(), self._loop)
            return future.result(timeout=15)
        except Exception as exc:
            logger.error("Error fetching recent personal Telegram messages: %s", exc)
            return []

    def send_message(self, recipient: str, text: str) -> dict:
        if not self.enabled or not self._client or not self._loop:
            return {"status": "failed", "detail": "Personal Telegram is not authenticated"}

        async def _send():
            target = await self._resolve_target_async(recipient)
            sent = await self._client.send_message(target, text)
            return sent

        try:
            future = asyncio.run_coroutine_threadsafe(_send(), self._loop)
            sent = future.result(timeout=15)
            return {"status": "ok", "detail": f"Sent Telegram message to {recipient} (id: {sent.id})"}
        except Exception as exc:
            logger.error("Failed to send personal Telegram message: %s", exc)
            return {"status": "failed", "detail": str(exc)}

    def send_file(self, recipient: str, path: str, caption: str = "") -> dict:
        if not self.enabled or not self._client or not self._loop:
            return {"status": "failed", "detail": "Personal Telegram is not authenticated"}

        async def _send():
            target = await self._resolve_target_async(recipient)
            sent = await self._client.send_file(target, path, caption=caption)
            return sent

        try:
            future = asyncio.run_coroutine_threadsafe(_send(), self._loop)
            sent = future.result(timeout=15)
            return {"status": "ok", "detail": f"Sent Telegram file to {recipient} (id: {getattr(sent, 'id', 'sent')})"}
        except Exception as exc:
            logger.error("Failed to send personal Telegram file: %s", exc)
            return {"status": "failed", "detail": str(exc)}
