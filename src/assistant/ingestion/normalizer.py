"""
Normalizer
============
Converts a raw, platform-specific payload (dict) into a UnifiedMessage.
The Agent Core never sees Gmail/WhatsApp/Telegram/SMS-native shapes.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage


def parse_payload_timestamp(raw_ts: Any) -> float:
    """Safely parse various timestamp representations (seconds, ms, ISO string) into float unix seconds."""
    if raw_ts is None:
        return time.time()
    if isinstance(raw_ts, (int, float)):
        if raw_ts > 1e11:
            return float(raw_ts) / 1000.0
        return float(raw_ts)
    if isinstance(raw_ts, str):
        raw_str = raw_ts.strip()
        if not raw_str:
            return time.time()
        try:
            val = float(raw_str)
            if val > 1e11:
                return val / 1000.0
            return val
        except ValueError:
            pass
        try:
            clean_str = raw_str.replace("Z", "+00:00")
            return datetime.fromisoformat(clean_str).timestamp()
        except Exception:
            pass
    return time.time()


def normalize_gmail(payload: dict[str, Any]) -> UnifiedMessage:
    raw_ts = payload.get("internalDate") or payload.get("date") or payload.get("timestamp")
    ts = parse_payload_timestamp(raw_ts) if raw_ts is not None else time.time()
    return UnifiedMessage(
        source=Source.GMAIL,
        conversation_id=f"gmail:{payload.get('thread_id', payload.get('from'))}",
        sender=payload["from"],
        recipient=payload.get("to"),
        content=payload.get("snippet") or payload.get("body", ""),
        metadata={"subject": payload.get("subject", ""), "message_id": payload.get("id")},
        platform_message_id=payload.get("id"),
        timestamp=ts,
    )


def normalize_telegram(payload: dict[str, Any]) -> UnifiedMessage:
    metadata = {"update_id": payload.get("update_id")}
    if payload.get("input_type"):
        metadata["input_type"] = payload["input_type"]
    if payload.get("media_type"):
        metadata["media_type"] = payload["media_type"]
    if payload.get("message_type"):
        metadata["message_type"] = payload["message_type"]

    raw_ts = payload.get("timestamp") or payload.get("date")
    ts = parse_payload_timestamp(raw_ts) if raw_ts is not None else time.time()

    return UnifiedMessage(
        source=Source.TELEGRAM,
        conversation_id=f"telegram:{payload['chat_id']}",
        sender=str(payload.get("from_id", payload.get("from", "unknown"))),
        content=payload.get("text", ""),
        metadata=metadata,
        platform_message_id=str(payload.get("message_id")) if payload.get("message_id") else None,
        origin=Origin.AGENT if payload.get("from_bot") else Origin.USER,
        timestamp=ts,
    )


def normalize_whatsapp(payload: dict[str, Any]) -> UnifiedMessage:
    metadata = {"from_me": payload.get("from_me", False)}
    if payload.get("input_type"):
        metadata["input_type"] = payload["input_type"]
    if payload.get("media_type"):
        metadata["media_type"] = payload["media_type"]
    if payload.get("message_type"):
        metadata["message_type"] = payload["message_type"]

    raw_ts = payload.get("timestamp") or payload.get("date") or payload.get("date_ms")
    ts = parse_payload_timestamp(raw_ts) if raw_ts is not None else time.time()

    return UnifiedMessage(
        source=Source.WHATSAPP,
        conversation_id=f"whatsapp:{payload['chat_id']}",
        sender=payload.get("sender", "unknown"),
        content=payload.get("text", ""),
        metadata=metadata,
        platform_message_id=payload.get("id"),
        # Baileys marks self-sent echoes with from_me=True — tag origin here
        # too, as a first line of defense in addition to the Echo Filter.
        origin=Origin.AGENT if payload.get("from_me") else Origin.USER,
        timestamp=ts,
    )


def normalize_sms(payload: dict[str, Any]) -> UnifiedMessage:
    raw_ts = payload.get("date") or payload.get("timestamp")
    ts = parse_payload_timestamp(raw_ts) if raw_ts is not None else time.time()
    return UnifiedMessage(
        source=Source.SMS,
        conversation_id=f"sms:{payload['number']}",
        sender=payload["number"],
        content=payload.get("body", ""),
        metadata={"sim_slot": payload.get("sim_slot")},
        platform_message_id=payload.get("sms_id"),
        timestamp=ts,
    )


NORMALIZERS = {
    Source.GMAIL: normalize_gmail,
    Source.TELEGRAM: normalize_telegram,
    Source.WHATSAPP: normalize_whatsapp,
    Source.SMS: normalize_sms,
}


def normalize(source: Source, payload: dict[str, Any]) -> UnifiedMessage:
    return NORMALIZERS[source](payload)

