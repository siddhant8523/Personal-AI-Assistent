"""
Contact Cache
=============
Lightweight, SQLite-backed cache for resolved Android contacts to avoid
repeatedly querying the Android device over the device gateway.

Privacy & Scope:
- Only stores minimal metadata for recipient resolution:
  normalized_name, display_name, phone_number, resolved_jid, cached_at, expires_at.
- Treats the phone number as the durable identity; resolved_jid is an optimization.
- Lazy expiration: entries are evaluated and purged upon access if expired.
- Configurable TTL: default 10 days, overridable via CONTACT_CACHE_TTL_DAYS.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from assistant.storage.db import get_connection

logger = logging.getLogger("assistant.contacts.cache")


def mask_phone(phone: str) -> str:
    """Mask phone number to prevent sensitive PII leakage in application logs."""
    p = str(phone or "").strip()
    if len(p) > 6:
        return p[:3] + "******" + p[-4:]
    return "***"


def phone_to_jid(phone: str) -> str:
    """Derive standard WhatsApp JID from a phone number string."""
    clean = phone.replace("+", "").replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
    if len(clean) == 10:
        clean = f"91{clean}"
    return f"{clean}@s.whatsapp.net"


@dataclass
class CachedContact:
    normalized_name: str
    display_name: str
    phone_number: str
    resolved_jid: str
    cached_at: float
    expires_at: float

    def is_expired(self, now: float | None = None) -> bool:
        if now is None:
            now = time.time()
        return now >= self.expires_at

    def to_dict(self) -> dict:
        return {
            "normalized_name": self.normalized_name,
            "display_name": self.display_name,
            "phone_number": self.phone_number,
            "resolved_jid": self.resolved_jid,
            "cached_at": self.cached_at,
            "expires_at": self.expires_at,
        }


class ContactCache:
    def __init__(self, ttl_days: float | None = None, ttl_seconds: float | None = None):
        if ttl_seconds is not None:
            self.ttl_seconds = float(ttl_seconds)
        elif ttl_days is not None:
            self.ttl_seconds = float(ttl_days) * 86400.0
        else:
            env_ttl = os.environ.get("CONTACT_CACHE_TTL_DAYS", "10")
            try:
                self.ttl_seconds = float(env_ttl) * 86400.0
            except ValueError:
                self.ttl_seconds = 10.0 * 86400.0

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalized contact name: lowercase and whitespace stripped."""
        return str(name or "").strip().lower()

    def get(self, name: str, now: float | None = None) -> CachedContact | None:
        """Look up contact by name. Returns non-expired entry or None."""
        normalized = self.normalize_name(name)
        if not normalized:
            return None

        if now is None:
            now = time.time()

        conn = get_connection()
        row = conn.execute(
            "SELECT normalized_name, display_name, phone_number, resolved_jid, cached_at, expires_at "
            "FROM contact_cache WHERE normalized_name = ?",
            (normalized,),
        ).fetchone()

        if not row:
            logger.info("[Contact Cache] miss name='%s'", normalized)
            return None

        entry = CachedContact(
            normalized_name=row["normalized_name"],
            display_name=row["display_name"],
            phone_number=row["phone_number"],
            resolved_jid=row["resolved_jid"],
            cached_at=float(row["cached_at"]),
            expires_at=float(row["expires_at"]),
        )

        if entry.is_expired(now):
            logger.info(
                "[Contact Cache] expired name='%s' expired_ago=%ds",
                normalized,
                int(now - entry.expires_at),
            )
            # Lazy expiration: remove expired entry
            self.delete(normalized)
            return None

        logger.info(
            "[Contact Cache] hit name='%s' phone=%s (expires_in=%ds)",
            normalized,
            mask_phone(entry.phone_number),
            max(0, int(entry.expires_at - now)),
        )
        return entry

    def set(
        self,
        name: str,
        display_name: str,
        phone_number: str,
        resolved_jid: str | None = None,
        now: float | None = None,
    ) -> CachedContact:
        """Store or update contact entry in the cache."""
        normalized = self.normalize_name(name)
        if now is None:
            now = time.time()

        expires_at = now + self.ttl_seconds
        phone_clean = str(phone_number or "").strip()
        jid = resolved_jid or phone_to_jid(phone_clean)

        conn = get_connection()
        # Check if updating an existing entry
        existing = conn.execute(
            "SELECT phone_number FROM contact_cache WHERE normalized_name = ?",
            (normalized,),
        ).fetchone()

        conn.execute(
            "INSERT INTO contact_cache (normalized_name, display_name, phone_number, resolved_jid, cached_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(normalized_name) DO UPDATE SET "
            "display_name=excluded.display_name, "
            "phone_number=excluded.phone_number, "
            "resolved_jid=excluded.resolved_jid, "
            "cached_at=excluded.cached_at, "
            "expires_at=excluded.expires_at",
            (normalized, display_name.strip(), phone_clean, jid, now, expires_at),
        )
        conn.commit()

        if existing:
            logger.info(
                "[Contact Cache] refreshed name='%s' phone=%s ttl_days=%.1f",
                normalized,
                mask_phone(phone_clean),
                self.ttl_seconds / 86400.0,
            )
        else:
            logger.info(
                "[Contact Cache] stored name='%s' phone=%s ttl_days=%.1f",
                normalized,
                mask_phone(phone_clean),
                self.ttl_seconds / 86400.0,
            )

        return CachedContact(
            normalized_name=normalized,
            display_name=display_name.strip(),
            phone_number=phone_clean,
            resolved_jid=jid,
            cached_at=now,
            expires_at=expires_at,
        )

    def delete(self, name: str) -> None:
        """Delete an entry by contact name."""
        normalized = self.normalize_name(name)
        if not normalized:
            return
        conn = get_connection()
        conn.execute("DELETE FROM contact_cache WHERE normalized_name = ?", (normalized,))
        conn.commit()
        logger.info("[Contact Cache] invalidated name='%s'", normalized)

    def clear(self) -> None:
        """Clear all entries in the contact cache."""
        conn = get_connection()
        conn.execute("DELETE FROM contact_cache")
        conn.commit()
        logger.info("[Contact Cache] cleared all entries")

    def all_entries(self) -> list[CachedContact]:
        """List all entries currently in the cache (primarily for tests/auditing)."""
        conn = get_connection()
        rows = conn.execute(
            "SELECT normalized_name, display_name, phone_number, resolved_jid, cached_at, expires_at "
            "FROM contact_cache"
        ).fetchall()
        return [
            CachedContact(
                normalized_name=r["normalized_name"],
                display_name=r["display_name"],
                phone_number=r["phone_number"],
                resolved_jid=r["resolved_jid"],
                cached_at=float(r["cached_at"]),
                expires_at=float(r["expires_at"]),
            )
            for r in rows
        ]
