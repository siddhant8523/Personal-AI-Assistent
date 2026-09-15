"""
Mobile Path Resolver
====================
Handles path semantics for Android device storage:
- Mobile Storage Root: /storage/emulated/0
- Translates paths like /Books/cptopic.pdf or Books/cptopic.pdf into
  /storage/emulated/0/Books/cptopic.pdf for execution on Android.
"""

from __future__ import annotations

import os

MOBILE_STORAGE_ROOT = "/storage/emulated/0"

MOBILE_TOP_LEVEL_DIRS = {
    "book", "books", "download", "downloads", "documents", "dcim", "pictures",
    "movies", "music", "podcasts", "android", "alarm", "alarms", "notifications",
    "ringtones", "telegram", "whatsapp"
}



def is_mobile_path(path: str) -> bool:
    """Determine whether a path or file query explicitly targets the mobile device filesystem."""
    p_lower = path.strip().lower()
    if not p_lower:
        return False
    if p_lower.startswith("/storage/") or p_lower.startswith("/sdcard"):
        return True
    
    parts = [part for part in p_lower.split("/") if part]
    if parts and parts[0] in MOBILE_TOP_LEVEL_DIRS:
        return True

    return False


def canonical_mobile_path(path: str) -> str:
    """Convert an Android path (e.g. /Books/cptopic.pdf) into a full mobile path (/storage/emulated/0/Books/cptopic.pdf)."""
    p = path.strip()
    if not p:
        return MOBILE_STORAGE_ROOT
    if p.startswith("/storage/emulated/0") or p.startswith("/sdcard"):
        return p

    # Ensure leading slash relative to mobile root
    rel_path = p.lstrip("/")
    return f"{MOBILE_STORAGE_ROOT}/{rel_path}"
