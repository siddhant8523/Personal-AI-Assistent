"""
Alarm Time Parser
=================
Converts natural-language time descriptions ("7:57 pm", "8:30 am", "22:15", "in 10 minutes")
into structured parameters expected by Android's alarm.set capability:
- hour (int 0..23)
- minute (int 0..59)
- delay_ms (int ms)
- label (str)
- skip_ui (bool)
"""

from __future__ import annotations

import datetime
import re
from typing import Any


def parse_alarm_time(time_str: str, label: str = "") -> dict[str, Any]:
    t_str = (time_str or "").strip().lower()
    if not t_str:
        return {}

    # Relative time like "in 10 minutes", "10 mins", "in 1 hour"
    rel_match = re.search(r'(?:in\s+)?(\d+)\s*(min|minute|sec|second|hr|hour)s?', t_str)
    if rel_match and ("in" in t_str or "min" in t_str or "hour" in t_str or "sec" in t_str):
        amount = int(rel_match.group(1))
        unit = rel_match.group(2)
        if "hr" in unit or "hour" in unit:
            delay_ms = amount * 3600 * 1000
        elif "sec" in unit:
            delay_ms = amount * 1000
        else:  # minute
            delay_ms = amount * 60 * 1000

        now = datetime.datetime.now()
        target_dt = now + datetime.timedelta(milliseconds=delay_ms)
        return {
            "hour": target_dt.hour,
            "minute": target_dt.minute,
            "delay_ms": delay_ms,
            "label": label,
            "skip_ui": True,
        }

    # 12-hour format: e.g. "7:57 pm", "7:57pm", "8:30 am", "8 am"
    match_12 = re.search(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', t_str)
    if match_12:
        hr = int(match_12.group(1))
        minute = int(match_12.group(2) or 0)
        ampm = match_12.group(3)
        if ampm == "pm" and hr < 12:
            hr += 12
        elif ampm == "am" and hr == 12:
            hr = 0
        return {
            "hour": hr,
            "minute": minute,
            "label": label,
            "skip_ui": True,
        }

    # 24-hour format: e.g. "19:57", "07:57", "22:15"
    match_24 = re.search(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', t_str)
    if match_24:
        hr = int(match_24.group(1))
        minute = int(match_24.group(2))
        return {
            "hour": hr,
            "minute": minute,
            "label": label,
            "skip_ui": True,
        }

    return {}
