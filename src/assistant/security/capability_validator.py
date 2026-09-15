"""
Capability Validator (Rule 18, Rule 19)
===========================================
The Android device and cloud connectors must not execute arbitrary
capabilities — only what's explicitly allow-listed in
config/capabilities.yaml.
"""

from __future__ import annotations


ANDROID_CAPABILITY_ALIASES: dict[str, str] = {
    "make_call": "call.make",
    "send_sms": "sms.send",
    "read_sms": "sms.read",
    "read_file": "file.read",
    "upload_file": "file.upload",
    "list_files": "file.list",
    "find_files": "file.find",
    "run_intent": "intent.execute",
    "open_app": "intent.open_app",
    "set_alarm": "alarm.set",
    "list_alarms": "alarm.list",
    "cancel_alarm": "alarm.cancel",
    "set_timer": "timer.set",
    "find_contact": "contact.find",
    "search_contacts": "contact.find",
}



REV_ANDROID_CAPABILITY_ALIASES: dict[str, str] = {v: k for k, v in ANDROID_CAPABILITY_ALIASES.items()}



def normalize_android_capability(capability: str) -> str:
    """Normalize legacy capability names (make_call, set_alarm, ...) to canonical Android dot-namespaced capability names (call.make, alarm.set, ...)."""
    cap = capability.strip().lower()
    return ANDROID_CAPABILITY_ALIASES.get(cap, cap)


class CapabilityValidator:
    def __init__(self, android_capabilities: list[str], cloud_capabilities: list[str]):
        self.raw_android = list(android_capabilities)
        self.cloud_capabilities = set(cloud_capabilities)
        self.android_capabilities = set()
        self._normalized_android = set()

        for cap in android_capabilities:
            norm = normalize_android_capability(cap)
            self.android_capabilities.add(norm)
            self.android_capabilities.add(cap)
            self._normalized_android.add(norm.upper().replace(".", "_"))
            self._normalized_android.add(cap.upper().replace(".", "_"))

    def validate_android(self, capability: str) -> bool:
        norm_cap = normalize_android_capability(capability)
        if norm_cap in self.android_capabilities or capability in self.android_capabilities:
            return True
        norm = norm_cap.upper().replace(".", "_")
        return norm in self._normalized_android

    def validate_cloud(self, capability: str) -> bool:
        return capability in self.cloud_capabilities


