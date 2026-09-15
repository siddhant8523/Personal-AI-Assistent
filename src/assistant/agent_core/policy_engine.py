"""
Policy Engine (Part 1, Section 7.6; Part 3, Section 4)
==========================================================
Decides whether a capability requires explicit approval before execution.
Loaded from config/policies.yaml.
"""

from __future__ import annotations


from assistant.security.capability_validator import normalize_android_capability, REV_ANDROID_CAPABILITY_ALIASES


class PolicyEngine:
    def __init__(self, capability_policies: dict[str, dict]):
        self._policies = capability_policies

    def requires_approval(self, capability: str) -> bool:
        policy = self._policies.get(capability)
        if policy is None:
            norm = normalize_android_capability(capability)
            policy = self._policies.get(norm)
        if policy is None:
            rev = REV_ANDROID_CAPABILITY_ALIASES.get(capability)
            if rev:
                policy = self._policies.get(rev)
        if policy is None:
            # Fail safe: unknown capability -> require approval.
            return True
        return bool(policy.get("requires_approval", True))

