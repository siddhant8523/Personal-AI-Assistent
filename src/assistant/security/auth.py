"""Device auth token check for the Device Gateway (Rule 18)."""

from __future__ import annotations

import hmac


class DeviceAuth:
    def __init__(self, expected_token: str):
        self._expected = expected_token

    def verify(self, provided_token: str) -> bool:
        return hmac.compare_digest(self._expected, provided_token or "")
