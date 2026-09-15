"""Android Device Gateway SMS payload -> common inbound message."""

from assistant.ingestion.normalizer import normalize_sms


def adapt_sms(payload: dict):
    return normalize_sms(payload)
