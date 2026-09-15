"""Gmail API payload -> common inbound message."""

from assistant.ingestion.normalizer import normalize_gmail


def adapt_gmail(payload: dict):
    return normalize_gmail(payload)
