"""Telegram personal-account payload -> common inbound message."""

from assistant.ingestion.normalizer import normalize_telegram


def adapt_telegram(payload: dict):
    return normalize_telegram(payload)
