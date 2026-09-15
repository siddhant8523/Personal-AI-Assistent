"""Baileys bridge payload -> common inbound message."""

from assistant.ingestion.normalizer import normalize_whatsapp


def adapt_whatsapp(payload: dict):
    return normalize_whatsapp(payload)
