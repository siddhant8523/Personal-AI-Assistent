"""Prompt template for intent understanding."""

SYSTEM = """You are the reasoning layer of a personal AI assistant.
Given a user's message, extract structured intent as compact JSON with keys:
intent (one of: SEND_MESSAGE, SEND_FILE, SEND_EMAIL, MAKE_CALL, QUERY_PRIORITY, GENERAL_CHAT),
platform (whatsapp|telegram|sms|gmail|none), recipient (string or null), content (string or null).
Return ONLY the JSON object, nothing else."""


def build_user_prompt(message: str, context_summary: str) -> str:
    return f"Context:\n{context_summary}\n\nUser message:\n{message}"
