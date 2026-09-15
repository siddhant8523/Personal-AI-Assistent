"""Prompt template for complex/semantic priority analysis (escalation from rules)."""

SYSTEM = """You analyze a single incoming message for a priority inbox.
Return compact JSON: {"urgency": "HIGH|MEDIUM|LOW", "importance": "HIGH|MEDIUM|LOW", "reason": "..."}"""


def build_user_prompt(sender: str, content: str) -> str:
    return f"Sender: {sender}\nMessage: {content}"
