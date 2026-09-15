"""Prompt template for task planning."""

SYSTEM = """You convert a structured intent + context into an ordered list of
plan steps needed to fulfill it. Return compact JSON: {"steps": ["...", "..."]}"""


def build_user_prompt(intent_json: str, context_summary: str) -> str:
    return f"Intent:\n{intent_json}\n\nContext:\n{context_summary}"
