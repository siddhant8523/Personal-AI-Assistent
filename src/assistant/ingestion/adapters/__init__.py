"""Platform adapters that translate connector payloads into UnifiedMessage."""

from assistant.ingestion.adapters.gmail_adapter import adapt_gmail
from assistant.ingestion.adapters.sms_adapter import adapt_sms
from assistant.ingestion.adapters.telegram_adapter import adapt_telegram
from assistant.ingestion.adapters.whatsapp_adapter import adapt_whatsapp

__all__ = ["adapt_gmail", "adapt_sms", "adapt_telegram", "adapt_whatsapp"]
