"""
One-time Telegram User Client (Telethon) setup script.

1. Ensures TELEGRAM_API_ID and TELEGRAM_API_HASH are set in .env.
2. Prompts for phone number & login code in terminal.
3. Saves authorized session file to TELEGRAM_SESSION_FILE
   (default ./config/secrets/telegram_user.session).
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from telethon.sync import TelegramClient

# Required for Python 3.14 + older Telethon event-loop behavior
asyncio.set_event_loop(asyncio.new_event_loop())


if __name__ == "__main__":
    load_dotenv()

    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    session_file = os.environ.get(
        "TELEGRAM_SESSION_FILE",
        "./config/secrets/telegram_user.session"
    )

    if not api_id or not api_hash:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH is missing in .env")
        sys.exit(1)

    os.makedirs(os.path.dirname(session_file), exist_ok=True)

    print(f"Connecting to Telegram with API ID: {api_id}...")

    with TelegramClient(session_file, int(api_id), api_hash) as client:
        if client.is_user_authorized():
            me = client.get_me()
            print(
                f"Already authorized as "
                f"{me.first_name} (@{me.username or me.id})!"
            )
        else:
            print("Starting interactive login...")
            client.start()

            me = client.get_me()

            print(
                f"Successfully logged in as "
                f"{me.first_name} (@{me.username or me.id})!"
            )

    print(
        f"Session saved to {session_file}. "
        f"Set TELEGRAM_USER_ENABLED=true in .env."
    )