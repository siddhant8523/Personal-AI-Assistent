# Telegram Skill

## Overview
This skill defines the assistant's Telegram integration capabilities, distinguishing between the Telegram Bot API and the Personal Telegram Account (MTProto/Telethon).

---

## 1. Bot API vs. Personal Telegram Account

### Telegram Bot API
- **Token**: `TELEGRAM_BOT_TOKEN`
- **Channel**: `TelegramChannel` (@task_9_demo_bot)
- **Role**: Serves as an interactive agent control channel for receiving commands and approval responses.

### Personal Telegram Account
- **Auth**: `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_SESSION_FILE`
- **Client**: `TelegramUserClient` (Telethon / MTProto)
- **Role**: Accesses user's actual Telegram account to perform:
  - Account information queries (`telegram_get_account`)
  - Contact and entity searches (`telegram_find_contact`)
  - Dialog / chat listing (`telegram_list_dialogs`)
  - Contact / chat counting (`telegram_count_chats`)
  - Message reading (`telegram_read_messages`)
  - Message search (`telegram_search_messages`)
  - Group listing (`telegram_list_groups`)
  - Channel listing (`telegram_list_channels`)
  - Personal messaging (`send_telegram_message`)
  - Personal file sending (`send_telegram_file`)

---

## 2. Operating Principles & Constraints

### Contact Resolution & Ambiguity
- When asked to message or find a person (e.g. "Find Rahul"), search dialogs and contacts.
- If **multiple contacts match** (e.g., Rahul Sharma, Rahul Verma), ask the user to clarify which person they mean. Never guess or select an unrelated person.
- If **no match exists**, state clearly: "I couldn't find [Name] on Telegram."

### Approval System Requirements
- Read-only operations (`telegram_get_account`, `telegram_find_contact`, `telegram_list_dialogs`, `telegram_count_chats`, `telegram_read_messages`, `telegram_search_messages`, `telegram_list_groups`, `telegram_list_channels`) execute immediately without approval.
- Outbound actions (`send_telegram_message`, `send_telegram_file`) **ALWAYS require explicit user approval** before sending.

### Content Generation vs. Literal Content
- **Literal Content**: If the user specifies exact text in quotes (e.g. `Send Sidd "Hello how are you"`), preserve the text verbatim.
- **Generated Content**: If the user requests content generation (e.g. `Send Sidd a 10-line paragraph about machine learning` or `Send Rahul a professional apology`), generate the complete text FIRST and display the actual generated message in the approval prompt before sending.

### File Restrictions
- Sending files checks `FilePolicy` (10 MB maximum file size limit, allowed outbound directories).
