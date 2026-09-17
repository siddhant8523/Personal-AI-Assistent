# Telegram Skill

## Important: Skills Are NOT Tools

A skill file is **instructions and domain knowledge** for the Agent. It does not execute anything by itself.

```text
Skill MD
   ↓
Agent understands capability / rules / constraints
   ↓
Agent selects the appropriate registered tool
   ↓
Tool Router
   ↓
Connector / Device Gateway
   ↓
External system
```

Never treat reading a skill as performing an action. Always use the registered LangChain tools and the Task Manager / Approval flow for side effects.

---

## Purpose

Guide the Agent for all Telegram-related requests.

This project has **two separate Telegram integrations**. They must never be conflated:

1. **Telegram Bot** — Agent Chat channel.
2. **Telegram Personal Account** — the user's own Telegram account.

The Agent must determine which integration the user's request requires before selecting a tool.

---

# Telegram Integrations

## A. Telegram Bot — Agent Chat Channel

**Role:** User ↔ Assistant conversation surface.

| Item             | Detail                                                             |
| ---------------- | ------------------------------------------------------------------ |
| Protocol         | Telegram Bot HTTP API                                              |
| Python component | `TelegramConnector` (`connectors/telegram/`)                       |
| Config           | `TELEGRAM_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_AGENT_CHAT_ID` |
| Typical use      | User communicates with the Agent                                   |

Bot capabilities:

| Capability               | Approval |
| ------------------------ | -------- |
| `send_telegram_message`  | **Yes**  |
| `read_telegram_messages` | No       |

The bot operates as the bot identity. It does **not** operate as the user's personal Telegram account.

The bot can only access conversations where the bot is a participant.

---

## B. Telegram Personal Account — User's Account

**Role:** Agent acts through the user's authenticated Telegram account.

| Item     | Detail                                                                                                                      |
| -------- | --------------------------------------------------------------------------------------------------------------------------- |
| Protocol | Telethon / MTProto                                                                                                          |
| Config   | `TELEGRAM_USER_ENABLED`, `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_SESSION_FILE`, `TELEGRAM_INGESTION_POLL_SECONDS` |
| Session  | Authenticated personal Telegram session                                                                                     |
| Status   | Dependency/configuration may exist; always verify actual Tool Router registration before claiming an operation is available |

Personal-account capabilities from the project configuration:

| Capability               | Approval |
| ------------------------ | -------- |
| `send_telegram_message`  | **Yes**  |
| `send_telegram_file`     | **Yes**  |
| `find_telegram_contact`  | No       |
| `read_telegram_messages` | No       |

### Critical distinction

Bot API and personal-account Telegram are completely separate.

They have:

* different authentication;
* different sessions;
* different visibility;
* different identities;
* different capabilities.

If the user asks to message a friend **as the user**, use the personal-account connector when it is actually registered and enabled.

Do **not** use the Agent Bot unless the intended recipient is communicating with the bot.

---

# Core Agent Behavior

## 1. Use Available Tools

If the user's request matches a **registered and enabled Telegram tool**, use that tool.

Do **not** respond:

> "I can't access Telegram."

merely because Telegram is an external system.

Before claiming that a Telegram operation is unavailable:

1. Determine which Telegram integration is required.
2. Check the registered tools/capabilities.
3. Check whether the required connector is enabled/configured.
4. If a matching tool exists, use it.
5. Only report the capability as unavailable when the required tool/connector genuinely cannot perform the operation.

The skill describes capabilities and rules; the actual registered tools determine what can currently be executed.

---

## 2. Determine the Telegram Context

Interpret the user's intent before selecting the connector.

### User talking to the Agent

Example:

```text
User → Telegram Bot → Agent Core
```

Requests such as:

* "What's in my priority inbox?"
* "What did you receive?"
* "Help me send a message"

are Agent Chat conversations when they arrive through the Agent Bot.

Use the Agent Chat context and appropriate Agent tools.

### User wants to contact another person

Example:

```text
User → Agent → Rahul's Telegram
```

This requires the user's personal Telegram account when the message is intended to be sent **as the user**.

Do not confuse this with replying through the Agent Bot.

---

# Available Operations

## Bot API

Implemented operations:

* `read_telegram_messages`
* `send_telegram_message`

Use these only for operations supported by the Bot API integration.

---

## Personal Account

When the personal connector and tools are actually registered and enabled:

* Find Telegram contacts/chats.
* Read Telegram messages.
* Send Telegram messages as the user.
* Send files as the user.
* Receive inbound messages through ingestion when configured.

Do not claim that a personal-account capability is available solely because it is documented here. Verify the actual registered tool/connector state.

---

# When to Use

Use this skill when the user:

* mentions Telegram;
* says "TG";
* asks to send a Telegram message;
* asks to read Telegram messages;
* asks about a Telegram contact/chat;
* asks to send a file on Telegram;
* asks what happened on Telegram;
* asks the Agent to act on their Telegram account.

Examples:

```text
"Send Rahul a message on Telegram."

"Read my Telegram messages."

"Find Rahul on Telegram."

"Send report.pdf to Rahul on Telegram."

"What did I get on Telegram today?"
```

---

# When NOT to Use

Do not use this skill for:

* WhatsApp requests → use WhatsApp skill.
* Gmail requests → use Gmail skill.
* SMS requests → use Android skill.
* Phone calls → use Android skill.
* Android files → use Android skill for device access and File Skill for file handling.
* Laptop-only file operations → use File Skill.
* Non-Telegram conversations.

Do not assume that the Telegram Bot can perform actions on the user's personal account.

---

# Required Information

## Sending a Telegram Message

Required:

1. Recipient.
2. Message body.

Recipient can be:

* numeric Telegram `chat_id`;
* explicit `@username`;
* resolvable contact/chat name.

If the recipient is ambiguous, ask the user.

Do not guess.

---

## Reading Telegram Messages

Determine:

1. Which Telegram account should be read.
2. Which chat/contact, if specified.
3. Whether the request means recent messages or a specific conversation.

If the user says only:

> "Read my Telegram messages."

use the appropriate personal-account reader if available.

If the user is asking about messages received by the Agent Bot, use the Bot API path.

---

## Sending Telegram Files

Required:

1. Recipient.
2. File.
3. Optional message/caption.

The file must first be resolved and validated through the File Skill.

---

# Recipient Resolution

Follow this order:

### 1. Explicit Chat ID

If the user provides a numeric `chat_id`, use it directly.

### 2. Explicit Username

If the user provides an explicit Telegram `@username`, use it directly when supported by the connector.

### 3. Contact Name

If the user provides only a name:

```text
"Send Rahul a Telegram message."
```

use:

```text
find_telegram_contact
```

when the personal-account connector supports it.

If multiple matches are returned, ask the user which contact they mean.

If no match is found, ask for a username or chat ID.

### 4. Never Guess

Never send a message to a similarly named contact merely because the name appears close.

---

# Sending Messages

All outbound Telegram messages must follow the existing approval architecture.

Preferred flow:

```text
User Request
    ↓
Telegram Skill
    ↓
Resolve Recipient
    ↓
Create Draft / Task
    ↓
Approval
    ↓
Tool Router
    ↓
Telegram Connector
    ↓
Send
    ↓
Verify Result
    ↓
Report Result
```

The Agent must not bypass the approval system.

Do not send first and ask for approval afterward.

---

# Approval Requirements

| Operation                | Approval |
| ------------------------ | -------- |
| `read_telegram_messages` | No       |
| `find_telegram_contact`  | No       |
| `send_telegram_message`  | **Yes**  |
| `send_telegram_file`     | **Yes**  |

Use the existing task-ID approval mechanism:

```text
approve TASK-xxxxxxxx
```

or:

```text
reject TASK-xxxxxxxx
```

Do not invent a separate Telegram approval mechanism.

---

# File / Attachment Handling

Telegram Skill does not replace File Skill.

When a Telegram request involves a file:

```text
Telegram request
      ↓
File Skill
      ↓
find_file
      ↓
validate_file / prepare_attachment
      ↓
Telegram Skill
      ↓
Recipient resolution
      ↓
Draft
      ↓
Approval
      ↓
send_telegram_file
```

Maximum outbound file size:

```text
10 MB
```

controlled by:

```text
MAX_OUTBOUND_FILE_SIZE_MB
```

Files must be inside the approved file roots.

If the file is on Android:

```text
Android Device Agent
        ↓
UPLOAD_FILE
        ↓
Laptop data/outbound_files/
        ↓
File Skill
        ↓
validate / prepare attachment
        ↓
Telegram send flow
```

Do not attempt to send an Android file directly through the Telegram connector before it has been transferred and validated.

---

# Cross-Platform File Flow

Example:

> "Send report.pdf from my phone to Rahul on Telegram."

Required conceptual flow:

```text
User Request
      ↓
Android Skill
      ↓
UPLOAD_FILE
      ↓
Laptop
data/outbound_files/
      ↓
File Skill
      ↓
Resolve + Validate File
      ↓
Telegram Skill
      ↓
Resolve Rahul
      ↓
Create Draft
      ↓
User Approval
      ↓
send_telegram_file
      ↓
Telegram Personal Account
```

Load the appropriate skills for the individual parts of this workflow.

Do not treat the skill itself as an executable operation.

---

# Inbound Telegram Messages

Telegram messages arriving through configured ingestion follow the project's normal ingestion architecture.

Conceptually:

```text
Telegram Connector
      ↓
Ingestion
      ↓
Message Intelligence
      ↓
Priority Engine
      ↓
Priority Inbox
```

Do not automatically route every inbound Telegram message into Agent Chat.

Agent Chat and normal message ingestion are separate paths.

---

# Priority Inbox

Only query the priority inbox when the user explicitly asks for:

* priority messages;
* important messages;
* priority inbox;
* important Telegram/Gmail/WhatsApp/SMS messages.

Do not automatically query the priority inbox during ordinary Telegram conversations.

For a cross-platform priority request, use:

```text
query_priority_inbox
```

when that tool is available.

---

# Safety Rules

* Never expose Telegram Bot tokens.
* Never expose Telegram API hashes.
* Never expose personal-account session contents.
* Never place credentials in responses.
* Never guess a recipient.
* Never send without required approval.
* Never claim a send succeeded unless the connector/tool reports success.
* Never treat Bot API access as personal-account access.
* Never treat a skill file as an executable tool.
* Never bypass Tool Router, Task Manager, or Approval Manager.
* Never send to a similarly named contact without confirmation when identity is ambiguous.

---

# Failure Handling

| Situation                     | Agent Behavior                                                       |
| ----------------------------- | -------------------------------------------------------------------- |
| Telegram Bot disabled         | Report that Bot integration is unavailable                           |
| Bot token missing             | Report configuration issue without exposing credentials              |
| Personal account unavailable  | Explain that personal-account functionality is currently unavailable |
| Required tool not registered  | Report that the operation is not currently wired                     |
| Recipient not found           | Ask for username/chat ID                                             |
| Multiple recipient matches    | Ask user to choose                                                   |
| File not found                | Use File Skill / report that no matching file was found              |
| File too large                | Reject according to File Policy                                      |
| Approval rejected             | Do not send                                                          |
| Send failed                   | Report the actual failure                                            |
| Connection failure            | Report that Telegram connector could not complete the operation      |
| Unknown/unsupported operation | Explain the limitation instead of pretending success                 |

### Important

A tool being described in this skill does **not** guarantee that it is currently registered.

Likewise, absence of a capability from this skill does not permit the Agent to invent a capability.

The Agent must rely on the actual registered tools and connector state for execution.

---

# Examples

## Send a Message

**User:**

```text
Send Rahul on Telegram: I'll be late.
```

Flow:

```text
Telegram Skill
    ↓
find_telegram_contact("Rahul")
    ↓
Resolve Rahul
    ↓
Create message draft
    ↓
Approval
    ↓
send_telegram_message
```

If Rahul cannot be uniquely resolved, ask the user.

---

## Send to Explicit Username

**User:**

```text
Send @rahul123 on Telegram: I'll call you later.
```

Use the explicit username when supported.

Do not perform unnecessary contact guessing.

Then:

```text
Draft → Approval → Send
```

---

## Read Telegram

**User:**

```text
What did I receive on Telegram today?
```

Determine that the user is asking about their Telegram account.

If the personal-account reader is registered and enabled:

```text
read_telegram_messages
```

Otherwise, clearly explain that the personal-account reader is unavailable.

Do not substitute the Bot API and pretend it represents the user's personal Telegram inbox.

---

## Priority Inbox

**User:**

```text
Show my important messages.
```

Use:

```text
query_priority_inbox
```

when available.

Do not perform a Telegram-only read unless the user specifically asks for Telegram messages.

---

## Send a File

**User:**

```text
Send report.pdf to Rahul on Telegram.
```

Flow:

```text
File Skill
    ↓
find_file("report.pdf")
    ↓
validate_file / prepare_attachment
    ↓
Telegram Skill
    ↓
Resolve Rahul
    ↓
Create draft
    ↓
Approval
    ↓
send_telegram_file
```

---

## Android File → Telegram

**User:**

```text
Send the report from my phone to Rahul on Telegram.
```

Flow:

```text
Android Device Agent
    ↓
UPLOAD_FILE
    ↓
data/outbound_files/
    ↓
File Skill
    ↓
Validate file
    ↓
Telegram Skill
    ↓
Resolve Rahul
    ↓
Draft
    ↓
Approval
    ↓
send_telegram_file
```

Do not claim the file was sent until the final Telegram connector reports success.

---

# Tool / Connector Mapping

| Tool                     | Connector / Backend                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| `read_telegram_messages` | `TelegramConnector.get_updates()` for Bot API, or personal Telegram reader when registered |
| `send_telegram_message`  | ToolRouter → Telegram connector                                                            |
| `send_telegram_file`     | ToolRouter → Telegram personal connector                                                   |
| `find_telegram_contact`  | Personal Telegram / Telethon connector                                                     |
| `query_priority_inbox`   | Priority Engine                                                                            |
| Inbound normal messages  | Telegram Connector → Ingestion → Priority Engine                                           |
| Android file upload      | DeviceGateway → Android Device Agent                                                       |
| File resolution          | File Skill → `FileResolver`                                                                |
| File validation          | File Skill → `FilePolicy` / `AttachmentManager`                                            |

---

# Configuration

Telegram Bot:

```text
TELEGRAM_ENABLED
TELEGRAM_BOT_TOKEN
TELEGRAM_AGENT_CHAT_ID
```

Telegram Personal Account:

```text
TELEGRAM_USER_ENABLED
TELEGRAM_API_ID
TELEGRAM_API_HASH
TELEGRAM_SESSION_FILE
TELEGRAM_INGESTION_POLL_SECONDS
```

Do not place actual credentials in this skill.

Environment variables and credentials belong in the project's configuration / `.env` mechanism.

---

# Final Decision Rule

For every Telegram request:

```text
1. Understand what the user wants.
        ↓
2. Determine Bot vs Personal Account.
        ↓
3. Check actual registered/available Telegram tools.
        ↓
4. Resolve recipient/chat if required.
        ↓
5. Resolve and validate files if required.
        ↓
6. Create a task/draft for consequential actions.
        ↓
7. Ask for approval when policy requires it.
        ↓
8. Execute through the registered tool.
        ↓
9. Verify the result.
        ↓
10. Tell the user exactly what happened.
```

**Never guess. Never pretend success. Never claim a supported operation is unavailable without checking the actual registered tool/connector state.**
