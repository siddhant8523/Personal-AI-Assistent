# Personal AI Assistant — Laptop Agent

Laptop-side "brain" of the personal AI assistant described in
`docs/architecture/`. This is the Agent Core, Memory, Message Intelligence,
Execution, and Connector planes — everything except the Android Device
Agent (a separate project).

## What's implemented

- Full pipeline: Connector -> Echo Filter -> Conversation Router ->
  (Agent Chat -> Agent Core) or (Normal Chat -> Ingestion -> Message
  Intelligence -> Priority Engine -> Priority Inbox)
- **Agent Core orchestration is now LangChain + LangGraph** (see
  `agent_core/graph/` and `docs/architecture/05_langgraph_migration.md`)
  -- a `StateGraph` with explicit nodes/edges replaces the old if/elif
  chain, wrapping the same existing TaskPlanner/PolicyEngine/
  ApprovalManager/ToolRouter as LangChain tools. Online (with
  `MISTRAL_API_KEY`) uses real `ChatMistralAI` tool-calling; offline
  falls back to the same deterministic regex intent parser as before.
- Memory: SOUL.md, USER.md, session/daily/long-term/semantic memory
  (SQLite-backed)
- Execution: task lifecycle state machine, Tool Router, Outbound Registry
- **The self-echo architecture fix**: `router/echo_filter.py` +
  `execution/outbound_registry.py` -- runs entirely outside the graph,
  before Agent Chat / Normal Chat routing even happens. See
  `docs/architecture/04_bug_and_fix.md`
- Connectors: Gmail (OAuth + send + bounded polling), Telegram Bot (Agent
  Chat) and separate personal Telegram (Telethon/MTProto), WhatsApp
  (Python interface + Baileys bridge for text/files)
- Device Gateway: authenticated websocket server started when
  `DEVICE_GATEWAY_ENABLED=true`; Android remains a separate device app
- CLI channel -- fully runnable without any external credentials

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # already done for you if you're reading this
python scripts/init_memory.py # creates SOUL.md / USER.md / sqlite db

python -m assistant.main       # or: PYTHONPATH=src python src/assistant/main.py
```

No API keys are required to try it — with no `MISTRAL_API_KEY` set, the
LLM client runs in a deterministic offline mode so the full pipeline
(routing, priority scoring, task lifecycle, approvals, the echo filter)
still works end-to-end. Add a real key to `.env` to get real reasoning and
free-text intent parsing.

### CLI commands

```
<anything>                          -> sent to Agent Chat
send Rahul on whatsapp: hello        -> creates a draft + asks for approval
approve TASK-xxxxxxxx                -> approves & executes a pending task
reject TASK-xxxxxxxx                 -> rejects a pending task
/sim whatsapp Rahul Can you call me urgently   -> simulate an inbound normal message
/priority                            -> show today's priority inbox
/echo-demo                           -> demonstrates the self-echo bug being caught
/quit
```

## Enabling real connectors

- **Gmail**: follow `scripts/setup_gmail_oauth.py`, then set
  `GMAIL_ENABLED=true` in `.env`.
- **Telegram Bot control chat**: create a bot via @BotFather, put the token
  in `TELEGRAM_BOT_TOKEN`, set `TELEGRAM_ENABLED=true` and set the allowed
  `TELEGRAM_AGENT_CHAT_ID`.
- **Personal Telegram messaging**: set `TELEGRAM_USER_ENABLED=true`, API ID,
  API hash, and session-file path. Authorize the Telethon session once before
  starting the agent; it is intentionally independent of the Bot token.
- **WhatsApp**: `cd src/assistant/connectors/whatsapp/baileys_bridge &&
  npm install && npm start`, scan the QR code once, then set
  `WHATSAPP_ENABLED=true`.
- **Android device**: set `DEVICE_GATEWAY_ENABLED=true` and configure a
  strong `DEVICE_GATEWAY_AUTH_TOKEN`. The Android app connects to the
  laptop websocket and is still a separate project.

## Running tests

```bash
pytest tests/unit -v      # 20 tests: router, echo filter, priority engine,
                           # task manager, and the LangGraph orchestrator
```

## Operational notes

- Inbound Gmail currently uses bounded polling. Gmail Pub/Sub push delivery
  still requires deployment-specific webhook infrastructure and a Google
  Cloud Pub/Sub topic, so it is intentionally not enabled by default.
- Personal Telegram must already have an authorised Telethon session. This
  avoids an unattended laptop process prompting for a phone-login code.
- The Android gateway dispatches only allow-listed capabilities. It can accept
  `SMS_INBOUND` events, but the Android application remains responsible for
  OS permissions and actual SMS/call/alarm/file APIs.

## Project layout

See `docs/architecture/` for the full design docs, and the directory
structure explanation given alongside this project for what each module
maps to (Channel / Connector / Router / Ingestion / Message Intelligence /
Agent Core / Memory / Execution / Device Gateway planes).
