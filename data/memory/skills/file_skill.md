# File Skill

## Important: Skills Are NOT Tools

A skill file is **instructions and domain knowledge** for the Agent. It does
not execute anything by itself.

```
Skill MD
   ↓
Agent understands capability / rules / constraints
   ↓
Agent selects the appropriate tool (or asks for clarification)
   ↓
Tool Router
   ↓
Connector / Device Gateway
   ↓
External system (Gmail, Telegram, WhatsApp, Android, filesystem)
```

Never treat reading a skill as performing an action. Always use the
registered LangChain tools and the Task Manager / Approval flow for side
effects.


## Purpose

Guide the Agent whenever a request involves finding, validating, attaching,
or transferring files — on the laptop or from Android — before sending via
Gmail, Telegram, or WhatsApp.

## Capabilities

Related tool capabilities (not the skill itself):

| Tool | Role |
|------|------|
| `find_file` | Fuzzy search in allowed directories |
| `validate_file` | Check path policy + size |
| `prepare_attachment` | Same validation, attachment-ready metadata |
| `send_*_file` / attachments | Platform sends after approval |

File access is **not** arbitrary filesystem access.

## Available Operations

- **find_file** — fuzzy search by filename/description within allowed roots
- **validate_file** / **prepare_attachment** — enforce FilePolicy (roots + 10 MB)
- **upload_file** (via Android) — stage phone files into `data/outbound_files/`
- Platform **send_*_file** tools — after validation, draft, and approval

## Approved File Directories

From `config/capabilities.yaml` and runtime wiring:

- `./data/files`
- `./data/outbound_files` (primary staging for outbound attachments)
- `~/Documents/AssistantFiles`

At runtime, `main.py` configures `FilePolicy` with
`OUTBOUND_FILES_DIR` (default `./data/outbound_files`) and
`MAX_OUTBOUND_FILE_SIZE_MB` (default **10**).

Only paths under allowed roots may be read or attached. Paths outside are
**rejected** (`PermissionError`).

## Components

| Component | Responsibility |
|-----------|----------------|
| **FilePolicy** | Allowed roots + max size (10 MB default) |
| **FileResolver** | Walk allowed roots; fuzzy match filename; ambiguity → error |
| **AttachmentManager** | `prepare(path)` — policy check + metadata |

## Maximum Outbound Size

**10 MB** per file (configurable via `MAX_OUTBOUND_FILE_SIZE_MB`).

Files above the limit are rejected in **validate_file** /
**prepare_attachment** with `ValueError`. Do not attempt to send.

## Android → Laptop File Transfer

When the user references a file on the phone:

1. Android Device Agent locates file on device (user-provided path hint).
2. **upload_file** capability pulls file to laptop
   `data/outbound_files/`.
3. **FileResolver** finds the uploaded file by name/query.
4. **AttachmentManager** validates size ≤ 10 MB.
5. Recipient resolution on target platform.
6. Draft → user approval → connector send.

See **android_skill** for the full cross-platform example.

## Telegram / WhatsApp / Gmail Attachments

Common pattern:

```
find_file(query)
   ↓
validate_file(path) or prepare_attachment(path)
   ↓
include in send task (platform-specific)
   ↓
approval
   ↓
Tool Router → connector
```

Each platform skill describes send-specific details.

## When to Use

- User mentions a filename, PDF, document, attachment, "send the file",
  "find report", path under Documents/Work, etc.
- Before any `send_*_file` or email attachment.

## When NOT to Use

- Pure text messages with no file — skip file tools.
- Reading Android SMS or call logs — **android_skill**, not file resolver.
- Do not use `find_file` to scan entire disk — only approved roots.

## Required Information

1. **File identification** — name, partial name, or description
   ("Q3 report pdf")
2. **Platform + recipient** — if sending
3. **Optional message** — caption/body accompanying file

If **find_file** returns multiple candidates (`AmbiguousFileError`), present
choices and ask the user to pick one.

## Recipient Resolution

File skill does not resolve recipients — delegate to platform skill after
file is validated.

## File/Attachment Handling (detailed)

### Missing file

- **find_file** returns none → tell user file not found in allowed dirs;
  suggest correct name or upload from Android.

### Unsupported file

- Binary/type restrictions may apply per connector — if send fails, report
  error; do not claim success.

### Ambiguous match

- List candidates; require user disambiguation.

### Over size limit

- "File exceeds 10 MB limit" — suggest compression, cloud link, or split.

## Approval Requirements

Sending files always requires approval on WhatsApp, Telegram, Gmail
attachment capabilities. Validation alone does not.

## Safety Rules

- Never read `/etc`, `~/.ssh`, or paths outside allowed roots.
- Never embed file bytes in LLM context for large files — use metadata only.
- Do not exfiltrate directory listings of user home — search by query only.

## Failure Handling

| Error | Meaning |
|-------|---------|
| `PermissionError` | Path outside allowed roots |
| `ValueError` | File too large |
| `AmbiguousFileError` | Multiple matches — ask user |
| `None` from find | No match |

## Examples

**User:** "Send report.pdf to Rahul on Telegram"
→ find_file("report.pdf") → validate → telegram send flow (see
**telegram_skill** + **android_skill** if file on phone).

**User:** "Is the deck under 10 megabytes?"
→ find_file → validate_file → report size from metadata.

## Tool/Connector Mapping

| Tool | Module |
|------|--------|
| `find_file` | `FileResolver.find()` |
| `validate_file` | `AttachmentManager.prepare()` |
| `prepare_attachment` | `AttachmentManager.prepare()` |
| Policy | `FilePolicy.is_allowed()`, `check_size()` |

Env: `OUTBOUND_FILES_DIR`, `MAX_OUTBOUND_FILE_SIZE_MB`.
