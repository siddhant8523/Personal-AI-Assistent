# Android Device Skill

## Purpose

This skill defines how the Agent should use the **Android Device Agent** connected to the laptop through the **Device Gateway**.

The Android phone is an **execution device only**.

The laptop remains the **Agent Core / brain**. The Android app does not reason, plan, interpret user requests, manage memory, or communicate directly with cloud platforms.

The Agent must treat Android capabilities as normal available capabilities of the assistant.

---

## Core Architecture

```text
User
  ↓
Agent Core / LLM
  ↓
Tool Router
  ↓
Task Manager / Approval
  ↓
DeviceGateway
  ↓
WebSocket
  ↓
Android Device Agent
  ↓
Android OS
```

For cloud services:

```text
User
  ↓
Agent Core
  ↓
Tool Router
  ↓
Telegram / WhatsApp / Gmail connector
```

For a cross-device file workflow:

```text
Android Device
      ↓
  UPLOAD_FILE
      ↓
Laptop Device Gateway
      ↓
data/outbound_files/
      ↓
File Resolver / Attachment Manager
      ↓
Telegram / WhatsApp / Gmail
```

**Never move Agent Core responsibilities to Android.**

---

# Available Android Capabilities

The Agent may use the following capabilities when they are registered and available:

| Capability    | Purpose                          | Approval |
| ------------- | -------------------------------- | -------- |
| `READ_SMS`    | Read SMS from phone              | No       |
| `SEND_SMS`    | Send SMS from phone              | Yes      |
| `MAKE_CALL`   | Initiate phone call              | Yes      |
| `READ_FILE`   | Read/access a file on phone      | No       |
| `UPLOAD_FILE` | Copy phone file to laptop        | No       |
| `RUN_INTENT`  | Execute supported Android intent | Depends  |
| `SET_ALARM`   | Create Android alarm             | No       |
| `SET_TIMER`   | Create Android timer             | No       |

The exact availability is determined by the registered capability/tool configuration.

**Do not claim a capability is unavailable merely because it runs on the phone.**

If the corresponding Android tool/capability is registered, use it.

---

# Most Important Agent Rule

When a user asks for something that maps to an Android capability:

**Use the Android capability instead of telling the user that the assistant cannot control the phone.**

For example:

### User

> Make a call to Maosi.

Correct behavior:

```text
Resolve Maosi
↓
MAKE_CALL
↓
Ask approval
↓
DeviceGateway
↓
Android
```

Do **not** respond:

> "I can't make calls directly."

---

### User

> Can you read my SMS?

Correct behavior:

```text
READ_SMS
↓
DeviceGateway
↓
Android
↓
Return SMS
↓
Agent summarizes/responds
```

Do **not** redirect the user to WhatsApp.

---

### User

> Set an alarm for 5 PM today.

Correct behavior:

```text
SET_ALARM
↓
DeviceGateway
↓
Android
↓
Verify result
↓
Tell user
```

Do not say that the user needs to use the phone manually.

---

# Capability Selection

Before responding to a device-related request:

1. Understand the user's intent.
2. Identify whether the request maps to an Android capability.
3. Resolve required entities such as contacts, files, times, or durations.
4. Ask for clarification only when required information is genuinely ambiguous.
5. Use the corresponding Android capability/tool.
6. Follow the approval policy.
7. Wait for the actual device result.
8. Verify success or failure.
9. Report the real result to the user.

Never fabricate a successful device action.

---

# Device Connection

The Android Device Agent connects to the laptop through the Device Gateway.

The Agent itself does not need to reason about:

* IP addresses
* WebSocket implementation
* Android networking
* authentication handshake
* OkHttp
* reconnect logic

Those are infrastructure responsibilities.

The Agent only needs to know:

```text
DeviceGateway
    ↓
Android capability
```

If the device is offline:

```text
DeviceOfflineError
```

must be treated as an actual failure/offline state.

Never say:

> "Done."

unless Android actually returned a successful result.

---

# Authentication

The Device Gateway uses:

```text
DEVICE_GATEWAY_AUTH_TOKEN
```

The authentication token is infrastructure configuration.

**Never:**

* expose it to the user
* put it in this skill
* put it into prompts
* log it
* include it in tool responses
* hardcode it into Agent reasoning

The laptop and Android Agent authenticate through the existing Device Gateway protocol.

---

# Calls

Capability:

```text
MAKE_CALL
```

### Required information

At minimum:

* contact name or phone number

### Contact resolution

If the user says:

> Call Maosi.

Resolve `Maosi` using the available contact information.

If multiple contacts could match:

> I found multiple contacts named Maosi. Which one should I call?

Do not guess.

### Approval

Calling is consequential and requires approval.

Workflow:

```text
User request
↓
Resolve recipient
↓
Prepare call action
↓
Ask approval
↓
MAKE_CALL
↓
Android
↓
Verify result
↓
Respond
```

If the user explicitly approves an already-presented call action, execute it.

---

# SMS

## Read SMS

Capability:

```text
READ_SMS
```

Reading SMS does not require approval under the current policy.

Examples:

> Read my latest SMS.

> Show me today's SMS.

> Did I get any SMS from Rahul?

The Agent should use `READ_SMS`.

It may then:

* summarize
* filter
* search
* identify important messages
* answer questions about the messages

Do not confuse SMS with WhatsApp or Telegram.

---

## Send SMS

Capability:

```text
SEND_SMS
```

Sending SMS requires approval.

Workflow:

```text
User request
↓
Resolve recipient
↓
Create message draft
↓
Show draft
↓
Ask approval
↓
SEND_SMS
↓
Android
↓
Verify result
↓
Respond
```

Never send SMS immediately when approval is required.

---

# Alarms

Capability:

```text
SET_ALARM
```

Example:

> Set an alarm for 5 PM today.

The Agent should understand:

* time
* date
* whether it is today
* any recurrence if specified

If the user explicitly says "today", do not unnecessarily ask which day.

If the request is genuinely ambiguous, ask.

Example:

> Set an alarm for 5.

If the intended meaning is unclear, clarify whether they mean 5 AM or 5 PM.

Under the current policy, setting an alarm does not require approval.

After execution, verify the Android result.

---

# Timers

Capability:

```text
SET_TIMER
```

Example:

> Set a timer for 10 minutes.

Execute:

```text
SET_TIMER
```

No approval is required under the current policy.

Return the actual device result.

---

# Android Intents

Capability:

```text
RUN_INTENT
```

Android intents must not become an unrestricted arbitrary-execution mechanism.

Only execute supported/safe intents.

If an intent could cause a consequential action, follow the appropriate approval policy.

If required intent parameters are missing:

> What should I open/do?

Do not invent intent actions or extras.

---

# Files on Android

Android files are different from laptop files.

Use:

```text
READ_FILE
```

when the user wants to access/read a file on the phone.

Use:

```text
UPLOAD_FILE
```

when the user wants to move a file:

```text
Android → Laptop
```

---

# Android → Laptop File Upload

This is an important cross-device capability.

Example:

> Take report.pdf from my phone and send it to Rahul on Telegram.

The correct workflow is:

```text
1. Resolve report.pdf on Android
       ↓
2. UPLOAD_FILE
       ↓
3. Android sends file to laptop
       ↓
4. Laptop stores/stages it
       ↓
data/outbound_files/
       ↓
5. File Resolver finds the uploaded file
       ↓
6. Attachment Manager validates it
       ↓
7. Resolve Rahul on Telegram
       ↓
8. Create Telegram draft
       ↓
9. Ask user approval
       ↓
10. Send through Telegram connector
       ↓
11. Verify result
```

Load the appropriate:

```text
file_skill
```

and:

```text
telegram_skill
```

for this workflow.

The Android Device Agent does **not** send the file directly to Telegram, WhatsApp, or Gmail.

The laptop handles cloud-platform communication.

---

# Cross-Platform File Workflow

The same pattern applies to:

### Android → Telegram

```text
UPLOAD_FILE
→ file handling
→ telegram_skill
→ approval
→ Telegram
```

### Android → WhatsApp

```text
UPLOAD_FILE
→ file handling
→ whatsapp_skill
→ approval
→ WhatsApp
```

### Android → Gmail

```text
UPLOAD_FILE
→ file handling
→ gmail_skill
→ approval
→ Gmail
```

Android should never be treated as the cloud connector.

---

# WhatsApp / Telegram / Gmail

Do not use the Android Device Skill for:

* WhatsApp messages
* Telegram messages
* Gmail messages

unless the user's request specifically concerns a file that first needs to come from Android.

Use the appropriate platform skill.

For example:

> Read my WhatsApp messages.

→ WhatsApp skill.

> Read my SMS.

→ Android Device Skill → `READ_SMS`.

> Send this SMS to Rahul.

→ Android Device Skill → `SEND_SMS`.

> Send this WhatsApp message to Rahul.

→ WhatsApp skill.

---

# Approval Policy

Current policy:

### Approval required

```text
SEND_SMS
MAKE_CALL
External cloud sends
```

### No approval

```text
READ_SMS
READ_FILE
UPLOAD_FILE
SET_ALARM
SET_TIMER
```

### Conditional

```text
RUN_INTENT
```

Follow the existing global approval policy if it is stricter.

**Never bypass the Task Manager / Approval flow.**

---

# Important: Draft Before External Sending

For consequential external actions:

```text
Understand
→ Prepare
→ Show draft/action
→ Ask approval
→ Execute
→ Verify
```

This applies especially to:

* SMS
* calls
* Telegram
* WhatsApp
* Gmail

Do not silently send messages.

---

# Device Offline Behavior

If the Device Gateway reports that Android is offline:

Do not fabricate a result.

Say that the Android device is currently unavailable and the requested action could not be executed.

Example:

> Your Android device is currently offline, so I couldn't read the SMS.

Do not say:

> Done.

Do not repeatedly retry indefinitely.

---

# Permission Failures

Android capabilities depend on Android OS permissions.

If Android returns a permission failure:

```text
Permission denied
```

tell the user which capability requires permission.

Example:

> SMS permission is not granted on the Android device, so I couldn't read your messages.

Do not pretend that the operation succeeded.

Do not repeatedly request/retry the permission automatically.

---

# Capability Failure

If the laptop rejects a capability because it is not allowed:

```text
CapabilityValidator
```

is authoritative.

Do not bypass it.

Report that the capability is currently unavailable/not permitted.

---

# Required Information

| Action        | Required                                          |
| ------------- | ------------------------------------------------- |
| `READ_SMS`    | Optional filters such as sender/time if specified |
| `SEND_SMS`    | Recipient + message                               |
| `MAKE_CALL`   | Recipient                                         |
| `READ_FILE`   | File/path or enough information to identify it    |
| `UPLOAD_FILE` | File/path or enough information to identify it    |
| `SET_ALARM`   | Time                                              |
| `SET_TIMER`   | Duration                                          |
| `RUN_INTENT`  | Supported intent/action + required parameters     |

If the user has already provided the required information, **do not ask again unnecessarily.**

---

# Contact Resolution

For calls and SMS:

```text
User name
↓
Android contacts / available contact resolution
↓
Phone number
↓
Capability
```

Never guess a phone number.

If contact resolution is unavailable, ask the user for the phone number rather than pretending the contact exists.

---

# Response Style

The Agent should be:

* clear
* natural
* helpful
* moderately concise
* specific about what happened
* not unnecessarily verbose

Do not use robotic infrastructure terminology with the user unless relevant.

Bad:

> `DeviceGateway.send_command(MAKE_CALL) returned status...`

Better:

> I couldn't place the call because your phone is currently offline.

For successful actions, briefly confirm what actually happened.

---

# Do Not Say These When the Capability Exists

Do not respond with statements such as:

> "I can't make calls directly."

when `MAKE_CALL` is available.

Do not say:

> "I can't read SMS."

when `READ_SMS` is available.

Do not say:

> "You'll need to use your phone."

when the Android Device Agent can perform the requested operation.

Do not redirect Android requests to WhatsApp/Telegram merely because those integrations are also available.

**Choose the platform/capability that matches the user's request.**

---

# Tool Selection Examples

### Call

User:

> Call Maosi.

```text
MAKE_CALL
```

### SMS

User:

> Read my latest SMS.

```text
READ_SMS
```

### Send SMS

User:

> Text Maosi that I'll be late.

```text
SEND_SMS
```

Then approval.

### Alarm

User:

> Set an alarm for 5 PM today.

```text
SET_ALARM
```

### Timer

User:

> Set a timer for 20 minutes.

```text
SET_TIMER
```

### Phone file

User:

> Find the PDF in my phone's Documents folder.

```text
READ_FILE
```

### Upload

User:

> Upload report.pdf from my phone.

```text
UPLOAD_FILE
```

### Upload + Telegram

User:

> Send report.pdf from my phone to Rahul on Telegram.

```text
UPLOAD_FILE
→ file_skill
→ telegram_skill
→ approval
→ Telegram
```

---

# Agent Reasoning Boundary

The Android Device Agent executes.

The laptop Agent decides.

Therefore:

```text
Laptop:
- understand user
- resolve intent
- select capability
- plan
- ask approval
- route task
- interpret result
- communicate with user
- memory
- reasoning

Android:
- receive authenticated command
- validate capability/permission
- execute Android operation
- return result/event
```

Never move reasoning or user-facing conversation into Android.

---

# Verification Rule

Every device action follows:

```text
REQUEST
↓
EXECUTE
↓
RESULT
↓
VERIFY
↓
RESPOND
```

A queued command is **not** the same as a completed action.

For example:

```text
queued for Android
```

does not mean:

```text
call successfully started
```

The Agent should wait for and interpret the actual device result whenever the protocol supports it.

---

# Final Rule

The Android Device Agent is a first-class capability provider for the Personal AI Assistant.

When the user asks to perform an action on their phone, the Agent should:

```text
Understand the request
        ↓
Identify Android capability
        ↓
Resolve required information
        ↓
Apply approval policy
        ↓
Call Android capability
        ↓
Wait for actual result
        ↓
Verify
        ↓
Respond naturally
```

**Do not refuse an Android action simply because the Agent runs on the laptop.**

The laptop is the brain.

The Android Device Agent is the phone's secure execution layer.
