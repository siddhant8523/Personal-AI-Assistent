/**
 * Baileys WhatsApp Bridge
 * ========================
 *
 * Loads configuration from the project's ROOT .env file.
 *
 * Project structure:
 *
 * personal-ai-assistant/
 * ├── .env
 * └── src/
 *     └── assistant/
 *         └── connectors/
 *             └── whatsapp/
 *                 └── baileys_bridge/
 *                     └── index.js
 *
 * Start:
 *   cd src/assistant/connectors/whatsapp/baileys_bridge
 *   npm install
 *   npm start
 */

const path = require("path");
const fs = require("fs");

// ============================================================
// ENVIRONMENT
// ============================================================

// index.js:
// baileys_bridge/
// whatsapp/
// connectors/
// assistant/
// src/
// PROJECT ROOT
const PROJECT_ROOT = path.resolve(__dirname, "../../../../..");

require("dotenv").config({
  path: path.join(PROJECT_ROOT, ".env"),
  quiet: true,
});

// ============================================================
// DEPENDENCIES
// ============================================================

const express = require("express");
const multer = require("multer");
const qrcode = require("qrcode-terminal");

const {
  default: makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  areJidsSameUser,
  isLidUser,
  isJidUser,
  jidNormalizedUser,
  downloadMediaMessage,
} = require("@whiskeysockets/baileys");

const { NodeCache } = require("@cacheable/node-cache");

// ============================================================
// BOUNDED MESSAGE STORE & RETRY CACHE
// ============================================================

class BoundedMessageStore {
  constructor(maxSize = 1000) {
    this.maxSize = maxSize;
    this.store = new Map();
  }

  set(id, message) {
    if (!id || !message) return;
    if (this.store.has(id)) {
      this.store.delete(id);
    } else if (this.store.size >= this.maxSize) {
      const oldestKey = this.store.keys().next().value;
      this.store.delete(oldestKey);
    }
    this.store.set(id, message);
  }

  get(id) {
    if (!id) return undefined;
    const item = this.store.get(id);
    if (item) {
      this.store.delete(id);
      this.store.set(id, item);
    }
    return item;
  }

  has(id) {
    return this.store.has(id);
  }

  get size() {
    return this.store.size;
  }

  clear() {
    this.store.clear();
  }
}

const messageStore = new BoundedMessageStore(1000);
const msgRetryCounterCache = new NodeCache({
  stdTTL: 3600,
  useClones: false,
});

// ============================================================
// CONFIGURATION
// ============================================================

const PORT = Number(
  process.env.WHATSAPP_BRIDGE_PORT || 3000
);

const BRIDGE_URL =
  process.env.WHATSAPP_BRIDGE_URL ||
  `http://localhost:${PORT}`;

const WHATSAPP_ENABLED =
  String(process.env.WHATSAPP_ENABLED || "false")
    .toLowerCase() === "true";

const AGENT_CHAT_ID =
  (process.env.WHATSAPP_AGENT_CHAT_ID || "").trim();

const SELF_NUMBER =
  (process.env.WHATSAPP_SELF_NUMBER || "").trim();

const INGESTION_POLL_SECONDS =
  Number(
    process.env.WHATSAPP_INGESTION_POLL_SECONDS || 5
  );

function normalizePnJid(jidOrPhone) {
  if (!jidOrPhone) return "";
  let s = String(jidOrPhone).trim();
  if (s.includes("@")) {
    return jidNormalizedUser(s);
  }
  const digits = s.replace(/\D/g, "");
  return digits ? `${digits}@s.whatsapp.net` : "";
}

const TARGET_AGENT_PN_JID = normalizePnJid(AGENT_CHAT_ID);

// ============================================================
// VALIDATION / CONFIG LOGGING
// ============================================================

console.log("[WhatsApp] Bridge started");
if (process.env.DEBUG === "true") {
  console.log(`[WhatsApp] Configuration: Agent Chat: ${AGENT_CHAT_ID ? "configured" : "not set"}, Self: ${SELF_NUMBER ? "configured" : "not set"}`);
}

if (!WHATSAPP_ENABLED) {
  console.warn(
    "[WhatsApp] WARNING: WHATSAPP_ENABLED=false"
  );
}

if (!AGENT_CHAT_ID) {
  console.warn(
    "[WhatsApp] WARNING: WHATSAPP_AGENT_CHAT_ID is not configured."
  );
}

// ============================================================
// EXPRESS
// ============================================================

const app = express();

app.use(express.json());

// ============================================================
// FILE UPLOAD
// ============================================================

const upload = multer({
  storage: multer.memoryStorage(),

  limits: {
    // Maximum outbound file size = 10 MB
    fileSize: 10 * 1024 * 1024,
  },
});

// ============================================================
// AUTH SESSION & PROCESS LOCK
// ============================================================

const AUTH_STATE_DIR = path.join(
  __dirname,
  "auth_state"
);

let lockFd = null;

function acquireProcessLock() {
  if (!fs.existsSync(AUTH_STATE_DIR)) {
    fs.mkdirSync(AUTH_STATE_DIR, { recursive: true });
  }
  const lockFile = path.join(AUTH_STATE_DIR, ".bridge.lock");
  const lockData = JSON.stringify({
    pid: process.pid,
    created_at: Date.now(),
  });

  try {
    lockFd = fs.openSync(lockFile, "wx");
    fs.writeFileSync(lockFd, lockData);
    if (process.env.DEBUG === "true") {
      console.debug("[WhatsApp] Bridge lock acquired");
    }
    return true;
  } catch (err) {
    if (err.code === "EEXIST") {
      try {
        const content = fs.readFileSync(lockFile, "utf8");
        const parsed = JSON.parse(content);
        const existingPid = parsed.pid;
        if (existingPid && existingPid !== process.pid) {
          try {
            process.kill(existingPid, 0);
            console.error(
              `[ProcessLock] Another bridge instance (PID ${existingPid}) is active. Refusing duplicate startup.`
            );
            process.exit(1);
          } catch (killErr) {
            if (killErr.code === "ESRCH") {
              console.warn(
                `[ProcessLock] Reclaiming stale lock from dead PID ${existingPid}.`
              );
              try {
                fs.unlinkSync(lockFile);
              } catch (_) {}
              lockFd = fs.openSync(lockFile, "wx");
              fs.writeFileSync(lockFd, lockData);
              if (process.env.DEBUG === "true") {
                console.debug("[WhatsApp] Bridge lock acquired");
              }
              return true;
            }
            throw killErr;
          }
        }
      } catch (parseErr) {
        console.warn("[ProcessLock] Corrupt lock file detected, recreating:", parseErr.message);
        try {
          fs.unlinkSync(lockFile);
        } catch (_) {}
        lockFd = fs.openSync(lockFile, "wx");
        fs.writeFileSync(lockFd, lockData);
        return true;
      }
    }
    throw err;
  }
}

function releaseProcessLock() {
  if (lockFd !== null) {
    try {
      fs.closeSync(lockFd);
    } catch (_) {}
    lockFd = null;
  }
  const lockFile = path.join(AUTH_STATE_DIR, ".bridge.lock");
  try {
    if (fs.existsSync(lockFile)) {
      const content = fs.readFileSync(lockFile, "utf8");
      const parsed = JSON.parse(content);
      if (parsed.pid === process.pid) {
        fs.unlinkSync(lockFile);
        console.log("[WhatsApp] Bridge lock released");
      }
    }
  } catch (_) {}
}

process.on("exit", () => {
  releaseProcessLock();
});

process.on("SIGINT", () => {
  releaseProcessLock();
  process.exit(0);
});

process.on("SIGTERM", () => {
  releaseProcessLock();
  process.exit(0);
});

// ============================================================
// STATE & LID CACHE
// ============================================================

let sock = null;
let reconnecting = false;
let socketInstanceId = 0;
let isConnected = false;
let latestQr = null;

const recentBuffer = [];
const MAX_RECENT_MESSAGES = 200;
const seenMsgIds = new Set();

const lidToPnMap = {};
const pnToLidMap = {};

async function resolveJidToPn(rawJid) {
  if (!rawJid) return "";
  const normalized = jidNormalizedUser(rawJid);

  if (isJidUser(normalized)) {
    return normalized;
  }

  if (isLidUser(normalized)) {
    if (lidToPnMap[normalized]) {
      return lidToPnMap[normalized];
    }

    if (sock && TARGET_AGENT_PN_JID) {
      try {
        const results = await sock.onWhatsApp(TARGET_AGENT_PN_JID);
        if (Array.isArray(results)) {
          for (const res of results) {
            if (res?.jid && res?.lid) {
              const resPn = jidNormalizedUser(res.jid);
              const resLid = jidNormalizedUser(res.lid);
              lidToPnMap[resLid] = resPn;
              pnToLidMap[resPn] = resLid;
            }
          }
          if (lidToPnMap[normalized]) {
            return lidToPnMap[normalized];
          }
        }
      } catch (err) {
        console.error("[WhatsApp] Error resolving LID via onWhatsApp:", err.message);
      }
    }
  }

  return normalized;
}

async function getTargetJid(chatId) {
  if (!chatId) return chatId;
  const normalized = jidNormalizedUser(String(chatId).trim());

  if (isLidUser(normalized) || normalized.endsWith("@g.us") || normalized.endsWith("@broadcast")) {
    return normalized;
  }

  if (pnToLidMap[normalized]) {
    return pnToLidMap[normalized];
  }

  if (sock && isJidUser(normalized)) {
    try {
      const results = await sock.onWhatsApp(normalized);
      if (Array.isArray(results)) {
        for (const res of results) {
          if (res?.jid && res?.lid) {
            const resPn = jidNormalizedUser(res.jid);
            const resLid = jidNormalizedUser(res.lid);
            lidToPnMap[resLid] = resPn;
            pnToLidMap[resPn] = resLid;
          }
        }
        if (pnToLidMap[normalized]) {
          return pnToLidMap[normalized];
        }
      }
    } catch (err) {
      console.error("[WhatsApp] Error resolving target LID via onWhatsApp:", err.message);
    }
  }

  return normalized;
}

// ============================================================
// HELPERS
// ============================================================

async function isAgentChat(chatId) {
  if (!TARGET_AGENT_PN_JID) {
    return false;
  }

  const resolved = await resolveJidToPn(chatId);
  return (
    areJidsSameUser(resolved, TARGET_AGENT_PN_JID) ||
    chatId === AGENT_CHAT_ID ||
    chatId === TARGET_AGENT_PN_JID
  );
}

function requireConnection(res) {
  if (!sock) {
    res.status(503).json({
      status: "failed",
      detail: "WhatsApp is not connected",
    });

    return false;
  }

  return true;
}

// ============================================================
// SAFE LOGGING & SANITIZATION
// ============================================================

function redactText(text) {
  if (!text || typeof text !== "string") return "";
  return text
    .replace(/[0-9]{8,15}@s\.whatsapp\.net/g, "[REDACTED_JID]")
    .replace(/[0-9]{12,18}@lid/g, "[REDACTED_LID]")
    .replace(/(?<=\D|^)\+?[0-9]{10,15}(?=\D|$)/g, "[REDACTED_NUMBER]");
}

function createSafeBaileysLogger() {
  const isDebug = process.env.DEBUG === "true" || process.env.LOG_LEVEL === "DEBUG";

  const DISCARD_FIELDS = new Set([
    "histNotification",
    "initialHistBootstrapInlinePayload",
    "mediaKey",
    "fileSha256",
    "fileEncSha256",
    "directPath",
    "encHandle",
    "ephemeral",
    "clientHello",
    "helloMsg",
    "node",
    "auth",
    "creds",
    "keys",
    "signal",
    "message",
    "messages",
    "ws",
    "socket",
  ]);

  const USELESS_MSGS = new Set([
    "connected to WA",
    "logging in...",
    "handled 0 offline messages/notifications",
    "Connection is now AwaitingInitialSync, buffering events",
    "History sync is disabled by config, not waiting for notification. Transitioning to Online.",
    "15 pre-keys found on server",
    "pre-keys found on server",
    "uploading pre-keys",
    "uploaded pre-keys",
    "opened connection to WA",
    "clean dirty bits account_sync",
    "resyncing critical_block from v0",
    "resyncing regular from v0",
    "failed to sync state from version",
    "got history notification",
    "no name present, ignoring presence update request...",
  ]);

  function isUseless(msg, obj) {
    if (typeof msg === "string") {
      if (USELESS_MSGS.has(msg)) return true;
      for (const u of USELESS_MSGS) {
        if (msg.includes(u)) return true;
      }
    }
    if (obj && typeof obj === "object") {
      for (const k of Object.keys(obj)) {
        if (DISCARD_FIELDS.has(k)) return true;
      }
      if (typeof obj.msg === "string") {
        if (USELESS_MSGS.has(obj.msg)) return true;
        for (const u of USELESS_MSGS) {
          if (obj.msg.includes(u)) return true;
        }
      }
    }
    return false;
  }

  function formatSafe(msg, obj) {
    if (typeof msg === "string" && msg) {
      return redactText(msg);
    }
    if (obj && typeof obj === "object" && typeof obj.msg === "string") {
      return redactText(obj.msg);
    }
    return null;
  }

  const logger = {
    level: isDebug ? "debug" : "warn",
    trace: () => {},
    debug: (arg1, arg2) => {
      if (!isDebug) return;
      const obj = typeof arg1 === "object" ? arg1 : null;
      const msg = typeof arg1 === "string" ? arg1 : (typeof arg2 === "string" ? arg2 : "");
      if (isUseless(msg, obj)) return;
      const safe = formatSafe(msg, obj);
      if (safe) console.debug(`[WhatsApp Debug] ${safe}`);
    },
    info: (arg1, arg2) => {
      const obj = typeof arg1 === "object" ? arg1 : null;
      const msg = typeof arg1 === "string" ? arg1 : (typeof arg2 === "string" ? arg2 : "");
      if (isUseless(msg, obj)) return;
      const safe = formatSafe(msg, obj);
      if (safe && isDebug) {
        console.log(`[WhatsApp] ${safe}`);
      }
    },
    warn: (arg1, arg2) => {
      const obj = typeof arg1 === "object" ? arg1 : null;
      const msg = typeof arg1 === "string" ? arg1 : (typeof arg2 === "string" ? arg2 : "");
      if (isUseless(msg, obj)) return;
      const safe = formatSafe(msg, obj);
      if (safe) {
        console.warn(`[WhatsApp] Warning: ${safe}`);
      }
    },
    error: (arg1, arg2) => {
      const obj = typeof arg1 === "object" ? arg1 : null;
      const msg = typeof arg1 === "string" ? arg1 : (typeof arg2 === "string" ? arg2 : "");
      if (isUseless(msg, obj)) return;
      let errText = formatSafe(msg, obj);
      if (!errText && obj && obj.error) {
        errText = typeof obj.error.message === "string" ? obj.error.message : "Internal error";
      }
      if (errText) {
        console.error(`[WhatsApp] Error: ${redactText(errText)}`);
      }
    },
    fatal: (arg1, arg2) => {
      const msg = typeof arg1 === "string" ? arg1 : "Fatal error";
      console.error(`[WhatsApp] Fatal: ${redactText(msg)}`);
    },
    child: () => logger,
  };

  return logger;
}

// ============================================================
// START WHATSAPP
// ============================================================

async function start() {
  socketInstanceId++;
  const currentSocketInstanceId = socketInstanceId;

  console.log("[WhatsApp] Connecting...");

  const {
    state,
    saveCreds,
  } = await useMultiFileAuthState(
    AUTH_STATE_DIR
  );

  // Safe SignalKeyStore instrumentation (debug only)
  const originalKeysGet = state.keys.get;
  const originalKeysSet = state.keys.set;

  state.keys.get = async (type, ids) => {
    const startTs = Date.now();
    const result = await originalKeysGet(type, ids);
    const count = Array.isArray(ids) ? ids.length : 0;
    if (process.env.DEBUG === "true" && (type === "session" || type === "pre-key")) {
      console.debug(
        `[SignalKeyStore] get category=${type} ids=${count} durationMs=${Date.now() - startTs}`
      );
    }
    return result;
  };

  state.keys.set = async (data) => {
    const startTs = Date.now();
    await originalKeysSet(data);
    if (process.env.DEBUG === "true") {
      for (const category of Object.keys(data || {})) {
        const count = Object.keys(data[category] || {}).length;
        if (category === "session" || category === "pre-key" || category === "sender-key") {
          console.debug(
            `[SignalKeyStore] set category=${category} ids=${count} durationMs=${Date.now() - startTs}`
          );
        }
      }
    }
  };

  sock = makeWASocket({
    auth: state,
    logger: createSafeBaileysLogger(),
    msgRetryCounterCache,
    getMessage: async (key) => {
      const msgId = key?.id;
      if (!msgId) {
        return undefined;
      }
      const stored = messageStore.get(msgId);
      if (stored) {
        if (process.env.DEBUG === "true") {
          console.debug(`[WhatsApp Bridge] Retry getMessage HIT: id=${msgId}`);
        }
        return stored.message || stored;
      }
      return undefined;
    },
  });

  // Save authentication credentials.
  sock.ev.on(
    "creds.update",
    saveCreds
  );

  sock.ev.on("contacts.upsert", (contacts) => {
    for (const c of contacts) {
      if (c.id && c.lid) {
        const pnJid = jidNormalizedUser(c.id);
        const lidJid = jidNormalizedUser(c.lid);
        lidToPnMap[lidJid] = pnJid;
        pnToLidMap[pnJid] = lidJid;
      }
      if (c.lid && c.phoneNumber) {
        const lidJid = jidNormalizedUser(c.lid);
        const pnJid = jidNormalizedUser(c.phoneNumber);
        lidToPnMap[lidJid] = pnJid;
        pnToLidMap[pnJid] = lidJid;
      }
    }
  });

  sock.ev.on("contacts.update", (updates) => {
    for (const c of updates) {
      if (c.id && c.lid) {
        const pnJid = jidNormalizedUser(c.id);
        const lidJid = jidNormalizedUser(c.lid);
        lidToPnMap[lidJid] = pnJid;
        pnToLidMap[pnJid] = lidJid;
      }
    }
  });

  // ==========================================================
  // CONNECTION UPDATE
  // ==========================================================

  sock.ev.on(
    "connection.update",
    async (update) => {
      const {
        connection,
        lastDisconnect,
        qr,
      } = update;

      // ------------------------------------------------------
      // QR CODE
      // ------------------------------------------------------

      if (qr) {
        latestQr = qr;
        isConnected = false;
        console.log("[WhatsApp] Authentication required");
        console.log("[WhatsApp] Waiting for QR scan");
        qrcode.generate(qr, { small: true });
      }

      // ------------------------------------------------------
      // CONNECTED
      // ------------------------------------------------------

      if (connection === "open") {
        reconnecting = false;
        isConnected = true;
        latestQr = null;
        console.log("[WhatsApp] Connected");

        if (TARGET_AGENT_PN_JID && sock) {
          try {
            const results = await sock.onWhatsApp(TARGET_AGENT_PN_JID);
            if (Array.isArray(results)) {
              for (const res of results) {
                if (res?.jid && res?.lid) {
                  const resPn = jidNormalizedUser(res.jid);
                  const resLid = jidNormalizedUser(res.lid);
                  lidToPnMap[resLid] = resPn;
                  pnToLidMap[resPn] = resLid;
                }
              }
            }
          } catch (err) {
            if (process.env.DEBUG === "true") {
              console.debug("[WhatsApp Debug] Initial LID resolution error:", err.message);
            }
          }
        }
      }

      // ------------------------------------------------------
      // CLOSED
      // ------------------------------------------------------

      if (connection === "close") {
        isConnected = false;
        const statusCode =
          lastDisconnect?.error?.output?.statusCode;

        const shouldReconnect =
          statusCode !==
          DisconnectReason.loggedOut;

        if (sock) {
          try {
            sock.ev.removeAllListeners();
          } catch (_) {}
          sock = null;
        }

        if (
          shouldReconnect &&
          !reconnecting
        ) {
          reconnecting = true;
          console.log("[WhatsApp] Connection lost; reconnecting...");

          setTimeout(
            () => {
              start().catch(
                (error) => {
                  reconnecting = false;
                  console.error(
                    "[WhatsApp] Reconnection failed:",
                    error?.message || "unknown error"
                  );
                }
              );
            },
            3000
          );
        }

        if (!shouldReconnect) {
          console.log(
            "[WhatsApp] Authentication required"
          );
        }
      }
    }
  );

  // ==========================================================
  // INCOMING WHATSAPP MESSAGES
  // ==========================================================

  sock.ev.on(
    "messages.upsert",
    async ({ messages }) => {
      for (const m of messages) {
        if (!m?.key?.remoteJid) {
          continue;
        }

        console.log("[WhatsApp] Incoming message received");

        if (m.key?.id && m.message) {
          messageStore.set(m.key.id, { key: m.key, message: m.message });
          console.log("[WhatsApp] Incoming message stored");
        }

        if (m.key?.id) {
          if (seenMsgIds.has(m.key.id)) {
            if (process.env.DEBUG === "true") {
              console.debug("[WhatsApp] Duplicate message ignored");
            }
            continue;
          }
          seenMsgIds.add(m.key.id);
          if (seenMsgIds.size > 1000) {
            const firstKey = seenMsgIds.values().next().value;
            seenMsgIds.delete(firstKey);
          }
        }

        const rawChatId = m.key.remoteJid;
        const fromMe = !!m.key.fromMe;

        if (m.key.remoteJidAlt && isJidUser(m.key.remoteJidAlt) && isLidUser(rawChatId)) {
          const altPn = jidNormalizedUser(m.key.remoteJidAlt);
          const lidNorm = jidNormalizedUser(rawChatId);
          lidToPnMap[lidNorm] = altPn;
          pnToLidMap[altPn] = lidNorm;
        }

        const resolvedPnJid = await resolveJidToPn(rawChatId);

        let text = "";
        let mediaType = null;
        let audioMimetype = null;
        if (m.message?.conversation) {
          text = m.message.conversation;
        } else if (m.message?.extendedTextMessage?.text) {
          text = m.message.extendedTextMessage.text;
        } else if (m.message?.audioMessage) {
          mediaType = "audio";
          audioMimetype = m.message.audioMessage.mimetype || "audio/ogg; codecs=opus";
          if (process.env.DEBUG === "true") {
            console.debug(`[WhatsApp] Inbound audio message detected`);
          }
        }

        if (process.env.DEBUG === "true") {
          console.debug(`[WhatsApp] IN raw=${redactText(rawChatId)} resolved=${redactText(resolvedPnJid || rawChatId)}`);
        }

        recentBuffer.push({
          id: m.key.id || "",
          chat_id: resolvedPnJid || rawChatId,
          raw_chat_id: rawChatId,
          sender: m.key.participant || rawChatId,
          text,
          media_type: mediaType,
          mimetype: audioMimetype,
          from_me: fromMe,
          timestamp: new Date().toISOString(),
        });

        if (
          recentBuffer.length >
          MAX_RECENT_MESSAGES
        ) {
          recentBuffer.shift();
        }
      }
    }
  );
}

// ============================================================
// HEALTH
// ============================================================

app.get(
  "/health",
  (req, res) => {
    res.json({
      status: "ok",

      enabled:
        WHATSAPP_ENABLED,

      connected:
        isConnected,

      reconnecting:
        reconnecting,

      qr:
        latestQr,

      agent_chat_id:
        AGENT_CHAT_ID || null,

      message_store_size:
        messageStore.size,
    });
  }
);

app.get(
  "/qr",
  (req, res) => {
    res.json({
      qr: latestQr,
      connected: isConnected,
    });
  }
);

// ============================================================
// SEND MESSAGE
// ============================================================

app.post(
  "/send",
  async (req, res) => {
    try {
      if (!requireConnection(res)) {
        return;
      }

      const {
        chat_id,
        text,
      } = req.body;

      if (!chat_id) {
        return res.status(400).json({
          status: "failed",
          detail:
            "chat_id is required",
        });
      }

      if (!text) {
        return res.status(400).json({
          status: "failed",
          detail:
            "text is required",
        });
      }

      const cid = String(chat_id).trim();
      if (!cid.includes("@s.whatsapp.net") && !cid.includes("@g.us") && !cid.includes("@lid")) {
        return res.status(400).json({
          status: "failed",
          detail: "Invalid WhatsApp recipient",
        });
      }

      const targetJid = await getTargetJid(cid);
      if (process.env.DEBUG === "true") {
        console.debug(`[WhatsApp] Sending message`);
      }
      const result =
        await sock.sendMessage(
          targetJid,
          {
            text: String(text),
          }
        );
      console.log("[WhatsApp] Message sent");

      if (result?.key?.id && result?.message) {
        messageStore.set(result.key.id, { key: result.key, message: result.message });
      }

      return res.json({
        status: "ok",

        detail:
          "Message sent",

        message_id:
          result?.key?.id ||
          null,
      });
    } catch (error) {
      console.error(
        "[WhatsApp] Message send failed:",
        error.message
      );

      return res.status(500).json({
        status: "failed",
        detail:
          error.message,
      });
    }
  }
);

// ============================================================
// PRESENCE UPDATE
// ============================================================

app.post(
  "/presence",
  async (req, res) => {
    try {
      if (!requireConnection(res)) {
        return;
      }

      const {
        chat_id,
        state = "composing",
      } = req.body;

      if (!chat_id) {
        return res.status(400).json({
          status: "failed",
          detail: "chat_id is required",
        });
      }

      const cid = String(chat_id).trim();
      if (!cid.includes("@s.whatsapp.net") && !cid.includes("@g.us") && !cid.includes("@lid")) {
        return res.status(400).json({
          status: "failed",
          detail: "Invalid WhatsApp recipient",
        });
      }

      if (process.env.DEBUG === "true") {
        console.debug(`[WhatsApp Bridge] presence update: ${state}`);
      }
      await sock.sendPresenceUpdate(state, cid);

      return res.json({
        status: "ok",
        detail: `Presence updated to ${state}`,
      });
    } catch (error) {
      console.error(
        "[WhatsApp] Presence update error:",
        error.message
      );

      return res.status(500).json({
        status: "failed",
        detail: error.message,
      });
    }
  }
);

// ============================================================
// SEND FILE
// ============================================================

app.post(
  "/send-file",
  upload.single("file"),
  async (req, res) => {
    try {
      if (!requireConnection(res)) {
        return;
      }

      const {
        chat_id,
        caption = "",
      } = req.body;

      if (!chat_id) {
        return res.status(400).json({
          status: "failed",
          detail:
            "chat_id is required",
        });
      }

      if (!req.file) {
        return res.status(400).json({
          status: "failed",
          detail:
            "file is required",
        });
      }

      const targetJid = await getTargetJid(chat_id);
      if (process.env.DEBUG === "true") {
        console.debug(`[WhatsApp] Sending file`);
      }
      const result =
        await sock.sendMessage(
          targetJid,
          {
            document:
              req.file.buffer,

            mimetype:
              req.file.mimetype ||
              "application/octet-stream",

            fileName:
              req.file.originalname,

            caption:
              caption || "",
          }
        );
      console.log("[WhatsApp] File sent");

      if (result?.key?.id && result?.message) {
        messageStore.set(result.key.id, { key: result.key, message: result.message });
      }

      return res.json({
        status: "ok",

        detail:
          "File sent",

        message_id:
          result?.key?.id ||
          null,
      });
    } catch (error) {
      console.error(
        "[WhatsApp] File send error:",
        error.message
      );

      return res.status(500).json({
        status: "failed",
        detail:
          error.message,
      });
    }
  }
);

// ============================================================
// RECENT MESSAGES
// ============================================================

app.get(
  "/recent",
  (req, res) => {
    const messages =
      recentBuffer.splice(
        0,
        recentBuffer.length
      );

    res.json(messages);
  }
);

// ============================================================
// PEEK RECENT MESSAGES
// ============================================================

app.get(
  "/recent/peek",
  (req, res) => {
    res.json(
      recentBuffer
    );
  }
);

// ============================================================
// MEDIA DOWNLOAD
// ============================================================

app.get("/media/:id", async (req, res) => {
  try {
    const msgId = req.params.id;
    if (!msgId) {
      return res.status(400).json({ status: "failed", detail: "Message id is required" });
    }
    const stored = messageStore.get(msgId);
    if (!stored) {
      return res.status(404).json({ status: "failed", detail: `Message ${msgId} not found in store` });
    }

    const fullMsg = stored.message ? stored : { key: { id: msgId }, message: stored };
    const audioMsg = fullMsg.message?.audioMessage;
    const docMsg = fullMsg.message?.documentMessage;
    const imgMsg = fullMsg.message?.imageMessage;

    const mimetype = audioMsg?.mimetype || docMsg?.mimetype || imgMsg?.mimetype || "application/octet-stream";

    const buffer = await downloadMediaMessage(
      fullMsg,
      "buffer",
      {},
      sock ? { logger: sock.logger, reuploadRequest: sock.updateMediaMessage } : undefined
    );

    res.setHeader("Content-Type", mimetype);
    res.setHeader("Content-Length", buffer.length);
    return res.send(buffer);
  } catch (err) {
    console.error("[WhatsApp Bridge] Media download error:", err.message);
    return res.status(500).json({ status: "failed", detail: err.message });
  }
});

// ============================================================
// RESOLVE CONTACT
// ============================================================

app.get("/contacts/resolve", (req, res) => {
  const nameQuery = String(req.query.name || "").trim().toLowerCase();
  if (!nameQuery) {
    return res.status(400).json({ status: "failed", detail: "query parameter 'name' is required" });
  }

  // Direct JID check
  if (nameQuery.includes("@s.whatsapp.net") || nameQuery.includes("@lid")) {
    return res.json({ status: "ok", matches: [{ name: nameQuery, jid: nameQuery }] });
  }
  const digits = nameQuery.replace(/\D/g, "");
  if (digits.length >= 10) {
    const pnJid = `${digits}@s.whatsapp.net`;
    return res.json({ status: "ok", matches: [{ name: nameQuery, jid: pnJid }] });
  }

  const matches = [];
  const seenJids = new Set();

  for (const [lid, pn] of Object.entries(lidToPnMap)) {
    if (lid.toLowerCase().includes(nameQuery) || pn.toLowerCase().includes(nameQuery)) {
      if (!seenJids.has(pn)) {
        seenJids.add(pn);
        matches.push({ name: pn, jid: pn });
      }
    }
  }

  for (const msg of recentBuffer) {
    const sender = (msg.sender || "").toLowerCase();
    const chat = (msg.chat_id || "").toLowerCase();
    if (sender.includes(nameQuery) || chat.includes(nameQuery)) {
      const targetJid = msg.chat_id || msg.sender;
      if (targetJid && !seenJids.has(targetJid)) {
        seenJids.add(targetJid);
        matches.push({ name: msg.sender || msg.chat_id, jid: targetJid });
      }
    }
  }

  res.json({ status: "ok", matches });
});

// ============================================================
// START SERVER
// ============================================================

if (require.main === module) {
  acquireProcessLock();
  app.listen(
    PORT,
    () => {
      if (process.env.DEBUG === "true") {
        console.debug(`[WhatsApp] Bridge listening on :${PORT}`);
      }

      start().catch(
        (error) => {
          console.error(
            "[WhatsApp] Failed to start:",
            error
          );
        }
      );
    }
  );
}

module.exports = {
  app,
  BoundedMessageStore,
  messageStore,
  msgRetryCounterCache,
  getTargetJid,
  resolveJidToPn,
  acquireProcessLock,
  releaseProcessLock,
  AUTH_STATE_DIR,
  lidToPnMap,
  pnToLidMap,
  createSafeBaileysLogger,
  redactText,
};