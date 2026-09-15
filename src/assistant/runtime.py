"""
Assistant Runtime (src/assistant/runtime.py)
============================================
Central runtime coordinator and lifecycle manager for the Personal AI Assistant.
Encapsulates all service construction, dependency wiring, background worker supervision,
and graceful shutdown.

Allows both the CLI channel and future Streamlit / UI layers to consume the exact
same underlying application core without duplicating setup or launching redundant workers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
from typing import Any

import yaml
from dotenv import load_dotenv

from assistant.agent_core.agent_command_handler import AgentCommandHandler
from assistant.agent_core.approval_manager import ApprovalManager
from assistant.agent_core.context_builder import ContextBuilder
from assistant.agent_core.orchestrator import AgentOrchestrator
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.agent_core.task_planner import TaskPlanner
from assistant.channels.cli_channel import CLIChannel
from assistant.channels.telegram_channel import TelegramChannel
from assistant.channels.whatsapp_channel import WhatsAppChannel
from assistant.connectors.gmail.gmail_connector import GmailConnector
from assistant.connectors.telegram.telegram_connector import TelegramConnector
from assistant.connectors.telegram.telegram_user_client import TelegramUserClient
from assistant.connectors.whatsapp.whatsapp_connector import WhatsAppConnector
from assistant.contacts.contact_cache import ContactCache
from assistant.device_gateway.device_gateway import DeviceGateway
from assistant.device_gateway.transport.ws_transport import serve as serve_device_gateway
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.files.attachment_manager import AttachmentManager
from assistant.files.file_policy import FilePolicy
from assistant.files.file_resolver import FileResolver
from assistant.files.mobile_path_resolver import canonical_mobile_path
from assistant.ingestion.adapters import adapt_gmail, adapt_sms, adapt_telegram, adapt_whatsapp
from assistant.ingestion.live_ingestion import PollingWorker
from assistant.intelligence.eligibility_filter import MessageEligibilityFilter
from assistant.intelligence.normal_message_pipeline import NormalMessagePipeline
from assistant.intelligence.priority_analyzer import PriorityAnalyzer, PriorityAnalyzerWorker
from assistant.intelligence.priority_engine import PriorityEngine
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.intelligence.priority_rules import PriorityRulesManager
from assistant.llm.llm_client import LLMClient, get_priority_llm_client
from assistant.memory.memory_service import MemoryService
from assistant.router.conversation_registry import ConversationClass, ConversationRegistry
from assistant.router.conversation_router import ConversationRouter
from assistant.router.echo_filter import EchoFilter
from assistant.security.auth import DeviceAuth
from assistant.security.capability_validator import CapabilityValidator
from assistant.storage.db import init_db
from assistant.tools.web_information import WebInformationService

logger = logging.getLogger("assistant.runtime")

DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_network_connectivity(
    targets: list[tuple[str, int]] | None = None,
    timeout: float = 1.0,
) -> bool:
    """Probes global network/internet availability via low-overhead socket connection.

    Standard targets: 8.8.8.8:53 and 1.1.1.1:53 (DNS port, fast connection).
    Returns False immediately when the network interface is down/unreachable.
    """
    env_override = os.environ.get("NETWORK_CONNECTIVITY_OVERRIDE")
    if env_override is not None:
        return env_override.strip().lower() in ("true", "1", "yes", "online")

    check_targets = targets or [("8.8.8.8", 53), ("1.1.1.1", 53)]
    for host, port in check_targets:
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.close()
            return True
        except OSError:
            continue
    return False


class AssistantRuntime:
    """Central lifecycle coordinator for the Personal AI Assistant."""

    def __init__(self, root_dir: str | None = None, auto_load_env: bool = True):
        self._root = root_dir or DEFAULT_ROOT
        self._lock = threading.RLock()
        self._started = False
        self._device_gateway_thread: threading.Thread | None = None
        self._last_network_check_ts: float = 0.0
        self._last_network_check_val: bool = False
        self._network_check_ttl: float = 2.0

        if auto_load_env:
            load_dotenv(os.path.join(self._root, ".env"), override=True)

        self.settings = self._load_yaml("settings.yaml")
        self.policies = self._load_yaml("policies.yaml")
        self.capabilities_cfg = self._load_yaml("capabilities.yaml")

        self._initialize_core()
        self._initialize_connectors()
        self._initialize_execution_and_tools()
        self._initialize_agent_core()
        self._initialize_router_and_channels()
        self._initialize_intelligence_and_analyzer()

    def _load_yaml(self, name: str) -> dict:
        path = os.path.join(self._root, "config", name)
        if not os.path.exists(path):
            logger.debug("Config file %s not found, using empty mapping.", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _initialize_core(self) -> None:
        # --- Storage ---
        init_db()

        # --- Memory ---
        memory_dir = os.environ.get("MEMORY_DIR", os.path.join(self._root, "data", "memory"))
        session_ttl = self.settings.get("memory", {}).get("session_ttl_minutes", 60)
        self.memory = MemoryService(memory_dir=memory_dir, session_ttl_minutes=session_ttl)

        # --- Normal Conversational LLM ---
        self.llm = LLMClient()
        logger.info("LLM client online=%s model=%s", self.llm.online, self.llm.model)

    def _initialize_connectors(self) -> None:
        # --- Gmail ---
        self.gmail_connector = GmailConnector(
            enabled=os.environ.get("GMAIL_ENABLED", "false").lower() == "true",
        )


        # --- Telegram Bot Connector ---
        self.telegram_connector = TelegramConnector(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            enabled=os.environ.get("TELEGRAM_ENABLED", "false").lower() == "true",
        )

        # Ensure asyncio event loop exists for Telethon (Python 3.14 compatibility)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # --- Telegram Personal Account (Telethon) ---
        self.telegram_personal = TelegramUserClient(
            api_id=os.environ.get("TELEGRAM_API_ID", ""),
            api_hash=os.environ.get("TELEGRAM_API_HASH", ""),
            session_file=os.environ.get("TELEGRAM_SESSION_FILE", ""),
            enabled=os.environ.get("TELEGRAM_USER_ENABLED", "false").lower() == "true",
        )

        # --- STT ---
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        groq_stt_model = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
        self.stt_service = None
        if groq_api_key:
            try:
                from assistant.stt.groq_stt import GroqSTTService

                self.stt_service = GroqSTTService(api_key=groq_api_key, model=groq_stt_model)
                logger.info("Groq STT Service initialized with model %s", groq_stt_model)
            except Exception as exc:
                logger.warning("Failed to initialize Groq STT Service: %s", exc)

        # --- Contact Cache & WhatsApp ---
        contact_cache_ttl = float(os.environ.get("CONTACT_CACHE_TTL_DAYS", "10"))
        self.contact_cache = ContactCache(ttl_days=contact_cache_ttl)

        self.whatsapp_connector = WhatsAppConnector(
            bridge_url=os.environ.get("WHATSAPP_BRIDGE_URL", "http://localhost:3000"),
            enabled=os.environ.get("WHATSAPP_ENABLED", "false").lower() == "true",
            stt_service=self.stt_service,
            contact_cache=self.contact_cache,
        )

    def _initialize_execution_and_tools(self) -> None:
        echo_window = int(os.environ.get("ECHO_FILTER_WINDOW_SECONDS", "120"))
        self.outbound_registry = OutboundRegistry(window_seconds=echo_window)
        self.task_manager = TaskManager(self.outbound_registry)
        self.tool_router = ToolRouter(self.task_manager)

        # Outbound messaging capability routes
        self.tool_router.register(
            "send_whatsapp_message",
            lambda task: self.whatsapp_connector.send_message(task.target, task.parameters.get("content", "")),
        )
        self.tool_router.register(
            "send_telegram_message",
            lambda task: (
                self.telegram_personal.send_message(task.target, task.parameters.get("content", ""))
                if self.telegram_personal.enabled
                else self.telegram_connector.send_message(task.target, task.parameters.get("content", ""))
            ),
        )
        self.tool_router.register(
            "send_email",
            lambda task: self.gmail_connector.send(task.target, "Message from your assistant", task.parameters.get("content", "")),
        )

        # Outbound file tools
        outbound_dir = os.environ.get("OUTBOUND_FILES_DIR", os.path.join(self._root, "data", "outbound_files"))
        max_file_mb = int(os.environ.get("MAX_OUTBOUND_FILE_SIZE_MB", "10"))
        os.makedirs(outbound_dir, exist_ok=True)
        self.file_policy = FilePolicy(allowed_roots=[outbound_dir], max_file_size_mb=max_file_mb)
        self.file_resolver = FileResolver(self.file_policy)
        self.attachment_manager = AttachmentManager(self.file_policy)

        self.tool_router.register(
            "send_whatsapp_file",
            lambda task: self.whatsapp_connector.send_file(task.target, task.parameters["path"], task.parameters.get("content", "")),
        )
        self.tool_router.register(
            "send_telegram_file",
            lambda task: self.telegram_personal.send_file(task.target, task.parameters["path"], task.parameters.get("content", "")),
        )
        self.tool_router.register(
            "send_gmail_attachment",
            lambda task: self.gmail_connector.send_attachment(
                task.target, "Message from your assistant", task.parameters.get("content", ""), task.parameters["path"]
            ),
        )

        # Device Gateway
        self.device_auth = DeviceAuth(os.environ.get("DEVICE_GATEWAY_AUTH_TOKEN", ""))
        self.device_gateway = DeviceGateway(
            CapabilityValidator(self.capabilities_cfg.get("android_capabilities", []), self.capabilities_cfg.get("cloud_capabilities", [])),
            on_event=None,
        )
        self.whatsapp_connector.set_device_gateway(self.device_gateway)

        self._register_device_handlers()

    def _register_device_handlers(self) -> None:
        def _send_sms_handler(task):
            phone_num = task.parameters.get("phone_number") or task.parameters.get("number")
            if not phone_num:
                target_str = str(task.target or "").strip()
                import re

                m = re.search(r"\(([\+\d\s\-]+)\)", target_str)
                if m:
                    phone_num = m.group(1).strip()
                elif target_str.startswith("+") or target_str.replace("-", "").replace(" ", "").isdigit():
                    phone_num = target_str
            if not phone_num and self.memory is not None and hasattr(self.memory, "user_profile"):
                phone_num = self.memory.user_profile.resolve_contact_number(task.target) or ""

            content = task.parameters.get("text") or task.parameters.get("content") or task.parameters.get("message") or ""
            masked_num = (phone_num[:3] + "******" + phone_num[-4:]) if phone_num and len(phone_num) > 7 else (phone_num or "unknown")
            logger.info("[device] dispatching sms.send request_id=%s recipient=%s content_len=%d",
                        getattr(task, "task_id", "n/a"), masked_num, len(content))

            params = {
                "recipient": task.target,
                "phone_number": phone_num or "",
                "number": phone_num or "",
                "text": content,
                "content": content,
                "message": content,
                **task.parameters,
            }
            return self.device_gateway.send_command("sms.send", params).to_dict()

        def _make_call_handler(task):
            phone_num = (
                task.parameters.get("phone_number")
                or task.parameters.get("number")
                or (task.target if (task.target.startswith("+") or task.target.replace("-", "").replace(" ", "").isdigit()) else "")
            )
            if not phone_num and self.memory is not None and hasattr(self.memory, "user_profile"):
                phone_num = self.memory.user_profile.resolve_contact_number(task.target) or ""

            masked_num = phone_num[:3] + "******" + phone_num[-4:] if len(phone_num) > 7 else phone_num
            logger.info("[device] dispatching call.make request_id=%s target=%s recipient=%s",
                        getattr(task, "task_id", "n/a"), task.target, masked_num)

            if not phone_num:
                return {"status": "failed", "detail": f"I couldn't find a phone number for {task.target}. Please provide the number."}

            params = {
                "recipient": task.target,
                "phone_number": phone_num,
                "number": phone_num,
                **task.parameters,
            }
            return self.device_gateway.send_command("call.make", params).to_dict()

        def _upload_file_handler(task):
            path = task.parameters.get("path") or task.target or ""
            m_path = canonical_mobile_path(path)
            logger.info("[device] mobile file request detected: mobile path=%s", m_path)
            return self.device_gateway.send_command("file.upload", {"path": m_path}).to_dict()

        self.tool_router.register("send_sms", _send_sms_handler)
        self.tool_router.register("sms.send", _send_sms_handler)
        self.tool_router.register("read_sms", lambda task: self.device_gateway.send_command("sms.read", task.parameters).to_dict())
        self.tool_router.register("sms.read", lambda task: self.device_gateway.send_command("sms.read", task.parameters).to_dict())
        self.tool_router.register("make_call", _make_call_handler)
        self.tool_router.register("call.make", _make_call_handler)
        self.tool_router.register("set_alarm", lambda task: self.device_gateway.send_command("alarm.set", task.parameters).to_dict())
        self.tool_router.register("alarm.set", lambda task: self.device_gateway.send_command("alarm.set", task.parameters).to_dict())
        self.tool_router.register("list_alarms", lambda task: self.device_gateway.send_command("alarm.list", task.parameters).to_dict())
        self.tool_router.register("alarm.list", lambda task: self.device_gateway.send_command("alarm.list", task.parameters).to_dict())
        self.tool_router.register("cancel_alarm", lambda task: self.device_gateway.send_command("alarm.cancel", task.parameters).to_dict())
        self.tool_router.register("alarm.cancel", lambda task: self.device_gateway.send_command("alarm.cancel", task.parameters).to_dict())
        self.tool_router.register("find_contact", lambda task: self.device_gateway.send_command("contact.find", task.parameters).to_dict())
        self.tool_router.register("contact.find", lambda task: self.device_gateway.send_command("contact.find", task.parameters).to_dict())
        self.tool_router.register("set_timer", lambda task: self.device_gateway.send_command("timer.set", task.parameters).to_dict())
        self.tool_router.register("timer.set", lambda task: self.device_gateway.send_command("timer.set", task.parameters).to_dict())
        self.tool_router.register("run_intent", lambda task: self.device_gateway.send_command("intent.execute", task.parameters).to_dict())
        self.tool_router.register("intent.execute", lambda task: self.device_gateway.send_command("intent.execute", task.parameters).to_dict())
        self.tool_router.register("intent.open_app", lambda task: self.device_gateway.send_command("intent.open_app", task.parameters).to_dict())
        self.tool_router.register("read_file", lambda task: self.device_gateway.send_command("file.read", task.parameters).to_dict())
        self.tool_router.register("file.read", lambda task: self.device_gateway.send_command("file.read", task.parameters).to_dict())
        self.tool_router.register(
            "file.list",
            lambda task: self.device_gateway.send_command("file.list", {"path": canonical_mobile_path(task.parameters.get("path", "/Books"))}).to_dict(),
        )
        self.tool_router.register(
            "file.find",
            lambda task: self.device_gateway.send_command(
                "file.find", {"path": canonical_mobile_path(task.parameters.get("path", "/storage/emulated/0")), "query": task.parameters.get("query", "")}
            ).to_dict(),
        )
        self.tool_router.register("upload_file", _upload_file_handler)
        self.tool_router.register("file.upload", _upload_file_handler)

    def _initialize_agent_core(self) -> None:
        self.policy_engine = PolicyEngine(self.policies.get("capabilities", {}))
        self.context_builder = ContextBuilder(self.memory)
        self.task_planner = TaskPlanner(self.task_manager, self.policy_engine, whatsapp_connector=self.whatsapp_connector)
        self.approval_manager = ApprovalManager(self.task_manager)
        self.priority_inbox = PriorityInbox()

        self.orchestrator = AgentOrchestrator(
            llm=self.llm,
            memory=self.memory,
            context_builder=self.context_builder,
            task_planner=self.task_planner,
            approval_manager=self.approval_manager,
            tool_router=self.tool_router,
            priority_inbox=self.priority_inbox,
            telegram_connector=self.telegram_connector,
            telegram_personal=self.telegram_personal,
            whatsapp_connector=self.whatsapp_connector,
            gmail_connector=self.gmail_connector,
            file_resolver=self.file_resolver,
            attachment_manager=self.attachment_manager,
            web_information=WebInformationService(),
            device_gateway=self.device_gateway,
        )

    def _initialize_router_and_channels(self) -> None:
        self.echo_filter = EchoFilter(self.outbound_registry, self_identities={"me", "assistant"})
        self.conversation_registry = ConversationRegistry(mapping=self.settings.get("conversations", {}))

        telegram_agent_chat_id = os.environ.get("TELEGRAM_AGENT_CHAT_ID", "")
        whatsapp_agent_chat_id = os.environ.get("WHATSAPP_AGENT_CHAT_ID", "919172767219@s.whatsapp.net").strip()
        if telegram_agent_chat_id:
            self.conversation_registry.set(f"telegram_bot:{telegram_agent_chat_id}", ConversationClass.AGENT_CHAT)
        if whatsapp_agent_chat_id:
            self.conversation_registry.set(f"whatsapp:{whatsapp_agent_chat_id}", ConversationClass.AGENT_CHAT)
            if "@" in whatsapp_agent_chat_id:
                raw_phone = whatsapp_agent_chat_id.split("@")[0]
                self.conversation_registry.set(f"whatsapp:{raw_phone}", ConversationClass.AGENT_CHAT)
            else:
                self.conversation_registry.set(f"whatsapp:{whatsapp_agent_chat_id}@s.whatsapp.net", ConversationClass.AGENT_CHAT)

        self.conversation_registry.set("whatsapp:52909752496163@lid", ConversationClass.AGENT_CHAT)
        self.conversation_registry.set("whatsapp:52909752496163", ConversationClass.AGENT_CHAT)

        self.eligibility_filter = MessageEligibilityFilter(
            conversation_registry=self.conversation_registry,
            whatsapp_connector=self.whatsapp_connector,
        )

        priority_cfg = self.settings.get("priority", {})
        self.priority_engine = PriorityEngine(weights=priority_cfg.get("weights"), thresholds=priority_cfg.get("thresholds"))

        self.normal_pipeline = NormalMessagePipeline(
            priority_engine=self.priority_engine,
            priority_inbox=self.priority_inbox,
            user_profile=self.memory.user_profile,
            orchestrator=self.orchestrator,
            eligibility_filter=self.eligibility_filter,
        )

        cli_channel_holder: dict = {}
        telegram_channel_holder: dict = {}
        whatsapp_channel_holder: dict = {}

        def reply_sender(original_message, text: str) -> None:
            cid = original_message.conversation_id or ""
            if cid.startswith("telegram_bot:") and telegram_channel_holder.get("channel"):
                telegram_channel_holder["channel"].reply(original_message, text)
            elif cid.startswith("whatsapp:") and whatsapp_channel_holder.get("channel"):
                whatsapp_channel_holder["channel"].reply(original_message, text)
            elif cli_channel_holder.get("channel"):
                cli_channel_holder["channel"].reply(original_message, text)

        def presence_provider(original_message):
            cid = original_message.conversation_id or ""
            req_id = getattr(original_message, "message_id", "") or ""
            if cid.startswith("telegram_bot:") and telegram_channel_holder.get("channel"):
                chat_id = cid.removeprefix("telegram_bot:")
                return telegram_channel_holder["channel"].get_presence_indicator(chat_id=chat_id, request_id=req_id)
            elif cid.startswith("whatsapp:") and whatsapp_channel_holder.get("channel"):
                jid = cid.removeprefix("whatsapp:")
                return whatsapp_channel_holder["channel"].get_presence_indicator(jid=jid, request_id=req_id)
            from assistant.channels.presence import NullPresenceIndicator

            return NullPresenceIndicator()

        self.agent_command_handler = AgentCommandHandler(self.orchestrator, reply_sender, presence_provider=presence_provider)

        self.router = ConversationRouter(
            registry=self.conversation_registry,
            echo_filter=self.echo_filter,
            agent_command_handler=self.agent_command_handler,
            normal_message_handler=self.normal_pipeline.handle,
        )

        self.device_gateway._on_event = lambda event: self.router.route(adapt_sms(event)) if event.get("type") == "SMS_INBOUND" else None

        self.cli_channel = CLIChannel(
            self.router,
            device_gateway=self.device_gateway,
            tool_router=self.tool_router,
            llm_model=self.llm.model,
        )
        cli_channel_holder["channel"] = self.cli_channel

        self.telegram_channel = TelegramChannel(
            self.telegram_connector, self.router, agent_chat_id=telegram_agent_chat_id, stt_service=self.stt_service
        )
        self.whatsapp_channel = WhatsAppChannel(
            self.whatsapp_connector, self.router, agent_chat_id=whatsapp_agent_chat_id
        )
        telegram_channel_holder["channel"] = self.telegram_channel
        whatsapp_channel_holder["channel"] = self.whatsapp_channel

        self.gmail_worker = PollingWorker(
            "gmail",
            float(os.environ.get("GMAIL_INGESTION_POLL_SECONDS", "60")),
            self.gmail_connector.fetch_recent,
            adapt_gmail,
            self.router.route,
        )
        self.telegram_personal_worker = PollingWorker(
            "telegram-personal",
            float(os.environ.get("TELEGRAM_INGESTION_POLL_SECONDS", "5")),
            self.telegram_personal.fetch_recent,
            adapt_telegram,
            self.router.route,
        )
        self.whatsapp_worker = PollingWorker(
            "whatsapp",
            float(os.environ.get("WHATSAPP_INGESTION_POLL_SECONDS", "5")),
            self.whatsapp_connector.fetch_recent,
            adapt_whatsapp,
            self.router.route,
        )
        self.ingestion_workers = [self.gmail_worker, self.telegram_personal_worker, self.whatsapp_worker]

    def _initialize_intelligence_and_analyzer(self) -> None:
        self.priority_rules_manager = PriorityRulesManager()
        analyzer_interval = float(os.environ.get("PRIORITY_ANALYZER_INTERVAL_MINUTES", "30"))
        analyzer_batch_size = int(os.environ.get("PRIORITY_ANALYZER_BATCH_SIZE", "50"))
        analyzer_max_batches = int(os.environ.get("PRIORITY_ANALYZER_MAX_BATCHES_PER_CYCLE", "3"))
        analyzer_startup = os.environ.get("PRIORITY_ANALYZER_RUN_ON_STARTUP", "false").lower() in ("true", "1", "yes")

        self.priority_llm = get_priority_llm_client()

        self.priority_analyzer = PriorityAnalyzer(
            priority_inbox=self.priority_inbox,
            rules_manager=self.priority_rules_manager,
            user_profile=self.memory.user_profile,
            llm_client=self.priority_llm,
            batch_size=analyzer_batch_size,
            max_batches_per_cycle=analyzer_max_batches,
        )
        self.priority_analyzer_worker = PriorityAnalyzerWorker(
            analyzer=self.priority_analyzer,
            interval_minutes=analyzer_interval,
            run_on_startup=analyzer_startup,
        )

    # --------------------------------------------------------------------------
    # Public Properties & Aliases
    # --------------------------------------------------------------------------

    @property
    def agent_orchestrator(self) -> AgentOrchestrator:
        return self.orchestrator

    @property
    def memory_service(self) -> MemoryService:
        return self.memory

    @property
    def normal_llm(self) -> LLMClient:
        return self.llm

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def connectors(self) -> dict[str, Any]:
        return {
            "whatsapp": self.whatsapp_connector,
            "telegram": self.telegram_connector,
            "telegram_personal": self.telegram_personal,
            "gmail": self.gmail_connector,
        }

    # --------------------------------------------------------------------------
    # Lifecycle Management
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Starts all enabled background services idempotently."""
        with self._lock:
            if self._started:
                logger.debug("AssistantRuntime is already started.")
                return

            # 1. Ingestion Pollers
            if os.environ.get("GMAIL_ENABLED", "false").lower() == "true":
                self.gmail_worker.start()
            if os.environ.get("TELEGRAM_USER_ENABLED", "false").lower() == "true":
                self.telegram_personal_worker.start()
            if os.environ.get("WHATSAPP_ENABLED", "false").lower() == "true":
                self.whatsapp_connector.start_bridge()
                self.whatsapp_worker.start()

            # 2. Telegram Bot Channel Listener
            if self.telegram_channel is not None:
                self.telegram_channel.start()

            # 3. Priority Analyzer Background Worker
            if os.environ.get("PRIORITY_ANALYZER_ENABLED", "true").lower() in ("true", "1", "yes"):
                if self.priority_analyzer_worker is not None:
                    self.priority_analyzer_worker.start()

            # 4. Android Device Gateway WebSocket Server
            if os.environ.get("DEVICE_GATEWAY_ENABLED", "false").lower() == "true":
                self._start_device_gateway()

            self._started = True
            logger.info("AssistantRuntime background services started successfully.")

    def _start_device_gateway(self) -> None:
        if self._device_gateway_thread and self._device_gateway_thread.is_alive():
            return
        host = os.environ.get("DEVICE_GATEWAY_HOST", "0.0.0.0")
        port = int(os.environ.get("DEVICE_GATEWAY_PORT", "8765"))

        def _run_server() -> None:
            try:
                asyncio.run(serve_device_gateway(self.device_gateway, self.device_auth, host, port))
            except OSError as err:
                logger.warning("[Device Gateway] Could not bind WebSocket server on %s:%s (already in use): %s", host, port, err)
            except Exception as exc:
                logger.warning("[Device Gateway] WebSocket server stopped with error: %s", exc)

        self._device_gateway_thread = threading.Thread(
            target=_run_server,
            name="device-gateway",
            daemon=True,
        )
        self._device_gateway_thread.start()

    def is_network_available(self, force_refresh: bool = False) -> bool:
        """Returns True if the runtime has global network/internet connectivity.

        Uses a short 2.0s TTL cache so multiple calls in a single render pass
        remain ultra fast, while allowing immediate forced checks on reruns.
        """
        now = time.monotonic()
        if not force_refresh and (now - self._last_network_check_ts < self._network_check_ttl):
            return self._last_network_check_val

        val = check_network_connectivity()
        self._last_network_check_val = val
        self._last_network_check_ts = now
        return val

    def get_chat_status(self, force_refresh: bool = False) -> dict[str, Any]:
        """Returns current status of the local Chat application runtime.

        Chat is 'Online' only when the local application runtime is started
        AND global network connectivity is available.
        """
        if not self._started:
            return {
                "connected": False,
                "status": "offline",
                "label": "Offline",
                "details": "Runtime not started",
            }

        if not self.is_network_available(force_refresh=force_refresh):
            return {
                "connected": False,
                "status": "offline",
                "label": "Offline",
                "details": "Network unavailable",
            }

        return {
            "connected": True,
            "status": "online",
            "label": "Online",
            "details": "Operational",
        }

    def get_whatsapp_status(self, force_refresh: bool = False) -> dict[str, Any]:
        """Returns current status of WhatsApp connector with network freshness guard."""
        raw: dict[str, Any] = {"running": False, "connected": False, "reconnecting": False, "qr": None}
        if hasattr(self, "whatsapp_connector") and self.whatsapp_connector:
            try:
                raw = self.whatsapp_connector.get_status()
            except Exception as exc:
                logger.debug("Error querying WhatsApp connector status: %s", exc)

        # Global network guard: network loss forces Offline to prevent stale states
        if not self.is_network_available(force_refresh=force_refresh):
            return {
                **raw,
                "connected": False,
                "reconnecting": False,
                "status": "offline",
                "label": "Offline",
            }

        # Network is available: evaluate actual bridge state
        is_conn = bool(raw.get("connected"))
        is_reconn = bool(raw.get("reconnecting"))
        has_qr = bool(raw.get("qr"))

        if is_conn:
            status = "connected"
            label = "Connected"
        elif is_reconn:
            status = "connecting"
            label = "Reconnecting"
        elif has_qr:
            status = "auth_required"
            label = "Auth Required"
        else:
            status = "disconnected"
            label = "Not connected"

        return {
            **raw,
            "connected": is_conn,
            "status": status,
            "label": label,
        }

    def get_telegram_status(self, force_refresh: bool = False) -> dict[str, Any]:
        """Returns current status of Telegram connectors (Bot & User) with network guard."""
        bot_status = (
            self.telegram_connector.get_status()
            if hasattr(self, "telegram_connector") and self.telegram_connector
            else {"connected": False}
        )
        user_status = (
            self.telegram_personal.get_status()
            if hasattr(self, "telegram_personal") and self.telegram_personal
            else {"connected": False}
        )

        # Global network guard: network loss forces Offline
        if not self.is_network_available(force_refresh=force_refresh):
            return {
                "connected": False,
                "status": "offline",
                "label": "Offline",
                "bot": {**bot_status, "connected": False},
                "user": {**user_status, "connected": False},
            }

        # Network is available: evaluate actual connection states
        is_conn = bool(bot_status.get("connected") or user_status.get("connected"))
        is_reconn = bool(user_status.get("reconnecting") and not is_conn)

        if is_conn:
            status = "connected"
            label = "Connected"
        elif is_reconn:
            status = "connecting"
            label = "Connecting"
        else:
            status = "disconnected"
            label = "Not connected"

        return {
            "connected": is_conn,
            "status": status,
            "label": label,
            "bot": bot_status,
            "user": user_status,
        }

    def get_android_status(self, force_refresh: bool = False) -> dict[str, Any]:
        """Returns current status of Android Device Gateway with live-socket requirement."""
        is_running = bool(self._device_gateway_thread and self._device_gateway_thread.is_alive())
        dev_status = (
            self.device_gateway.get_device_status()
            if hasattr(self, "device_gateway") and self.device_gateway
            else {"connected": False}
        )
        host = os.environ.get("DEVICE_GATEWAY_HOST", "0.0.0.0")
        port = int(os.environ.get("DEVICE_GATEWAY_PORT", "8765"))
        display_host = "localhost" if host == "0.0.0.0" else host
        ws_url = f"ws://{display_host}:{port}"

        # Global network guard: network loss forces Offline
        if not self.is_network_available(force_refresh=force_refresh):
            return {
                "server_running": is_running,
                "connected": False,
                "device_name": dev_status.get("device_name"),
                "ws_url": ws_url,
                "status": "offline",
                "label": "Offline",
            }

        # Network is available: evaluate actual authenticated DeviceHandle state
        is_conn = bool(dev_status.get("connected", False))
        is_reconn = bool(dev_status.get("reconnecting", False) and not is_conn)

        if is_conn:
            status = "connected"
            label = "Connected"
        elif is_reconn:
            status = "connecting"
            label = "Reconnecting"
        else:
            status = "disconnected"
            label = "Not connected"

        return {
            "server_running": is_running,
            "connected": is_conn,
            "device_name": dev_status.get("device_name") if is_conn else None,
            "ws_url": ws_url,
            "status": status,
            "label": label,
        }

    def stop(self) -> None:
        """Shuts down all active background services idempotently."""
        with self._lock:
            logger.info("Stopping AssistantRuntime...")

            # 1. Stop Priority Analyzer Worker
            if getattr(self, "priority_analyzer_worker", None) is not None:
                try:
                    self.priority_analyzer_worker.stop()
                except Exception as exc:
                    logger.debug("Error stopping priority analyzer worker: %s", exc)

            # 2. Stop Ingestion Pollers
            for worker in getattr(self, "ingestion_workers", []):
                try:
                    worker.stop()
                except Exception as exc:
                    logger.debug("Error stopping worker %s: %s", getattr(worker, "name", "unknown"), exc)

            # 3. Stop Telegram Bot Channel
            if getattr(self, "telegram_channel", None) is not None:
                try:
                    self.telegram_channel.stop()
                except Exception as exc:
                    logger.debug("Error stopping telegram channel: %s", exc)

            # 4. Stop WhatsApp Bridge Process
            if getattr(self, "whatsapp_connector", None) is not None:
                try:
                    self.whatsapp_connector.stop_bridge()
                except Exception as exc:
                    logger.debug("Error stopping WhatsApp bridge: %s", exc)

            # 5. Detach Device Gateway
            if getattr(self, "device_gateway", None) is not None:
                try:
                    self.device_gateway.detach_device()
                except Exception as exc:
                    logger.debug("Error detaching device gateway: %s", exc)

            self._started = False
            logger.info("AssistantRuntime stopped.")

    def __enter__(self) -> AssistantRuntime:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def to_dict(self) -> dict[str, Any]:
        """Provides backwards-compatible dictionary representation identical to old main.py:build_app()."""
        return {
            "router": self.router,
            "cli_channel": self.cli_channel,
            "orchestrator": self.orchestrator,
            "task_manager": self.task_manager,
            "tool_router": self.tool_router,
            "priority_inbox": self.priority_inbox,
            "priority_analyzer": self.priority_analyzer,
            "priority_analyzer_worker": self.priority_analyzer_worker,
            "priority_rules_manager": self.priority_rules_manager,
            "memory": self.memory,
            "ingestion_workers": self.ingestion_workers,
            "telegram_channel": self.telegram_channel,
            "whatsapp_channel": self.whatsapp_channel,
            "whatsapp_connector": self.whatsapp_connector,
            "device_gateway": self.device_gateway,
            "device_auth": self.device_auth,
        }

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()


_global_runtime: AssistantRuntime | None = None
_global_lock = threading.Lock()


def get_runtime() -> AssistantRuntime:
    """Returns or creates the process-wide AssistantRuntime singleton."""
    global _global_runtime
    with _global_lock:
        if _global_runtime is None:
            _global_runtime = AssistantRuntime()
        return _global_runtime


def reset_runtime() -> None:
    """Stops and resets the process-wide AssistantRuntime (primarily for test teardowns)."""
    global _global_runtime
    with _global_lock:
        if _global_runtime is not None:
            _global_runtime.stop()
            _global_runtime = None
