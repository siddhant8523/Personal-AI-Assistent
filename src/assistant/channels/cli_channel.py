"""
CLI Channel (Part 1, Section 5.1)
=====================================
The laptop CLI is inherently an Agent-Control channel -- there's no
"normal conversation" concept for a local terminal. It also doubles as a
demo harness for the Message Intelligence pipeline via `/sim`, and for
the Echo Filter fix via `/echo-demo`.

Presents modern Rich terminal UX (CLI Presentation Layer) while keeping 100% of
underlying router and Agent Core mechanics unchanged.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from rich.console import Console

from assistant.channels.cli_ui_renderer import (
    CLISpinner,
    render_assistant_reply,
    render_header,
    render_user_input,
)
from assistant.ingestion.unified_message import Origin, Source, UnifiedMessage
from assistant.router.conversation_router import ConversationRouter

console = Console()

_SOURCE_MAP = {"whatsapp": Source.WHATSAPP, "telegram": Source.TELEGRAM, "sms": Source.SMS, "gmail": Source.GMAIL}


class CLIChannel:
    def __init__(
        self,
        router: ConversationRouter,
        agent_chat_conversation_id: str = "cli_default",
        device_gateway: Any = None,
        tool_router: Any = None,
        llm_model: str | None = None,
    ):
        self.router = router
        self.agent_chat_conversation_id = agent_chat_conversation_id
        self.device_gateway = device_gateway
        self.tool_router = tool_router
        self.llm_model = llm_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.spinner = CLISpinner()
        self._attach_listeners()

    def _attach_listeners(self) -> None:
        if self.tool_router is not None:
            self.tool_router.on_tool_start = self._on_tool_start
            self.tool_router.on_tool_finish = self._on_tool_finish
        if self.device_gateway is not None:
            self.device_gateway.on_command_start = self._on_device_command_start
            self.device_gateway.on_command_finish = self._on_device_command_finish

    def _on_tool_start(self, capability: str, task: Any) -> None:
        self.spinner.update(f"Executing tool {capability}...")

    def _on_tool_finish(self, capability: str, task: Any, result: dict) -> None:
        st = "✓" if result.get("status") in ("ok", "success") or "detail" in result else "✗"
        self.spinner.update(f"Tool {capability} {st}")

    def _on_device_command_start(self, capability: str, params: dict) -> None:
        self.spinner.update(f"Android · Dispatched {capability}...")

    def _on_device_command_finish(self, capability: str, result: Any) -> None:
        status_str = getattr(result, "status", "completed")
        st = "✓" if status_str == "ok" else "✗"
        self.spinner.update(f"Android · {capability} {st}")

    def reply(self, original: UnifiedMessage, text: str) -> None:
        self.spinner.stop()
        render_assistant_reply(text)

    def run(self) -> None:
        device_connected = False
        if self.device_gateway is not None and hasattr(self.device_gateway, "is_device_connected"):
            device_connected = bool(self.device_gateway.is_device_connected())

        render_header(model_name=self.llm_model, device_connected=device_connected, online=True)

        while True:
            try:
                raw_input = console.input("[bold bright_cyan]You › [/bold bright_cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold green]✓ Session ended.[/bold green] [dim]Goodbye.[/dim]\n")
                break

            text = raw_input.strip()
            if not text:
                continue

            if text in ("/quit", "exit", "quit"):
                console.print("\n[bold green]✓ Session ended.[/bold green] [dim]Goodbye.[/dim]\n")
                break

            if text == "/priority":
                text = "what are my important messages today?"

            if text.startswith("/sim "):
                self._handle_sim(text)
                continue

            if text == "/echo-demo":
                self._handle_echo_demo()
                continue

            self.spinner.start("Thinking...")
            try:
                message = UnifiedMessage(
                    source=Source.CLI,
                    conversation_id=self.agent_chat_conversation_id,
                    sender="user",
                    content=text,
                    origin=Origin.USER,
                )
                self.router.route(message)
            except Exception:
                self.spinner.stop()
                render_assistant_reply("Sorry, something went wrong while processing that request. Please try again.")
            finally:
                self.spinner.stop()

    def _handle_sim(self, text: str) -> None:
        parts = text.split(maxsplit=3)
        if len(parts) < 4:
            console.print("[bold red]usage: /sim <platform> <sender> <text>[/bold red]")
            return
        _, platform, sender, content = parts
        source = _SOURCE_MAP.get(platform.lower())
        if source is None:
            console.print(f"[bold red]unknown platform: {platform}[/bold red]")
            return
        message = UnifiedMessage(
            source=source,
            conversation_id=f"{platform.lower()}:{sender.lower()}_chat",
            sender=sender,
            content=content,
            origin=Origin.USER,
        )
        outcome = self.router.route(message)
        console.print(f"[dim]simulated {platform} message from {sender} -> route: {outcome}[/dim]")

    def _handle_echo_demo(self) -> None:
        """Sends an AGENT-origin message with the same content/conversation
        as a recent outbound send, proving the Echo Filter drops it instead
        of letting it loop back into the Agent Core."""
        console.print("[bold yellow]Simulating a WhatsApp platform echo of the agent's own last reply...[/bold yellow]")
        echoed = UnifiedMessage(
            source=Source.WHATSAPP,
            conversation_id=self.agent_chat_conversation_id,
            sender="me",
            content="[echo-demo] this is an echoed agent message",
            origin=Origin.AGENT,
        )
        outcome = self.router.route(echoed)
        console.print(f"[dim]route: {outcome} (expected: dropped_echo)[/dim]")
