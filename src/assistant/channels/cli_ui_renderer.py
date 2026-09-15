"""
CLI Presentation Layer / UI Renderer
===================================
Provides modern, polished Rich terminal UI components for the Personal AI Assistant.
Purely visual layer -- consumes existing events/results and formats terminal output.
No business logic, tool routing, LLM handling, or state modification.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED, DOUBLE_EDGE

console = Console()


def render_header(model_name: str | None = None, device_connected: bool = False, online: bool = True) -> None:
    resolved_model = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    status_text = "[bold green]● ONLINE[/bold green]" if online else "[bold yellow]○ OFFLINE[/bold yellow]"
    device_status = "[bold green]● Connected[/bold green]" if device_connected else "[dim]○ Offline[/dim]"

    content = Text()
    content.append("  ✦ JOE AI", style="bold cyan")
    content.append(" " * 34)
    content.append(status_text)
    content.append("\n  Personal AI Assistant\n\n", style="dim white")

    content.append("  ✓ Agent Core     ", style="bold green")
    content.append("· Running\n", style="dim white")

    content.append("  ✓ LLM Engine     ", style="bold green")
    content.append(f"· {resolved_model}\n", style="dim white")

    content.append("  ✓ Device Gateway ", style="bold green")
    content.append("· Enabled\n", style="dim white")

    content.append("  " + ("✓" if device_connected else "○") + " Android Device  ", style="bold green" if device_connected else "dim white")
    content.append(f"· {device_status}\n\n", style="dim white")

    content.append("  Commands:\n", style="bold yellow")
    content.append("    /sim <platform> <sender> <text>   Simulate inbound message\n", style="dim white")
    content.append("    /priority                         Show priority inbox\n", style="dim white")
    content.append("    /echo-demo                        Demonstrate self-echo filter\n", style="dim white")
    content.append("    /quit                             Exit session\n", style="dim white")

    panel = Panel(
        content,
        border_style="bright_cyan",
        box=ROUNDED,
        title="[bold bright_white]Assistant Terminal[/bold bright_white]",
        subtitle="[dim]Press Ctrl+C or type /quit to exit[/dim]",
    )
    console.print()
    console.print(panel)
    console.print()


def render_user_input(text: str) -> None:
    console.print()
    console.print("[bold bright_cyan]You[/bold bright_cyan]")
    console.print(f"[bright_white]› {text}[/bright_white]")
    console.print()


def render_approval_card(text: str) -> None:
    """Renders a structured approval box if the text contains an approval prompt."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    task_type = "Action"
    target = ""
    draft = ""

    # Parse standard approval string format
    # Proposed action via CLI:
    # make_call → mummy
    # Draft:
    # Make phone call to mummy
    for i, line in enumerate(lines):
        if "→" in line or "->" in line:
            parts = line.replace("->", "→").split("→")
            task_type = parts[0].strip()
            if len(parts) > 1:
                target = parts[1].strip()
        elif line.startswith("Draft:"):
            draft = "\n".join(lines[i+1:]).replace("Approve or reject?", "").strip()
            break

    if not draft and "Draft:" in text:
        draft = text.split("Draft:")[1].split("Approve or reject?")[0].strip()
    if not draft:
        draft = text

    icon_map = {
        "make_call": "☎",
        "call.make": "☎",
        "send_sms": "✉",
        "sms.send": "✉",
        "send_whatsapp_message": "💬",
        "send_telegram_message": "✈",
        "upload_file": "📄",
        "file.upload": "📄",
        "set_alarm": "🔔",
        "alarm.set": "🔔",
    }
    icon = icon_map.get(task_type.lower(), "⚙")

    card_text = Text()
    card_text.append(f"  {icon}  Requesting Approval: ", style="bold yellow")
    card_text.append(f"{task_type.upper()}\n\n", style="bold bright_white")

    if target:
        card_text.append("  Target / Recipient: ", style="dim white")
        card_text.append(f"{target}\n\n", style="bold cyan")

    card_text.append("  Draft Details:\n", style="dim white")
    card_text.append(f"  ──────────────────────────────────────────────────────────\n", style="dim border_style")
    card_text.append(f"  {draft}\n", style="italic white")

    panel = Panel(
        card_text,
        title="[bold yellow]╭─ Approval Required ─╮[/bold yellow]",
        border_style="yellow",
        box=ROUNDED,
    )
    console.print(panel)


def render_assistant_reply(text: str) -> None:
    if "Proposed action via" in text or "Approve or reject?" in text:
        render_approval_card(text)
        return

    console.print("[bold bright_magenta]✦ Joe[/bold bright_magenta]")
    if text.startswith("Approved and executed"):
        console.print(f"[bold green]✓ Done[/bold green]")
        console.print(f"[white]{text}[/white]")
    elif text.startswith("Approved, but failed"):
        console.print(f"[bold red]✗ Execution Failed[/bold red]")
        console.print(f"[white]{text}[/white]")
    elif text.startswith("Rejected"):
        console.print(f"[bold red]✗ Action Rejected[/bold red]")
        console.print(f"[dim white]{text}[/dim white]")
    elif text.startswith("Error") or "failed" in text.lower() and len(text) < 120:
        console.print(f"[white]{text}[/white]")
    else:
        console.print(f"[white]{text}[/white]")
    console.print()


def render_success(message: str) -> None:
    console.print(f"[bold green]✓ {message}[/bold green]")


def render_error(message: str) -> None:
    console.print(f"[bold red]✗ {message}[/bold red]")


class CLISpinner:
    """Thread-safe thinking spinner and activity logger for CLI UI."""
    def __init__(self):
        self._status = None
        self._lock = threading.Lock()
        self._active = False

    def start(self, initial_text: str = "Thinking...") -> None:
        with self._lock:
            if self._active and self._status:
                return
            self._status = console.status(f"[bold cyan]✦ Joe[/bold cyan]  [dim]{initial_text}[/dim]", spinner="dots")
            self._status.start()
            self._active = True

    def update(self, text: str) -> None:
        with self._lock:
            if self._active and self._status:
                self._status.update(f"[bold cyan]✦ Joe[/bold cyan]  [dim]{text}[/dim]")

    def stop(self) -> None:
        with self._lock:
            if self._active and self._status:
                self._status.stop()
                self._status = None
                self._active = False
