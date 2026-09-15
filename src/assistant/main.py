"""
Bootstrap / Entrypoint
==========================
Wires every plane together via AssistantRuntime:

  storage -> memory -> security/policy -> llm -> connectors ->
  execution (outbound registry, task manager, tool router) ->
  agent core (context builder, planner, approval, orchestrator) ->
  router (echo filter -> conversation router) -> channels

Run:  python -m assistant.main
"""

from __future__ import annotations

import logging
import os

from assistant.runtime import AssistantRuntime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.logging_config import (
    DEFAULT_ASSISTANT_LOG as assistant_log_file,
    DEFAULT_LOGS_DIR as logs_dir,
    LOG_FORMAT as log_format,
    configure_logging,
)


# Initialize default logging on import (clean CLI, logs to file)
configure_logging()

logger = logging.getLogger("assistant.main")


def build_app(runtime: AssistantRuntime | None = None) -> AssistantRuntime:
    """Builds and returns the application runtime.

    Backwards-compatible interface for callers expecting build_app().
    Returns an AssistantRuntime instance which also supports dict-like subscripting.
    """
    configure_logging()
    if runtime is None:
        runtime = AssistantRuntime(root_dir=ROOT)
    return runtime


def main() -> None:
    """CLI application entrypoint."""
    configure_logging()
    runtime = AssistantRuntime(root_dir=ROOT)
    runtime.start()
    try:
        runtime.cli_channel.run()
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
