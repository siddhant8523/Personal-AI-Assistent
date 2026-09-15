"""
LLM Provider Router with Automatic Failover
===========================================
Manages automatic, bidirectional failover between primary (Groq) and
fallback (Mistral) LLM providers for the normal conversational agent.

Key Guarantees:
- Groq is primary; Mistral is fallback.
- Failover triggered on temporary failures: HTTP 429 (rate-limit), 5xx, timeouts, network drops.
- Permanent failures (401, 403, 400, missing credentials) fail fast without provider bouncing.
- Independent, thread-safe monotonic cooldowns per provider (prevents ping-pong).
- Groq is automatically preferred again once its cooldown expires.
- Maximum provider attempts bounded by LLM_MAX_PROVIDER_ATTEMPTS (default: 2).
- Streaming safety: failover is ONLY allowed before any token or tool-call chunk has been emitted.
- Tool-call safety: tool calls are normalized and never re-executed or replayed.
- Concurrency safety: provider state lock is NEVER held during network I/O.
- Strict isolation: STT (Whisper) and Priority Analyzer are completely separate and never routed here.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Iterator, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolCallChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, SkipValidation

from assistant.llm.llm_client import (
    LLMAuthenticationError,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMServiceError,
    classify_llm_error,
)

logger = logging.getLogger("assistant.llm")


# ==============================================================================
# Error Classification Helper
# ==============================================================================

class LLMResponseValidationError(LLMServiceError):
    """Raised when an LLM provider returns an empty, malformed, or unusable response."""
    category = "VALIDATION_ERROR"


def is_temporary_failover_error(exc: Exception) -> bool:
    """Classify whether an exception represents a temporary failure eligible for failover.

    Temporary (failover eligible):
    - Rate limit (HTTP 429)
    - Upstream service error (HTTP 500, 502, 503, 504)
    - Network timeout, connection drop, DNS failure
    - Empty, malformed, or unusable provider response

    Permanent (not eligible for failover; fast fail):
    - Authentication / Authorization (HTTP 401, 403)
    - Client error / Bad Request / Validation (HTTP 400, 422, invalid tool schema)
    - Not Found / Model unavailable (HTTP 404)
    """
    classified = classify_llm_error(exc) if not isinstance(exc, LLMError) else exc

    # 1. Validation errors (empty/malformed response) -> temporary/retryable with next provider
    if isinstance(classified, LLMResponseValidationError):
        return True

    # 2. Authentication & permissions -> permanent
    if isinstance(classified, LLMAuthenticationError):
        return False

    # 3. Check HTTP / gRPC status code if present on classified or underlying exception
    status_code = (
        getattr(classified, "status_code", None)
        or getattr(exc, "status_code", None)
        or getattr(exc, "raw_status_code", None)
        or (exc.code if isinstance(getattr(exc, "code", None), int) else None)
    )
    if status_code in (400, 401, 403, 404, 422):
        return False
    if status_code == 429:
        return True
    if status_code is not None and status_code >= 500:
        return True

    # 4. Rate limits -> temporary
    if isinstance(classified, LLMRateLimitError):
        return True

    # 5. Network and timeouts -> temporary
    if isinstance(classified, LLMNetworkError):
        return True

    # 6. Service errors (check status code or message)
    if isinstance(classified, LLMServiceError):
        msg = str(classified).lower()
        if any(kw in msg for kw in ("500", "502", "503", "504", "service unavailable", "bad gateway", "gateway timeout")):
            return True
        return False

    # 7. Check error message keywords for permanent errors
    msg = str(exc).lower()
    if any(kw in msg for kw in ("invalid_request_error", "bad request", "invalid api key", "unauthorized", "model_not_found", "permission_denied")):
        return False

    return False


def _normalize_content_to_text(content: Any) -> str:
    """Normalize provider content (str or Gemini-style list) to a plain text string.

    Gemini (ChatGoogleGenerativeAI) may return AIMessage/AIMessageChunk.content as:
      - str: "Hello world"  (Groq / Mistral style)
      - list: [{"type": "text", "text": "Hello world"}, ...]  (Gemini style)

    This helper extracts the concatenated plain text from either form without
    losing structured tool-call information (tool calls live in .tool_calls /
    .tool_call_chunks, not in content blocks).

    NEVER call str() on the raw list: that produces a Python repr like
    "[{'type': 'text', 'text': 'Hello world'}]" which would crash any downstream
    regex operation that expects a real string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Gemini text block: {"type": "text", "text": "..."}
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    # Fallback for unexpected types – coerce to string
    return str(content) if content is not None else ""


def _extract_error_details(exc: Exception, classified: LLMError) -> dict[str, Any]:
    """Safely extract error metadata for internal diagnostics without exposing secrets."""
    status_code = (
        getattr(classified, "status_code", None)
        or getattr(exc, "status_code", None)
        or getattr(exc, "raw_status_code", None)
        or (exc.code if isinstance(getattr(exc, "code", None), int) else None)
    )
    error_code = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error_code = body.get("code")
    elif isinstance(body, str):
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                error_code = parsed.get("code")
        except Exception:
            pass
    if not error_code:
        error_code = getattr(exc, "code", None)

    return {
        "exception_class": type(exc).__name__,
        "status_code": status_code,
        "error_code": error_code,
        "category": getattr(classified, "category", "UNKNOWN"),
    }


def _log_provider_failure(
    provider: str,
    raw_exc: Exception,
    classified: LLMError,
    model_name: str | None = None,
) -> None:
    """Log safe internal diagnostic for provider failure (no secrets, prompts, or sensitive payloads)."""
    details = _extract_error_details(raw_exc, classified)
    logger.warning(
        "[LLM] Diagnostic: provider=%s exception_class=%s status=%s error_code=%s category=%s model=%s",
        provider,
        details["exception_class"],
        details["status_code"],
        details["error_code"],
        details["category"],
        model_name or "unknown",
    )


def validate_chat_response(res: Any) -> AIMessage:
    """Validate that provider output is present, well-formed, and usable by Agent/LangGraph.

    At minimum validates:
    - Response exists
    - AI message is present/extractable
    - Content is non-empty OR valid tool calls exist
    - Tool calls can be normalized into {"name": ..., "args": ..., "id": ..., "type": "tool_call"}
    - Each tool call has non-empty name and valid args dict
    """
    if res is None:
        raise LLMResponseValidationError("Provider returned None response")

    if isinstance(res, AIMessage):
        ai_msg = res
    elif hasattr(res, "generations") and res.generations:
        gen = res.generations[0]
        ai_msg = getattr(gen, "message", None)
        if not isinstance(ai_msg, AIMessage):
            raw_content = getattr(gen, "text", "") or getattr(ai_msg, "content", "")
            # Normalize list content (Gemini) to plain text for the AIMessage wrapper
            ai_msg = AIMessage(content=_normalize_content_to_text(raw_content))
    elif hasattr(res, "content"):
        # Preserve list content as-is if it is an AIMessage subtype (Gemini returns one);
        # otherwise flatten to text for generic content-bearing objects.
        if isinstance(res, AIMessage):
            ai_msg = res
        else:
            ai_msg = AIMessage(
                content=_normalize_content_to_text(res.content),
                tool_calls=getattr(res, "tool_calls", None) or [],
            )
    else:
        ai_msg = AIMessage(content=str(res))

    raw_content = getattr(ai_msg, "content", "")
    raw_tool_calls = getattr(ai_msg, "tool_calls", None) or []

    # If tool calls are present, normalize and validate each tool call
    if raw_tool_calls:
        try:
            normalized_tcs = normalize_tool_calls(raw_tool_calls)
        except Exception as exc:
            raise LLMResponseValidationError(f"Failed to normalize tool calls: {exc}") from exc

        for tc in normalized_tcs:
            if not tc.get("name") or not isinstance(tc.get("args"), dict):
                raise LLMResponseValidationError(f"Malformed tool call in response: {tc}")

        ai_msg.tool_calls = normalized_tcs

    # Determine whether content is non-empty, handling both str and list-of-blocks (Gemini).
    # IMPORTANT: we check the normalized text, NOT str(list), to avoid false positives.
    text_content = _normalize_content_to_text(raw_content)
    has_text = bool(text_content.strip())
    has_tools = bool(ai_msg.tool_calls)

    if not has_text and not has_tools:
        raise LLMResponseValidationError("Provider returned empty content with no tool calls")

    # Normalize list-style content to plain text on the returned AIMessage so that
    # all downstream code (respond_node, strip_markdown_formatting, etc.) receives a
    # plain string and never encounters a list where a str is expected.
    if isinstance(raw_content, list):
        ai_msg.content = text_content

    return ai_msg


# ==============================================================================
# Provider State Tracker
# ==============================================================================

class ProviderStateTracker:
    """Thread-safe provider state and cooldown tracker using monotonic time.

    Invariants:
    - Independent cooldown timers per provider.
    - Global lock is held ONLY during in-memory state reads/writes (microseconds),
      NEVER during network requests.
    """

    def __init__(
        self,
        cooldown_seconds: float = 60.0,
        providers: Sequence[str] = ("groq", "mistral", "gemini"),
    ):
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {p: 0.0 for p in providers}
        self._failure_counts: dict[str, int] = {p: 0 for p in providers}

    def is_available(self, provider: str) -> bool:
        """Check if a specific provider is currently available (not in cooldown)."""
        now = time.monotonic()
        with self._lock:
            cd = self._cooldown_until.get(provider, 0.0)
            if now >= cd:
                if cd > 0.0:
                    logger.info("[LLM] %s available again", provider.capitalize())
                    self._cooldown_until[provider] = 0.0
                return True
            return False

    def select_provider(
        self,
        preferred_order: tuple[str, ...] = ("groq", "mistral", "gemini"),
        configured_providers: tuple[str, ...] = ("groq", "mistral", "gemini"),
    ) -> str | None:
        """Select preferred available provider in thread-safe manner.

        Returns provider name or None if all configured providers are cooling down.
        """
        now = time.monotonic()
        with self._lock:
            # Check for any expired cooldowns
            for p in configured_providers:
                cd = self._cooldown_until.get(p, 0.0)
                if cd > 0.0 and now >= cd:
                    logger.info("[LLM] %s available again", p.capitalize())
                    self._cooldown_until[p] = 0.0

            # Iterate through preferred_order; pick first configured provider not in cooldown
            for candidate in preferred_order:
                if candidate in configured_providers and now >= self._cooldown_until.get(candidate, 0.0):
                    return candidate

            # All cooling down
            return None

    def mark_failure(
        self,
        provider: str,
        cooldown_override: float | None = None,
    ) -> None:
        """Mark a provider cooling down after a temporary failure."""
        duration = cooldown_override if cooldown_override is not None else self.cooldown_seconds
        duration = max(0.0, float(duration))
        now = time.monotonic()
        with self._lock:
            self._cooldown_until[provider] = now + duration
            self._failure_counts[provider] = self._failure_counts.get(provider, 0) + 1
            logger.info(
                "[LLM] %s cooldown started (%ds)",
                provider.capitalize(),
                int(duration),
            )

    def mark_success(self, provider: str) -> None:
        """Clear transient failures for a provider upon success."""
        with self._lock:
            self._failure_counts[provider] = 0
            self._cooldown_until[provider] = 0.0

    def get_state(self) -> dict[str, Any]:
        """Return snapshot of provider availability and remaining cooldowns."""
        now = time.monotonic()
        with self._lock:
            state: dict[str, Any] = {}
            for p in self._cooldown_until.keys():
                cd = self._cooldown_until.get(p, 0.0)
                remaining = max(0.0, cd - now)
                state[p] = {
                    "available": remaining <= 0.0,
                    "cooldown_remaining": round(remaining, 1),
                    "failures": self._failure_counts.get(p, 0),
                }
            return state

    def reset(self) -> None:
        """Reset all provider cooldowns (for test isolation)."""
        with self._lock:
            for p in list(self._cooldown_until.keys()):
                self._cooldown_until[p] = 0.0
                self._failure_counts[p] = 0


# ==============================================================================
# Tool Call Normalization Helper
# ==============================================================================

def normalize_tool_calls(tool_calls: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Normalize tool calls into standard dict structure:
    {"name": str, "args": dict, "id": str, "type": "tool_call"}
    """
    if not tool_calls:
        return []
    normalized: list[dict[str, Any]] = []
    for idx, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            name = getattr(tc, "name", "")
            raw_args = getattr(tc, "args", {})
            tc_id = getattr(tc, "id", None) or f"call_{idx}"
        else:
            name = tc.get("name", "")
            raw_args = tc.get("args", {})
            tc_id = tc.get("id") or f"call_{idx}"

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        normalized.append(
            {
                "name": name,
                "args": args,
                "id": str(tc_id),
                "type": "tool_call",
            }
        )
    return normalized


# ==============================================================================
# LLM Provider Router
# ==============================================================================

class LLMProviderRouter(BaseChatModel):
    """LangChain-compatible ChatModel with automatic Groq <-> Mistral <-> Gemini failover.

    Operates strictly on the Normal Conversational Agent LLM path.
    """

    state_tracker: SkipValidation[ProviderStateTracker] = Field(default_factory=ProviderStateTracker)
    groq_model: SkipValidation[Any] = None
    mistral_model: SkipValidation[Any] = None
    gemini_model: SkipValidation[Any] = None
    provider_order: tuple[str, ...] = ("groq", "mistral", "gemini")
    failover_enabled: bool = True
    max_provider_attempts: int = 4
    cooldown_seconds: float = 60.0
    bound_tools: list[Any] = Field(default_factory=list)

    def __init__(
        self,
        groq_model: Any = None,
        mistral_model: Any = None,
        gemini_model: Any = None,
        state_tracker: ProviderStateTracker | None = None,
        provider_order: tuple[str, ...] | None = None,
        failover_enabled: bool = True,
        max_provider_attempts: int = 4,
        cooldown_seconds: float = 60.0,
        bound_tools: list[Any] | None = None,
        **kwargs: Any,
    ):
        order = tuple(provider_order) if provider_order else ("groq", "mistral", "gemini")
        tracker = state_tracker or ProviderStateTracker(cooldown_seconds=cooldown_seconds, providers=order)
        super().__init__(
            state_tracker=tracker,
            groq_model=groq_model,
            mistral_model=mistral_model,
            gemini_model=gemini_model,
            provider_order=order,
            failover_enabled=failover_enabled,
            max_provider_attempts=max_provider_attempts,
            cooldown_seconds=cooldown_seconds,
            bound_tools=bound_tools or [],
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "llm-provider-router"

    def _get_configured_providers(self) -> tuple[str, ...]:
        providers = []
        if self.groq_model is not None:
            providers.append("groq")
        if self.mistral_model is not None:
            providers.append("mistral")
        if self.gemini_model is not None:
            providers.append("gemini")
        return tuple(providers)

    def _get_model_for_provider(self, provider: str) -> Any:
        if provider == "groq":
            return self.groq_model
        if provider == "mistral":
            return self.mistral_model
        if provider == "gemini":
            return self.gemini_model
        raise ValueError(f"Unknown provider: {provider}")

    def _get_next_provider(self, current_provider: str) -> str | None:
        """Get the next provider in configured cyclic order."""
        configured = self._get_configured_providers()
        available_order = [p for p in self.provider_order if p in configured]
        if not available_order or len(available_order) <= 1:
            return None
        if current_provider in available_order:
            idx = available_order.index(current_provider)
            return available_order[(idx + 1) % len(available_order)]
        return available_order[0]

    def select_provider(self) -> str:
        """Select the preferred available provider respecting cooldowns."""
        configured = self._get_configured_providers()
        if not configured:
            raise LLMServiceError("No LLM providers are configured.")

        if not self.failover_enabled:
            # When failover is disabled, strictly use primary (Groq)
            if "groq" in configured:
                return "groq"
            return configured[0]

        selected = self.state_tracker.select_provider(
            preferred_order=self.provider_order,
            configured_providers=configured,
        )
        if selected is None:
            raise LLMServiceError(
                "All AI services are temporarily unavailable. Please try again shortly."
            )
        return selected

    def mark_failure(self, provider: str, exc: Exception | None = None) -> None:
        classified = classify_llm_error(exc) if exc and not isinstance(exc, LLMError) else exc
        cooldown_override = None
        if isinstance(classified, LLMRateLimitError) and classified.retry_after is not None:
            # If provider explicitly specified retry_after, use max(retry_after, cooldown)
            if classified.retry_after > self.cooldown_seconds:
                cooldown_override = classified.retry_after
        self.state_tracker.mark_failure(provider, cooldown_override=cooldown_override)

    def mark_success(self, provider: str) -> None:
        self.state_tracker.mark_success(provider)

    def get_provider_state(self) -> dict[str, Any]:
        return self.state_tracker.get_state()

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> BaseChatModel:
        """Return a new LLMProviderRouter with tools bound to all underlying models."""
        bound_groq = None
        if self.groq_model is not None:
            bound_groq = self.groq_model.bind_tools(tools, **kwargs) if hasattr(self.groq_model, "bind_tools") else self.groq_model

        bound_mistral = None
        if self.mistral_model is not None:
            bound_mistral = self.mistral_model.bind_tools(tools, **kwargs) if hasattr(self.mistral_model, "bind_tools") else self.mistral_model

        bound_gemini = None
        if self.gemini_model is not None:
            bound_gemini = self.gemini_model.bind_tools(tools, **kwargs) if hasattr(self.gemini_model, "bind_tools") else self.gemini_model

        return LLMProviderRouter(
            groq_model=bound_groq,
            mistral_model=bound_mistral,
            gemini_model=bound_gemini,
            state_tracker=self.state_tracker,
            provider_order=self.provider_order,
            failover_enabled=self.failover_enabled,
            max_provider_attempts=self.max_provider_attempts,
            cooldown_seconds=self.cooldown_seconds,
            bound_tools=tools,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Invoke provider with automatic failover bounded by max_provider_attempts."""
        last_exc: Exception | None = None
        attempted_sequence: list[str] = []

        # 1. Select initial provider (for a new request, respects cooldowns)
        try:
            current_provider = self.select_provider()
        except LLMServiceError as exc:
            raise exc

        while len(attempted_sequence) < self.max_provider_attempts:
            attempted_sequence.append(current_provider)
            logger.info("[LLM] Provider selected: %s", current_provider)
            model = self._get_model_for_provider(current_provider)
            model_name = (
                getattr(model, "model_name", None)
                or getattr(model, "model", None)
                or getattr(model, "name", None)
            )

            # 2. Perform network invocation (LOCK IS NEVER HELD HERE)
            try:
                raw_res = model.invoke(messages, stop=stop, **kwargs)
                ai_msg = validate_chat_response(raw_res)
                self.mark_success(current_provider)
                return ChatResult(generations=[ChatGeneration(message=ai_msg)])

            except Exception as raw_exc:
                last_exc = raw_exc
                classified = classify_llm_error(raw_exc)
                is_temp = is_temporary_failover_error(classified)

                # Mark provider cooling down and log diagnostic
                self.mark_failure(current_provider, classified)
                _log_provider_failure(current_provider, raw_exc, classified, model_name)

                # Permanent error -> fail immediately, no failover!
                if not is_temp:
                    raise classified from raw_exc

                # Check if failover is allowed and next provider can be attempted
                next_provider = self._get_next_provider(current_provider)
                can_failover = (
                    self.failover_enabled
                    and next_provider is not None
                    and len(attempted_sequence) < self.max_provider_attempts
                )

                if can_failover:
                    reason = "rate limited" if isinstance(classified, LLMRateLimitError) else "temporary failure"
                    logger.warning(
                        "[LLM] %s %s; switching to %s",
                        current_provider.capitalize(),
                        reason,
                        next_provider.capitalize(),
                    )
                    current_provider = next_provider
                    continue

                # Attempts exhausted or no fallback available
                if len(attempted_sequence) > 1 or len(self._get_configured_providers()) > 1:
                    logger.warning("[LLM] All providers unavailable")
                    raise LLMServiceError(
                        "All AI services are temporarily unavailable. Please try again shortly.",
                        original_error=raw_exc,
                    )

                raise classified from raw_exc

        if last_exc:
            raise LLMServiceError(
                "All AI services are temporarily unavailable. Please try again shortly.",
                original_error=last_exc,
            )
        raise LLMServiceError("All AI services are temporarily unavailable. Please try again shortly.")

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream provider chunks with safe failover.

        CRITICAL STREAMING INVARIANT:
        Failover is allowed ONLY before any tokens or tool-call chunks have been
        emitted (`emitted_any == False`). Once user-visible output begins, the
        stream will NEVER fail over or replay to prevent duplicated output.
        """
        last_exc: Exception | None = None
        attempted_sequence: list[str] = []

        try:
            current_provider = self.select_provider()
        except LLMServiceError as exc:
            raise exc

        while len(attempted_sequence) < self.max_provider_attempts:
            attempted_sequence.append(current_provider)
            logger.info("[LLM] Provider selected: %s", current_provider)
            model = self._get_model_for_provider(current_provider)
            model_name = (
                getattr(model, "model_name", None)
                or getattr(model, "model", None)
                or getattr(model, "name", None)
            )

            emitted_any = False

            # 2. Perform streaming (LOCK IS NEVER HELD HERE)
            try:
                stream_iter = model.stream(messages, stop=stop, run_manager=run_manager, **kwargs)
                for chunk in stream_iter:
                    if isinstance(chunk, ChatGenerationChunk):
                        gen_chunk = chunk
                        msg_chunk = chunk.message
                    elif isinstance(chunk, AIMessageChunk):
                        msg_chunk = chunk
                        gen_chunk = ChatGenerationChunk(message=msg_chunk)
                    else:
                        msg_chunk = AIMessageChunk(content=str(getattr(chunk, "content", chunk)))
                        gen_chunk = ChatGenerationChunk(message=msg_chunk)

                    raw_content = getattr(msg_chunk, "content", None)
                    # Gemini may emit content as a list of structured blocks.
                    # Extract plain text for run_manager (which expects a str) while
                    # yielding the original chunk unchanged so LangGraph can accumulate
                    # tool_calls and other structured data from it.
                    text_content = _normalize_content_to_text(raw_content) if raw_content is not None else ""
                    tc_chunks = getattr(msg_chunk, "tool_call_chunks", None) or []

                    if text_content or tc_chunks:
                        emitted_any = True

                    if run_manager and text_content:
                        run_manager.on_llm_new_token(text_content, chunk=gen_chunk)

                    yield gen_chunk

                # If stream finished without emitting ANY content or tool chunks, it is empty/unusable
                if not emitted_any:
                    raise LLMResponseValidationError("Provider stream completed with empty content and no tool calls")

                # Stream completed successfully
                self.mark_success(current_provider)
                return

            except Exception as raw_exc:
                last_exc = raw_exc
                classified = classify_llm_error(raw_exc)
                is_temp = is_temporary_failover_error(classified)

                if emitted_any:
                    # CRITICAL: Stream output already started. NEVER switch providers or replay.
                    logger.warning(
                        "[LLM Stream] Failure after output started provider=%s error=%s",
                        current_provider,
                        type(raw_exc).__name__,
                    )
                    self.mark_failure(current_provider, classified)
                    _log_provider_failure(current_provider, raw_exc, classified, model_name)
                    raise classified from raw_exc

                # Failed before any output was emitted
                self.mark_failure(current_provider, classified)
                _log_provider_failure(current_provider, raw_exc, classified, model_name)

                if not is_temp:
                    raise classified from raw_exc

                next_provider = self._get_next_provider(current_provider)
                can_failover = (
                    self.failover_enabled
                    and next_provider is not None
                    and len(attempted_sequence) < self.max_provider_attempts
                )

                if can_failover:
                    reason = "rate limited" if isinstance(classified, LLMRateLimitError) else "temporary failure"
                    logger.warning(
                        "[LLM] %s %s; switching to %s",
                        current_provider.capitalize(),
                        reason,
                        next_provider.capitalize(),
                    )
                    current_provider = next_provider
                    continue

                if len(attempted_sequence) > 1 or len(self._get_configured_providers()) > 1:
                    logger.warning("[LLM] All providers unavailable")
                    raise LLMServiceError(
                        "All AI services are temporarily unavailable. Please try again shortly.",
                        original_error=raw_exc,
                    )

                raise classified from raw_exc

        if last_exc:
            raise LLMServiceError(
                "All AI services are temporarily unavailable. Please try again shortly.",
                original_error=last_exc,
            )
        raise LLMServiceError("All AI services are temporarily unavailable. Please try again shortly.")
