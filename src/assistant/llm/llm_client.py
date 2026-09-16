"""
LLM Client & Model Factory
===========================
Thin wrapper and model factory so the rest of the app never imports a specific SDK directly.

Supports Groq, Mistral, and easily extensible to other providers.
If no valid API key is set for the selected provider, falls back to offline mode.
"""

from __future__ import annotations
from collections.abc import Sequence

import json
import logging
import os
import re
import time
from typing import Any, Iterator

from pydantic import SkipValidation

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCallChunk,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger("assistant.llm")


# ==============================================================================
# LLM Exception Hierarchy
# ==============================================================================

class LLMError(Exception):
    """Base exception for LLM-related failures."""
    category: str = "UNKNOWN"

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class LLMNetworkError(LLMError):
    """Network connection, DNS resolution, or socket timeout failure."""
    category = "NETWORK"


class LLMRateLimitError(LLMError):
    """Rate limit exceeded (HTTP 429)."""
    category = "RATE_LIMIT"

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message, original_error=original_error)
        self.retry_after = retry_after


class LLMAuthenticationError(LLMError):
    """Authentication or authorization failure (HTTP 401/403)."""
    category = "AUTH"


class LLMServiceError(LLMError):
    """Upstream service or model error (HTTP 5xx, 404)."""
    category = "SERVICE_ERROR"

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message, original_error=original_error)
        self.status_code = status_code


def classify_llm_error(exc: Exception) -> LLMError:
    """Classify provider, SDK, or HTTP transport exceptions into structured LLMErrors."""
    if isinstance(exc, LLMError):
        return exc

    exc_name = type(exc).__name__
    msg = str(exc)
    msg_lower = msg.lower()
    status_code = (
        getattr(exc, "status_code", None)
        or getattr(exc, "raw_status_code", None)
        or (exc.code if isinstance(getattr(exc, "code", None), int) else None)
    )

    # 1. Rate limits (HTTP 429)
    if status_code == 429 or "ratelimit" in exc_name.lower() or "rate limit" in msg_lower or "429" in msg_lower:
        retry_after: float | None = None
        response = getattr(exc, "response", None)
        if response is not None and hasattr(response, "headers"):
            ra_val = response.headers.get("retry-after")
            if ra_val:
                try:
                    retry_after = float(ra_val)
                except (ValueError, TypeError):
                    pass
        if retry_after is None:
            m = re.search(r"try again in ([\d\.]+)s", msg_lower) or re.search(r"in ([\d\.]+) seconds", msg_lower)
            if m:
                try:
                    retry_after = float(m.group(1))
                except (ValueError, TypeError):
                    pass
        return LLMRateLimitError(
            message="The AI service is temporarily rate-limited.",
            retry_after=retry_after,
            original_error=exc,
        )

    # 2. Authentication & Authorization (HTTP 401/403)
    if status_code in (401, 403) or "auth" in exc_name.lower() or "permission" in exc_name.lower() or "unauthorized" in msg_lower or "api key" in msg_lower:
        return LLMAuthenticationError(
            message="The AI service authentication failed. Please check your API key configuration.",
            original_error=exc,
        )

    # 3. Network, DNS, timeout, connection failures
    network_keywords = (
        "connection",
        "connect",
        "socket",
        "dns",
        "name resolution",
        "not known",
        "unreachable",
        "refused",
        "reset",
        "timed out",
        "timeout",
        "network",
        "gaierror",
    )
    is_network = (
        "connect" in exc_name.lower()
        or "timeout" in exc_name.lower()
        or "network" in exc_name.lower()
        or "socket" in exc_name.lower()
        or any(kw in msg_lower for kw in network_keywords)
    )
    if not is_network:
        try:
            import httpx
            if isinstance(exc, (httpx.NetworkError, httpx.TimeoutException)):
                is_network = True
        except ImportError:
            pass

    if is_network:
        return LLMNetworkError(
            message="Sorry, I can't reach the AI service right now. It looks like your internet connection is unavailable. Please check your connection and try again.",
            original_error=exc,
        )

    # 4. Service / Upstream failures (HTTP 5xx, 404)
    if (status_code and status_code >= 500) or "internalserver" in exc_name.lower() or "badgateway" in exc_name.lower() or "serviceunavailable" in exc_name.lower() or "gatewaytimeout" in exc_name.lower():
        return LLMServiceError(
            message="The AI service is temporarily unavailable. Please try again shortly.",
            status_code=status_code,
            original_error=exc,
        )

    if status_code == 404 or "notfound" in exc_name.lower() or ("model" in msg_lower and "not found" in msg_lower):
        return LLMServiceError(
            message="The requested AI model is currently unavailable.",
            status_code=404,
            original_error=exc,
        )

    return LLMError(
        message="Sorry, something went wrong while processing that request. Please try again.",
        original_error=exc,
    )


def llm_error_to_user_message(err: Exception) -> str:
    """Produce safe, human-readable user message without exposing internal details."""
    classified = classify_llm_error(err) if not isinstance(err, LLMError) else err

    if isinstance(classified, LLMNetworkError):
        return "Sorry, I can't reach the AI service right now. It looks like your internet connection is unavailable. Please check your connection and try again."

    if isinstance(classified, LLMRateLimitError):
        if classified.retry_after is not None and classified.retry_after > 0:
            secs = max(1, round(classified.retry_after))
            return f"The AI service is temporarily rate-limited. Please try again in about {secs} seconds."
        return "The AI service is temporarily rate-limited. Please try again in a moment."

    if isinstance(classified, LLMAuthenticationError):
        return "The AI service authentication failed. Please check your API key configuration."

    if isinstance(classified, LLMServiceError):
        return "The AI service is temporarily unavailable. Please try again shortly."

    return "Sorry, something went wrong while processing that request. Please try again."


# ==============================================================================
# LangChain Custom Chat Model for Groq
# ==============================================================================

class ChatGroqCustom(BaseChatModel):
    client: SkipValidation[Any] = None
    model_name: str = "openai/gpt-oss-120b"
    bound_tools: list = []

    def _messages_to_groq(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        groq_msgs = []
        for m in messages:
            if isinstance(m, SystemMessage):
                groq_msgs.append({"role": "system", "content": str(m.content)})
            elif isinstance(m, HumanMessage):
                groq_msgs.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage):
                msg_dict: dict[str, Any] = {"role": "assistant", "content": str(m.content or "")}
                if m.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.get("id") or f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("args", {}))
                                if isinstance(tc.get("args"), dict)
                                else str(tc.get("args")),
                            },
                        }
                        for idx, tc in enumerate(m.tool_calls)
                    ]
                groq_msgs.append(msg_dict)
            elif isinstance(m, ToolMessage):
                groq_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": str(m.content),
                    }
                )
        return groq_msgs

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        groq_msgs = self._messages_to_groq(messages)
        params: dict[str, Any] = {"model": self.model_name, "messages": groq_msgs}
        if self.bound_tools:
            params["tools"] = [convert_to_openai_tool(t) for t in self.bound_tools]

        max_attempts = 2
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            try:
                res = self.client.chat.completions.create(**params)
                break
            except Exception as raw_exc:
                last_exc = raw_exc
                classified = classify_llm_error(raw_exc)
                logger.warning(
                    "[LLM] %s provider=groq model=%s attempt=%d/%d error=%s",
                    classified.category,
                    self.model_name,
                    attempt + 1,
                    max_attempts,
                    type(raw_exc).__name__,
                )

                # Bounded retry: retry rate-limit once only if retry_after <= 2.0 seconds
                if isinstance(classified, LLMRateLimitError):
                    if attempt == 0 and classified.retry_after is not None and 0 < classified.retry_after <= 2.0:
                        time.sleep(classified.retry_after)
                        continue
                    raise classified from raw_exc

                # Bounded retry: fast fail on network errors with at most 1 quick transient retry
                if isinstance(classified, LLMNetworkError):
                    if attempt == 0:
                        time.sleep(0.5)
                        continue
                    raise classified from raw_exc

                raise classified from raw_exc
        else:
            if last_exc:
                raise classify_llm_error(last_exc)

        msg = res.choices[0].message
        tcs = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "call_unknown")
                tc_func = getattr(tc, "function", None)
                if tc_func is not None:
                    name = getattr(tc_func, "name", "")
                    raw_args = getattr(tc_func, "arguments", "{}")
                elif isinstance(tc, dict):
                    name = tc.get("name", "")
                    tc_f = tc.get("function")
                    if isinstance(tc_f, dict):
                        name = tc_f.get("name") or name
                        raw_args = tc_f.get("arguments", "{}")
                    else:
                        raw_args = tc.get("args", "{}")
                else:
                    name = getattr(tc, "name", "")
                    raw_args = getattr(tc, "args", "{}")

                if isinstance(raw_args, dict):
                    args = raw_args
                else:
                    try:
                        args = json.loads(str(raw_args) or "{}")
                    except Exception:
                        args = {}

                tcs.append(
                    {
                        "name": name,
                        "args": args,
                        "id": tc_id,
                        "type": "tool_call",
                    }
                )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=msg.content or "", tool_calls=tcs)
                )
            ]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        groq_msgs = self._messages_to_groq(messages)
        params: dict[str, Any] = {"model": self.model_name, "messages": groq_msgs, "stream": True}
        if self.bound_tools:
            params["tools"] = [convert_to_openai_tool(t) for t in self.bound_tools]

        max_attempts = 2
        emitted_any = False

        for attempt in range(max_attempts):
            try:
                stream_resp = self.client.chat.completions.create(**params)
                for chunk in stream_resp:
                    choices = getattr(chunk, "choices", None) or (chunk.get("choices") if isinstance(chunk, dict) else None)
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = getattr(choice, "delta", None) or (choice.get("delta") if isinstance(choice, dict) else None)
                    if delta is None:
                        continue
                    content = getattr(delta, "content", None) or (delta.get("content") if isinstance(delta, dict) else "") or ""

                    tc_chunks: list[ToolCallChunk] = []
                    delta_tcs = getattr(delta, "tool_calls", None) or (delta.get("tool_calls") if isinstance(delta, dict) else None)
                    if delta_tcs:
                        for tc in delta_tcs:
                            tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
                            tc_index = getattr(tc, "index", 0) if not isinstance(tc, dict) else tc.get("index", 0)
                            tc_func = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function")
                            if tc_func is not None:
                                name = getattr(tc_func, "name", None) if not isinstance(tc_func, dict) else tc_func.get("name")
                                args = getattr(tc_func, "arguments", None) if not isinstance(tc_func, dict) else tc_func.get("arguments")
                            else:
                                name = None
                                args = None
                            tc_chunks.append(
                                ToolCallChunk(
                                    name=name,
                                    args=args,
                                    id=tc_id,
                                    index=tc_index,
                                )
                            )

                    if content or tc_chunks:
                        emitted_any = True
                        msg_chunk = AIMessageChunk(content=content, tool_call_chunks=tc_chunks)
                        gen_chunk = ChatGenerationChunk(message=msg_chunk)
                        if run_manager:
                            run_manager.on_llm_new_token(content, chunk=gen_chunk)
                        yield gen_chunk
                break
            except Exception as raw_exc:
                classified = classify_llm_error(raw_exc)
                logger.warning(
                    "[LLM Stream] %s provider=groq model=%s attempt=%d/%d emitted=%s error=%s",
                    classified.category,
                    self.model_name,
                    attempt + 1,
                    max_attempts,
                    emitted_any,
                    type(raw_exc).__name__,
                )
                if emitted_any:
                    # CRITICAL: Never restart the stream after tokens/chunks have been emitted
                    raise classified from raw_exc

                # Bounded retry before any output:
                if isinstance(classified, LLMRateLimitError):
                    if attempt == 0 and classified.retry_after is not None and 0 < classified.retry_after <= 2.0:
                        time.sleep(classified.retry_after)
                        continue
                    raise classified from raw_exc

                if isinstance(classified, LLMNetworkError):
                    if attempt == 0:
                        time.sleep(0.5)
                        continue
                    raise classified from raw_exc

                raise classified from raw_exc

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> BaseChatModel:
        return self.__class__(
            client=self.client, model_name=self.model_name, bound_tools=tools
        )

    @property
    def _llm_type(self) -> str:
        return "groq-custom"


# ==============================================================================
# LLMClient & Model Factory
# ==============================================================================

class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        mistral_api_key: str | None = None,
        mistral_model: str | None = None,
        google_api_key: str | None = None,
        gemini_model: str | None = None,
        provider_order: Sequence[str] | None = None,
        failover_enabled: bool | None = None,
        cooldown_seconds: float | None = None,
        max_provider_attempts: int | None = None,
    ):
        # Read environment variables directly in __init__ so .env loaded after module import is respected
        env_provider = os.getenv("LLM_PROVIDER", "").lower()
        if provider:
            self.provider = provider.lower()
            self.explicit_provider = True
        elif env_provider:
            self.provider = env_provider
            self.explicit_provider = False
        else:
            self.provider = "groq"
            self.explicit_provider = False

        if failover_enabled is not None:
            self.failover_enabled = failover_enabled
        elif self.explicit_provider:
            # Explicit single provider instances (e.g. Priority Analyzer) default to no failover
            self.failover_enabled = False
        else:
            self.failover_enabled = os.getenv("LLM_FAILOVER_ENABLED", "true").lower() in ("true", "1", "yes")

        self.provider_order = (
            tuple(provider_order)
            if provider_order is not None
            else tuple(
                p.strip().lower()
                for p in os.getenv("LLM_PROVIDER_ORDER", "groq,mistral,gemini").split(",")
                if p.strip()
            )
        )

        self.cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else float(os.getenv("LLM_PROVIDER_COOLDOWN_SECONDS", "60"))
        )
        self.max_provider_attempts = (
            max_provider_attempts
            if max_provider_attempts is not None
            else int(os.getenv("LLM_MAX_PROVIDER_ATTEMPTS", "4"))
        )

        # Explicit empty api_key ("") indicates caller requested offline mode
        is_explicit_offline = (api_key == "")

        # Groq configuration (Primary)
        if is_explicit_offline:
            self.groq_api_key = ""
        elif api_key is not None and (self.provider == "groq" or not self.explicit_provider):
            self.groq_api_key = api_key
        else:
            self.groq_api_key = os.getenv("GROQ_API_KEY", "")

        self.groq_model = (
            model
            if (model is not None and self.provider == "groq")
            else os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        )
        self._groq_client = None
        if self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key, timeout=30.0, max_retries=0)
            except ImportError:
                self._groq_client = None
            except Exception as exc:
                logger.warning("Failed to initialize Groq client: %s", exc)
                self._groq_client = None

        # Mistral configuration (Fallback)
        if is_explicit_offline:
            self.mistral_api_key = ""
        elif mistral_api_key is not None:
            self.mistral_api_key = mistral_api_key
        elif api_key is not None and self.provider == "mistral" and self.explicit_provider:
            self.mistral_api_key = api_key
        else:
            self.mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
        self.mistral_model = (
            mistral_model
            if mistral_model is not None
            else (
                model
                if (model is not None and self.provider == "mistral")
                else os.environ.get("MISTRAL_MODEL", "mistral-small-2603")
            )
        )
        self._mistral_client = None
        if self.mistral_api_key:
            try:
                from mistralai.client import Mistral
                self._mistral_client = Mistral(api_key=self.mistral_api_key)
            except ImportError:
                self._mistral_client = None
            except Exception as exc:
                logger.warning("Failed to initialize Mistral client: %s", exc)
                self._mistral_client = None

        # Google Gemini configuration (Third provider)
        if is_explicit_offline:
            self.google_api_key = ""
        elif google_api_key is not None:
            self.google_api_key = google_api_key
        elif api_key is not None and self.provider == "gemini" and self.explicit_provider:
            self.google_api_key = api_key
        else:
            self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.gemini_model = (
            gemini_model
            if gemini_model is not None
            else (
                model
                if (model is not None and self.provider == "gemini")
                else os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            )
        )

        # Backwards compatibility attributes
        if self.provider == "groq":
            self.api_key = self.groq_api_key
            self.model = self.groq_model
        elif self.provider == "mistral":
            self.api_key = self.mistral_api_key
            self.model = self.mistral_model
        else:
            self.api_key = self.google_api_key
            self.model = self.gemini_model

        self._state_tracker = None

    @property
    def _client(self) -> Any:
        if self.provider == "groq":
            return self._groq_client
        return self._mistral_client

    @_client.setter
    def _client(self, val: Any) -> None:
        if self.provider == "groq":
            self._groq_client = val
        else:
            self._mistral_client = val

    @property
    def online(self) -> bool:
        return (
            self._groq_client is not None
            or self._mistral_client is not None
            or bool(self.google_api_key)
        )

    def get_langchain_model(self, tools: list[Any] | None = None) -> BaseChatModel:
        """Returns a LangChain-compatible ChatModel for the active provider or failover router."""
        if not self.online:
            raise RuntimeError("LLMClient is offline (no API key configured).")

        groq_chat_model = None
        if self._groq_client is not None:
            groq_chat_model = ChatGroqCustom(client=self._groq_client, model_name=self.groq_model)
            if tools:
                groq_chat_model = groq_chat_model.bind_tools(tools)

        mistral_chat_model = None
        if self.mistral_api_key:
            try:
                from langchain_mistralai import ChatMistralAI
                mistral_chat_model = ChatMistralAI(
                    model=self.mistral_model,
                    api_key=self.mistral_api_key,
                    max_retries=1,
                    timeout=30.0,
                )
                if tools:
                    mistral_chat_model = mistral_chat_model.bind_tools(tools)
            except Exception as exc:
                logger.warning("Failed to construct ChatMistralAI model: %s", exc)
                mistral_chat_model = None

        gemini_chat_model = None
        if self.google_api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                gemini_chat_model = ChatGoogleGenerativeAI(
                    model=self.gemini_model,
                    google_api_key=self.google_api_key,
                    max_retries=1,
                    timeout=30.0,
                )
                if tools:
                    gemini_chat_model = gemini_chat_model.bind_tools(tools)
            except Exception as exc:
                logger.warning("Failed to construct ChatGoogleGenerativeAI model: %s", exc)
                gemini_chat_model = None

        if self.failover_enabled and (
            groq_chat_model is not None
            or mistral_chat_model is not None
            or gemini_chat_model is not None
        ):
            from assistant.llm.provider_router import LLMProviderRouter, ProviderStateTracker
            if self._state_tracker is None:
                self._state_tracker = ProviderStateTracker(
                    cooldown_seconds=self.cooldown_seconds,
                    providers=self.provider_order,
                )
            return LLMProviderRouter(
                groq_model=groq_chat_model,
                mistral_model=mistral_chat_model,
                gemini_model=gemini_chat_model,
                state_tracker=self._state_tracker,
                provider_order=self.provider_order,
                failover_enabled=True,
                max_provider_attempts=self.max_provider_attempts,
                cooldown_seconds=self.cooldown_seconds,
                bound_tools=tools,
            )

        # Failover disabled or single provider mode
        if self.provider == "groq":
            if groq_chat_model is None:
                raise RuntimeError("Groq client is not initialized.")
            return groq_chat_model
        elif self.provider == "mistral":
            if mistral_chat_model is None:
                raise RuntimeError("Mistral client is not initialized.")
            return mistral_chat_model
        elif self.provider == "gemini":
            if gemini_chat_model is None:
                raise RuntimeError("Gemini client is not initialized.")
            return gemini_chat_model
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(
        self, system: str, user_message: str, max_tokens: int = 600
    ) -> str:
        if not self.online:
            return self._offline_fallback(user_message)

        if self.failover_enabled and (
            self._groq_client is not None
            or self._mistral_client is not None
            or bool(self.google_api_key)
        ):
            try:
                model = self.get_langchain_model()
                res = model.invoke([
                    SystemMessage(content=system),
                    HumanMessage(content=user_message),
                ])
                return getattr(res, "content", str(res)) or ""
            except Exception as exc:
                classified = classify_llm_error(exc)
                logger.warning(
                    "[LLM] %s in generate(): %s",
                    classified.category,
                    exc,
                )
                return self._offline_fallback(user_message)

        if self._client is not None:
            try:
                if self.provider == "groq":
                    response = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_message},
                        ],
                    )
                    return response.choices[0].message.content or ""
                elif self.provider == "mistral":
                    response = self._client.chat.complete(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_message},
                        ],
                    )
                    return response.choices[0].message.content or ""
            except Exception as exc:
                classified = classify_llm_error(exc)
                logger.warning(
                    "[LLM] %s in generate() provider=%s model=%s: %s",
                    classified.category,
                    self.provider,
                    self.model,
                    exc,
                )
                return self._offline_fallback(user_message)

        return self._offline_fallback(user_message)

    def _offline_fallback(self, user_message: str) -> str:
        """Deterministic stand-in so demos/tests work with no API key."""
        return (
            f"[offline mode — no {self.provider.upper()}_API_KEY set]\n"
            f"Acknowledged: {user_message.strip()[:200]}"
        )


def get_priority_llm_client() -> LLMClient | None:
    """Return a dedicated LLMClient for the Message Intelligence / Priority Analyzer.

    Strict Requirements:
    - Uses PRIORITY_GROQ_API_KEY as the dedicated key.
    - Uses PRIORITY_GROQ_MODEL as the dedicated model (default: openai/gpt-oss-120b).
    - NEVER silently falls back to the normal conversational GROQ_API_KEY if
      PRIORITY_GROQ_API_KEY is missing.
    - Returns None if PRIORITY_GROQ_API_KEY is missing or empty so the analyzer
      can fail gracefully and defer pending items without consuming normal chat quota.
    """
    key = os.environ.get("PRIORITY_GROQ_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("PRIORITY_GROQ_MODEL", "").strip() or "openai/gpt-oss-120b"
    return LLMClient(provider="groq", api_key=key, model=model)

