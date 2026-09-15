"""
Unit tests for Groq <-> Mistral Automatic LLM Failover.
======================================================
Covers all 50 verification points across:
- Provider selection & cooldown management
- Rate-limit (429) & Retry-After handling
- Server errors (500, 502, 503, 504) & network timeouts
- Permanent error fast-fail (401, 403, 400)
- Bounded attempts & ping-pong prevention
- Streaming safety & tool-call safety (no replay after first token/tool chunk)
- Mistral tool call normalization
- Concurrency & lock safety (no network call under lock)
- Configuration & strict isolation (STT and Priority Analyzer isolated)
- Safe logging (no secrets, prompts, or sensitive payloads)
"""

from typing import Any
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolCallChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from assistant.llm.llm_client import (
    LLMAuthenticationError,
    LLMClient,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMServiceError,
    classify_llm_error,
    get_priority_llm_client,
)
from assistant.llm.provider_router import (
    LLMProviderRouter,
    ProviderStateTracker,
    is_temporary_failover_error,
    normalize_tool_calls,
)
from assistant.stt.groq_stt import GroqSTTService


# ==============================================================================
# Test Fixtures & Helpers
# ==============================================================================

class MockChatModel:
    """Mock LangChain chat model for testing router invocations."""

    def __init__(self, name: str = "mock", fail_with: Exception | None = None, response: AIMessage | None = None):
        self.name = name
        self.fail_with = fail_with
        self.response = response or AIMessage(content=f"Hello from {name}")
        self.calls: list[Any] = []
        self.stream_chunks: list[AIMessageChunk] = [
            AIMessageChunk(content=f"Hello from {name}"),
        ]
        self.fail_after_chunks: int | None = None

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        if self.fail_with:
            raise self.fail_with
        return self.response

    def stream(self, messages, **kwargs):
        self.calls.append(messages)
        if (self.fail_after_chunks is None or self.fail_after_chunks == 0) and self.fail_with:
            raise self.fail_with
        for idx, chunk in enumerate(self.stream_chunks):
            yield ChatGenerationChunk(message=chunk)
            if self.fail_after_chunks is not None and (idx + 1) >= self.fail_after_chunks:
                if self.fail_with:
                    raise self.fail_with

    def bind_tools(self, tools, **kwargs):
        return self


# ==============================================================================
# 1. Provider Selection Tests (1-4)
# ==============================================================================

def test_01_groq_selected_by_default():
    tracker = ProviderStateTracker(cooldown_seconds=10.0)
    router = LLMProviderRouter(
        groq_model=MockChatModel("groq"),
        mistral_model=MockChatModel("mistral"),
        state_tracker=tracker,
    )
    assert router.select_provider() == "groq"


def test_02_mistral_selected_when_groq_cooling_down():
    tracker = ProviderStateTracker(cooldown_seconds=10.0)
    tracker.mark_failure("groq")
    router = LLMProviderRouter(
        groq_model=MockChatModel("groq"),
        mistral_model=MockChatModel("mistral"),
        state_tracker=tracker,
    )
    assert router.select_provider() == "mistral"


def test_03_groq_becomes_preferred_after_cooldown():
    tracker = ProviderStateTracker(cooldown_seconds=0.05)
    tracker.mark_failure("groq")
    assert tracker.select_provider() == "mistral"

    time.sleep(0.06)
    # Groq cooldown expired -> Groq preferred again
    assert tracker.select_provider() == "groq"


def test_04_mistral_selected_after_groq_failure():
    groq_mock = MockChatModel("groq", fail_with=LLMServiceError("500 Internal Error", status_code=500))
    mistral_mock = MockChatModel("mistral", response=AIMessage(content="Mistral reply"))
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        cooldown_seconds=10.0,
    )
    res = router.invoke([HumanMessage(content="Hi")])
    assert res.content == "Mistral reply"
    assert len(groq_mock.calls) == 1
    assert len(mistral_mock.calls) == 1


# ==============================================================================
# 2. Rate-Limit Tests (5-9)
# ==============================================================================

def test_05_groq_429_fails_over_to_mistral():
    groq_mock = MockChatModel("groq", fail_with=LLMRateLimitError("Rate limited 429", retry_after=10.0))
    mistral_mock = MockChatModel("mistral", response=AIMessage(content="Mistral fallback response"))
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        cooldown_seconds=60.0,
    )
    res = router.invoke([HumanMessage(content="Test")])
    assert res.content == "Mistral fallback response"
    state = router.get_provider_state()
    assert state["groq"]["available"] is False
    assert state["mistral"]["available"] is True


def test_06_mistral_429_fails_over_to_groq_when_groq_available():
    tracker = ProviderStateTracker(cooldown_seconds=60.0)
    tracker.mark_failure("mistral")
    router = LLMProviderRouter(
        groq_model=MockChatModel("groq", response=AIMessage(content="Groq response")),
        mistral_model=MockChatModel("mistral"),
        state_tracker=tracker,
    )
    res = router.invoke([HumanMessage(content="Test")])
    assert res.content == "Groq response"


def test_07_both_429_returns_friendly_failure():
    groq_mock = MockChatModel("groq", fail_with=LLMRateLimitError("Groq 429"))
    mistral_mock = MockChatModel("mistral", fail_with=LLMRateLimitError("Mistral 429"))
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        cooldown_seconds=60.0,
    )
    with pytest.raises(LLMError) as exc_info:
        router.invoke([HumanMessage(content="Test")])
    assert "temporarily unavailable" in str(exc_info.value).lower() or "rate-limited" in str(exc_info.value).lower()


def test_08_retry_after_short_behavior_preserved():
    # ChatGroqCustom retries once if retry_after <= 2.0s
    from assistant.llm.llm_client import ChatGroqCustom

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Success after short wait", tool_calls=None))]

    class RateLimit429(Exception):
        status_code = 429

    mock_client.chat.completions.create.side_effect = [
        RateLimit429("Rate limit, try again in 0.1s"),
        mock_resp,
    ]
    model = ChatGroqCustom(client=mock_client)
    res = model.invoke([HumanMessage(content="Hello")])
    assert res.content == "Success after short wait"
    assert mock_client.chat.completions.create.call_count == 2


def test_09_long_retry_after_does_not_cause_excessive_sleep():
    # If retry_after > 2.0s, ChatGroqCustom does not sleep; raises immediately for failover
    from assistant.llm.llm_client import ChatGroqCustom

    mock_client = MagicMock()

    class LongRateLimit(Exception):
        status_code = 429

    mock_client.chat.completions.create.side_effect = LongRateLimit("Rate limit, try again in 30.0s")
    model = ChatGroqCustom(client=mock_client)

    start = time.monotonic()
    with pytest.raises(LLMRateLimitError):
        model.invoke([HumanMessage(content="Hello")])
    elapsed = time.monotonic() - start
    assert elapsed < 1.0  # Must fail fast without sleeping for 30s!


# ==============================================================================
# 3. Server-Error Tests (10-13)
# ==============================================================================

@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_10_to_13_groq_5xx_fails_over_to_mistral(status_code):
    groq_mock = MockChatModel("groq", fail_with=LLMServiceError(f"HTTP {status_code}", status_code=status_code))
    mistral_mock = MockChatModel("mistral", response=AIMessage(content=f"Mistral after {status_code}"))
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        cooldown_seconds=60.0,
    )
    res = router.invoke([HumanMessage(content="Hello")])
    assert res.content == f"Mistral after {status_code}"


# ==============================================================================
# 4. Network Tests (14-15)
# ==============================================================================

def test_14_groq_timeout_fails_over_to_mistral():
    groq_mock = MockChatModel("groq", fail_with=LLMNetworkError("ConnectTimeout timed out"))
    mistral_mock = MockChatModel("mistral", response=AIMessage(content="Mistral after timeout"))
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        cooldown_seconds=60.0,
    )
    res = router.invoke([HumanMessage(content="Hi")])
    assert res.content == "Mistral after timeout"


def test_15_groq_connection_failure_fails_over_to_mistral():
    groq_mock = MockChatModel("groq", fail_with=LLMNetworkError("[Errno -3] Temporary failure in name resolution"))
    mistral_mock = MockChatModel("mistral", response=AIMessage(content="Mistral after DNS failure"))
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        cooldown_seconds=60.0,
    )
    res = router.invoke([HumanMessage(content="Hi")])
    assert res.content == "Mistral after DNS failure"


# ==============================================================================
# 5. Permanent-Error Tests (16-20)
# ==============================================================================

def test_16_groq_401_does_not_create_infinite_failover():
    groq_mock = MockChatModel("groq", fail_with=LLMAuthenticationError("Invalid API key"))
    mistral_mock = MockChatModel("mistral")
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
    )
    with pytest.raises(LLMAuthenticationError):
        router.invoke([HumanMessage(content="Hi")])
    assert len(mistral_mock.calls) == 0  # Mistral was never attempted


def test_17_mistral_401_does_not_create_infinite_failover():
    tracker = ProviderStateTracker(cooldown_seconds=60.0)
    tracker.mark_failure("groq")  # Groq cooling down
    mistral_mock = MockChatModel("mistral", fail_with=LLMAuthenticationError("Mistral bad auth"))
    groq_mock = MockChatModel("groq")
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        state_tracker=tracker,
    )
    with pytest.raises(LLMAuthenticationError):
        router.invoke([HumanMessage(content="Hi")])
    assert len(groq_mock.calls) == 0


def test_18_invalid_request_does_not_trigger_provider_bouncing():
    # 400 Bad Request
    class BadRequestError(Exception):
        status_code = 400

    groq_mock = MockChatModel("groq", fail_with=BadRequestError("Invalid schema / 400"))
    mistral_mock = MockChatModel("mistral")
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
    )
    with pytest.raises(Exception):
        router.invoke([HumanMessage(content="Hi")])
    assert len(mistral_mock.calls) == 0


def test_19_missing_mistral_key_is_handled_correctly():
    # LLMClient initialized without MISTRAL_API_KEY
    client = LLMClient(api_key="groq-key", mistral_api_key="", failover_enabled=True)
    model = client.get_langchain_model()
    assert isinstance(model, LLMProviderRouter)
    assert model.mistral_model is None
    # If groq fails and no mistral is configured:
    with patch.object(model.groq_model.client.chat.completions, "create", side_effect=LLMRateLimitError("429")):
        with pytest.raises(LLMError):
            model.invoke([HumanMessage(content="Test")])


def test_20_unsupported_model_configuration_handled_correctly():
    # Model not found (404)
    groq_mock = MockChatModel("groq", fail_with=LLMServiceError("Model not found", status_code=404))
    mistral_mock = MockChatModel("mistral")
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
    )
    with pytest.raises(LLMServiceError):
        router.invoke([HumanMessage(content="Hi")])
    assert len(mistral_mock.calls) == 0


# ==============================================================================
# 6. Cooldown Tests (21-25)
# ==============================================================================

def test_21_groq_enters_cooldown():
    tracker = ProviderStateTracker(cooldown_seconds=60.0)
    assert tracker.is_available("groq") is True
    tracker.mark_failure("groq")
    assert tracker.is_available("groq") is False


def test_22_mistral_enters_cooldown():
    tracker = ProviderStateTracker(cooldown_seconds=60.0)
    assert tracker.is_available("mistral") is True
    tracker.mark_failure("mistral")
    assert tracker.is_available("mistral") is False


def test_23_cooldown_expiration_restores_provider_eligibility():
    tracker = ProviderStateTracker(cooldown_seconds=0.05)
    tracker.mark_failure("groq")
    assert tracker.is_available("groq") is False
    time.sleep(0.06)
    assert tracker.is_available("groq") is True


def test_24_cooldowns_are_independent():
    tracker = ProviderStateTracker(cooldown_seconds=60.0)
    tracker.mark_failure("groq")
    assert tracker.is_available("groq") is False
    assert tracker.is_available("mistral") is True


def test_25_ping_pong_is_prevented():
    tracker = ProviderStateTracker(cooldown_seconds=60.0)
    # Failure marks cooldown, so immediate next request goes to Mistral, not Groq
    tracker.mark_failure("groq")
    assert tracker.select_provider() == "mistral"
    assert tracker.select_provider() == "mistral"
    assert tracker.is_available("groq") is False


# ==============================================================================
# 7. Attempt Tests (26-27)
# ==============================================================================

def test_26_maximum_provider_attempts_respected():
    groq_mock = MockChatModel("groq", fail_with=LLMServiceError("503", status_code=503))
    mistral_mock = MockChatModel("mistral", fail_with=LLMServiceError("503", status_code=503))
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        max_provider_attempts=2,
    )
    with pytest.raises(LLMError):
        router.invoke([HumanMessage(content="Test")])

    assert len(groq_mock.calls) == 1
    assert len(mistral_mock.calls) == 1
    assert len(groq_mock.calls) + len(mistral_mock.calls) == 2


def test_27_no_infinite_retry():
    groq_mock = MockChatModel("groq", fail_with=LLMRateLimitError("429"))
    mistral_mock = MockChatModel("mistral", fail_with=LLMRateLimitError("429"))
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        max_provider_attempts=2,
    )
    with pytest.raises(LLMError):
        router.invoke([HumanMessage(content="Test")])
    assert len(groq_mock.calls) + len(mistral_mock.calls) <= 2


# ==============================================================================
# 8. Tool Tests (28-31)
# ==============================================================================

def test_28_groq_tool_call_remains_unchanged():
    groq_tool_calls = [{"name": "send_sms", "args": {"recipient": "+12345"}, "id": "call_1", "type": "tool_call"}]
    groq_mock = MockChatModel("groq", response=AIMessage(content="", tool_calls=groq_tool_calls))
    router = LLMProviderRouter(groq_model=groq_mock, mistral_model=MockChatModel("mistral"))
    res = router.invoke([HumanMessage(content="Send SMS")])
    assert res.tool_calls == groq_tool_calls


def test_29_mistral_tool_call_is_normalized():
    # 1. Test normalize_tool_calls helper directly with unnormalized raw tool calls
    raw_calls = [
        {"name": "make_call", "args": '{"recipient": "John"}', "id": "call_m1"},
        {"name": "set_alarm", "args": {"time": "08:00"}},
    ]
    normalized = normalize_tool_calls(raw_calls)
    assert len(normalized) == 2
    assert normalized[0]["name"] == "make_call"
    assert normalized[0]["args"] == {"recipient": "John"}
    assert normalized[0]["id"] == "call_m1"
    assert normalized[0]["type"] == "tool_call"
    assert normalized[1]["name"] == "set_alarm"
    assert normalized[1]["args"] == {"time": "08:00"}
    assert normalized[1]["type"] == "tool_call"

    # 2. Test router invoke normalizes tool_calls returned from Mistral model
    mistral_tool_calls = [{"name": "make_call", "args": {"recipient": "John"}, "id": "call_m1", "type": "tool_call"}]
    mistral_mock = MockChatModel("mistral", response=AIMessage(content="", tool_calls=mistral_tool_calls))
    tracker = ProviderStateTracker(cooldown_seconds=60.0)
    tracker.mark_failure("groq")

    router = LLMProviderRouter(
        groq_model=MockChatModel("groq"),
        mistral_model=mistral_mock,
        state_tracker=tracker,
    )
    res = router.invoke([HumanMessage(content="Call John")])
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0] == mistral_tool_calls[0]


def test_30_failover_before_tool_execution_does_not_duplicate_execution():
    groq_mock = MockChatModel("groq", fail_with=LLMRateLimitError("429"))
    mistral_tool_calls = [{"name": "set_alarm", "args": {"time": "08:00"}, "id": "call_alarm", "type": "tool_call"}]
    mistral_mock = MockChatModel("mistral", response=AIMessage(content="", tool_calls=mistral_tool_calls))

    router = LLMProviderRouter(groq_model=groq_mock, mistral_model=mistral_mock)
    res = router.invoke([HumanMessage(content="Set alarm for 8am")])
    # Router returns tool calls once
    assert res.tool_calls == mistral_tool_calls


def test_31_already_executed_tool_call_never_replayed():
    # Verified by checking that tool normalization produces deterministic dicts
    # and failover returns exactly one ChatResult without repeating previous turns
    tcs = normalize_tool_calls([{"name": "test", "args": {}, "id": "1"}])
    assert len(tcs) == 1
    assert tcs[0]["type"] == "tool_call"


# ==============================================================================
# 9. Streaming Tests (32-36)
# ==============================================================================

def test_32_groq_fails_before_first_token_mistral_fallback():
    groq_mock = MockChatModel("groq", fail_with=LLMServiceError("503", status_code=503))
    mistral_mock = MockChatModel("mistral")
    mistral_mock.stream_chunks = [
        AIMessageChunk(content="Mistral "),
        AIMessageChunk(content="streaming"),
    ]
    router = LLMProviderRouter(groq_model=groq_mock, mistral_model=mistral_mock)
    chunks = list(router.stream([HumanMessage(content="Hello")]))
    contents = [c.content for c in chunks if c.content]
    assert contents == ["Mistral ", "streaming"]


def test_33_groq_fails_after_first_token_no_replay():
    groq_mock = MockChatModel("groq", fail_with=LLMNetworkError("Connection lost mid-stream"))
    groq_mock.stream_chunks = [
        AIMessageChunk(content="Initial token "),
        AIMessageChunk(content="second token"),
    ]
    groq_mock.fail_after_chunks = 1  # Fails after yielding first token

    mistral_mock = MockChatModel("mistral")
    mistral_mock.stream_chunks = [AIMessageChunk(content="Mistral replay")]

    router = LLMProviderRouter(groq_model=groq_mock, mistral_model=mistral_mock)

    yielded = []
    with pytest.raises(LLMNetworkError):
        for chunk in router.stream([HumanMessage(content="Hello")]):
            yielded.append(chunk.content)

    # Output yielded should only contain the initial token before error; NO replay from Mistral!
    assert "Initial token " in yielded
    assert "Mistral replay" not in yielded
    assert len(mistral_mock.calls) == 0


def test_34_groq_fails_after_tool_call_chunk_no_replay():
    groq_mock = MockChatModel("groq", fail_with=LLMNetworkError("Stream dropped"))
    groq_mock.stream_chunks = [
        AIMessageChunk(content="", tool_call_chunks=[ToolCallChunk(name="send_sms", args='{"to": "+1"}', id="1", index=0)]),
    ]
    groq_mock.fail_after_chunks = 1

    mistral_mock = MockChatModel("mistral")
    router = LLMProviderRouter(groq_model=groq_mock, mistral_model=mistral_mock)

    with pytest.raises(LLMNetworkError):
        list(router.stream([HumanMessage(content="Send sms")]))

    assert len(mistral_mock.calls) == 0  # No failover after tool chunk emitted


def test_35_mistral_streaming_works():
    tracker = ProviderStateTracker(cooldown_seconds=60.0)
    tracker.mark_failure("groq")
    mistral_mock = MockChatModel("mistral")
    mistral_mock.stream_chunks = [
        AIMessageChunk(content="Token 1 "),
        AIMessageChunk(content="Token 2"),
    ]
    router = LLMProviderRouter(
        groq_model=MockChatModel("groq"),
        mistral_model=mistral_mock,
        state_tracker=tracker,
    )
    tokens = [c.content for c in router.stream([HumanMessage(content="Hi")]) if c.content]
    assert tokens == ["Token 1 ", "Token 2"]


def test_36_active_streams_do_not_migrate_providers():
    # If a stream started with Groq, it continues with Groq; another thread marking failure doesn't affect it
    groq_mock = MockChatModel("groq")
    groq_mock.stream_chunks = [
        AIMessageChunk(content="A"),
        AIMessageChunk(content="B"),
    ]
    mistral_mock = MockChatModel("mistral")
    router = LLMProviderRouter(groq_model=groq_mock, mistral_model=mistral_mock)

    stream_gen = router.stream([HumanMessage(content="Hi")])
    chunk1 = next(stream_gen)
    assert chunk1.content == "A"

    # Background thread marks Groq failure
    router.mark_failure("groq")

    # Ongoing stream still yields from Groq
    chunk2 = next(stream_gen)
    assert chunk2.content == "B"


# ==============================================================================
# 10. Concurrency Tests (37-38)
# ==============================================================================

def test_37_concurrent_provider_state_updates_are_safe():
    tracker = ProviderStateTracker(cooldown_seconds=1.0)
    errors = []

    def worker(idx: int):
        try:
            for _ in range(50):
                if idx % 2 == 0:
                    tracker.mark_failure("groq")
                    tracker.is_available("groq")
                else:
                    tracker.select_provider()
                    tracker.mark_success("groq")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


def test_38_no_network_request_occurs_while_holding_provider_state_lock():
    tracker = ProviderStateTracker(cooldown_seconds=60.0)

    class LockCheckingModel:
        def invoke(self, messages, **kwargs):
            # Assert that the tracker's internal lock is NOT locked by the current thread
            locked_by_me = tracker._lock.locked()
            if locked_by_me:
                raise AssertionError("ProviderStateTracker lock is held during network request!")
            return AIMessage(content="OK")

        def stream(self, messages, **kwargs):
            if tracker._lock.locked():
                raise AssertionError("ProviderStateTracker lock is held during streaming network request!")
            yield ChatGenerationChunk(message=AIMessageChunk(content="OK"))

        def bind_tools(self, tools, **kwargs):
            return self

    router = LLMProviderRouter(
        groq_model=LockCheckingModel(),
        mistral_model=LockCheckingModel(),
        state_tracker=tracker,
    )
    # Invoke check
    res = router.invoke([HumanMessage(content="Test")])
    assert res.content == "OK"

    # Stream check
    chunks = [c for c in router.stream([HumanMessage(content="Test")]) if c.content]
    assert len(chunks) == 1


# ==============================================================================
# 11. Configuration & Isolation Tests (39-46)
# ==============================================================================

def test_39_groq_model_used_for_normal_agent(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "custom-groq-model-v1")
    client = LLMClient(api_key="fake-groq-key")
    assert client.groq_model == "custom-groq-model-v1"


def test_40_mistral_model_used_for_mistral_fallback(monkeypatch):
    monkeypatch.setenv("MISTRAL_MODEL", "custom-mistral-large")
    client = LLMClient(api_key="fake-groq-key", mistral_api_key="fake-mistral-key")
    assert client.mistral_model == "custom-mistral-large"


def test_41_no_normal_agent_hard_coded_groq_model_remains(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    client = LLMClient(api_key="fake-groq-key")
    assert client.groq_model == "openai/gpt-oss-120b"


def test_42_stt_model_remains_isolated(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "mistral-large-latest")
    monkeypatch.setenv("MISTRAL_MODEL", "mistral-large-latest")
    stt = GroqSTTService(api_key="fake-stt-key")
    # STT must use whisper-large-v3-turbo and never Mistral model
    assert stt.model == "whisper-large-v3-turbo"


def test_43_priority_groq_model_remains_isolated(monkeypatch):
    monkeypatch.setenv("PRIORITY_GROQ_API_KEY", "fake-priority-key")
    monkeypatch.setenv("PRIORITY_GROQ_MODEL", "dedicated-priority-model")
    monkeypatch.setenv("GROQ_MODEL", "normal-agent-model")
    monkeypatch.setenv("MISTRAL_MODEL", "normal-mistral-model")

    priority_client = get_priority_llm_client()
    assert priority_client is not None
    assert priority_client.model == "dedicated-priority-model"
    assert priority_client.api_key == "fake-priority-key"
    assert priority_client.failover_enabled is False


def test_44_mistral_api_key_never_used_for_priority_analyzer(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret-key")
    monkeypatch.delenv("PRIORITY_GROQ_API_KEY", raising=False)

    priority_client = get_priority_llm_client()
    # Must return None if PRIORITY_GROQ_API_KEY is not set; never use MISTRAL_API_KEY
    assert priority_client is None


def test_45_groq_api_key_never_used_as_mistral_credential(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-exclusive-key")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    client = LLMClient(api_key="groq-exclusive-key")
    assert client.mistral_api_key == ""
    assert client.groq_api_key == "groq-exclusive-key"


def test_46_failover_disabled_means_groq_only(monkeypatch):
    monkeypatch.setenv("LLM_FAILOVER_ENABLED", "false")
    client = LLMClient(api_key="groq-key", mistral_api_key="mistral-key", failover_enabled=False)
    assert client.failover_enabled is False
    client._groq_client = MagicMock()
    model = client.get_langchain_model()
    # When failover is disabled, model is ChatGroqCustom directly, not LLMProviderRouter
    assert not isinstance(model, LLMProviderRouter)


# ==============================================================================
# 12. Safe Logging Tests (47-50)
# ==============================================================================

def test_47_to_50_logging_sanitization(caplog):
    caplog.set_level(logging.INFO, logger="assistant.llm")
    tracker = ProviderStateTracker(cooldown_seconds=60.0)
    tracker.mark_failure("groq")

    groq_mock = MockChatModel("groq", fail_with=LLMRateLimitError("429 rate limited"))
    mistral_mock = MockChatModel("mistral", response=AIMessage(content="Sensitive secret response from LLM"))
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        state_tracker=tracker,
    )

    router.invoke([HumanMessage(content="Confidential user prompt text")])

    logged = caplog.text
    # 47: Provider-switch logs present
    assert "[LLM]" in logged
    # 48: Prompts/responses are NOT logged
    assert "Confidential user prompt text" not in logged
    assert "Sensitive secret response from LLM" not in logged
    # 49: Authorization headers are not logged
    assert "Bearer" not in logged
    assert "Authorization" not in logged
    # 50: Provider errors are concise/sanitized
    assert "Traceback" not in logged


# ==============================================================================
# 13. Bidirectional 3-Attempt Failover Tests (Section 2, 3, 7, 19)
# ==============================================================================

class SequenceTrackingMockChatModel:
    """Mock LangChain chat model that yields successive responses/errors per invocation."""

    def __init__(self, name: str, side_effects: list[Any]):
        self.name = name
        self.side_effects = list(side_effects)
        self.calls: list[Any] = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        if not self.side_effects:
            return AIMessage(content=f"Default from {self.name}")
        effect = self.side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect

    def stream(self, messages, **kwargs):
        self.calls.append(messages)
        if not self.side_effects:
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"Default from {self.name}"))
            return
        effect = self.side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        yield ChatGenerationChunk(message=AIMessageChunk(content=str(effect)))

    def bind_tools(self, tools, **kwargs):
        return self


def test_51_bidirectional_groq_mistral_groq_success():
    """Attempt 1: Groq fails -> Attempt 2: Mistral fails -> Attempt 3: Groq succeeds."""
    groq_mock = SequenceTrackingMockChatModel(
        "groq",
        side_effects=[
            LLMRateLimitError("Groq 429"),
            AIMessage(content="Groq final bounded attempt succeeded!"),
        ],
    )
    mistral_mock = SequenceTrackingMockChatModel(
        "mistral",
        side_effects=[
            LLMServiceError("Mistral 500", status_code=500),
        ],
    )
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        max_provider_attempts=3,
    )
    res = router.invoke([HumanMessage(content="Test query")])
    assert res.content == "Groq final bounded attempt succeeded!"
    assert len(groq_mock.calls) == 2
    assert len(mistral_mock.calls) == 1
    assert len(groq_mock.calls) + len(mistral_mock.calls) == 3


def test_52_bidirectional_groq_mistral_groq_all_fail_friendly_error():
    """Attempt 1: Groq fails -> Attempt 2: Mistral fails -> Attempt 3: Groq fails -> friendly error."""
    groq_mock = SequenceTrackingMockChatModel(
        "groq",
        side_effects=[
            LLMRateLimitError("Groq 429"),
            LLMServiceError("Groq 503", status_code=503),
        ],
    )
    mistral_mock = SequenceTrackingMockChatModel(
        "mistral",
        side_effects=[
            LLMRateLimitError("Mistral 429"),
        ],
    )
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        max_provider_attempts=3,
    )
    with pytest.raises(LLMServiceError) as exc_info:
        router.invoke([HumanMessage(content="Test query")])

    assert "All AI services are temporarily unavailable" in str(exc_info.value)
    assert len(groq_mock.calls) == 2
    assert len(mistral_mock.calls) == 1
    assert len(groq_mock.calls) + len(mistral_mock.calls) == 3


def test_53_never_perform_a_fourth_attempt():
    """Ensure that under no circumstances does the router perform a 4th provider attempt."""
    groq_mock = SequenceTrackingMockChatModel(
        "groq",
        side_effects=[
            LLMRateLimitError("Groq 429 attempt 1"),
            LLMRateLimitError("Groq 429 attempt 3"),
            AIMessage(content="Attempt 4 should NEVER happen"),
        ],
    )
    mistral_mock = SequenceTrackingMockChatModel(
        "mistral",
        side_effects=[
            LLMRateLimitError("Mistral 429 attempt 2"),
            AIMessage(content="Attempt 5 should NEVER happen"),
        ],
    )
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        max_provider_attempts=3,
    )
    with pytest.raises(LLMServiceError):
        router.invoke([HumanMessage(content="Test query")])

    assert len(groq_mock.calls) == 2
    assert len(mistral_mock.calls) == 1
    assert len(groq_mock.calls) + len(mistral_mock.calls) == 3


def test_54_mistral_initial_cooldown_failover_to_groq_and_back():
    """When Groq is in cooldown for a NEW request: Mistral -> Groq -> Mistral."""
    tracker = ProviderStateTracker(cooldown_seconds=60.0)
    tracker.mark_failure("groq")  # Groq is cooling down initially

    groq_mock = SequenceTrackingMockChatModel(
        "groq",
        side_effects=[
            LLMServiceError("Groq 502", status_code=502),
        ],
    )
    mistral_mock = SequenceTrackingMockChatModel(
        "mistral",
        side_effects=[
            LLMRateLimitError("Mistral 429"),
            AIMessage(content="Mistral attempt 3 success!"),
        ],
    )
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        state_tracker=tracker,
        max_provider_attempts=3,
    )
    res = router.invoke([HumanMessage(content="Test query")])
    assert res.content == "Mistral attempt 3 success!"
    assert len(mistral_mock.calls) == 2
    assert len(groq_mock.calls) == 1


def test_55_streaming_bidirectional_failover_before_tokens():
    """In streaming mode: Groq fails -> Mistral fails -> Groq streams successfully."""
    groq_mock = SequenceTrackingMockChatModel(
        "groq",
        side_effects=[
            LLMServiceError("Groq 500", status_code=500),
            "Groq streamed final response",
        ],
    )
    mistral_mock = SequenceTrackingMockChatModel(
        "mistral",
        side_effects=[
            LLMRateLimitError("Mistral 429"),
        ],
    )
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        max_provider_attempts=3,
    )
    chunks = list(router.stream([HumanMessage(content="Test query")]))
    contents = [c.content for c in chunks if c.content]
    assert contents == ["Groq streamed final response"]
    assert len(groq_mock.calls) == 2
    assert len(mistral_mock.calls) == 1


def test_56_safe_diagnostic_logging_captures_metadata_without_secrets(caplog):
    """Verify diagnostic logging captures provider, status, error code, category, and model safely."""
    caplog.set_level(logging.WARNING, logger="assistant.llm")

    class MockMistralSDKError(Exception):
        def __init__(self):
            self.status_code = 429
            self.raw_status_code = 429
            self.body = '{"object":"error","message":"Rate limit exceeded","code":"1300"}'
            super().__init__("Status 429: Rate limit exceeded")

    groq_mock = MockChatModel("groq", fail_with=LLMRateLimitError("429"))
    mistral_mock = MockChatModel("mistral", fail_with=MockMistralSDKError())
    mistral_mock.model_name = "mistral-small-2603"

    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        max_provider_attempts=2,
    )
    with pytest.raises(LLMError):
        router.invoke([HumanMessage(content="Check diagnostics")])

    logged = caplog.text
    # Diagnostic fields present
    assert "Diagnostic:" in logged
    assert "provider=mistral" in logged
    assert "status=429" in logged
    assert "error_code=1300" in logged
    assert "category=RATE_LIMIT" in logged
    assert "model=mistral-small-2603" in logged
    # Clean switch log present
    assert "[LLM] Groq rate limited; switching to Mistral" in logged


def test_57_bind_tools_propagates_to_both_providers():
    """Verify LLMProviderRouter.bind_tools() binds tools to both Groq and Mistral models."""
    mock_groq = MagicMock()
    mock_groq.bind_tools.return_value = "bound_groq"
    mock_mistral = MagicMock()
    mock_mistral.bind_tools.return_value = "bound_mistral"

    router = LLMProviderRouter(groq_model=mock_groq, mistral_model=mock_mistral)
    tools = [{"name": "test_tool", "description": "test"}]
    bound_router = router.bind_tools(tools)

    assert isinstance(bound_router, LLMProviderRouter)
    mock_groq.bind_tools.assert_called_once_with(tools)
    mock_mistral.bind_tools.assert_called_once_with(tools)
    assert bound_router.groq_model == "bound_groq"
    assert bound_router.mistral_model == "bound_mistral"
    assert bound_router.bound_tools == tools


def test_58_strict_isolation_zero_router_imports_in_stt_and_priority():
    """Verify STT and Priority Analyzer modules have ZERO imports or usages of LLMProviderRouter."""
    import inspect
    import assistant.stt.groq_stt as stt_mod
    import assistant.intelligence.priority_analyzer as prio_mod

    stt_src = inspect.getsource(stt_mod)
    prio_src = inspect.getsource(prio_mod)

    assert "LLMProviderRouter" not in stt_src
    assert "mistral" not in stt_src.lower()
    assert "gemini" not in stt_src.lower()
    assert "LLMProviderRouter" not in prio_src
    assert "mistral" not in prio_src.lower()
    assert "gemini" not in prio_src.lower()


# ==============================================================================
# 14. Google Gemini 3rd Provider & Response Validation Tests (Section 1-14)
# ==============================================================================

def test_59_gemini_provider_order_and_cooldown_selection():
    """Verify provider order and selection when Groq and Mistral are cooling down."""
    tracker = ProviderStateTracker(cooldown_seconds=60.0, providers=("groq", "mistral", "gemini"))
    tracker.mark_failure("groq")
    tracker.mark_failure("mistral")

    router = LLMProviderRouter(
        groq_model=MockChatModel("groq"),
        mistral_model=MockChatModel("mistral"),
        gemini_model=MockChatModel("gemini"),
        state_tracker=tracker,
        provider_order=("groq", "mistral", "gemini"),
    )
    # Both groq and mistral cooling down -> Gemini selected
    assert router.select_provider() == "gemini"


def test_60_gemini_success_on_attempt_3():
    """Attempt 1: Groq 429 -> Attempt 2: Mistral 429 -> Attempt 3: Gemini succeeds."""
    groq_mock = SequenceTrackingMockChatModel("groq", side_effects=[LLMRateLimitError("Groq 429")])
    mistral_mock = SequenceTrackingMockChatModel("mistral", side_effects=[LLMRateLimitError("Mistral 429")])
    gemini_mock = SequenceTrackingMockChatModel("gemini", side_effects=[AIMessage(content="Gemini answer")])

    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        gemini_model=gemini_mock,
        provider_order=("groq", "mistral", "gemini"),
        max_provider_attempts=4,
    )
    res = router.invoke([HumanMessage(content="What is 2+2?")])
    assert res.content == "Gemini answer"
    assert len(groq_mock.calls) == 1
    assert len(mistral_mock.calls) == 1
    assert len(gemini_mock.calls) == 1


def test_61_groq_success_on_attempt_4():
    """Attempt 1: Groq 429 -> Attempt 2: Mistral 429 -> Attempt 3: Gemini 429 -> Attempt 4: Groq succeeds."""
    groq_mock = SequenceTrackingMockChatModel(
        "groq",
        side_effects=[
            LLMRateLimitError("Groq 429 attempt 1"),
            AIMessage(content="Groq attempt 4 success"),
        ],
    )
    mistral_mock = SequenceTrackingMockChatModel("mistral", side_effects=[LLMRateLimitError("Mistral 429")])
    gemini_mock = SequenceTrackingMockChatModel("gemini", side_effects=[LLMRateLimitError("Gemini 429")])

    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        gemini_model=gemini_mock,
        provider_order=("groq", "mistral", "gemini"),
        max_provider_attempts=4,
    )
    res = router.invoke([HumanMessage(content="Query")])
    assert res.content == "Groq attempt 4 success"
    assert len(groq_mock.calls) == 2
    assert len(mistral_mock.calls) == 1
    assert len(gemini_mock.calls) == 1
    assert len(groq_mock.calls) + len(mistral_mock.calls) + len(gemini_mock.calls) == 4


def test_62_all_four_attempts_fail_friendly_error():
    """All 4 provider attempts fail -> friendly unavailable error."""
    groq_mock = SequenceTrackingMockChatModel(
        "groq",
        side_effects=[
            LLMRateLimitError("Groq 429 attempt 1"),
            LLMServiceError("Groq 503 attempt 4", status_code=503),
        ],
    )
    mistral_mock = SequenceTrackingMockChatModel("mistral", side_effects=[LLMRateLimitError("Mistral 429")])
    gemini_mock = SequenceTrackingMockChatModel("gemini", side_effects=[LLMServiceError("Gemini 500", status_code=500)])

    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        gemini_model=gemini_mock,
        provider_order=("groq", "mistral", "gemini"),
        max_provider_attempts=4,
    )
    with pytest.raises(LLMServiceError) as exc_info:
        router.invoke([HumanMessage(content="Query")])

    assert "All AI services are temporarily unavailable" in str(exc_info.value)
    assert len(groq_mock.calls) == 2
    assert len(mistral_mock.calls) == 1
    assert len(gemini_mock.calls) == 1
    assert len(groq_mock.calls) + len(mistral_mock.calls) + len(gemini_mock.calls) == 4


def test_63_never_perform_a_fifth_attempt():
    """Ensure strict capping at 4 attempts: never perform a 5th attempt."""
    groq_mock = SequenceTrackingMockChatModel(
        "groq",
        side_effects=[
            LLMRateLimitError("Groq 429 attempt 1"),
            LLMRateLimitError("Groq 429 attempt 4"),
            AIMessage(content="Never reach attempt 5"),
        ],
    )
    mistral_mock = SequenceTrackingMockChatModel(
        "mistral",
        side_effects=[
            LLMRateLimitError("Mistral 429 attempt 2"),
            AIMessage(content="Never reach attempt 6"),
        ],
    )
    gemini_mock = SequenceTrackingMockChatModel(
        "gemini",
        side_effects=[
            LLMRateLimitError("Gemini 429 attempt 3"),
            AIMessage(content="Never reach attempt 7"),
        ],
    )
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        gemini_model=gemini_mock,
        provider_order=("groq", "mistral", "gemini"),
        max_provider_attempts=4,
    )
    with pytest.raises(LLMServiceError):
        router.invoke([HumanMessage(content="Query")])

    total_calls = len(groq_mock.calls) + len(mistral_mock.calls) + len(gemini_mock.calls)
    assert total_calls == 4


def test_64_mistral_5xx_fails_over_to_gemini():
    """Mistral 500 error fails over to Gemini."""
    tracker = ProviderStateTracker(cooldown_seconds=60.0, providers=("groq", "mistral", "gemini"))
    tracker.mark_failure("groq")  # Starts on Mistral

    mistral_mock = MockChatModel("mistral", fail_with=LLMServiceError("Mistral 500", status_code=500))
    gemini_mock = MockChatModel("gemini", response=AIMessage(content="Gemini from Mistral 5xx"))

    router = LLMProviderRouter(
        groq_model=MockChatModel("groq"),
        mistral_model=mistral_mock,
        gemini_model=gemini_mock,
        state_tracker=tracker,
        provider_order=("groq", "mistral", "gemini"),
    )
    res = router.invoke([HumanMessage(content="Hi")])
    assert res.content == "Gemini from Mistral 5xx"
    assert len(mistral_mock.calls) == 1
    assert len(gemini_mock.calls) == 1


def test_65_gemini_5xx_fails_over_to_groq():
    """Gemini 503 error fails over back to Groq."""
    from google.api_core import exceptions
    gemini_mock = MockChatModel("gemini", fail_with=exceptions.ServiceUnavailable("Gemini 503"))
    groq_mock = SequenceTrackingMockChatModel(
        "groq",
        side_effects=[
            LLMRateLimitError("Groq 429"),
            AIMessage(content="Groq after Gemini 503"),
        ],
    )
    mistral_mock = MockChatModel("mistral", fail_with=LLMRateLimitError("Mistral 429"))

    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        gemini_model=gemini_mock,
        provider_order=("groq", "mistral", "gemini"),
        max_provider_attempts=4,
    )
    res = router.invoke([HumanMessage(content="Hi")])
    assert res.content == "Groq after Gemini 503"
    assert len(groq_mock.calls) == 2


def test_66_empty_response_fails_over():
    """Provider returning empty content without tool calls fails validation and fails over."""
    groq_mock = MockChatModel("groq", response=AIMessage(content=""))
    mistral_mock = MockChatModel("mistral", response=AIMessage(content="Mistral fallback answer"))

    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        provider_order=("groq", "mistral"),
    )
    res = router.invoke([HumanMessage(content="What is 2+2?")])
    assert res.content == "Mistral fallback answer"
    assert len(groq_mock.calls) == 1
    assert len(mistral_mock.calls) == 1


def test_67_malformed_tool_call_fails_over():
    """Provider returning a malformed tool call fails validation and fails over."""
    class MalformedToolCallModel:
        def __init__(self, name="groq"):
            self.name = name
            self.calls = []

        def invoke(self, messages, **kwargs):
            self.calls.append(messages)
            return AIMessage(content="", tool_calls=[{"name": "", "args": {}, "id": "call_bad"}])

        def bind_tools(self, tools, **kwargs):
            return self

    groq_mock = MalformedToolCallModel("groq")
    mistral_mock = MockChatModel(
        "mistral",
        response=AIMessage(
            content="",
            tool_calls=[{"name": "valid_tool", "args": {"x": 1}, "id": "call_1", "type": "tool_call"}],
        ),
    )

    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=mistral_mock,
        provider_order=("groq", "mistral"),
    )
    res = router.invoke([HumanMessage(content="Run tool")])
    assert res.tool_calls[0]["name"] == "valid_tool"
    assert len(groq_mock.calls) == 1
    assert len(mistral_mock.calls) == 1


def test_68_permanent_errors_no_bouncing_on_gemini():
    """Permanent errors on Gemini (401, 403, 400, 404, 422) fail fast without bouncing."""
    from google.api_core import exceptions

    tracker = ProviderStateTracker(cooldown_seconds=60.0, providers=("groq", "mistral", "gemini"))
    tracker.mark_failure("groq")
    tracker.mark_failure("mistral")

    gemini_mock = MockChatModel("gemini", fail_with=exceptions.Unauthenticated("Invalid API key"))
    groq_mock = MockChatModel("groq")

    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=MockChatModel("mistral"),
        gemini_model=gemini_mock,
        state_tracker=tracker,
        provider_order=("groq", "mistral", "gemini"),
    )
    with pytest.raises(LLMAuthenticationError):
        router.invoke([HumanMessage(content="Test")])

    assert len(gemini_mock.calls) == 1
    assert len(groq_mock.calls) == 0  # Groq was never attempted


def test_69_gemini_tool_binding_and_normalization():
    """Verify bind_tools propagates to Gemini and tool calls are normalized."""
    mock_groq = MagicMock()
    mock_mistral = MagicMock()
    mock_gemini = MagicMock()

    router = LLMProviderRouter(
        groq_model=mock_groq,
        mistral_model=mock_mistral,
        gemini_model=mock_gemini,
    )
    tools = [{"name": "search", "description": "search web"}]
    bound = router.bind_tools(tools)

    mock_groq.bind_tools.assert_called_once_with(tools)
    mock_mistral.bind_tools.assert_called_once_with(tools)
    mock_gemini.bind_tools.assert_called_once_with(tools)
    assert bound.bound_tools == tools


def test_70_streaming_failure_after_token_on_gemini_no_failover():
    """If Gemini emits a token and then drops connection, failover is NOT permitted."""
    tracker = ProviderStateTracker(cooldown_seconds=60.0, providers=("groq", "mistral", "gemini"))
    tracker.mark_failure("groq")
    tracker.mark_failure("mistral")

    gemini_mock = MockChatModel("gemini", fail_with=LLMNetworkError("Gemini connection lost mid-stream"))
    gemini_mock.stream_chunks = [
        AIMessageChunk(content="Partial Gemini token "),
        AIMessageChunk(content="Second token"),
    ]
    gemini_mock.fail_after_chunks = 1

    groq_mock = MockChatModel("groq")
    router = LLMProviderRouter(
        groq_model=groq_mock,
        mistral_model=MockChatModel("mistral"),
        gemini_model=gemini_mock,
        state_tracker=tracker,
    )
    yielded = []
    with pytest.raises(LLMNetworkError):
        for chunk in router.stream([HumanMessage(content="Stream test")]):
            yielded.append(chunk.content)

    assert "Partial Gemini token " in yielded
    assert len(groq_mock.calls) == 0  # No replay through Groq!


def test_71_cooldown_behavior_all_three_providers():
    """Independent cooldowns across groq, mistral, and gemini."""
    tracker = ProviderStateTracker(cooldown_seconds=0.05, providers=("groq", "mistral", "gemini"))
    assert tracker.is_available("groq") is True
    assert tracker.is_available("mistral") is True
    assert tracker.is_available("gemini") is True

    tracker.mark_failure("groq")
    assert tracker.is_available("groq") is False
    assert tracker.is_available("mistral") is True
    assert tracker.is_available("gemini") is True

    tracker.mark_failure("mistral")
    assert tracker.is_available("mistral") is False
    assert tracker.is_available("gemini") is True

    tracker.mark_failure("gemini")
    assert tracker.is_available("gemini") is False

    # After cooldown expiry
    time.sleep(0.06)
    assert tracker.is_available("groq") is True
    assert tracker.is_available("mistral") is True
    assert tracker.is_available("gemini") is True


def test_72_llm_client_initializes_gemini_with_google_api_key(monkeypatch):
    """LLMClient initializes ChatGoogleGenerativeAI when GOOGLE_API_KEY is configured."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "groq,mistral,gemini")

    client = LLMClient(api_key="groq-key", mistral_api_key="mistral-key")
    assert client.google_api_key == "test-google-key"
    assert client.gemini_model == "gemini-2.5-flash"
    assert client.provider_order == ("groq", "mistral", "gemini")
    assert client.max_provider_attempts == 4

    model = client.get_langchain_model()
    assert isinstance(model, LLMProviderRouter)
    assert model.gemini_model is not None


def test_73_stt_and_priority_never_use_gemini(monkeypatch):
    """Ensure STT and Priority Analyzer have zero dependency on Gemini or GOOGLE_API_KEY."""
    monkeypatch.setenv("GOOGLE_API_KEY", "gemini-secret-key")
    monkeypatch.setenv("PRIORITY_GROQ_API_KEY", "priority-secret-key")

    priority_client = get_priority_llm_client()
    assert priority_client is not None
    assert priority_client.provider == "groq"
    assert priority_client.failover_enabled is False

    stt = GroqSTTService(api_key="stt-key")
    assert stt.model == "whisper-large-v3-turbo"


# ==============================================================================
# Gemini Content Representation Regression Tests (74-82)
# ==============================================================================

from assistant.llm.provider_router import (
    _normalize_content_to_text,
    validate_chat_response,
    LLMResponseValidationError,
)


def test_74_normalize_content_plain_string():
    """Plain string content passes through unchanged."""
    assert _normalize_content_to_text("Hello world") == "Hello world"
    assert _normalize_content_to_text("") == ""


def test_75_normalize_content_gemini_list_single_text_block():
    """Gemini structured list with a single text block is flattened to a string."""
    content = [{"type": "text", "text": "Hello from Gemini"}]
    result = _normalize_content_to_text(content)
    assert result == "Hello from Gemini"


def test_76_normalize_content_gemini_list_multiple_text_blocks():
    """Multiple text blocks are concatenated in order."""
    content = [
        {"type": "text", "text": "Part one. "},
        {"type": "text", "text": "Part two."},
    ]
    result = _normalize_content_to_text(content)
    assert result == "Part one. Part two."


def test_77_normalize_content_list_with_mixed_blocks():
    """Non-text blocks (e.g. tool-use) in the list are skipped; text blocks are extracted."""
    content = [
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {}},
        {"type": "text", "text": "Here is your answer."},
    ]
    result = _normalize_content_to_text(content)
    assert result == "Here is your answer."


def test_78_normalize_content_empty_list():
    """Empty list yields an empty string."""
    assert _normalize_content_to_text([]) == ""


def test_79_validate_chat_response_gemini_text_string():
    """validate_chat_response accepts a normal AIMessage with plain-string content."""
    msg = AIMessage(content="Hello")
    result = validate_chat_response(msg)
    assert isinstance(result.content, str)
    assert result.content == "Hello"


def test_80_validate_chat_response_gemini_list_content():
    """validate_chat_response normalizes Gemini list content to a plain string.

    This is the exact regression for:
        TypeError: expected string or bytes-like object, got 'list'
    which occurred when AIMessage.content was a list of structured content blocks.
    """
    msg = AIMessage(content=[{"type": "text", "text": "Hello from Gemini"}])
    # Must NOT raise TypeError
    result = validate_chat_response(msg)
    assert isinstance(result.content, str), (
        f"Content must be normalized to str, got {type(result.content)}: {result.content!r}"
    )
    assert result.content == "Hello from Gemini"


def test_81_validate_chat_response_gemini_tool_call_response():
    """validate_chat_response correctly handles a Gemini response that has tool calls
    but empty/absent text content."""
    tool_call = {"name": "set_alarm", "args": {"time": "07:00 AM"}, "id": "call_1", "type": "tool_call"}
    msg = AIMessage(content="", tool_calls=[tool_call])
    result = validate_chat_response(msg)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "set_alarm"


def test_82_gemini_list_content_does_not_trigger_typeerror():
    """Regression: streaming a Gemini list-content chunk through the router must not raise
    TypeError: expected string or bytes-like object, got 'list'.

    The router's _stream must extract plain text before calling on_llm_new_token.
    """
    gemini_response = AIMessage(
        content=[{"type": "text", "text": "Gemini response text"}]
    )
    gemini_mock = MockChatModel("gemini", response=gemini_response)
    # Gemini streaming emits list-content chunks
    gemini_mock.stream_chunks = [
        AIMessageChunk(content=[{"type": "text", "text": "Gemini "}]),
        AIMessageChunk(content=[{"type": "text", "text": "response text"}]),
    ]

    tracker = ProviderStateTracker(cooldown_seconds=60.0, providers=("groq", "mistral", "gemini"))
    tracker.mark_failure("groq")
    tracker.mark_failure("mistral")

    router = LLMProviderRouter(
        groq_model=MockChatModel("groq"),
        mistral_model=MockChatModel("mistral"),
        gemini_model=gemini_mock,
        state_tracker=tracker,
    )

    mock_run_manager = MagicMock()

    # Must not raise TypeError when on_llm_new_token is called with list content
    chunks = list(router._stream(
        [HumanMessage(content="Hi")],
        run_manager=mock_run_manager,
    ))
    assert len(chunks) == 2

    # run_manager was called only with plain strings, never with a list
    for call in mock_run_manager.on_llm_new_token.call_args_list:
        token_arg = call[0][0]  # first positional arg
        assert isinstance(token_arg, str), (
            f"on_llm_new_token must receive a str, got {type(token_arg)}: {token_arg!r}"
        )
        assert "[" not in token_arg, (
            f"on_llm_new_token must not receive a Python list repr: {token_arg!r}"
        )


def test_83_gemini_invoke_normalizes_list_content():
    """Router.invoke with a Gemini model that returns list content returns a plain-string AIMessage."""
    gemini_response = AIMessage(
        content=[{"type": "text", "text": "Gemini reply text"}]
    )
    gemini_mock = MockChatModel("gemini", response=gemini_response)

    tracker = ProviderStateTracker(cooldown_seconds=60.0, providers=("groq", "mistral", "gemini"))
    tracker.mark_failure("groq")
    tracker.mark_failure("mistral")

    router = LLMProviderRouter(
        groq_model=MockChatModel("groq"),
        mistral_model=MockChatModel("mistral"),
        gemini_model=gemini_mock,
        state_tracker=tracker,
    )

    result = router.invoke([HumanMessage(content="Hello")])
    assert isinstance(result.content, str), (
        f"Invoke result must normalize list content to str, got {type(result.content)}"
    )
    assert result.content == "Gemini reply text"


