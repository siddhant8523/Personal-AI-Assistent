"""
Unit tests for Gmail OAuth credential loading from environment variables.

Verifies:
    - Credentials can be constructed from env vars (no files needed).
    - refresh_token is read from GMAIL_REFRESH_TOKEN.
    - Correct Gmail scope is used.
    - Missing required vars raise EnvironmentError with a safe message.
    - No credential file is required or written.
    - No secret is written to disk.
    - No secret is emitted in log output.
    - No network calls are made (everything mocked).
"""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from assistant.connectors.gmail.gmail_connector import GmailConnector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_ENV = {
    "GOOGLE_CLIENT_ID": "test-client-id",
    "GOOGLE_CLIENT_SECRET": "test-client-secret",
    "GMAIL_REFRESH_TOKEN": "test-refresh-token",
    "GOOGLE_TOKEN_URI": "https://oauth2.googleapis.com/token",
    "GMAIL_SCOPES": "https://www.googleapis.com/auth/gmail.modify",
}


def _make_connector(env: dict, enabled: bool = True) -> GmailConnector:
    """Build a GmailConnector with mocked Google libraries and given env."""
    mock_service = MagicMock()
    with patch.dict(os.environ, env, clear=False), \
         patch("google.oauth2.credentials.Credentials", return_value=MagicMock()), \
         patch("googleapiclient.discovery.build", return_value=mock_service):
        return GmailConnector(enabled=enabled)


# ---------------------------------------------------------------------------
# 1. Credentials constructed from env vars — no files
# ---------------------------------------------------------------------------

def test_credentials_built_from_env_vars():
    """GmailConnector initialises successfully using only env vars — no JSON file required."""
    mock_creds = MagicMock()
    mock_service = MagicMock()

    with patch.dict(os.environ, VALID_ENV, clear=False), \
         patch("google.oauth2.credentials.Credentials", return_value=mock_creds) as MockCreds, \
         patch("googleapiclient.discovery.build", return_value=mock_service):

        connector = GmailConnector(enabled=True)

    assert connector.enabled is True
    assert connector._service is mock_service
    # Credentials must have been constructed (not loaded from a file)
    MockCreds.assert_called_once()
    call_kwargs = MockCreds.call_args.kwargs
    assert call_kwargs["client_id"] == "test-client-id"
    assert call_kwargs["client_secret"] == "test-client-secret"
    assert call_kwargs["refresh_token"] == "test-refresh-token"
    assert call_kwargs["token"] is None          # no access token — let library refresh


# ---------------------------------------------------------------------------
# 2. Refresh token read from GMAIL_REFRESH_TOKEN
# ---------------------------------------------------------------------------

def test_refresh_token_sourced_from_env():
    """The refresh_token passed to Credentials must come from GMAIL_REFRESH_TOKEN."""
    mock_creds = MagicMock()

    with patch.dict(os.environ, VALID_ENV, clear=False), \
         patch("google.oauth2.credentials.Credentials", return_value=mock_creds) as MockCreds, \
         patch("googleapiclient.discovery.build", return_value=MagicMock()):

        GmailConnector(enabled=True)

    assert MockCreds.call_args.kwargs["refresh_token"] == "test-refresh-token"


# ---------------------------------------------------------------------------
# 3. Correct Gmail scope is used
# ---------------------------------------------------------------------------

def test_correct_gmail_scope_used():
    """The scope passed to Credentials must match GMAIL_SCOPES."""
    mock_creds = MagicMock()
    scope = "https://www.googleapis.com/auth/gmail.modify"

    with patch.dict(os.environ, {**VALID_ENV, "GMAIL_SCOPES": scope}, clear=False), \
         patch("google.oauth2.credentials.Credentials", return_value=mock_creds) as MockCreds, \
         patch("googleapiclient.discovery.build", return_value=MagicMock()):

        GmailConnector(enabled=True)

    assert MockCreds.call_args.kwargs["scopes"] == [scope]


# ---------------------------------------------------------------------------
# 4. Missing required vars → EnvironmentError with safe message
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_var", [
    "GOOGLE_CLIENT_ID",
    "GOOGLE_CLIENT_SECRET",
    "GMAIL_REFRESH_TOKEN",
])
def test_missing_required_var_raises_environment_error(missing_var, monkeypatch):
    """Any missing required OAuth env var raises EnvironmentError."""
    env = {k: v for k, v in VALID_ENV.items() if k != missing_var}
    monkeypatch.delenv(missing_var, raising=False)

    connector = GmailConnector.__new__(GmailConnector)

    with patch.dict(os.environ, env, clear=False):
        # Forcibly remove the variable so it's absent even if it exists in the process env
        os.environ.pop(missing_var, None)
        with pytest.raises(EnvironmentError) as exc_info:
            connector._build_credentials()

    msg = str(exc_info.value)
    # Message must be safe — no secrets
    assert "GOOGLE_CLIENT_ID" in msg or "GOOGLE_CLIENT_SECRET" in msg or "GMAIL_REFRESH_TOKEN" in msg
    # No actual secret values in the error
    assert "test-client-id" not in msg
    assert "test-client-secret" not in msg
    assert "test-refresh-token" not in msg


def test_missing_vars_error_message_is_safe(monkeypatch):
    """EnvironmentError message must NOT contain any credential value."""
    for key in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    connector = GmailConnector.__new__(GmailConnector)
    with pytest.raises(EnvironmentError) as exc_info:
        connector._build_credentials()

    msg = str(exc_info.value)
    assert "test-client-id" not in msg
    assert "test-client-secret" not in msg
    assert "test-refresh-token" not in msg


# ---------------------------------------------------------------------------
# 5. No credential file required
# ---------------------------------------------------------------------------

def test_no_credential_file_required(tmp_path):
    """GmailConnector must not read or require any JSON file on disk."""
    mock_creds = MagicMock()
    original_open = open  # keep reference

    def fail_if_json_opened(path, *args, **kwargs):
        assert not str(path).endswith(".json"), (
            f"GmailConnector must not open a JSON credential file: {path}"
        )
        return original_open(path, *args, **kwargs)

    with patch.dict(os.environ, VALID_ENV, clear=False), \
         patch("google.oauth2.credentials.Credentials", return_value=mock_creds), \
         patch("googleapiclient.discovery.build", return_value=MagicMock()), \
         patch("builtins.open", side_effect=fail_if_json_opened):

        connector = GmailConnector(enabled=True)

    assert connector.enabled is True


# ---------------------------------------------------------------------------
# 6. No secret written to disk
# ---------------------------------------------------------------------------

def test_no_secret_written_to_disk(tmp_path):
    """_build_service must not write any file — no token serialisation."""
    files_written: list[str] = []
    original_open = open

    def spy_open(path, mode="r", *args, **kwargs):
        if "w" in mode:
            files_written.append(str(path))
        return original_open(path, mode, *args, **kwargs)

    mock_creds = MagicMock()

    with patch.dict(os.environ, VALID_ENV, clear=False), \
         patch("google.oauth2.credentials.Credentials", return_value=mock_creds), \
         patch("googleapiclient.discovery.build", return_value=MagicMock()), \
         patch("builtins.open", side_effect=spy_open):

        GmailConnector(enabled=True)

    json_writes = [f for f in files_written if f.endswith(".json")]
    assert json_writes == [], f"Unexpected JSON file writes: {json_writes}"


# ---------------------------------------------------------------------------
# 7. No secret emitted in log output
# ---------------------------------------------------------------------------

def test_no_secret_in_log_output(caplog):
    """Log messages from _build_service must not contain credential values."""
    mock_creds = MagicMock()

    with patch.dict(os.environ, VALID_ENV, clear=False), \
         patch("google.oauth2.credentials.Credentials", return_value=mock_creds), \
         patch("googleapiclient.discovery.build", return_value=MagicMock()), \
         caplog.at_level(logging.DEBUG, logger="assistant.connectors.gmail"):

        GmailConnector(enabled=True)

    full_log = "\n".join(caplog.messages)
    assert "test-client-id" not in full_log
    assert "test-client-secret" not in full_log
    assert "test-refresh-token" not in full_log


# ---------------------------------------------------------------------------
# 8. Mock mode when GMAIL_ENABLED=false
# ---------------------------------------------------------------------------

def test_mock_mode_when_disabled():
    """When enabled=False, connector stays in mock mode regardless of env vars."""
    connector = GmailConnector(enabled=False)

    assert connector.enabled is False
    assert connector._service is None
    messages = connector.fetch_recent()
    assert len(messages) == 2
    assert messages[0]["id"] == "mock-1"


# ---------------------------------------------------------------------------
# 9. Mock mode when env vars absent and enabled=True
# ---------------------------------------------------------------------------

def test_mock_mode_when_env_vars_absent(monkeypatch):
    """enabled=True with missing env vars → mock mode (no crash at init time)."""
    for key in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    connector = GmailConnector(enabled=True)

    assert connector.enabled is False
    assert connector._service is None


# ---------------------------------------------------------------------------
# 10. Service built ONCE — not per operation
# ---------------------------------------------------------------------------

def test_service_built_once_not_per_operation():
    """_build_service is called exactly once; the same service object is reused."""
    mock_service = MagicMock()
    mock_creds = MagicMock()
    build_call_count = {"n": 0}

    def counting_build(*args, **kwargs):
        build_call_count["n"] += 1
        return mock_service

    with patch.dict(os.environ, VALID_ENV, clear=False), \
         patch("google.oauth2.credentials.Credentials", return_value=mock_creds), \
         patch("googleapiclient.discovery.build", side_effect=counting_build):

        connector = GmailConnector(enabled=True)

    # Trigger several operations
    mock_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {"messages": []}
    connector.fetch_recent()
    connector.fetch_recent()

    assert build_call_count["n"] == 1, "build() must be called exactly once"
