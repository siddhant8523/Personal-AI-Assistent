"""
Gmail Connector (Part 1, Section 8; Part 3, Section 8)
==========================================================
Real implementation uses google-api-python-client with OAuth (see
scripts/setup_gmail_oauth.py for the one-time auth flow). When
GMAIL_ENABLED=false or required OAuth environment variables are absent,
runs in mock mode so the rest of the pipeline (priority engine, agent core)
can still be exercised without live credentials.

Required environment variables (runtime):
    GOOGLE_CLIENT_ID      – OAuth 2.0 client ID
    GOOGLE_CLIENT_SECRET  – OAuth 2.0 client secret
    GMAIL_REFRESH_TOKEN   – long-lived refresh token obtained during setup
    GOOGLE_TOKEN_URI      – (optional) defaults to https://oauth2.googleapis.com/token
    GMAIL_SCOPES          – (optional) defaults to https://www.googleapis.com/auth/gmail.modify
"""

from __future__ import annotations

import os


class GmailConnector:
    def __init__(self, enabled: bool = False):
        # enabled=True requires the three OAuth env vars to be set.
        # If any are missing we stay in mock mode rather than crashing at
        # import time; _build_service() raises a clear EnvironmentError.
        self.enabled = enabled and self._oauth_vars_present()
        self._service = None
        if self.enabled:
            self._service = self._build_service()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _oauth_vars_present() -> bool:
        """Return True only when all three required OAuth env vars are non-empty."""
        return bool(
            os.environ.get("GOOGLE_CLIENT_ID")
            and os.environ.get("GOOGLE_CLIENT_SECRET")
            and os.environ.get("GMAIL_REFRESH_TOKEN")
        )

    def _build_credentials(self):
        """
        Construct a google.oauth2.credentials.Credentials object entirely from
        environment variables.  No file I/O is performed.  The access token is
        intentionally left as None — google-api-python-client will obtain and
        refresh it transparently the first time an API call is made, and will
        keep refreshing it automatically whenever it expires.
        """
        from google.oauth2.credentials import Credentials

        client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN", "")
        token_uri = os.environ.get(
            "GOOGLE_TOKEN_URI", "https://oauth2.googleapis.com/token"
        )
        scopes_raw = os.environ.get(
            "GMAIL_SCOPES", "https://www.googleapis.com/auth/gmail.modify"
        )
        scopes = [s.strip() for s in scopes_raw.split(",") if s.strip()]

        if not (client_id and client_secret and refresh_token):
            raise EnvironmentError(
                "Gmail OAuth credentials are not configured. "
                "Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET and GMAIL_REFRESH_TOKEN."
            )

        return Credentials(
            token=None,          # no access token; let the library refresh on first use
            refresh_token=refresh_token,
            token_uri=token_uri,
            client_id=client_id,
            client_secret=client_secret,
            scopes=scopes,
        )

    def _build_service(self):
        """
        Build the Gmail API service client (called once from __init__).

        Credentials are sourced entirely from environment variables — no JSON
        files are read or written.  The service object is stored as
        self._service and reused for every subsequent Gmail operation, matching
        the original lifecycle.

        Token refresh is handled transparently by google-api-python-client:
        when the access token is None (or later expires), the library exchanges
        the refresh token for a new access token automatically before each API
        call.  No explicit creds.refresh(Request()) call is needed here.
        """
        import logging

        from googleapiclient.discovery import build

        logger = logging.getLogger("assistant.connectors.gmail")
        creds = self._build_credentials()
        service = build("gmail", "v1", credentials=creds)
        logger.info("Gmail service initialised from environment-variable credentials.")
        return service

    # ------------------------------------------------------------------
    # Public API — unchanged from original
    # ------------------------------------------------------------------

    def fetch_recent(self, max_results: int = 10) -> list[dict]:
        if not self.enabled:
            return self._mock_messages()

        results = self._service.users().messages().list(userId="me", maxResults=max_results).execute()
        out = []
        for m in results.get("messages", []):
            msg = self._service.users().messages().get(userId="me", id=m["id"]).execute()
            headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
            out.append({
                "id": msg["id"], "thread_id": msg.get("threadId"),
                "from": headers.get("From", ""), "to": headers.get("To", ""),
                "subject": headers.get("Subject", ""), "snippet": msg.get("snippet", ""),
            })
        return out

    def send(self, to: str, subject: str, body: str) -> dict:
        if not self.enabled:
            return {"status": "ok", "detail": f"[MOCK] would send email to {to}: {subject}"}

        import base64
        from email.mime.text import MIMEText

        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        sent = self._service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return {"status": "ok", "detail": sent.get("id")}

    def send_attachment(self, to: str, subject: str, body: str, path: str) -> dict:
        if not self.enabled:
            return {"status": "ok", "detail": f"[MOCK] would send {path} to {to}"}
        import base64
        import mimetypes
        from email.message import EmailMessage

        message = EmailMessage()
        message["To"], message["Subject"] = to, subject
        message.set_content(body)
        mime, _ = mimetypes.guess_type(path)
        maintype, subtype = (mime or "application/octet-stream").split("/", 1)
        with open(path, "rb") as handle:
            message.add_attachment(handle.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(path))
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        sent = self._service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return {"status": "ok", "detail": sent.get("id")}

    def _mock_messages(self) -> list[dict]:
        return [
            {
                "id": "mock-1", "thread_id": "t1", "from": "recruiter@example.com", "to": "me@example.com",
                "subject": "Interview scheduled", "snippet": "Your interview is scheduled for tomorrow at 10 AM.",
            },
            {
                "id": "mock-2", "thread_id": "t2", "from": "deals@newsletter.example.com", "to": "me@example.com",
                "subject": "Weekend Sale", "snippet": "20% discount this weekend only.",
            },
        ]
