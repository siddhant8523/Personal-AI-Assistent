"""
One-time Gmail OAuth setup.

1. Create a project in Google Cloud Console, enable the Gmail API.
2. Create OAuth 2.0 credentials (Desktop app) and download the JSON as
   config/secrets/gmail_credentials.json (path from GMAIL_CREDENTIALS_FILE).
3. Run this script; it opens a browser to authorize, then prints the
   environment variables you need to set in .env.
4. Copy the printed values into your .env file and set GMAIL_ENABLED=true.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

if __name__ == "__main__":
    load_dotenv()
    from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: E402

    creds_file = os.environ["GMAIL_CREDENTIALS_FILE"]

    flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
    creds = flow.run_local_server(port=0)

    print("\n# ── Copy the following lines into your .env file ──────────────────")
    print(f"GOOGLE_CLIENT_ID={creds.client_id}")
    print(f"GOOGLE_CLIENT_SECRET={creds.client_secret}")
    print(f"GOOGLE_TOKEN_URI={creds.token_uri}")
    print(f"GMAIL_REFRESH_TOKEN={creds.refresh_token}")
    print(f"GMAIL_SCOPES={','.join(creds.scopes or SCOPES)}")
    print("# ───────────────────────────────────────────────────────────────────")
    print("\nSet GMAIL_ENABLED=true in .env once you have pasted the values above.")
    print("Do NOT commit your .env file.")
