"""
gmail_drafts.py — clean Gmail Draft creator (no gmail_meta issues)
"""

import os
import pickle
import base64
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Gmail Drafts scope
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

TOKEN_FILE = "token.json"
CLIENT_SECRET = "client_secret.json"


def _get_service():
    """Authenticate and return Gmail API service."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save token
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def create_draft(from_email, subject, body, in_reply_to, to_email):
    """
    Creates a Gmail draft. Returns True on success, False on failure.
    """
    try:
        service = _get_service()

        message = MIMEText(body)
        message["to"] = to_email
        message["from"] = from_email
        message["subject"] = subject
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
            message["References"] = in_reply_to

        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()

        draft = service.users().drafts().create(
            userId="me",
            body={"message": {"raw": encoded}}
        ).execute()

        print(f"[GMAIL] Draft created → ID: {draft.get('id')}")
        return True

    except Exception as e:
        print("[GMAIL ERROR]", e)
        return False
