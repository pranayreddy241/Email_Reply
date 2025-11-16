#!/usr/bin/env python3
"""
Baseline Restaurant Email Agent (Gmail API / OAuth)
- Fetch unread messages (INBOX) via Gmail API
- Use LLM (llm_agent.py) to understand intent (reservation / review / other)
- Auto-reply reservations (confirm or ask missing details)
- Stage non-urgent replies as local SQLite drafts
- Prevent double-processing by Message-ID

Usage:
  python agent.py
  python agent.py --send-pending

Env (example):
  RESTAURANT_NAME="Your Restaurant"
  RESERVATION_PHONE="+1..."
  RESERVATION_LINK="https://..."
  AGENT_DB_PATH="email_agent.sqlite"

Files required:
  - client_secret.json (OAuth client credentials)
  - token.json (created automatically on first run)
"""

from __future__ import print_function
import os
import re
import sqlite3
import email
import base64
import time
import random
import json
from datetime import datetime, timedelta

import pytz
from dateutil import parser as dateparser
from email.message import EmailMessage
from email.header import decode_header, make_header

from dotenv import load_dotenv
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Your LLM helper module
from llm_agent import summarize_thread, call_llm_extract, decide_action, received_local_dt

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")

# ---- OpenAI client (new SDK) ----
from openai import OpenAI
_oai = OpenAI(api_key=OPENAI_API_KEY)  # used inside llm_agent, mainly

# ----------------- Config -----------------
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant")
RESERVATION_PHONE = os.getenv("RESERVATION_PHONE", "")
RESERVATION_LINK = os.getenv("RESERVATION_LINK", "")
DB_PATH = os.getenv("AGENT_DB_PATH", "email_agent.sqlite")

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def _is_no_reply(addr, msg):
    """Filter obvious marketing / no-reply senders."""
    a = (addr or "").lower()
    if any(x in a for x in ["no-reply", "noreply", "notifications", "mailer-daemon"]):
        return True
    if msg.get("List-Unsubscribe"):
        return True
    return False


# ----------------- Gmail API auth -----------------
def get_gmail_service():
    """Authenticate (OAuth) and return a Gmail API service."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# ----------------- Helpers -----------------
def _dec(s):
    if not s:
        return ""
    try:
        return str(make_header(decode_header(s)))
    except Exception:
        return s


def _body(msg):
    """Extract best-effort text body from email.message.Message."""
    if msg.is_multipart():
        # Prefer text/plain
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition"))
            if ctype == "text/plain" and "attachment" not in disp:
                cs = part.get_content_charset() or "utf-8"
                return part.get_payload(decode=True).decode(cs, errors="replace")
        # Fallback to HTML stripped
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition"))
            if ctype == "text/html" and "attachment" not in disp:
                cs = part.get_content_charset() or "utf-8"
                html = part.get_payload(decode=True).decode(cs, errors="replace")
                text = re.sub(r'<br\s*/?>', '\n', html, flags=re.I)
                return re.sub(r'<[^>]+>', ' ', text)
    else:
        cs = msg.get_content_charset() or "utf-8"
        raw = msg.get_payload(decode=True)
        if raw is None:
            return msg.get_payload()
        text = raw.decode(cs, errors="replace")
        if msg.get_content_type() == "text/html":
            text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)
            text = re.sub(r'<[^>]+>', ' ', text)
        return text
    return ""


def _send(service, to_addr, subject, body, in_reply_to=None, thread_id=None):
    """Send a message via Gmail API."""
    m = EmailMessage()
    # Use authenticated user's email as sender
    sender_email = service.users().getProfile(userId="me").execute().get("emailAddress", "me")
    m["From"] = sender_email
    m["To"] = to_addr
    m["Subject"] = subject
    if in_reply_to:
        m["In-Reply-To"] = in_reply_to
        m["References"]  = in_reply_to
    m.set_content(body)

    raw = base64.urlsafe_b64encode(m.as_bytes()).decode("utf-8")
    payload = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id
    return service.users().messages().send(userId="me", body=payload).execute()


def _execute_with_retries(req, retries=5, base=1.0, cap=30.0, what="request"):
    """Generic Gmail API call with exponential backoff."""
    for attempt in range(retries):
        try:
            return req.execute()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in (500, 502, 503, 504):
                delay = min(cap, base * (2 ** attempt) + random.random())
                print(f"[RETRY] {what} failed with {status}, retrying in {delay:.1f}s (attempt {attempt+1}/{retries})")
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            delay = min(cap, base * (2 ** attempt) + random.random())
            print(f"[RETRY] {what} exception {e!r}, retrying in {delay:.1f}s (attempt {attempt+1}/{retries})")
            time.sleep(delay)
            continue
    raise RuntimeError(f"{what} failed after {retries} retries")


def _tpl_confirm(name, d, t, p):
    L = [
        f"Hi{(' ' + name) if name else ''},",
        "",
        f"Your reservation is confirmed at {RESTAURANT_NAME}.",
        f"• Date: {d}",
        f"• Time: {t}",
        f"• Party size: {p}",
    ]
    if RESERVATION_LINK:
        L.append(f"Modify/cancel: {RESERVATION_LINK}")
    if RESERVATION_PHONE:
        L.append(f"Phone: {RESERVATION_PHONE}")
    L += ["", "We look forward to hosting you!", f"— {RESTAURANT_NAME}"]
    return "\n".join(L)


def _tpl_missing(name, hd, ht, hp):
    miss = [x for x, v in {"date": hd, "time": ht, "party size": hp}.items() if not v]
    L = [
        f"Hi{(' ' + name) if name else ''},",
        f"Thanks for booking at {RESTAURANT_NAME}.",
        "Could you confirm your " + ", ".join(miss) + " so we can finalize your reservation?",
    ]
    if RESERVATION_LINK:
        L.append(f"You can also book directly here: {RESERVATION_LINK}")
    L += ["", "Best,", RESTAURANT_NAME]
    return "\n".join(L)


def _tpl_review(name):
    return (
        f"Hi{(' ' + name) if name else ''},\n\n"
        f"Thank you for your feedback about {RESTAURANT_NAME}. We appreciate it!\n\n"
        f"— {RESTAURANT_NAME}\n"
    )


def _tpl_other(name):
    return (
        f"Hi{(' ' + name) if name else ''},\n\n"
        f"Thanks for reaching out to {RESTAURANT_NAME}. We'll get back to you shortly.\n\n"
        f"— {RESTAURANT_NAME}\n"
    )


def _db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS processed("
        "  message_id TEXT PRIMARY KEY,"
        "  processed_at TEXT)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS drafts("
        "  id INTEGER PRIMARY KEY,"
        "  to_email TEXT,"
        "  subject TEXT,"
        "  body TEXT,"
        "  in_reply_to TEXT,"
        "  created_at TEXT,"
        "  sent_at TEXT)"
    )
    conn.commit()
    return conn


def _get_thread_bundle(service, thread_id):
    """
    Return a condensed list of dicts for the last few messages in a thread:
      {from, date, subject, body}
    """
    data = service.users().threads().get(userId="me", id=thread_id, format="full").execute()
    items = []
    for part in data.get("messages", [])[-4:]:  # last 4 messages
        payload = part.get("payload", {})
        headers = payload.get("headers", [])

        def hdr(name):
            for h in headers:
                if h.get("name", "").lower() == name.lower():
                    return h.get("value", "")
            return ""

        frm = hdr("From") or ""
        subj = hdr("Subject") or ""
        date = hdr("Date") or ""

        body = ""

        def walk(parts):
            nonlocal body
            for p in parts or []:
                mime = p.get("mimeType", "")
                if "text/plain" in mime and p.get("body", {}).get("data"):
                    import quopri
                    raw = base64.urlsafe_b64decode(p["body"]["data"])
                    try:
                        body = raw.decode("utf-8", "replace")
                    except Exception:
                        body = quopri.decodestring(raw).decode("utf-8", "replace")
                    return
                if "parts" in p:
                    walk(p["parts"])

        walk(payload.get("parts"))
        if not body:
            body = "(no text body)"
        items.append({"from": frm, "date": date, "subject": subj, "body": body})
    return items


def _fetch_unseen(service):
    """
    Return list of tuples: (msg_id, thread_id, raw_bytes) for unread messages.
    Uses robust retries.
    """

    def _pull(query=None, labelIds=None, limit=50):
        params = {"userId": "me", "maxResults": limit}
        if query is not None:
            params["q"] = query
        if labelIds is not None:
            params["labelIds"] = labelIds

        out, skipped = [], []
        resp = _execute_with_retries(
            service.users().messages().list(**params),
            what="messages.list",
        )
        for m in resp.get("messages", []):
            msg_id = m["id"]
            thread_id = m.get("threadId")
            try:
                raw_resp = _execute_with_retries(
                    service.users().messages().get(userId="me", id=msg_id, format="raw"),
                    what=f"messages.get(raw,{msg_id})",
                )
                raw_bytes = base64.urlsafe_b64decode(raw_resp["raw"])
                out.append((msg_id, thread_id, raw_bytes))
            except Exception as e:
                print(f"[SKIP message] id={msg_id} reason={e}")
                skipped.append(msg_id)
                continue
        if skipped:
            print(
                f"[INFO] skipped {len(skipped)} message(s) due to transient errors: "
                f"{skipped[:3]}{'...' if len(skipped) > 3 else ''}"
            )
        return out

    max_process = int(os.getenv("MAX_PROCESS", "10"))

    # Pass 1: unread in INBOX
    msgs = _pull(query=os.getenv("GMAIL_QUERY", "is:unread"),
                 labelIds=["INBOX"], limit=max_process)
    if msgs:
        print(f"[DEBUG] pass1 INBOX is:unread -> {len(msgs)}")
        return msgs

    # Pass 2: unread anywhere
    msgs = _pull(query="is:unread", labelIds=None, limit=max_process)
    if msgs:
        print(f"[DEBUG] pass2 ANY is:unread -> {len(msgs)}")
        return msgs

    # Pass 3: label UNREAD
    msgs = _pull(query=None, labelIds=["UNREAD"], limit=max_process)
    if msgs:
        print(f"[DEBUG] pass3 label:UNREAD -> {len(msgs)}")
        return msgs

    print("[DEBUG] no unread messages found by any strategy")
    return []


def handle(service, conn, raw, thread_id):
    """
    Process a single unread Gmail message given its raw bytes + thread_id.
    Uses llm_agent to decide what to do (confirm / ask_missing / draft / skip).
    """
    msg = email.message_from_bytes(raw)
    mid = msg.get("Message-ID")
    c = conn.cursor()

    # skip if already processed
    if mid and c.execute("SELECT 1 FROM processed WHERE message_id=?", (mid,)).fetchone():
        return

    subj = _dec(msg.get("Subject", ""))
    body = _body(msg)
    from_email = email.utils.parseaddr(msg.get("From", ""))[1]

    # skip marketing / no-reply
    if _is_no_reply(from_email, msg):
        print("[SKIP no-reply/marketing] ->", from_email)
        return

    # Build thread context using thread_id (NO gmail_meta)
    thread_bundle = _get_thread_bundle(service, thread_id)
    thread_text = summarize_thread(thread_bundle)

    # LLM structured extraction + decision
    extract = call_llm_extract(thread_text)

    # Use email's received time as reference for "today/tomorrow"
    ref_dt = received_local_dt(msg)

    plan = decide_action(extract, ref_dt)

    print(
        f"[LLM] plan={plan['action']} conf={plan['confidence']:.2f} "
        f"date={plan['date_iso']} time={plan['time_24']} party={plan['party_size']}"
    )

    name = (extract.get("name") or "").strip() or (from_email.split("@")[0])

    if plan["action"] == "confirm":
        body_text = _tpl_confirm(
            name,
            plan["date_iso"],
            plan["time_24"],
            str(plan["party_size"]),
        )
        _send(
            service,
            from_email,
            f"Re: {subj} — Reservation Confirmed",
            body_text,
            in_reply_to=mid,
            thread_id=thread_id,
        )
        print("[SENT] confirm ->", from_email)

    elif plan["action"] == "ask_missing":
        hd = bool(plan["date_iso"])
        ht = bool(plan["time_24"])
        hp = bool(plan["party_size"])
        _send(
            service,
            from_email,
            f"Re: {subj} — One quick detail",
            _tpl_missing(name, hd, ht, hp),
            in_reply_to=mid,
            thread_id=thread_id,
        )
        print("[SENT] missing ->", from_email)

    elif plan["action"] == "feedback":
    # FULL AUTO FEEDBACK SYSTEM
        sentiment, score = analyze_sentiment_with_backoff(body)
        discount = choose_discount(sentiment, score, body)

        prefix = "CARE" if sentiment == "negative" else "THANKS"
        code = _random_code(prefix, discount)
        persist_coupon(conn, from_email, code, discount, sentiment, score)

        reply_body = generate_personalized_reply(
            name,
            sentiment,
            score,
            discount,
            code,
            body
        )

        _send(service, from_email, f"Re: {subj}", reply_body, in_reply_to=mid, thread_id=thread_id)
        print(f"[SENT feedback] {sentiment}/{score} -> {discount}% code={code} to {from_email}")


    else:
        print("[SKIP other] ->", from_email)

    # mark as processed
    if mid:
        c.execute(
            "INSERT OR REPLACE INTO processed(message_id, processed_at) "
            "VALUES (?, datetime('now'))",
            (mid,),
        )
        conn.commit()


def send_pending(service, conn):
    """
    Send any staged drafts in the SQLite DB.
    Use: python agent.py --send-pending
    """
    c = conn.cursor()
    rows = c.execute(
        "SELECT id,to_email,subject,body,in_reply_to "
        "FROM drafts WHERE sent_at IS NULL ORDER BY id"
    ).fetchall()
    for did, to_email, subject, body, in_reply_to in rows:
        _send(service, to_email, subject, body, in_reply_to=in_reply_to, thread_id=None)
        c.execute("UPDATE drafts SET sent_at=datetime('now') WHERE id=?", (did,))
        conn.commit()
        print("[SENT draft]", did, "->", to_email)


def main():
    service = get_gmail_service()
    conn = _db()

    import sys
    if "--send-pending" in sys.argv:
        send_pending(service, conn)
        return

    messages = _fetch_unseen(service)  # list of (msg_id, thread_id, raw_bytes)
    print(f"[INFO] fetched {len(messages)} unread messages")

    for msg_id, thread_id, raw in messages:
        try:
            handle(service, conn, raw, thread_id)
        except Exception as e:
            print("[ERR]", e)


if __name__ == "__main__":
    main()
