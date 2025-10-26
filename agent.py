#!/usr/bin/env python3
"""
Baseline Restaurant Email Agent (Gmail API / OAuth)
- Fetch unread messages (INBOX) via Gmail API
- Classify: reservation/review/other (regex rules)
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
from llm_agent import summarize_thread, call_llm_extract, decide_action, received_local_dt
import os, re, sqlite3, email, base64
from email.message import EmailMessage
from email.header import decode_header, make_header

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import json, pytz
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")

# ---- OpenAI client (new SDK) ----
from openai import OpenAI
_oai = OpenAI(api_key=OPENAI_API_KEY)

# Utility: get local tz-aware datetime from RFC2822 email header
def _received_dt(msg):
    try:
        hdr = msg.get("Date")
        dt = email.utils.parsedate_to_datetime(hdr) if hdr else datetime.utcnow()
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.timezone(TIMEZONE))
    except Exception:
        return datetime.now(pytz.timezone(TIMEZONE))



# ----------------- Config -----------------
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant")
RESERVATION_PHONE = os.getenv("RESERVATION_PHONE", "")
RESERVATION_LINK = os.getenv("RESERVATION_LINK", "")
DB_PATH = os.getenv("AGENT_DB_PATH", "email_agent.sqlite")

RESERVATION_KEYWORDS = [r"\breservation\b", r"\bbook(?:ing)?\b", r"\btable\b", r"\bparty\b"]
REVIEW_KEYWORDS = [r"\breview\b", r"\bfeedback\b", r"\bcomplaint\b", r"\bpraise\b", r"\bexperience\b"]
DATE_RE = r"(?:(?P<date>(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?|today|tomorrow|mon|tue|wed|thu|fri|sat|sun))"
TIME_RE = r"(?:(?P<time>\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|\bo'clock\b)?))"
PARTY_RE = r"(?:(?:for|party of|group of|we are|we're|total of)\s*(?P<party>\d{1,2}))"

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def _is_no_reply(addr, msg):
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
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition"))
            if ctype == "text/plain" and "attachment" not in disp:
                cs = part.get_content_charset() or "utf-8"
                return part.get_payload(decode=True).decode(cs, errors="replace")
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
'''
def _classify(text):
    t = text.lower()
    if any(re.search(p, t) for p in RESERVATION_KEYWORDS): return "reservation"
    if any(re.search(p, t) for p in REVIEW_KEYWORDS):      return "review"
    return "other"

def _extract(text):
    d = re.search(DATE_RE, text, flags=re.I)
    t = re.search(TIME_RE, text, flags=re.I)
    p = re.search(PARTY_RE, text, flags=re.I)
    det = {
        "date": d.group("date") if d else None,
        "time": (t.group("time") if t else None),
        "party_size": p.group("party") if p else None
    }
    # If time is just an hour like "7" or "10", normalize to "7:00"
    if det["time"] and re.fullmatch(r"\d{1,2}", det["time"].strip()):
        det["time"] += ":00"
    return det
'''
def _tpl_confirm(name, d, t, p):
    L = [
        f"Hi{(' ' + name) if name else ''},",
        "",
        f"Your reservation is confirmed at {RESTAURANT_NAME}.",
        f"• Date: {d}",
        f"• Time: {t}",
        f"• Party size: {p}"
    ]
    if RESERVATION_LINK:  L.append(f"Modify/cancel: {RESERVATION_LINK}")
    if RESERVATION_PHONE: L.append(f"Phone: {RESERVATION_PHONE}")
    L += ["", "We look forward to hosting you!", f"— {RESTAURANT_NAME}"]
    return "\n".join(L)

def _tpl_missing(name, hd, ht, hp):
    miss = [x for x, v in {"date": hd, "time": ht, "party size": hp}.items() if not v]
    L = [
        f"Hi{(' ' + name) if name else ''},",
        f"Thanks for booking at {RESTAURANT_NAME}.",
        "Could you confirm your " + ", ".join(miss) + " so we can finalize your reservation?"
    ]
    if RESERVATION_LINK:
        L.append(f"You can also book directly here: {RESERVATION_LINK}")
    L += ["", "Best,", RESTAURANT_NAME]
    return "\n".join(L)

def _tpl_review(name):
    return f"Hi{(' ' + name) if name else ''},\n\nThank you for your feedback about {RESTAURANT_NAME}. We appreciate it!\n\n— {RESTAURANT_NAME}\n"

def _tpl_other(name):
    return f"Hi{(' ' + name) if name else ''},\n\nThanks for reaching out to {RESTAURANT_NAME}. We'll get back to you shortly.\n\n— {RESTAURANT_NAME}\n"

def _db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS processed(message_id TEXT PRIMARY KEY, processed_at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS drafts(id INTEGER PRIMARY KEY, to_email TEXT, subject TEXT, body TEXT, in_reply_to TEXT, created_at TEXT, sent_at TEXT)")
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
                if h.get("name","").lower() == name.lower():
                    return h.get("value","")
            return ""
        frm = hdr("From") or ""
        subj = hdr("Subject") or ""
        date = hdr("Date") or ""
        # Extract simple text/plain (fallback to text/html stripped)
        body = ""
        def walk(parts):
            nonlocal body
            for p in parts or []:
                mime = p.get("mimeType","")
                if "text/plain" in mime and p.get("body", {}).get("data"):
                    import base64, quopri
                    raw = base64.urlsafe_b64decode(p["body"]["data"])
                    try:
                        body = raw.decode("utf-8","replace")
                    except:
                        body = quopri.decodestring(raw).decode("utf-8","replace")
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
    Tries progressively broader queries so we don't miss anything.
    """
    import base64

    def _fetch_with(query=None, labelIds=None, limit=50):
        params = {"userId": "me", "maxResults": limit}
        if query is not None:
            params["q"] = query
        if labelIds is not None:
            params["labelIds"] = labelIds

        out = []
        resp = service.users().messages().list(**params).execute()
        for m in resp.get("messages", []):
            msg_id = m["id"]
            meta = service.users().messages().get(userId="me", id=msg_id, format="metadata").execute()
            thread_id = meta.get("threadId")
            raw_resp = service.users().messages().get(userId="me", id=msg_id, format="raw").execute()
            raw_bytes = base64.urlsafe_b64decode(raw_resp["raw"])
            out.append((msg_id, thread_id, raw_bytes))
        return out

    # Try 1: common case (unread in Inbox, any tab)
    msgs = _fetch_with(query="is:unread", labelIds=["INBOX"])
    if msgs:
        print(f"[DEBUG] pass1 INBOX is:unread -> {len(msgs)}")
        return msgs

    # Try 2: super broad (any unread anywhere, no label filter)
    msgs = _fetch_with(query="is:unread", labelIds=None)
    if msgs:
        print(f"[DEBUG] pass2 ANY is:unread -> {len(msgs)}")
        return msgs

    # Try 3: use UNREAD label directly (broadest)
    msgs = _fetch_with(query=None, labelIds=["UNREAD"])
    if msgs:
        print(f"[DEBUG] pass3 label:UNREAD -> {len(msgs)}")
        return msgs

    print("[DEBUG] no unread messages found by any strategy")
    return []

'''
Removed this as we are getting an error and our system is unable to read the messages
def _fetch_unseen(service):
    """Return list of tuples: (msg_id, thread_id, raw_bytes) for unread INBOX messages."""
    out = []
    resp = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        q="is:unread category:primary -from:(no-reply noreply notifications mailer-daemon)",
        maxResults=50
    ).execute()

    for m in resp.get("messages", []):
        msg_id = m["id"]
        # get threadId for proper threading
        meta = service.users().messages().get(userId="me", id=msg_id, format="metadata").execute()
        thread_id = meta.get("threadId")
        raw_resp = service.users().messages().get(userId="me", id=msg_id, format="raw").execute()
        raw_bytes = base64.urlsafe_b64decode(raw_resp["raw"])
        out.append((msg_id, thread_id, raw_bytes))
        # mark as read immediately (optional)
        service.users().messages().modify(userId="me", id=msg_id, body={"removeLabelIds": ["UNREAD"]}).execute()
    return out
'''
def handle(service, conn, raw, thread_id):
    msg = email.message_from_bytes(raw)
    mid = msg.get("Message-ID")
    c = conn.cursor()
    if mid and c.execute("SELECT 1 FROM processed WHERE message_id=?", (mid,)).fetchone():
        return

    subj = _dec(msg.get("Subject", ""))
    body = _body(msg)
    from_email = email.utils.parseaddr(msg.get("From", ""))[1]
    # Skip obvious automation
    if _is_no_reply(from_email, msg):
        print("[SKIP no-reply/marketing] ->", from_email)
        return

# Build thread context
    thread_bundle = _get_thread_bundle(service, gmail_meta.get("threadId"))
    thread_text = summarize_thread(thread_bundle)

# Call LLM for structured extraction
    extract = call_llm_extract(thread_text)

# Use email's received time as reference for relative dates
    ref_dt = received_local_dt(msg)

    plan = decide_action(extract, ref_dt)

    print(f"[LLM] plan={plan['action']} conf={plan['confidence']:.2f} "
          f"date={plan['date_iso']} time={plan['time_24']} party={plan['party_size']}")

    if plan["action"] == "confirm":
        body_text = _tpl_confirm(
            name,
            plan["date_iso"],
            plan["time_24"],
            str(plan["party_size"])
        )
        _send(service, from_email, f"Re: {subj} — Reservation Confirmed",
              body_text, in_reply_to=mid, thread_id=gmail_meta.get("threadId"))
        print("[SENT] confirm ->", from_email)

    elif plan["action"] == "ask_missing":
        hd = bool(plan["date_iso"])
        ht = bool(plan["time_24"])
        hp = bool(plan["party_size"])
        _send(service, from_email, f"Re: {subj} — One quick detail",
              _tpl_missing(name, hd, ht, hp), in_reply_to=mid, thread_id=gmail_meta.get("threadId"))
        print("[SENT] missing ->", from_email)

    elif plan["action"] == "draft":
        dsubj = f"Re: {subj} — Thank you"
        dbody = _tpl_review(name)
        c.execute(
            "INSERT INTO drafts(to_email,subject,body,in_reply_to,created_at) VALUES (?,?,?,?,datetime('now'))",
            (from_email, dsubj, dbody, mid)
        )
        conn.commit()
        print("[DRAFTED review] ->", from_email)

    else:
        print("[SKIP other] ->", from_email)
    c.execute("INSERT OR REPLACE INTO processed(message_id, processed_at) VALUES (?, datetime('now'))", (mid,))
    conn.commit()

def send_pending(service, conn):
    c = conn.cursor()
    for did, to_email, subject, body, in_reply_to in c.execute(
        "SELECT id,to_email,subject,body,in_reply_to FROM drafts WHERE sent_at IS NULL ORDER BY id"
    ):
        _send(service, to_email, subject, body, in_reply_to=in_reply_to, thread_id=None)
        c.execute("UPDATE drafts SET sent_at=datetime('now') WHERE id=?", (did,))
        conn.commit()
        print("[SENT draft]", did)

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
