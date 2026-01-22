#!/usr/bin/env python3
"""
Restaurant Email Agent (Postgres) + Gmail API (OAuth) + LLM

What it does:
- Fetch unread emails (uses the OLD/simple fetch: INBOX + is:unread query)
- Build recent thread context
- Use llm_agent.py to decide action: confirm / ask_missing / feedback / skip
- Confirm reservations (capacity-aware) into Postgres
- Update reservations: if email asks to update/change/reschedule, cancel latest confirmed reservation for that email, then book new one
- Best-effort phone extraction from email body + store with reservation
- Feedback: simple sentiment -> issue coupon -> log
- Dedupe using Postgres `processed` table keyed by Message-ID
- Mark emails as READ (remove UNREAD label)
- Write `system_status` heartbeats for dashboard

ENV:
  DATABASE_URL=postgresql://...
  OPENAI_API_KEY=... (optional; LLM is in llm_agent.py)
  RESTAURANT_NAME=...
  DEFAULT_CAPACITY=10
  MAX_PROCESS=10
  GMAIL_QUERY=is:unread
  GMAIL_TOKEN_JSON=<contents of token.json> (optional bootstrap, useful on Render)
"""

from __future__ import print_function

import os
import re
import time
import base64
import random
import string
import email
from email.message import EmailMessage
from email.header import decode_header, make_header

from dotenv import load_dotenv

from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from db_pg import get_pg_conn, ensure_schema_pg
from db_utils import reserve, next_available_slots, cancel_latest_reservation, DEFAULT_CAPACITY
from llm_agent import summarize_thread, call_llm_extract, decide_action, received_local_dt

load_dotenv()

# ----------------- Config -----------------
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant")
RESERVATION_PHONE = os.getenv("RESERVATION_PHONE", "")
RESERVATION_LINK = os.getenv("RESERVATION_LINK", "")
CAPACITY = int(os.getenv("DEFAULT_CAPACITY", str(DEFAULT_CAPACITY)))
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


# ----------------- Small helpers -----------------
def _bootstrap_token_json():
    """Optional: allow token.json content to be provided via env var (useful on Render)."""
    token_env = os.getenv("GMAIL_TOKEN_JSON")
    if token_env and not os.path.exists("token.json"):
        with open("token.json", "w") as f:
            f.write(token_env)


def _is_no_reply(addr, msg):
    """Filter obvious marketing / no-reply senders."""
    a = (addr or "").lower()
    if any(x in a for x in ["no-reply", "noreply", "notifications", "mailer-daemon"]):
        return True
    if msg.get("List-Unsubscribe"):
        return True
    return False


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
            disp = str(part.get("Content-Disposition") or "")
            if ctype == "text/plain" and "attachment" not in disp:
                cs = part.get_content_charset() or "utf-8"
                return part.get_payload(decode=True).decode(cs, errors="replace")
        # Fallback: text/html stripped
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if ctype == "text/html" and "attachment" not in disp:
                cs = part.get_content_charset() or "utf-8"
                html = part.get_payload(decode=True).decode(cs, errors="replace")
                text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
                return re.sub(r"<[^>]+>", " ", text)
    else:
        cs = msg.get_content_charset() or "utf-8"
        raw = msg.get_payload(decode=True)
        if raw is None:
            return msg.get_payload() or ""
        text = raw.decode(cs, errors="replace")
        if msg.get_content_type() == "text/html":
            text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
            text = re.sub(r"<[^>]+>", " ", text)
        return text
    return ""


def detect_update_request(subject: str, body: str) -> bool:
    """Heuristic: treat as update/reschedule if they mention change/update etc."""
    text = f"{subject}\n{body}".lower()
    keywords = [
        "update my reservation", "change my reservation", "modify my reservation",
        "reschedule", "change the time", "change the date", "move the reservation",
        "need to change", "can we change", "instead of", "correction", "correct my reservation",
        "change reservation", "update reservation",
    ]
    return any(k in text for k in keywords)


def extract_phone_best_effort(text: str):
    """Extract a phone-like string from email body. Requires >=9 digits."""
    if not text:
        return None
    m = re.search(r"(\+?\d[\d\-\(\)\s]{8,}\d)", text)
    if not m:
        return None
    raw = m.group(1)
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 9:
        return None
    return re.sub(r"\s+", " ", raw).strip()


def _random_code(prefix: str, pct: int) -> str:
    tail = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{prefix}{pct}-{tail}"


# ----------------- Gmail helpers -----------------
def get_gmail_service():
    """Authenticate (OAuth) and return a Gmail API service."""
    _bootstrap_token_json()

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("client_secret.json"):
                raise RuntimeError("client_secret.json missing in project root.")
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _execute_with_retries(req, retries=5, base=1.0, cap=30.0, what="request"):
    """Gmail API call with exponential backoff for transient errors."""
    for attempt in range(retries):
        try:
            return req.execute()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in (500, 502, 503, 504):
                delay = min(cap, base * (2 ** attempt) + random.random())
                print(f"[RETRY] {what} failed with {status}, retrying in {delay:.1f}s ({attempt+1}/{retries})")
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            delay = min(cap, base * (2 ** attempt) + random.random())
            print(f"[RETRY] {what} exception {e!r}, retrying in {delay:.1f}s ({attempt+1}/{retries})")
            time.sleep(delay)
            continue
    raise RuntimeError(f"{what} failed after {retries} retries")


def _send(service, to_addr, subject, body, in_reply_to=None, thread_id=None):
    """Send a message via Gmail API."""
    m = EmailMessage()
    sender_email = service.users().getProfile(userId="me").execute().get("emailAddress", "me")
    m["From"] = sender_email
    m["To"] = to_addr
    m["Subject"] = subject
    if in_reply_to:
        m["In-Reply-To"] = in_reply_to
        m["References"] = in_reply_to
    m.set_content(body)

    raw = base64.urlsafe_b64encode(m.as_bytes()).decode("utf-8")
    payload = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id
    return service.users().messages().send(userId="me", body=payload).execute()


def _mark_read(service, msg_id):
    """Remove UNREAD label so it doesn't come back next run."""
    try:
        service.users().messages().modify(
            userId="me",
            id=msg_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()
    except Exception as e:
        print("[WARN] failed to mark as read:", e)


# ----------------- OLD fetch (keep simple) -----------------
def _fetch_unseen(service):
    """
    OLD/simple behavior:
      - query (default is:unread)
      - labelIds=["INBOX"]
      - maxResults=MAX_PROCESS
    """
    max_process = int(os.getenv("MAX_PROCESS", "10"))
    query = os.getenv("GMAIL_QUERY", "is:unread")

    resp = _execute_with_retries(
        service.users().messages().list(
            userId="me",
            q=query,
            labelIds=["INBOX"],
            maxResults=max_process,
        ),
        what="messages.list",
    )

    out = []
    for m in (resp.get("messages") or []):
        msg_id = m["id"]
        thread_id = m.get("threadId")
        raw_resp = _execute_with_retries(
            service.users().messages().get(userId="me", id=msg_id, format="raw"),
            what="messages.get(raw)",
        )
        raw_bytes = base64.urlsafe_b64decode(raw_resp["raw"])
        out.append((msg_id, thread_id, raw_bytes))
    return out


def _get_thread_bundle(service, thread_id):
    """Return last few messages in thread as dicts: {from,date,subject,body}."""
    data = _execute_with_retries(
        service.users().threads().get(userId="me", id=thread_id, format="full"),
        what="threads.get",
    )
    items = []
    for part in (data.get("messages") or [])[-4:]:
        payload = part.get("payload", {})
        headers = payload.get("headers", [])

        def hdr(name):
            for h in headers:
                if (h.get("name") or "").lower() == name.lower():
                    return h.get("value", "") or ""
            return ""

        frm = hdr("From")
        subj = hdr("Subject")
        dt = hdr("Date")

        body = ""
        def walk(parts):
            nonlocal body
            for p in parts or []:
                mime = p.get("mimeType", "")
                if "text/plain" in mime and p.get("body", {}).get("data"):
                    raw = base64.urlsafe_b64decode(p["body"]["data"])
                    body = raw.decode("utf-8", "replace")
                    return
                if "parts" in p:
                    walk(p["parts"])

        walk(payload.get("parts"))
        if not body:
            body = "(no text body)"
        items.append({"from": frm, "date": dt, "subject": subj, "body": body})
    return items


# ----------------- Templates -----------------
def _tpl_confirm(name, d, t, p, confirmation_code=None, updated=False):
    head = "Your reservation is updated ✅" if updated else "Your reservation is confirmed ✅"
    L = [
        f"Hi{(' ' + name) if name else ''},",
        "",
        f"{head} at {RESTAURANT_NAME}.",
        f"• Date: {d}",
        f"• Time: {t}",
        f"• Party size: {p}",
    ]
    if confirmation_code:
        L.append(f"• Confirmation: {confirmation_code}")
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
        f"Thanks for reaching out to {RESTAURANT_NAME}.",
        "Could you confirm your " + ", ".join(miss) + " so we can finalize your reservation?",
    ]
    if RESERVATION_LINK:
        L.append(f"You can also book directly here: {RESERVATION_LINK}")
    L += ["", "Best,", RESTAURANT_NAME]
    return "\n".join(L)


def _tpl_slot_full(name, d, requested_t, alternatives):
    if alternatives:
        alt_lines = ["Here are the next available times:", *[f"• {t}" for t in alternatives]]
    else:
        alt_lines = ["Unfortunately, we are fully booked for that date. Could you share a different date or time?"]
    L = [
        f"Hi{(' ' + name) if name else ''},",
        "",
        f"Thanks for reaching out! We’re currently fully booked at {RESTAURANT_NAME} for:",
        f"• Date: {d}",
        f"• Time: {requested_t}",
        "",
        *alt_lines,
        "",
        "Reply with your preferred option and party size, and we’ll confirm it right away.",
        "",
        "Best,",
        RESTAURANT_NAME,
    ]
    return "\n".join([x for x in L if x])


# ----------------- Feedback (stable minimal, Postgres) -----------------
def analyze_sentiment_simple(text: str):
    t = (text or "").lower()
    neg_hits = sum(w in t for w in ["awful", "terrible", "horrible", "disgusting", "cold", "late", "rude", "worst", "refund", "angry", "disappointed"])
    pos_hits = sum(w in t for w in ["amazing", "great", "excellent", "love", "fantastic", "wonderful", "perfect", "delicious", "best"])
    if neg_hits >= 2:
        return "negative", 4
    if neg_hits == 1:
        return "negative", 3
    if pos_hits >= 2:
        return "positive", 1
    if pos_hits == 1:
        return "positive", 2
    return "neutral", 3


def choose_discount(sentiment: str, score: int) -> int:
    if sentiment == "positive":
        return 10 if score == 1 else 5
    if sentiment == "neutral":
        return 15
    return 25 if score >= 4 else 15


def persist_coupon(conn, email_addr: str, code: str, discount: int, sentiment: str, score: int):
    with conn.cursor() as c:
        c.execute(
            """INSERT INTO coupons(email, code, discount, sentiment, score)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (code) DO NOTHING""",
            (email_addr, code, discount, sentiment, score),
        )
    conn.commit()


def log_feedback(conn, email_addr: str, sentiment: str, score: int, discount: int, code: str, original_text: str, reply_text: str):
    with conn.cursor() as c:
        c.execute(
            """INSERT INTO feedback_log(email, sentiment, score, discount, code, original_text, reply_text)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (email_addr, sentiment, score, discount, code, original_text, reply_text),
        )
    conn.commit()


def generate_feedback_reply(name: str, sentiment: str, discount: int, code: str):
    if sentiment == "positive":
        return (
            f"Hi{(' ' + name) if name else ''},\n\n"
            f"Thank you for the kind note — it means a lot to our team.\n"
            f"As a thank you, here’s {discount}% off next time: {code}\n\n"
            f"— {RESTAURANT_NAME}"
        )
    if sentiment == "neutral":
        return (
            f"Hi{(' ' + name) if name else ''},\n\n"
            f"Thanks for sharing your feedback — we’re always improving.\n"
            f"Here’s {discount}% off your next visit: {code}\n\n"
            f"— {RESTAURANT_NAME}"
        )
    return (
        f"Hi{(' ' + name) if name else ''},\n\n"
        f"We’re sorry your experience fell short. We’d like to make it right.\n"
        f"Please accept {discount}% off your next visit: {code}\n\n"
        f"— {RESTAURANT_NAME}"
    )


# ----------------- Postgres DB + status + dedupe -----------------
def _db():
    conn = get_pg_conn()
    ensure_schema_pg(conn)
    return conn


def _status_set(conn, key: str, value: str):
    with conn.cursor() as c:
        c.execute(
            """INSERT INTO system_status(key, value, updated_at)
               VALUES (%s,%s,NOW())
               ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()""",
            (key, value),
        )
    conn.commit()


def _already_processed(conn, message_id: str) -> bool:
    if not message_id:
        return False
    with conn.cursor() as c:
        c.execute("SELECT 1 FROM processed WHERE message_id=%s", (message_id,))
        return c.fetchone() is not None


def _mark_processed(conn, message_id: str, action: str):
    if not message_id:
        return
    with conn.cursor() as c:
        c.execute(
            """INSERT INTO processed(message_id, action, processed_at)
               VALUES (%s,%s,NOW())
               ON CONFLICT (message_id) DO UPDATE SET action=EXCLUDED.action, processed_at=NOW()""",
            (message_id, action),
        )
    conn.commit()


# ----------------- Main handler -----------------
def handle(service, conn, raw_bytes, thread_id, gmail_msg_id):
    msg = email.message_from_bytes(raw_bytes)
    mid = msg.get("Message-ID", "")

    subj = _dec(msg.get("Subject", ""))
    body = _body(msg)
    from_email = email.utils.parseaddr(msg.get("From", ""))[1]

    if _is_no_reply(from_email, msg):
        print("[SKIP marketing/no-reply]", from_email)
        _mark_read(service, gmail_msg_id)
        return

    if mid and _already_processed(conn, mid):
        print("[SKIP already processed]", mid)
        _mark_read(service, gmail_msg_id)
        return

    # Build thread context and decide action
    thread_bundle = _get_thread_bundle(service, thread_id)
    thread_text = summarize_thread(thread_bundle)

    extract = call_llm_extract(thread_text)
    ref_dt = received_local_dt(msg)
    plan = decide_action(extract, ref_dt)

    action = (plan.get("action") or "skip").lower()
    conf = float(plan.get("confidence", 0.0) or 0.0)

    name = ""
    if isinstance(extract, dict):
        name = (extract.get("name") or "").strip()
    if not name:
        name = (from_email.split("@")[0] if from_email else "Guest")

    phone = None
    if isinstance(extract, dict):
        phone = (extract.get("phone") or "").strip() or None
    if not phone:
        phone = extract_phone_best_effort(body)

    wants_update = detect_update_request(subj, body)

    print(f"[EMAIL] from={from_email} action={action} conf={conf:.2f} update={wants_update}")
    _status_set(conn, "agent_last_email_from", from_email or "")
    _status_set(conn, "agent_last_subject", subj[:140])
    _status_set(conn, "agent_last_action", action)

    # 1) Reservation confirm/update
    if action == "confirm":
        cancelled_old = None
        if wants_update:
            cancelled_old = cancel_latest_reservation(conn, from_email)

        ok, code, reason = reserve(
            conn,
            name=name,
            email=from_email,
            phone=phone,
            party_size=plan.get("party_size"),
            date_iso=plan.get("date_iso"),
            time_24=plan.get("time_24"),
            source="email",
            capacity=CAPACITY,
        )

        if not ok:
            alternatives = next_available_slots(conn, plan.get("date_iso"), capacity=CAPACITY, limit=3)
            body_text = _tpl_slot_full(name, plan.get("date_iso"), plan.get("time_24"), alternatives)
            _send(service, from_email, f"Re: {subj} — Time Unavailable", body_text, in_reply_to=mid, thread_id=thread_id)
            _mark_processed(conn, mid, "slot_full")
            _mark_read(service, gmail_msg_id)
            return

        body_text = _tpl_confirm(
            name,
            plan.get("date_iso"),
            plan.get("time_24"),
            str(plan.get("party_size")),
            confirmation_code=code,
            updated=bool(wants_update and cancelled_old),
        )
        suffix = "— Reservation Updated" if (wants_update and cancelled_old) else "— Reservation Confirmed"
        _send(service, from_email, f"Re: {subj} {suffix}", body_text, in_reply_to=mid, thread_id=thread_id)

        _mark_processed(conn, mid, "update_confirm" if (wants_update and cancelled_old) else "confirm")
        _mark_read(service, gmail_msg_id)
        return

    # 2) Ask missing details
    if action == "ask_missing":
        hd = bool(plan.get("date_iso"))
        ht = bool(plan.get("time_24"))
        hp = bool(plan.get("party_size"))
        _send(service, from_email, f"Re: {subj} — One quick detail", _tpl_missing(name, hd, ht, hp), in_reply_to=mid, thread_id=thread_id)
        _mark_processed(conn, mid, "ask_missing")
        _mark_read(service, gmail_msg_id)
        return

    # 3) Feedback
    if action == "feedback":
        sentiment, score = analyze_sentiment_simple(body)
        discount = choose_discount(sentiment, score)
        prefix = "CARE" if sentiment == "negative" else "THANKS"
        code = _random_code(prefix, discount)

        persist_coupon(conn, from_email, code, discount, sentiment, score)
        reply_body = generate_feedback_reply(name, sentiment, discount, code)

        _send(service, from_email, f"Re: {subj}", reply_body, in_reply_to=mid, thread_id=thread_id)
        log_feedback(conn, from_email, sentiment, score, discount, code, body, reply_body)

        _mark_processed(conn, mid, "feedback")
        _mark_read(service, gmail_msg_id)
        return

    # 4) Skip
    print("[SKIP other]", from_email)
    _mark_processed(conn, mid, "skip")
    _mark_read(service, gmail_msg_id)


def main():
    print("✅ agent starting...")
    print("✅ DATABASE_URL set:", bool(os.getenv("DATABASE_URL")))
    print("✅ token.json exists:", os.path.exists("token.json"))
    print("✅ client_secret.json exists:", os.path.exists("client_secret.json"))

    service = get_gmail_service()
    try:
        profile = service.users().getProfile(userId="me").execute()
        print("✅ Gmail account:", profile.get("emailAddress"))
    except Exception as e:
        print("[WARN] Could not fetch Gmail profile:", e)

    conn = _db()
    _status_set(conn, "agent_last_start", time.strftime("%Y-%m-%d %H:%M:%S"))

    messages = _fetch_unseen(service)
    print(f"[INFO] fetched {len(messages)} unread messages (INBOX + {os.getenv('GMAIL_QUERY','is:unread')})")
    _status_set(conn, "agent_last_fetched", str(len(messages)))

    for gmail_msg_id, thread_id, raw_bytes in messages:
        try:
            handle(service, conn, raw_bytes, thread_id, gmail_msg_id)
        except Exception as e:
            print("[ERR]", repr(e))

    _status_set(conn, "agent_last_run", time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
