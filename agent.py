#!/usr/bin/env python3
"""
Restaurant Email Agent (Gmail API / OAuth + LLM)
- Fetch unread messages via Gmail API
- Use llm_agent.py to decide action: confirm / ask_missing / feedback / skip
- Auto-reply reservations (now capacity-aware + stores reservations)
- Auto-reply feedback with sentiment + coupon + personalized reply
- Prevent double-processing using SQLite `processed` table (Message-ID)
- Log feedback emails + replies in `feedback_log`
- Mark processed emails as READ in Gmail (remove UNREAD label)

Usage:
  python agent.py
  python agent.py --send-pending

Requires:
  client_secret.json (OAuth)
  token.json (auto-generated)
  .env with OPENAI_API_KEY, etc.
"""

from __future__ import print_function

import os
import re
import sqlite3
import email
import base64
import time
import random
import string
from email.message import EmailMessage
from email.header import decode_header, make_header
from dotenv import load_dotenv

from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# NEW: Reservations + schema helpers
from db_utils import ensure_schema, reserve, next_available_slots, DEFAULT_CAPACITY

# LLM helper module (your file)
from llm_agent import summarize_thread, call_llm_extract, decide_action, received_local_dt

load_dotenv()

# ----------------- Config -----------------
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant")
RESERVATION_PHONE = os.getenv("RESERVATION_PHONE", "")
RESERVATION_LINK = os.getenv("RESERVATION_LINK", "")
DB_PATH = os.getenv("AGENT_DB_PATH", "email_agent.sqlite")

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


# ----------------- Gmail helpers -----------------
def _is_no_reply(addr, msg):
    """Filter obvious marketing / no-reply senders."""
    a = (addr or "").lower()
    if any(x in a for x in ["no-reply", "noreply", "notifications", "mailer-daemon"]):
        return True
    if msg.get("List-Unsubscribe"):
        return True
    return False


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


# ----------------- Parsing helpers -----------------
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
                text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
                return re.sub(r"<[^>]+>", " ", text)
    else:
        cs = msg.get_content_charset() or "utf-8"
        raw = msg.get_payload(decode=True)
        if raw is None:
            return msg.get_payload()
        text = raw.decode(cs, errors="replace")
        if msg.get_content_type() == "text/html":
            text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
            text = re.sub(r"<[^>]+>", " ", text)
        return text
    return ""


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


def _tpl_confirm(name, d, t, p, confirmation_code=None):
    L = [
        f"Hi{(' ' + name) if name else ''},",
        "",
        f"Your reservation is confirmed at {RESTAURANT_NAME}.",
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
        f"Thanks for booking at {RESTAURANT_NAME}.",
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


# ----------------- Feedback helpers (sentiment + coupons) -----------------
def analyze_sentiment_with_backoff(message_text: str):
    """
    Returns (sentiment:str, score:int [1..5]).
    sentiment in {'positive','neutral','negative'}; score 1=very happy .. 5=furious.
    Uses OpenAI with retry, then heuristic fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except Exception:
            client = None

        if client:
            for attempt in range(4):
                try:
                    prompt = (
                        "Rate the customer's tone.\n"
                        "Return strict JSON: {\"sentiment\":\"positive|neutral|negative\",\"score\":1-5}.\n"
                        "Guidelines: 1 very happy, 3 neutral, 5 extremely upset.\n\n"
                        f"Message:\n{message_text}"
                    )
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=0.2,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=60,
                    )
                    txt = resp.choices[0].message.content.strip()
                    obj = __import__("json").loads(txt)
                    s = str(obj.get("sentiment", "neutral")).lower()
                    sc = int(obj.get("score", 3))
                    sc = max(1, min(5, sc))
                    if s not in {"positive", "neutral", "negative"}:
                        s = "neutral"
                    return s, sc
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"[WARN] OpenAI sentiment attempt {attempt+1} failed: {e} (retry {wait}s)")
                    time.sleep(wait)

    # Heuristic fallback
    t = (message_text or "").lower()
    neg_hits = sum(w in t for w in [
        "awful", "terrible", "horrible", "disgusting", "cold", "late",
        "rude", "bad", "worst", "never again", "refund", "angry", "disappointed"
    ])
    pos_hits = sum(w in t for w in [
        "amazing", "great", "excellent", "love", "loved", "fantastic",
        "wonderful", "perfect", "delicious", "best"
    ])
    if neg_hits >= 3:
        return "negative", 5
    if neg_hits == 2:
        return "negative", 4
    if neg_hits == 1:
        return "negative", 3
    if pos_hits >= 2:
        return "positive", 1
    if pos_hits == 1:
        return "positive", 2
    return "neutral", 3


def choose_discount(sentiment: str, score: int, message_text: str) -> int:
    text = (message_text or "").lower()
    if sentiment == "positive":
        enthusiastic = any(w in text for w in ["love", "loved", "amazing", "incredible", "fantastic", "perfect", "best"])
        return 10 if enthusiastic else 5
    if sentiment == "neutral":
        return 15
    mapping = {3: 15, 4: 25, 5: 30}
    disc = mapping.get(score, 15)
    if score == 5 and any(w in text for w in ["worst", "never again", "refund", "disgusting", "unacceptable", "furious"]):
        disc = 40
    return min(disc, 40)


def _random_code(prefix: str, pct: int) -> str:
    tail = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"{prefix}{pct}-{tail}"


def persist_coupon(conn, email_addr: str, code: str, discount: int, sentiment: str, score: int):
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO coupons(email, code, discount, sentiment, score) VALUES (?,?,?,?,?)",
        (email_addr, code, discount, sentiment, score),
    )
    conn.commit()


def log_feedback(conn, email_addr: str, sentiment: str, score: int, discount: int,
                 code: str, original_text: str, reply_text: str):
    c = conn.cursor()
    c.execute("""
        INSERT INTO feedback_log(email, sentiment, score, discount, code, original_text, reply_text)
        VALUES (?,?,?,?,?,?,?)
    """, (email_addr, sentiment, score, discount, code, original_text, reply_text))
    conn.commit()


def generate_personalized_reply(name: str, sentiment: str, score: int,
                                discount: int, code: str, message_text: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except Exception:
            client = None

        if client:
            for attempt in range(4):
                try:
                    sys = (
                        "You are a warm, concise customer service writer for a restaurant. "
                        "Write a SHORT, sincere, human reply (80–120 words). "
                        "Personalize tone to the customer's sentiment and upset score. "
                        "Always include the exact coupon code and % once. "
                        "Avoid generic boilerplates; vary the phrasing. "
                        "Sign off with the restaurant name only."
                    )
                    user = (
                        f"Restaurant: {RESTAURANT_NAME}\n"
                        f"Customer name: {name or 'Guest'}\n"
                        f"Sentiment: {sentiment}\n"
                        f"Upset score (1 very happy .. 5 furious): {score}\n"
                        f"Discount: {discount}%\n"
                        f"Coupon code: {code}\n"
                        f"Customer message:\n{message_text}\n"
                        "Write the reply body only."
                    )
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=0.7,
                        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                        max_tokens=240,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"[WARN] OpenAI reply attempt {attempt+1} failed: {e} (retry {wait}s)")
                    time.sleep(wait)

    # Fallback templates
    if sentiment == "positive":
        return (
            f"Hi{(' ' + name) if name else ''},\n\n"
            f"Thank you for the wonderful note—guests like you make our day. "
            f"As a small thank-you, here’s {discount}% off next time (code: {code}). "
            f"We can’t wait to welcome you back.\n\n{RESTAURANT_NAME}"
        )
    if sentiment == "neutral":
        return (
            f"Hi{(' ' + name) if name else ''},\n\n"
            f"Thanks for sharing your thoughts—your feedback helps us improve. "
            f"Please accept {discount}% off your next visit (code: {code}); we’d love another chance to impress.\n\n"
            f"{RESTAURANT_NAME}"
        )
    opener = "We’re truly sorry" if score >= 4 else "We’re sorry"
    return (
        f"Hi{(' ' + name) if name else ''},\n\n"
        f"{opener} that your experience fell short. You matter to us, and we’ve noted your concerns with the team. "
        f"Please allow us to make it right—here’s {discount}% off for your next visit (code: {code}). "
        f"We appreciate the chance to earn back your trust.\n\n{RESTAURANT_NAME}"
    )


# ----------------- DB (with migrations) -----------------
def _db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # processed table (needs action)
    c.execute("""
        CREATE TABLE IF NOT EXISTS processed(
            message_id TEXT PRIMARY KEY,
            processed_at TEXT
        )
    """)
    c.execute("PRAGMA table_info(processed)")
    cols = [row[1] for row in c.fetchall()]
    if "action" not in cols:
        c.execute("ALTER TABLE processed ADD COLUMN action TEXT")

    # drafts table
    c.execute("""
        CREATE TABLE IF NOT EXISTS drafts(
            id INTEGER PRIMARY KEY,
            to_email TEXT,
            subject TEXT,
            body TEXT,
            in_reply_to TEXT,
            created_at TEXT,
            sent_at TEXT
        )
    """)

    # coupons table
    c.execute("""
        CREATE TABLE IF NOT EXISTS coupons(
            id INTEGER PRIMARY KEY,
            email TEXT,
            code TEXT UNIQUE,
            discount INTEGER,
            sentiment TEXT,
            score INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # feedback log table
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback_log(
            id INTEGER PRIMARY KEY,
            email TEXT,
            sentiment TEXT,
            score INTEGER,
            discount INTEGER,
            code TEXT,
            original_text TEXT,
            reply_text TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()

    # NEW: ensure reservations + review scaffolding
    ensure_schema(conn)

    return conn


# ----------------- Thread + fetch helpers -----------------
def _get_thread_bundle(service, thread_id):
    """
    Return a condensed list of dicts for the last few messages in a thread:
      {from, date, subject, body}
    """
    data = service.users().threads().get(userId="me", id=thread_id, format="full").execute()
    items = []
    for part in data.get("messages", [])[-4:]:
        payload = part.get("payload", {})
        headers = payload.get("headers", [])

        def hdr(name):
            for h in headers:
                if h.get("name", "").lower() == name.lower():
                    return h.get("value", "")
            return ""

        frm = hdr("From") or ""
        subj = hdr("Subject") or ""
        dt = hdr("Date") or ""
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
        items.append({"from": frm, "date": dt, "subject": subj, "body": body})
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
        resp = _execute_with_retries(service.users().messages().list(**params), what="messages.list")
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
        if skipped:
            print(f"[INFO] skipped {len(skipped)} message(s) due to transient errors.")
        return out

    max_process = int(os.getenv("MAX_PROCESS", "10"))

    msgs = _pull(query=os.getenv("GMAIL_QUERY", "is:unread"), labelIds=["INBOX"], limit=max_process)
    if msgs:
        print(f"[DEBUG] pass1 INBOX is:unread -> {len(msgs)}")
        return msgs

    msgs = _pull(query="is:unread", labelIds=None, limit=max_process)
    if msgs:
        print(f"[DEBUG] pass2 ANY is:unread -> {len(msgs)}")
        return msgs

    msgs = _pull(query=None, labelIds=["UNREAD"], limit=max_process)
    if msgs:
        print(f"[DEBUG] pass3 label:UNREAD -> {len(msgs)}")
        return msgs

    print("[DEBUG] no unread messages found by any strategy")
    return []


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


# ----------------- Main handler -----------------
def handle(service, conn, raw, thread_id, msg_id):
    msg = email.message_from_bytes(raw)
    mid = msg.get("Message-ID")
    c = conn.cursor()

    # Always skip if already processed to prevent loops
    if mid and c.execute("SELECT 1 FROM processed WHERE message_id=?", (mid,)).fetchone():
        return

    subj = _dec(msg.get("Subject", ""))
    body = _body(msg)
    from_email = email.utils.parseaddr(msg.get("From", ""))[1]

    if _is_no_reply(from_email, msg):
        print("[SKIP no-reply/marketing] ->", from_email)
        _mark_read(service, msg_id)
        return

    # Thread context + LLM decision
    thread_bundle = _get_thread_bundle(service, thread_id)
    thread_text = summarize_thread(thread_bundle)

    extract = call_llm_extract(thread_text)
    ref_dt = received_local_dt(msg)
    plan = decide_action(extract, ref_dt)

    print(
        f"[LLM] plan={plan.get('action')} conf={plan.get('confidence', 0):.2f} "
        f"date={plan.get('date_iso')} time={plan.get('time_24')} party={plan.get('party_size')}"
    )

    name = (extract.get("name") or "").strip() or (from_email.split("@")[0])

    # 1) Confirm reservation (capacity-aware)
    if plan.get("action") == "confirm":
        ok, code, reason = reserve(
            conn,
            name=name,
            email=from_email,
            phone=None,
            party_size=plan.get("party_size"),
            date_iso=plan.get("date_iso"),
            time_24=plan.get("time_24"),
            source="email",
            capacity=DEFAULT_CAPACITY,
        )

        if not ok:
            alternatives = next_available_slots(conn, plan.get("date_iso"), capacity=DEFAULT_CAPACITY, limit=3)
            body_text = _tpl_slot_full(name, plan.get("date_iso"), plan.get("time_24"), alternatives)

            _send(
                service,
                from_email,
                f"Re: {subj} — Time Unavailable",
                body_text,
                in_reply_to=mid,
                thread_id=thread_id,
            )
            print("[SENT] slot full ->", from_email)

            if mid:
                c.execute(
                    "INSERT OR REPLACE INTO processed(message_id, action, processed_at) VALUES (?,?,datetime('now'))",
                    (mid, "slot_full"),
                )
                conn.commit()

            _mark_read(service, msg_id)
            return

        body_text = _tpl_confirm(name, plan.get("date_iso"), plan.get("time_24"), str(plan.get("party_size")), code)

        _send(
            service,
            from_email,
            f"Re: {subj} — Reservation Confirmed",
            body_text,
            in_reply_to=mid,
            thread_id=thread_id,
        )
        print("[SENT] confirm ->", from_email)

        if mid:
            c.execute(
                "INSERT OR REPLACE INTO processed(message_id, action, processed_at) VALUES (?,?,datetime('now'))",
                (mid, "confirm"),
            )
            conn.commit()

        _mark_read(service, msg_id)
        return

    # 2) Ask missing details
    if plan.get("action") == "ask_missing":
        hd = bool(plan.get("date_iso"))
        ht = bool(plan.get("time_24"))
        hp = bool(plan.get("party_size"))
        _send(
            service,
            from_email,
            f"Re: {subj} — One quick detail",
            _tpl_missing(name, hd, ht, hp),
            in_reply_to=mid,
            thread_id=thread_id,
        )
        print("[SENT] missing ->", from_email)

        if mid:
            c.execute(
                "INSERT OR REPLACE INTO processed(message_id, action, processed_at) VALUES (?,?,datetime('now'))",
                (mid, "ask_missing"),
            )
            conn.commit()

        _mark_read(service, msg_id)
        return

    # 3) Feedback (auto-reply with coupon)
    if plan.get("action") == "feedback":
        print("[INFO] Feedback detected → Sending coupon reply")

        sentiment, score = analyze_sentiment_with_backoff(body)
        discount = choose_discount(sentiment, score, body)

        prefix = "CARE" if sentiment == "negative" else "THANKS"
        code = _random_code(prefix, discount)
        persist_coupon(conn, from_email, code, discount, sentiment, score)

        reply_body = generate_personalized_reply(name, sentiment, score, discount, code, body)

        _send(service, from_email, f"Re: {subj}", reply_body, in_reply_to=mid, thread_id=thread_id)
        print(f"[SENT feedback] {sentiment}/{score} -> {discount}% code={code} to {from_email}")

        # log feedback (for dashboard)
        log_feedback(conn, from_email, sentiment, score, discount, code, body, reply_body)

        # mark processed + read
        if mid:
            c.execute(
                "INSERT OR REPLACE INTO processed(message_id, action, processed_at) VALUES (?,?,datetime('now'))",
                (mid, "feedback"),
            )
            conn.commit()

        _mark_read(service, msg_id)
        return

    # 4) Skip other
    print("[SKIP other] ->", from_email)
    if mid:
        c.execute(
            "INSERT OR REPLACE INTO processed(message_id, action, processed_at) VALUES (?,?,datetime('now'))",
            (mid, "skip"),
        )
        conn.commit()

    _mark_read(service, msg_id)


def send_pending(service, conn):
    """Send any staged drafts in the SQLite DB."""
    c = conn.cursor()
    rows = c.execute(
        "SELECT id,to_email,subject,body,in_reply_to FROM drafts WHERE sent_at IS NULL ORDER BY id"
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
            handle(service, conn, raw, thread_id, msg_id)
        except Exception as e:
            print("[ERR]", e)


if __name__ == "__main__":
    main()
