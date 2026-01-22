#!/usr/bin/env python3
"""
Agent (Postgres) + Gmail + LLM
Adds:
- Reservation updates: if user asks to change/update/reschedule, we cancel their latest confirmed res and book new one
- Phone capture from email body (best-effort)
"""

from __future__ import print_function

import os
import re
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

from db_pg import get_pg_conn, ensure_schema_pg
from db_utils import reserve, next_available_slots, cancel_latest_reservation, DEFAULT_CAPACITY

from llm_agent import summarize_thread, call_llm_extract, decide_action, received_local_dt

load_dotenv()

RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant")
RESERVATION_PHONE = os.getenv("RESERVATION_PHONE", "")
RESERVATION_LINK = os.getenv("RESERVATION_LINK", "")
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# ----------------- helpers -----------------
def _is_no_reply(addr, msg):
    a = (addr or "").lower()
    if any(x in a for x in ["no-reply", "noreply", "notifications", "mailer-daemon"]):
        return True
    if msg.get("List-Unsubscribe"):
        return True
    return False

def _bootstrap_token_json():
    token_env = os.getenv("GMAIL_TOKEN_JSON")
    if token_env and not os.path.exists("token.json"):
        with open("token.json", "w") as f:
            f.write(token_env)

def get_gmail_service():
    _bootstrap_token_json()

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
    try:
        service.users().messages().modify(
            userId="me",
            id=msg_id,
            body={"removeLabelIds": ["UNREAD"]}
        ).execute()
    except Exception as e:
        print("[WARN] failed to mark as read:", e)

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
            (key, value)
        )
    conn.commit()

def detect_update_request(subject: str, body: str) -> bool:
    """
    Lightweight heuristic: treat as update if they mention change/update/reschedule/correct etc.
    (LLM can later classify this explicitly.)
    """
    text = f"{subject}\n{body}".lower()
    keywords = [
        "update my reservation", "change my reservation", "modify my reservation", "reschedule",
        "change the time", "change the date", "move the reservation", "can we move",
        "can we change", "need to change", "instead of", "correction", "correct my reservation"
    ]
    return any(k in text for k in keywords)

def extract_phone_best_effort(text: str) -> str | None:
    """
    Extract a phone-like string from email body.
    Supports US-ish patterns and general digits.
    """
    if not text:
        return None
    # common patterns
    m = re.search(r"(\+?\d[\d\-\(\)\s]{8,}\d)", text)
    if not m:
        return None
    raw = m.group(1)
    # normalize spaces
    cleaned = re.sub(r"\s+", " ", raw).strip()
    # avoid matching order numbers etc by requiring at least 9 digits
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) < 9:
        return None
    return cleaned

def _get_thread_bundle(service, thread_id):
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

def _fetch_unseen(service):
    max_process = int(os.getenv("MAX_PROCESS", "10"))
    resp = service.users().messages().list(
        userId="me",
        q=os.getenv("GMAIL_QUERY", "is:unread"),
        labelIds=["INBOX"],
        maxResults=max_process
    ).execute()

    out = []
    for m in resp.get("messages", []):
        msg_id = m["id"]
        thread_id = m.get("threadId")
        raw_resp = service.users().messages().get(userId="me", id=msg_id, format="raw").execute()
        raw_bytes = base64.urlsafe_b64decode(raw_resp["raw"])
        out.append((msg_id, thread_id, raw_bytes))
    return out

# ----------------- templates -----------------
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
        RESTAURANT_NAME
    ]
    return "\n".join([x for x in L if x])

# ----------------- sentiment + coupons (same as before) -----------------
def analyze_sentiment_with_backoff(message_text: str):
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
                        "Return strict JSON: {\"sentiment\":\"positive|neutral|negative\",\"score\":1-5}.\n"
                        "1 very happy, 3 neutral, 5 extremely upset.\n\n"
                        f"Message:\n{message_text}"
                    )
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=0.2,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=80
                    )
                    txt = resp.choices[0].message.content.strip()
                    obj = __import__("json").loads(txt)
                    s = str(obj.get("sentiment", "neutral")).lower()
                    sc = int(obj.get("score", 3))
                    sc = max(1, min(5, sc))
                    if s not in {"positive", "neutral", "negative"}:
                        s = "neutral"
                    return s, sc
                except Exception:
                    time.sleep(2 ** attempt)

    t = (message_text or "").lower()
    neg_hits = sum(w in t for w in ["awful","terrible","horrible","disgusting","cold","late","rude","bad","worst","refund","angry","disappointed"])
    pos_hits = sum(w in t for w in ["amazing","great","excellent","love","loved","fantastic","wonderful","perfect","delicious","best"])
    if neg_hits >= 3: return "negative", 5
    if neg_hits == 2: return "negative", 4
    if neg_hits == 1: return "negative", 3
    if pos_hits >= 2: return "positive", 1
    if pos_hits == 1: return "positive", 2
    return "neutral", 3

def choose_discount(sentiment: str, score: int, message_text: str) -> int:
    text = (message_text or "").lower()
    if sentiment == "positive":
        enthusiastic = any(w in text for w in ["love","loved","amazing","incredible","fantastic","perfect","best"])
        return 10 if enthusiastic else 5
    if sentiment == "neutral":
        return 15
    mapping = {3: 15, 4: 25, 5: 30}
    disc = mapping.get(score, 15)
    if score == 5 and any(w in text for w in ["worst","never again","refund","disgusting","unacceptable","furious"]):
        disc = 40
    return min(disc, 40)

def _random_code(prefix: str, pct: int) -> str:
    tail = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"{prefix}{pct}-{tail}"

def persist_coupon(conn, email_addr: str, code: str, discount: int, sentiment: str, score: int):
    with conn.cursor() as c:
        c.execute(
            """INSERT INTO coupons(email, code, discount, sentiment, score)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (code) DO NOTHING""",
            (email_addr, code, discount, sentiment, score)
        )
    conn.commit()

def log_feedback(conn, email_addr: str, sentiment: str, score: int, discount: int,
                 code: str, original_text: str, reply_text: str):
    with conn.cursor() as c:
        c.execute("""
            INSERT INTO feedback_log(email, sentiment, score, discount, code, original_text, reply_text)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (email_addr, sentiment, score, discount, code, original_text, reply_text))
    conn.commit()

def generate_personalized_reply(name: str, sentiment: str, score: int,
                                discount: int, code: str, message_text: str) -> str:
    # keep simple fallback (you can re-add full OpenAI writer later)
    if sentiment == "positive":
        return (
            f"Hi{(' ' + name) if name else ''},\n\n"
            f"Thank you for the kind note! As a small thank-you, here’s {discount}% off next time (code: {code}).\n\n"
            f"{RESTAURANT_NAME}"
        )
    if sentiment == "neutral":
        return (
            f"Hi{(' ' + name) if name else ''},\n\n"
            f"Thanks for sharing this—your feedback helps us improve. Here’s {discount}% off your next visit (code: {code}).\n\n"
            f"{RESTAURANT_NAME}"
        )
    opener = "We’re truly sorry" if score >= 4 else "We’re sorry"
    return (
        f"Hi{(' ' + name) if name else ''},\n\n"
        f"{opener} your experience fell short. We’d like to make it right—here’s {discount}% off (code: {code}).\n\n"
        f"{RESTAURANT_NAME}"
    )

# ----------------- main email handler -----------------
def handle(service, conn, raw, thread_id, msg_id):
    msg = email.message_from_bytes(raw)
    mid = msg.get("Message-ID")
    subj = _dec(msg.get("Subject", ""))
    body = _body(msg)
    from_email = email.utils.parseaddr(msg.get("From", ""))[1]

    if _is_no_reply(from_email, msg):
        _mark_read(service, msg_id)
        return

    # dedupe
    if mid:
        with conn.cursor() as c:
            c.execute("SELECT 1 FROM processed WHERE message_id=%s", (mid,))
            if c.fetchone():
                return

    thread_bundle = _get_thread_bundle(service, thread_id)
    thread_text = summarize_thread(thread_bundle)

    extract = call_llm_extract(thread_text)
    ref_dt = received_local_dt(msg)
    plan = decide_action(extract, ref_dt)

    name = (extract.get("name") or "").strip() or (from_email.split("@")[0])

    # phone: from LLM extract if present else regex
    phone = (extract.get("phone") or "").strip() if isinstance(extract, dict) else ""
    if not phone:
        phone = extract_phone_best_effort(body) or None

    # detect update request
    wants_update = detect_update_request(subj, body)

    action = plan.get("action")

    _status_set(conn, "last_subject", subj)
    _status_set(conn, "last_email_from", from_email)
    _status_set(conn, "agent_last_seen_action", str(action))

    # 1) Reservation confirm/update
    if action == "confirm":
        # if update requested, cancel previous confirmed reservation first
        old = None
        if wants_update:
            old = cancel_latest_reservation(conn, from_email)

        ok, code, reason = reserve(
            conn,
            name=name,
            email=from_email,
            phone=phone,
            party_size=plan.get("party_size"),
            date_iso=plan.get("date_iso"),
            time_24=plan.get("time_24"),
            source="email",
            capacity=DEFAULT_CAPACITY
        )

        if not ok:
            # if we cancelled old but new booking failed, you may want to restore old.
            # For now: we’ll just suggest alternatives.
            alternatives = next_available_slots(conn, plan.get("date_iso"), capacity=DEFAULT_CAPACITY, limit=3)
            body_text = _tpl_slot_full(name, plan.get("date_iso"), plan.get("time_24"), alternatives)
            _send(service, from_email, f"Re: {subj} — Time Unavailable", body_text, in_reply_to=mid, thread_id=thread_id)

            if mid:
                with conn.cursor() as c:
                    c.execute(
                        """INSERT INTO processed(message_id, action, processed_at)
                           VALUES (%s,%s,NOW())
                           ON CONFLICT (message_id) DO UPDATE SET action=EXCLUDED.action, processed_at=NOW()""",
                        (mid, "slot_full")
                    )
                conn.commit()

            _mark_read(service, msg_id)
            return

        body_text = _tpl_confirm(
            name,
            plan.get("date_iso"),
            plan.get("time_24"),
            str(plan.get("party_size")),
            confirmation_code=code,
            updated=bool(wants_update and old)
        )

        subject_suffix = "— Reservation Updated" if (wants_update and old) else "— Reservation Confirmed"
        _send(service, from_email, f"Re: {subj} {subject_suffix}", body_text, in_reply_to=mid, thread_id=thread_id)

        if mid:
            with conn.cursor() as c:
                c.execute(
                    """INSERT INTO processed(message_id, action, processed_at)
                       VALUES (%s,%s,NOW())
                       ON CONFLICT (message_id) DO UPDATE SET action=EXCLUDED.action, processed_at=NOW()""",
                    (mid, "update_confirm" if (wants_update and old) else "confirm")
                )
            conn.commit()

        _mark_read(service, msg_id)
        return

    # 2) Ask missing details
    if action == "ask_missing":
        hd = bool(plan.get("date_iso"))
        ht = bool(plan.get("time_24"))
        hp = bool(plan.get("party_size"))

        _send(service, from_email, f"Re: {subj} — One quick detail", _tpl_missing(name, hd, ht, hp),
              in_reply_to=mid, thread_id=thread_id)

        if mid:
            with conn.cursor() as c:
                c.execute(
                    """INSERT INTO processed(message_id, action, processed_at)
                       VALUES (%s,%s,NOW())
                       ON CONFLICT (message_id) DO UPDATE SET action=EXCLUDED.action, processed_at=NOW()""",
                    (mid, "ask_missing")
                )
            conn.commit()

        _mark_read(service, msg_id)
        return

    # 3) Feedback
    if action == "feedback":
        sentiment, score = analyze_sentiment_with_backoff(body)
        discount = choose_discount(sentiment, score, body)
        prefix = "CARE" if sentiment == "negative" else "THANKS"
        code = _random_code(prefix, discount)
        persist_coupon(conn, from_email, code, discount, sentiment, score)

        reply_body = generate_personalized_reply(name, sentiment, score, discount, code, body)
        _send(service, from_email, f"Re: {subj}", reply_body, in_reply_to=mid, thread_id=thread_id)
        log_feedback(conn, from_email, sentiment, score, discount, code, body, reply_body)

        if mid:
            with conn.cursor() as c:
                c.execute(
                    """INSERT INTO processed(message_id, action, processed_at)
                       VALUES (%s,%s,NOW())
                       ON CONFLICT (message_id) DO UPDATE SET action=EXCLUDED.action, processed_at=NOW()""",
                    (mid, "feedback")
                )
            conn.commit()

        _mark_read(service, msg_id)
        return

    # 4) skip
    if mid:
        with conn.cursor() as c:
            c.execute(
                """INSERT INTO processed(message_id, action, processed_at)
                   VALUES (%s,%s,NOW())
                   ON CONFLICT (message_id) DO UPDATE SET action=EXCLUDED.action, processed_at=NOW()""",
                (mid, "skip")
            )
        conn.commit()

    _mark_read(service, msg_id)

def main():
    service = get_gmail_service()
    conn = _db()
    _status_set(conn, "agent_last_start", time.strftime("%Y-%m-%d %H:%M:%S"))

    import sys
    if "--send-pending" in sys.argv:
        # not used here, keep simple
        return

    messages = _fetch_unseen(service)
    _status_set(conn, "agent_last_fetched", str(len(messages)))

    for msg_id, thread_id, raw in messages:
        try:
            handle(service, conn, raw, thread_id, msg_id)
        except Exception as e:
            print("[ERR]", e)

    _status_set(conn, "agent_last_run", time.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
