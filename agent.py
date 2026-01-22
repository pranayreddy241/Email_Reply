#!/usr/bin/env python3
"""
Restaurant Email Agent (Gmail API / OAuth + LLM) — Postgres version

What it does:
- Fetch unread messages via Gmail API
- Use llm_agent.py to decide action: confirm / ask_missing / feedback / skip
- Auto-reply reservations (capacity-aware + stores reservations)
- Reservation update: if user asks to change/update/reschedule, cancel latest confirmed reservation and book new one
- Phone capture: from LLM extract if present else regex from email body
- Auto-reply feedback with sentiment + coupon + personalized reply
- Prevent double-processing using Postgres `processed` table (Message-ID)
- Log feedback emails + replies in `feedback_log`
- Mark processed emails as READ in Gmail (remove UNREAD label)

Requires:
- client_secret.json (OAuth)
- token.json (auto-generated)
- .env with OPENAI_API_KEY, etc.
- DATABASE_URL set for Postgres (Render will provide this automatically)
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

# Postgres connection + schema
from db_pg import get_pg_conn, ensure_schema_pg

# Reservation + business logic helpers (Postgres-safe inside these functions)
from db_utils import reserve, next_available_slots, cancel_latest_reservation, DEFAULT_CAPACITY

# LLM helper module
from llm_agent import summarize_thread, call_llm_extract, decide_action, received_local_dt

load_dotenv()

# ----------------- Config -----------------
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant")
RESERVATION_PHONE = os.getenv("RESERVATION_PHONE", "")
RESERVATION_LINK = os.getenv("RESERVATION_LINK", "")
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


# ----------------- Helpers -----------------
def _is_no_reply(addr, msg):
    """Filter obvious marketing / no-reply senders."""
    a = (addr or "").lower()
    if any(x in a for x in ["no-reply", "noreply", "notifications", "mailer-daemon"]):
        return True
    if msg.get("List-Unsubscribe"):
        return True
    return False


def _bootstrap_token_json():
    """
    Optional: if you store token JSON in env var (useful for Render),
    set GMAIL_TOKEN_JSON to the full token json string.
    """
    token_env = os.getenv("GMAIL_TOKEN_JSON")
    if token_env and not os.path.exists("token.json"):
        with open("token.json", "w") as f:
            f.write(token_env)


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


def _execute_with_retries(req, retries=5, base_delay=1.0, cap=30.0, what="request"):
    """Generic Gmail API call with exponential backoff."""
    for attempt in range(retries):
        try:
            return req.execute()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in (500, 502, 503, 504):
                delay = min(cap, base_delay * (2 ** attempt) + random.random())
                print(f"[RETRY] {what} failed with {status}, retrying in {delay:.1f}s (attempt {attempt+1}/{retries})")
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            delay = min(cap, base_delay * (2 ** attempt) + random.random())
            print(f"[RETRY] {what} exception {e!r}, retrying in {delay:.1f}s (attempt {attempt+1}/{retries})")
            time.sleep(delay)
            continue
    raise RuntimeError(f"{what} failed after {retries} retries")


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


# ----------------- Templates -----------------
def _tpl_confirm(name, d, t, p, confirmation_code=None, updated=False):
    header = "Your reservation is updated ✅" if updated else "Your reservation is confirmed ✅"
    L = [
        f"Hi{(' ' + name) if name else ''},",
        "",
        f"{header} at {RESTAURANT_NAME}.",
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


# ----------------- Update detection + phone extraction -----------------
def detect_update_request(subject: str, body: str) -> bool:
    text = f"{subject}\n{body}".lower()
    keywords = [
        "update my reservation", "change my reservation", "modify my reservation", "reschedule",
        "change the time", "change the date", "move the reservation", "can we move",
        "can we change", "need to change", "instead of", "correction", "correct my reservation",
        "update reservation", "change reservation",
    ]
    return any(k in text for k in keywords)


def extract_phone_best_effort(text: str):
    """
    Extract a phone-like string from email body.
    Requires at least 9 digits to reduce false matches.
    """
    if not text:
        return None
    m = re.search(r"(\+?\d[\d\-\(\)\s]{8,}\d)", text)
    if not m:
        return None
    raw = m.group(1)
    cleaned = re.sub(r"\s+", " ", raw).strip()
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) < 9:
        return None
    return cleaned


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
    with conn.cursor() as c:
        c.execute(
            """
            INSERT INTO coupons(email, code, discount, sentiment, score)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (code) DO NOTHING
            """,
            (email_addr, code, discount, sentiment, score),
        )
    conn.commit()


def log_feedback(conn, email_addr: str, sentiment: str, score: int, discount: int,
                 code: str, original_text: str, reply_text: str):
    with conn.cursor() as c:
        c.execute(
            """
            INSERT INTO feedback_log(email, sentiment, score, discount, code, original_text, reply_text)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            (email_addr, sentiment, score, discount, code, original_text, reply_text),
        )
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


# ----------------- DB -----------------
def _db():
    conn = get_pg_conn()
    ensure_schema_pg(conn)
    return conn


def _processed_has(conn, message_id: str) -> bool:
    if not message_id:
        return False
    with conn.cursor() as c:
        c.execute("SELECT 1 FROM processed WHERE message_id=%s", (message_id,))
        return c.fetchone() is not None


def _processed_put(conn, message_id: str, action: str):
    if not message_id:
        return
    with conn.cursor() as c:
        c.execute(
            """
            INSERT INTO processed(message_id, action, processed_at)
            VALUES (%s,%s,NOW())
            ON CONFLICT (message_id)
            DO UPDATE SET action=EXCLUDED.action, processed_at=NOW()
            """,
            (message_id, action),
        )
    conn.commit()


# ----------------- Thread + fetch helpers (KEEPING YOUR OLD STYLE) -----------------
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
    Your original robust fetch:
    Return list of tuples: (msg_id, thread_id, raw_bytes) for unread messages.
    Uses robust retries + multiple strategies.
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


# ----------------- Main handler -----------------
def handle(service, conn, raw, thread_id, msg_id):
    msg = email.message_from_bytes(raw)
    mid = msg.get("Message-ID")

    # Always skip if already processed to prevent loops
    if mid and _processed_has(conn, mid):
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

    name = (extract.get("name") or "").strip() if isinstance(extract, dict) else ""
    if not name:
        name = from_email.split("@")[0]

    # phone: from LLM extract if present else regex best-effort
    phone = None
    if isinstance(extract, dict):
        phone = (extract.get("phone") or "").strip() or None
    if not phone:
        phone = extract_phone_best_effort(body)

    wants_update = detect_update_request(subj, body)
    action = plan.get("action")

    # 1) Confirm reservation (capacity-aware) + update support
    if action == "confirm":
        old = None
        if wants_update:
            # cancel their latest confirmed reservation before booking new one
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

            _processed_put(conn, mid, "slot_full")
            _mark_read(service, msg_id)
            return

        updated = bool(wants_update and old)
        body_text = _tpl_confirm(
            name,
            plan.get("date_iso"),
            plan.get("time_24"),
            str(plan.get("party_size")),
            confirmation_code=code,
            updated=updated,
        )

        subject_suffix = "— Reservation Updated" if updated else "— Reservation Confirmed"
        _send(
            service,
            from_email,
            f"Re: {subj} {subject_suffix}",
            body_text,
            in_reply_to=mid,
            thread_id=thread_id,
        )
        print("[SENT] confirm ->", from_email)

        _processed_put(conn, mid, "update_confirm" if updated else "confirm")
        _mark_read(service, msg_id)
        return

    # 2) Ask missing details
    if action == "ask_missing":
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

        _processed_put(conn, mid, "ask_missing")
        _mark_read(service, msg_id)
        return

    # 3) Feedback (auto-reply with coupon)
    if action == "feedback":
        print("[INFO] Feedback detected → Sending coupon reply")

        sentiment, score = analyze_sentiment_with_backoff(body)
        discount = choose_discount(sentiment, score, body)

        prefix = "CARE" if sentiment == "negative" else "THANKS"
        code = _random_code(prefix, discount)
        persist_coupon(conn, from_email, code, discount, sentiment, score)

        reply_body = generate_personalized_reply(name, sentiment, score, discount, code, body)

        _send(service, from_email, f"Re: {subj}", reply_body, in_reply_to=mid, thread_id=thread_id)
        print(f"[SENT feedback] {sentiment}/{score} -> {discount}% code={code} to {from_email}")

        log_feedback(conn, from_email, sentiment, score, discount, code, body, reply_body)

        _processed_put(conn, mid, "feedback")
        _mark_read(service, msg_id)
        return

    # 4) Skip other
    print("[SKIP other] ->", from_email)
    _processed_put(conn, mid, "skip")
    _mark_read(service, msg_id)


def main():
    print("✅ agent starting...")
    print("✅ DATABASE_URL set:", bool(os.getenv("DATABASE_URL")))
    print("✅ token.json exists:", os.path.exists("token.json"))
    print("✅ client_secret.json exists:", os.path.exists("client_secret.json"))

    service = get_gmail_service()
    prof = service.users().getProfile(userId="me").execute()
    print("✅ Gmail account:", prof.get("emailAddress"))

    conn = _db()

    messages = _fetch_unseen(service)  # list of (msg_id, thread_id, raw_bytes)
    print(f"[INFO] fetched {len(messages)} unread messages")

    for msg_id, thread_id, raw in messages:
        try:
            handle(service, conn, raw, thread_id, msg_id)
        except Exception as e:
            print("[ERR]", e)


if __name__ == "__main__":
    main()
