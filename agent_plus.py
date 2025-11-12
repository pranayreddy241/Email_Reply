#!/usr/bin/env python3
"""
Enhanced Restaurant Email Agent
Adds Twilio alerts, Gmail draft creation, optional ML (joblib).
See README/guide for env vars.

import os, re, sqlite3, imaplib, smtplib, email
from email.message import EmailMessage
from email.header import decode_header, make_header
from email.utils import parseaddr, formatdate

EMAIL_ADDRESS=os.getenv("EMAIL_ADDRESS"); EMAIL_PASSWORD=os.getenv("EMAIL_PASSWORD")
IMAP_SERVER=os.getenv("IMAP_SERVER","imap.gmail.com"); SMTP_SERVER=os.getenv("SMTP_SERVER","smtp.gmail.com"); SMTP_PORT=int(os.getenv("SMTP_PORT","587"))
RESTAURANT_NAME=os.getenv("RESTAURANT_NAME","My Restaurant"); RESERVATION_PHONE=os.getenv("RESERVATION_PHONE",""); RESERVATION_LINK=os.getenv("RESERVATION_LINK","")
DB_PATH=os.getenv("AGENT_DB_PATH","email_agent.sqlite")
USE_TWILIO=os.getenv("USE_TWILIO_ALERTS","false").lower()=="true"
USE_GMAIL_DRAFTS=os.getenv("USE_GMAIL_DRAFTS","false").lower()=="true"
USE_ML=os.getenv("USE_ML_CLASSIFIER","false").lower()=="true"

notifier=None
if USE_TWILIO:
    try: import notifier as notifier
    except Exception as e: print("[WARN] Twilio disabled:", e); notifier=None

gd=None
if USE_GMAIL_DRAFTS:
    try: import gmail_drafts as gd
    except Exception as e: print("[WARN] Gmail drafts disabled:", e); gd=None

vec=clf=None
if USE_ML:
    try:
        import joblib
        vec=joblib.load("email_vectorizer.joblib"); clf=joblib.load("email_intent_model.joblib")
    except Exception as e:
        print("[WARN] ML disabled:", e); vec=clf=None; USE_ML=False

RESERVATION_KEYWORDS=[r"\breservation\b",r"\bbook(?:ing)?\b",r"\btable\b",r"\bparty\b"]
REVIEW_KEYWORDS=[r"\breview\b",r"\bfeedback\b",r"\bcomplaint\b",r"\bpraise\b",r"\bexperience\b"]
DATE_RE=r"(?:(?:on\s*)?(?P<date>(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?))"
TIME_RE=r"(?:(?:at\s*)?(?P<time>\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|\b)))"
PARTY_RE=r"(?:(?:for|party of)\s*(?P<party>\d{1,2}))"

def _dec(s): 
    if not s: return ""
    try: return str(make_header(decode_header(s)))
    except: return s

def _body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            ctype=part.get_content_type(); disp=str(part.get("Content-Disposition"))
            if ctype=="text/plain" and "attachment" not in disp:
                cs=part.get_content_charset() or "utf-8"; return part.get_payload(decode=True).decode(cs, errors="replace")
        for part in msg.walk():
            ctype=part.get_content_type(); disp=str(part.get("Content-Disposition"))
            if ctype=="text/html" and "attachment" not in disp:
                cs=part.get_content_charset() or "utf-8"
                html=part.get_payload(decode=True).decode(cs, errors="replace")
                import re; text=re.sub(r'<br\s*/?>','\n',html,flags=re.I); return re.sub(r'<[^>]+>',' ',text)
    else:
        cs=msg.get_content_charset() or "utf-8"; raw=msg.get_payload(decode=True)
        if raw is None: return msg.get_payload()
        text=raw.decode(cs, errors="replace")
        if msg.get_content_type()=="text/html":
            import re; text=re.sub(r'<br\s*/?>','\n',text,flags=re.I); text=re.sub(r'<[^>]+>',' ',text)
        return text
    return ""

def _send(to_addr, subject, body, in_reply_to=None):
    m=EmailMessage(); m["From"]=EMAIL_ADDRESS; m["To"]=to_addr; m["Subject"]=subject; m["Date"]=formatdate(localtime=True)
    if in_reply_to: m["In-Reply-To"]=in_reply_to; m["References"]=in_reply_to
    m.set_content(body)
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls(); s.login(EMAIL_ADDRESS, EMAIL_PASSWORD); s.send_message(m)

def _classify(text):
    t=text.lower()
    if vec and clf:
        X=vec.transform([t]); label=clf.predict(X)[0]; return label
    if any(re.search(p,t) for p in RESERVATION_KEYWORDS): return "reservation"
    if any(re.search(p,t) for p in REVIEW_KEYWORDS): return "review"
    return "other"

def _extract(text):
    import re
    d=re.search(DATE_RE,text,flags=re.I); t=re.search(TIME_RE,text,flags=re.I); p=re.search(PARTY_RE,text,flags=re.I)
    det={"date": d.group("date") if d else None,
         "time": (t.group("time") if t else None),
         "party_size": p.group("party") if p else None}
    if det["time"] and re.fullmatch(r"\\d{1,2}", det["time"].strip()): det["time"] += ":00"
    return det

def _tpl_confirm(name,d,t,p):
    L=[f"Hi{(' '+name) if name else ''},","",f"Your reservation is confirmed at {RESTAURANT_NAME}.",
       f"• Date: {d}",f"• Time: {t}",f"• Party size: {p}"]
    if RESERVATION_LINK: L.append(f"Modify/cancel: {RESERVATION_LINK}")
    if RESERVATION_PHONE: L.append(f"Phone: {RESTAURANT_PHONE}")
    L+=["","We look forward to hosting you!",f"— {RESTAURANT_NAME}"]; return "\\n".join(L)

def _tpl_missing(name,hd,ht,hp):
    miss=[x for x,v in {"date":hd,"time":ht,"party size":hp}.items() if not v]
    L=[f"Hi{(' '+name) if name else ''},",f"Thanks for booking at {RESTAURANT_NAME}.",
       "Could you confirm your "+", ".join(miss)+" so we can finalize your reservation?"]
    if RESERVATION_LINK: L.append(f"You can also book directly here: {RESERVATION_LINK}")
    L+=["","Best,",RESTAURANT_NAME]; return "\\n".join(L)

def _tpl_review(name):
    return f"Hi{(' '+name) if name else ''},\\n\\nThank you for your feedback about {RESTAURANT_NAME}. We appreciate it!\\n\\n— {RESTAURANT_NAME}\\n"

def _tpl_other(name):
    return f"Hi{(' '+name) if name else ''},\\n\\nThanks for reaching out to {RESTAURANT_NAME}. We'll get back to you shortly.\\n\\n— {RESTAURANT_NAME}\\n"

def _db():
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS processed(message_id TEXT PRIMARY KEY, processed_at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS drafts(id INTEGER PRIMARY KEY, to_email TEXT, subject TEXT, body TEXT, in_reply_to TEXT, created_at TEXT, sent_at TEXT)")
    conn.commit(); return conn

def _fetch_unseen():
    M=imaplib.IMAP4_SSL(IMAP_SERVER); M.login(EMAIL_ADDRESS, EMAIL_PASSWORD); M.select("INBOX")
    typ,data=M.search(None,'(UNSEEN)'); ids=data[0].split() if typ=="OK" else []
    out=[]
    for uid in ids:
        typ,msg_data=M.fetch(uid,"(RFC822)")
        if typ=="OK": out.append(msg_data[0][1])
    M.close(); M.logout(); return out

def _gmail_draft(to_email, subject, body, in_reply_to):
    if gd is None: return False
    try:
        gd.create_draft(EMAIL_ADDRESS, subject, body, in_reply_to, to_email); return True
    except Exception as e:
        print("[WARN] Gmail draft error:", e); return False

def handle(conn, raw):
    msg=email.message_from_bytes(raw); mid=msg.get("Message-ID"); c=conn.cursor()
    if mid and c.execute("SELECT 1 FROM processed WHERE message_id=?",(mid,)).fetchone(): return
    subj=_dec(msg.get("Subject","")); body=_body(msg); from_email=parseaddr(msg.get("From",""))[1]; name=parseaddr(msg.get("From",""))[0]
    label=_classify(subj+" "+body)
    if label=="reservation" and notifier:
        try: notifier.notify_owner(f"Reservation email from {from_email}: {subj}")
        except Exception as e: print("[WARN] Twilio notify:", e)
    if label=="reservation":
        det=_extract(subj+" "+body); hd,ht,hp=bool(det["date"]),bool(det["time"]),bool(det["party_size"])
        if hd and ht and hp:
            _send(from_email, f"Re: {subj} — Reservation Confirmed", _tpl_confirm(name,det["date"],det["time"],det["party_size"]), in_reply_to=mid)
            print("[SENT] confirm ->", from_email)
        else:
            _send(from_email, f"Re: {subj} — One quick detail", _tpl_missing(name,hd,ht,hp), in_reply_to=mid)
            print("[SENT] missing ->", from_email)
    else:
        if label=="review":
            dsubj=f"Re: {subj} — Thank you"; dbody=_tpl_review(name)
        else:
            dsubj=f"Re: {subj}"; dbody=_tpl_other(name)
        if _gmail_draft(from_email, dsubj, dbody, mid):
            print("[DRAFT] Gmail ->", from_email)
        else:
            c.execute("INSERT INTO drafts(to_email,subject,body,in_reply_to,created_at) VALUES (?,?,?,?,datetime('now'))",(from_email,dsubj,dbody,mid)); conn.commit(); print("[DRAFTED] local ->", from_email)
    if mid:
        c.execute("INSERT OR IGNORE INTO processed(message_id,processed_at) VALUES(?,datetime('now'))",(mid,)); conn.commit()

def send_pending(conn):
    c=conn.cursor()
    for did,to_email,subject,body,in_reply_to in c.execute("SELECT id,to_email,subject,body,in_reply_to FROM drafts WHERE sent_at IS NULL ORDER BY id"):
        _send(to_email,subject,body,in_reply_to); c.execute("UPDATE drafts SET sent_at=datetime('now') WHERE id=?",(did,)); conn.commit(); print("[SENT draft]",did)

def main():
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD: raise SystemExit("Set EMAIL_ADDRESS and EMAIL_PASSWORD")
    import sys; conn=_db()
    if "--send-pending" in sys.argv: send_pending(conn); return
    for raw in _fetch_unseen():
        try: handle(conn, raw)
        except Exception as e: print("[ERR]", e)

if __name__=="__main__":
    main()
'''

#!/usr/bin/env python3

Enhanced Restaurant Email Agent — Personalized Feedback Replies
---------------------------------------------------------------
Adds:
- Sentiment & upset score (1–5) for feedback emails via OpenAI (with backoff + rule fallback)
- Personalized, short, empathetic replies (no boilerplates)
- Dynamic coupon policy (5–40%) with unique codes stored in SQLite
- Twilio owner alerts for reservation emails (optional)
- Gmail draft creation for non-feedback "other" emails (optional)
- Existing reservation auto-reply flow preserved

Env (examples):
  EMAIL_ADDRESS=20131a0522@gvpce.ac.in
  EMAIL_PASSWORD=app-password
  IMAP_SERVER=imap.gmail.com
  SMTP_SERVER=smtp.gmail.com
  SMTP_PORT=587
  RESTAURANT_NAME="Your Restaurant"
  RESERVATION_PHONE="+1..."
  RESERVATION_LINK="https://..."
  AGENT_DB_PATH=email_agent.sqlite
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-4o-mini

Feature toggles:
  USE_TWILIO_ALERTS=true|false
  USE_GMAIL_DRAFTS=true|false
  USE_ML_CLASSIFIER=true|false   # still supported; not needed for feedback logic
"""

import os, re, sqlite3, imaplib, smtplib, email, time, random, string
from email.message import EmailMessage
from email.header import decode_header, make_header
from email.utils import parseaddr, formatdate

# Optional .env auto-load
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Config ---
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant")
RESERVATION_PHONE = os.getenv("RESERVATION_PHONE", "")
RESERVATION_LINK = os.getenv("RESERVATION_LINK", "")
DB_PATH = os.getenv("AGENT_DB_PATH", "email_agent.sqlite")

USE_TWILIO = os.getenv("USE_TWILIO_ALERTS", "false").lower() == "true"
USE_GMAIL_DRAFTS = os.getenv("USE_GMAIL_DRAFTS", "false").lower() == "true"
USE_ML = os.getenv("USE_ML_CLASSIFIER", "false").lower() == "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Optional Twilio & Gmail drafts helpers
notifier = None
if USE_TWILIO:
    try:
        import notifier as notifier
    except Exception as e:
        print("[WARN] Twilio disabled:", e)
        notifier = None

gd = None
if USE_GMAIL_DRAFTS:
    try:
        import gmail_drafts as gd
    except Exception as e:
        print("[WARN] Gmail drafts disabled:", e)
        gd = None

# Optional local ML (unused for feedback; still supported for classification fallback)
vec = clf = None
if USE_ML:
    try:
        import joblib
        vec = joblib.load("email_vectorizer.joblib")
        clf = joblib.load("email_intent_model.joblib")
    except Exception as e:
        print("[WARN] ML disabled:", e)
        vec = clf = None
        USE_ML = False

# --- Patterns / regex ---
RESERVATION_KEYWORDS = [r"\breservation\b", r"\bbook(?:ing)?\b", r"\btable\b", r"\bparty\b"]
REVIEW_KEYWORDS = [r"\breview\b", r"\bfeedback\b", r"\bcomplaint\b", r"\bpraise\b", r"\bexperience\b", r"\bbad\b", r"\bterrible\b", r"\bawful\b"]
DATE_RE = r"(?:(?:on\s*)?(?P<date>(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?))"
TIME_RE = r"(?:(?:at\s*)?(?P<time>\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|\b)))"
PARTY_RE = r"(?:(?:for|party of)\s*(?P<party>\d{1,2}))"

def _dec(s):
    if not s: return ""
    try: return str(make_header(decode_header(s)))
    except Exception: return s

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
                import re
                text = re.sub(r'<br\s*/?>', '\n', html, flags=re.I)
                return re.sub(r'<[^>]+>', ' ', text)
    else:
        cs = msg.get_content_charset() or "utf-8"
        raw = msg.get_payload(decode=True)
        if raw is None:
            return msg.get_payload()
        text = raw.decode(cs, errors="replace")
        if msg.get_content_type() == "text/html":
            import re
            text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)
            text = re.sub(r'<[^>]+>', ' ', text)
        return text
    return ""

def _send(to_addr, subject, body, in_reply_to=None):
    m = EmailMessage()
    m["From"] = EMAIL_ADDRESS
    m["To"] = to_addr
    m["Subject"] = subject
    m["Date"] = formatdate(localtime=True)
    if in_reply_to:
        m["In-Reply-To"] = in_reply_to
        m["References"] = in_reply_to
    m.set_content(body)
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls()
        s.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        s.send_message(m)

def _db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS processed(message_id TEXT PRIMARY KEY, processed_at TEXT)")
    c.execute("""CREATE TABLE IF NOT EXISTS drafts(
        id INTEGER PRIMARY KEY,
        to_email TEXT, subject TEXT, body TEXT, in_reply_to TEXT, created_at TEXT, sent_at TEXT
    )""")
    # New: coupons ledger
    c.execute("""CREATE TABLE IF NOT EXISTS coupons(
        id INTEGER PRIMARY KEY,
        email TEXT,
        code TEXT UNIQUE,
        discount INTEGER,
        sentiment TEXT,
        score INTEGER,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.commit()
    return conn

def _fetch_unseen():
    M = imaplib.IMAP4_SSL(IMAP_SERVER)
    M.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    M.select("INBOX")
    typ, data = M.search(None, '(UNSEEN)')
    ids = data[0].split() if typ == "OK" else []
    out = []
    for uid in ids:
        typ, msg_data = M.fetch(uid, "(RFC822)")
        if typ == "OK":
            out.append(msg_data[0][1])
    M.close(); M.logout()
    return out

# --- Classification (intent) ---
def _classify_intent(text: str) -> str:
    t = text.lower()
    if USE_ML and vec and clf:
        try:
            X = vec.transform([t])
            return clf.predict(X)[0]
        except Exception:
            pass
    if any(re.search(p, t) for p in RESERVATION_KEYWORDS): return "reservation"
    if any(re.search(p, t) for p in REVIEW_KEYWORDS): return "review"
    return "other"

# --- Reservation helpers ---
def _extract_res_details(text):
    d = re.search(DATE_RE, text, flags=re.I)
    t = re.search(TIME_RE, text, flags=re.I)
    p = re.search(PARTY_RE, text, flags=re.I)
    det = {
        "date": d.group("date") if d else None,
        "time": (t.group("time") if t else None),
        "party_size": p.group("party") if p else None
    }
    if det["time"] and re.fullmatch(r"\d{1,2}", det["time"].strip()):
        det["time"] += ":00"
    return det

def _tpl_confirm(name, d, t, p):
    L = [f"Hi{(' ' + name) if name else ''},", "",
         f"Your reservation is confirmed at {RESTAURANT_NAME}.",
         f"• Date: {d}", f"• Time: {t}", f"• Party size: {p}"]
    if RESERVATION_LINK: L.append(f"Modify/cancel: {RESERVATION_LINK}")
    if RESERVATION_PHONE: L.append(f"Phone: {RESERVATION_PHONE}")
    L += ["", "We look forward to hosting you!", f"— {RESTAURANT_NAME}"]
    return "\n".join(L)

def _tpl_missing(name, hd, ht, hp):
    miss = [x for x, v in {"date": hd, "time": ht, "party size": hp}.items() if not v]
    L = [f"Hi{(' ' + name) if name else ''},",
         f"Thanks for booking at {RESTAURANT_NAME}.",
         "Could you confirm your " + ", ".join(miss) + " so we can finalize your reservation?"]
    if RESERVATION_LINK: L.append(f"You can also book directly here: {RESERVATION_LINK}")
    L += ["", "Best,", RESTAURANT_NAME]
    return "\n".join(L)

# --- OpenAI helpers (sentiment + personalized reply) ---
def _openai_client():
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("[WARN] OpenAI client not available:", e)
        return None

def analyze_sentiment_with_backoff(message_text: str):
    """
    Returns (sentiment:str, score:int [1..5]).
    sentiment in {'positive','neutral','negative'}; score 1=very happy .. 5=furious
    Fallback to heuristic if API not available or quota exceeded.
    """
    client = _openai_client()
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
                    model=OPENAI_MODEL,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60
                )
                txt = resp.choices[0].message.content.strip()
                import json
                obj = json.loads(txt)
                s = str(obj.get("sentiment","neutral")).lower()
                sc = int(obj.get("score", 3))
                sc = max(1, min(5, sc))
                if s not in {"positive","neutral","negative"}:
                    s = "neutral"
                return s, sc
            except Exception as e:
                # backoff on 429/insufficient_quota/etc.
                wait = 2 ** attempt
                print(f"[WARN] OpenAI sentiment attempt {attempt+1} failed: {e} (retry {wait}s)")
                time.sleep(wait)
    # Heuristic fallback
    t = message_text.lower()
    neg_hits = sum(w in t for w in ["awful","terrible","horrible","disgusting","cold","late","rude","bad","worst","never again","refund","angry"])
    pos_hits = sum(w in t for w in ["amazing","great","excellent","love","loved","fantastic","wonderful","perfect","delicious"])
    if neg_hits >= 3: return "negative", 5
    if neg_hits == 2: return "negative", 4
    if neg_hits == 1: return "negative", 3
    if pos_hits >= 2: return "positive", 1
    if pos_hits == 1: return "positive", 2
    return "neutral", 3

def choose_discount(sentiment: str, score: int, message_text: str) -> int:
    if sentiment == "positive":
        # 10% if very enthusiastic, else 5%
        enthusiastic = any(w in message_text.lower() for w in ["love","loved","amazing","incredible","fantastic","perfect","best"])
        return 10 if enthusiastic else 5
    if sentiment == "neutral":
        return 15
    # negative: map score
    mapping = {3: 15, 4: 25, 5: 30}
    disc = mapping.get(score, 15)
    # If very harsh wording, consider bump up to 40
    if score == 5 and any(w in message_text.lower() for w in ["worst","never again","refund","disgusting","unacceptable","furious"]):
        disc = 40
    return min(disc, 40)

def _random_code(prefix: str, pct: int) -> str:
    tail = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"{prefix}{pct}-{tail}"

def persist_coupon(conn, email_addr: str, code: str, discount: int, sentiment: str, score: int):
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO coupons(email, code, discount, sentiment, score) VALUES (?,?,?,?,?)",
              (email_addr, code, discount, sentiment, score))
    conn.commit()

def generate_personalized_reply(name: str, sentiment: str, score: int, discount: int, code: str, message_text: str):
    """
    Use GPT to craft a short, kind, personalized reply (varies wording each time).
    Falls back to compact templates if OpenAI unavailable.
    """
    client = _openai_client()
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
                    model=OPENAI_MODEL,
                    temperature=0.7,
                    messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                    max_tokens=240
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                wait = 2 ** attempt
                print(f"[WARN] OpenAI reply attempt {attempt+1} failed: {e} (retry {wait}s)")
                time.sleep(wait)

    # Fallback compact templates (varies by sentiment)
    if sentiment == "positive":
        return (
            f"Hi{(' ' + name) if name else ''},\n\n"
            f"Thank you for the wonderful note—guests like you make our day. "
            f"As a small thank-you, here’s {discount}% off next time (code: {code}). "
            f"We can’t wait to welcome you back.\n\n— {RESTAURANT_NAME}"
        )
    if sentiment == "neutral":
        return (
            f"Hi{(' ' + name) if name else ''},\n\n"
            f"Thanks for sharing your thoughts—your feedback helps us improve. "
            f"Please accept {discount}% off your next visit (code: {code}); we’d love another chance to impress.\n\n"
            f"— {RESTAURANT_NAME}"
        )
    # negative
    opener = "We’re truly sorry" if score >= 4 else "We’re sorry"
    return (
        f"Hi{(' ' + name) if name else ''},\n\n"
        f"{opener} that your experience fell short. You matter to us, and we’ve noted your concerns with the team. "
        f"Please allow us to make it right—here’s {discount}% off for your next visit (code: {code}). "
        f"We appreciate the chance to earn back your trust.\n\n— {RESTAURANT_NAME}"
    )

# --- Gmail drafts fallback for "other" ---
def _gmail_draft(to_email, subject, body, in_reply_to):
    if gd is None:
        return False
    try:
        gd.create_draft(EMAIL_ADDRESS, subject, body, in_reply_to, to_email)
        return True
    except Exception as e:
        print("[WARN] Gmail draft error:", e)
        return False

# --- Main handlers ---
def handle_feedback(conn, from_email_addr, name, subject, body_text, in_reply_to):
    # 1) Sentiment / upset rating
    sentiment, score = analyze_sentiment_with_backoff(body_text)
    # 2) Discount decision
    discount = choose_discount(sentiment, score, body_text)
    # 3) Unique coupon code
    prefix = "CARE" if sentiment == "negative" else "THANKS"
    code = _random_code(prefix, discount)
    persist_coupon(conn, from_email_addr, code, discount, sentiment, score)
    # 4) Personalized reply
    reply_body = generate_personalized_reply(name, sentiment, score, discount, code, body_text)
    # 5) Send
    _send(from_email_addr, f"Re: {subject}", reply_body, in_reply_to=in_reply_to)
    print(f"[SENT feedback] {sentiment}/{score} -> {discount}% code={code} to {from_email_addr}")

def handle_message(conn, raw):
    msg = email.message_from_bytes(raw)
    mid = msg.get("Message-ID")
    c = conn.cursor()
    if mid:
        row = c.execute("SELECT 1 FROM processed WHERE message_id=?", (mid,)).fetchone()
        if row: return

    subj = _dec(msg.get("Subject", ""))
    body = _body(msg)
    from_email_addr = parseaddr(msg.get("From",""))[1]
    name = parseaddr(msg.get("From",""))[0]
    full_text = (subj + " " + body)

    label = _classify_intent(full_text)

    # Alerts for reservations
    if label == "reservation" and notifier:
        try:
            notifier.notify_owner(f"Reservation email from {from_email_addr}: {subj}")
        except Exception as e:
            print("[WARN] Twilio notify:", e)

    if label == "reservation":
        det = _extract_res_details(full_text)
        hd, ht, hp = bool(det["date"]), bool(det["time"]), bool(det["party_size"])
        if hd and ht and hp:
            _send(from_email_addr, f"Re: {subj} — Reservation Confirmed",
                  _tpl_confirm(name, det["date"], det["time"], det["party_size"]), in_reply_to=mid)
            print("[SENT] confirm ->", from_email_addr)
        else:
            _send(from_email_addr, f"Re: {subj} — One quick detail",
                  _tpl_missing(name, hd, ht, hp), in_reply_to=mid)
            print("[SENT] missing ->", from_email_addr)

    elif label == "review":
        # NEW: Immediate personalized reply with coupon logic
        handle_feedback(conn, from_email_addr, name, subj, body, mid)

    else:
        # "other" — keep draft behavior
        dsubj = f"Re: {subj}"
        dbody = (f"Hi{(' ' + name) if name else ''},\n\n"
                 f"Thanks for reaching out to {RESTAURANT_NAME}. We received your message and will get back to you shortly.\n\n"
                 f"— {RESTAURANT_NAME}\n")
        if _gmail_draft(from_email_addr, dsubj, dbody, mid):
            print("[DRAFT] Gmail ->", from_email_addr)
        else:
            c.execute("INSERT INTO drafts(to_email,subject,body,in_reply_to,created_at) VALUES (?,?,?,?,datetime('now'))",
                      (from_email_addr, dsubj, dbody, mid))
            conn.commit()
            print("[DRAFTED] local ->", from_email_addr)

    if mid:
        c.execute("INSERT OR IGNORE INTO processed(message_id,processed_at) VALUES(?,datetime('now'))", (mid,))
        conn.commit()

def send_pending(conn):
    c = conn.cursor()
    rows = c.execute("SELECT id,to_email,subject,body,in_reply_to FROM drafts WHERE sent_at IS NULL ORDER BY id").fetchall()
    for did, to_email, subject, body, in_reply_to in rows:
        _send(to_email, subject, body, in_reply_to)
        c.execute("UPDATE drafts SET sent_at=datetime('now') WHERE id=?", (did,))
        conn.commit()
        print(f"[SENT draft] {did} -> {to_email}")

def main():
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        raise SystemExit("Set EMAIL_ADDRESS and EMAIL_PASSWORD")
    conn = _db()
    import sys
    if "--send-pending" in sys.argv:
        send_pending(conn); return
    for raw in _fetch_unseen():
        try:
            handle_message(conn, raw)
        except Exception as e:
            print("[ERR]", e)

if __name__ == "__main__":
    main()
