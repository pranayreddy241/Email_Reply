#!/usr/bin/env python3
"""
Enhanced Restaurant Email Agent
Adds Twilio alerts, Gmail draft creation, optional ML (joblib).
See README/guide for env vars.
"""
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
