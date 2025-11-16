# llm_agent.py
from dotenv import load_dotenv
load_dotenv()   # <-- LOAD .env BEFORE ANYTHING ELSE

import os, json, email, pytz
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from openai import OpenAI

# ---- Load environment ----
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
TZ = pytz.timezone(TIMEZONE)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing — add it to your .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Utilities ----
def received_local_dt(msg) -> datetime:
    """Return the email's Date header as tz-aware local datetime (fallback: now)."""
    try:
        hdr = msg.get("Date")
        dt = email.utils.parsedate_to_datetime(hdr) if hdr else datetime.utcnow()
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(TZ)
    except Exception:
        return datetime.now(TZ)

def normalize_relative(date_text: str, time_text: str, ref_dt: datetime) -> Tuple[str, str]:
    """
    Convert phrases like 'today', 'tomorrow', 'next fri', 'in 2 hours' 
    into exact (YYYY-MM-DD, HH:MM). Uncertain → return None.
    """
    d_iso, t_24 = None, None
    if date_text:
        low = date_text.strip().lower()
        if low == "today":
            d_iso = ref_dt.strftime("%Y-%m-%d")
        elif low in ["tomorrow", "tmrw", "tomo"]:
            d_iso = (ref_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            weekdays = ["mon","tue","wed","thu","fri","sat","sun"]
            for i, w in enumerate(weekdays):
                if low.startswith(w):
                    delta = (i - ref_dt.weekday()) % 7
                    if delta == 0:
                        delta = 7
                    d_iso = (ref_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
                    break
            if d_iso is None:
                try:
                    d_iso = dateparser.parse(date_text, default=ref_dt).astimezone(TZ).strftime("%Y-%m-%d")
                except Exception:
                    d_iso = None
    if time_text:
        low = time_text.strip().lower()
        try:
            if low.startswith("in "):
                parts = low.split()
                val = int(parts[1])
                unit = parts[2]
                if "hour" in unit:
                    t_24 = (ref_dt + timedelta(hours=val)).strftime("%H:%M")
                elif "min" in unit:
                    t_24 = (ref_dt + timedelta(minutes=val)).strftime("%H:%M")
            if t_24 is None:
                t_ = dateparser.parse(time_text, default=ref_dt)
                t_24 = t_.astimezone(TZ).strftime("%H:%M")
        except Exception:
            t_24 = None
    return d_iso, t_24

# ---- LLM Instructions ----
SYSTEM = (
    "You are a maître d' agent that reads an email conversation and extracts structured booking info. "
    "Return strict JSON with fields: intent (reservation|review|other), confidence (0..1), "
    "name, party_size, date_text, time_text, notes. Use the conversation context to resolve vague phrases "
    "like 'tomorrow evening', 'in an hour', 'next Fri'. If uncertain, leave that field null and "
    "explain in notes. Do NOT hallucinate details."
)

JSON_INSTRUCTIONS = {
    "type": "json_schema",
    "json_schema": {
        "name": "reservation_extract",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent": {"type": "string", "enum": ["reservation","review","other"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "name": {"type": ["string","null"]},
                "party_size": {"type": ["integer","null"]},
                "date_text": {"type": ["string","null"]},
                "time_text": {"type": ["string","null"]},
                "notes": {"type": ["string","null"]},
            },
            "required": ["intent","confidence","name","party_size","date_text","time_text","notes"]
        },
        "strict": True
    }
}

def summarize_thread(messages: List[dict]) -> str:
    lines = []
    for m in messages:
        lines.append(
            f"From: {m.get('from','')}\n"
            f"Date: {m.get('date','')}\n"
            f"Subject: {m.get('subject','')}\n"
            f"Body:\n{m.get('body','')[:1000]}\n---"
        )
    return "\n".join(lines)

def call_llm_extract(thread_text: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        response_format=JSON_INSTRUCTIONS,
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":thread_text}
        ],
        temperature=0.2,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"intent": "other", "confidence": 0.5, "name": None, "party_size": None,
                "date_text": None, "time_text": None, "notes": None}

def decide_action(extract: Dict[str,Any], ref_dt: datetime) -> Dict[str,Any]:
    intent = extract.get("intent")
    conf = float(extract.get("confidence") or 0)
    name = extract.get("name")
    party = extract.get("party_size")
    date_text = extract.get("date_text")
    time_text = extract.get("time_text")
    notes = extract.get("notes")

    date_iso, time_24 = normalize_relative(date_text, time_text, ref_dt)

    have_all = bool(party) and bool(date_iso) and bool(time_24)

    if intent == "reservation":
        if have_all and conf >= 0.75:
            action = "confirm"
        else:
            action = "ask_missing"
    elif intent == "review":
        action = "draft"
    else:
        action = "skip"

    return {
        "intent": intent,
        "confidence": conf,
        "name": name,
        "party_size": party,
        "date_text": date_text,
        "time_text": time_text,
        "date_iso": date_iso,
        "time_24": time_24,
        "notes": notes,
        "action": action,
    }
