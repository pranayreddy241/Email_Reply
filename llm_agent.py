'''# llm_agent.py
import os, json, email, pytz
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from openai import OpenAI

TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
TZ = pytz.timezone(TIMEZONE)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
    Convert phrases like 'today', 'tomorrow', 'next fri', 'in 2 hours' to
    (YYYY-MM-DD, HH:MM 24h) when possible. If uncertain, return originals.
    """
    d_iso, t_24 = None, None
    if date_text:
        low = date_text.strip().lower()
        if low in ["today"]:
            d_iso = ref_dt.strftime("%Y-%m-%d")
        elif low in ["tomorrow", "tmrw", "tomo"]:
            d_iso = (ref_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            # weekdays like 'next fri' or plain 'fri'
            weekdays = ["mon","tue","wed","thu","fri","sat","sun"]
            for i, w in enumerate(weekdays):
                if low.startswith(w):
                    # find next weekday (>= tomorrow)
                    delta = (i - ref_dt.weekday()) % 7
                    if delta == 0:
                        delta = 7
                    d_iso = (ref_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
                    break
            if d_iso is None:
                try:
                    d_iso = dateparser.parse(date_text, dayfirst=False, yearfirst=False, default=ref_dt).astimezone(TZ).strftime("%Y-%m-%d")
                except Exception:
                    d_iso = None
    if time_text:
        low = time_text.strip().lower()
        # handle 'in X hours/minutes'
        try:
            if low.startswith("in "):
                # naive parse: 'in 2 hours', 'in 30 minutes'
                parts = low.split()
                val = int(parts[1])
                unit = parts[2]
                if "hour" in unit:
                    t_24 = (ref_dt + timedelta(hours=val)).strftime("%H:%M")
                elif "min" in unit:
                    t_24 = (ref_dt + timedelta(minutes=val)).strftime("%H:%M")
            if t_24 is None:
                # otherwise use dateutil to parse time in context of ref date
                t_ = dateparser.parse(time_text, default=ref_dt)
                t_24 = t_.astimezone(TZ).strftime("%H:%M")
        except Exception:
            t_24 = None
    return d_iso, t_24

# ---- LLM call ----
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
    """Make a compact plain-text thread to feed the LLM."""
    lines = []
    for m in messages:
        frm = m.get("from","")
        when = m.get("date","")
        subj = m.get("subject","")
        body = m.get("body","")[:1000]
        lines.append(f"From: {frm}\nDate: {when}\nSubject: {subj}\nBody:\n{body}\n---")
    return "\n".join(lines)

def call_llm_extract(thread_text: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # small, fast; swap to gpt-4.1 if you prefer
        response_format=JSON_INSTRUCTIONS,
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":thread_text}
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content
    return json.loads(raw)

def decide_action(extract: Dict[str,Any], ref_dt: datetime) -> Dict[str,Any]:
    """
    Convert date_text/time_text to normalized forms when possible, and produce a plan:
      action ∈ {confirm, ask_missing, draft, skip}
    """
    intent = extract.get("intent")
    conf = float(extract.get("confidence") or 0)
    name = extract.get("name")
    party = extract.get("party_size")
    date_text = extract.get("date_text")
    time_text = extract.get("time_text")
    notes = extract.get("notes")

    date_iso, time_24 = normalize_relative(date_text, time_text, ref_dt)

    have_all = bool(party) and bool(date_iso) and bool(time_24)
    action = "skip"
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
    }'''

# llm_agent.py
# Free, local LLM backend (Hugging Face transformers) for your Restaurant Email Agent

import os, re, json, email, pytz, math
from datetime import datetime, timedelta
from dateutil import parser as dateparser

# Load .env (so LLM_MODEL_PATH / CONFIRM_THRESHOLD / TIMEZONE are available)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- Transformers (local) ----------------
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "meta-llama/Llama-3.2-1B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model load — for CPU this can take a moment the first time
_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH).to(DEVICE)
_model.eval()

# Small, consistent generation config
_GEN_KW = dict(
    max_new_tokens=256,
    temperature=0.2,
    top_p=0.9,
    do_sample=False
)

# ---------------- Utility: generation wrapper ----------------
def _gen(prompt: str) -> str:
    inputs = _tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = _model.generate(**inputs, **_GEN_KW)
    text = _tokenizer.decode(out[0], skip_special_tokens=True)
    return text

# ---------------- Time helpers ----------------
def received_local_dt(msg) -> datetime:
    """Return timezone-aware datetime for the email's Date header."""
    tzname = os.getenv("TIMEZONE", "America/New_York")
    tz = pytz.timezone(tzname)
    try:
        hdr = msg.get("Date")
        dt = email.utils.parsedate_to_datetime(hdr) if hdr else datetime.utcnow()
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(tz)
    except Exception:
        return datetime.now(tz)

_DOW = ["mon","tue","wed","thu","fri","sat","sun"]

def _normalize_date_time(date_text: str | None, time_text: str | None, ref_dt: datetime):
    """Return (date_iso, time_24). Handles 'today', 'tomorrow', weekdays, and common formats."""
    date_iso, time_24 = None, None

    # --- time ---
    if time_text:
        t = time_text.strip().lower()
        # "in 2 hours" / "in an hour"
        m = re.search(r"in\s+(an|\d+)\s*hour", t)
        if m:
            hrs = 1 if m.group(1) == "an" else int(m.group(1))
            dt2 = ref_dt + timedelta(hours=hrs)
            time_24 = dt2.strftime("%H:%M")
            if date_text is None:  # if date missing, assume same day (or rollover handled below)
                date_iso = dt2.strftime("%Y-%m-%d")
        if time_24 is None:
            # Accept "7", "7pm", "7:30", "19:00"
            try:
                parsed_t = dateparser.parse(t, default=ref_dt)
                time_24 = parsed_t.strftime("%H:%M")
            except Exception:
                pass

    # --- date ---
    if date_text:
        d = date_text.strip().lower()
        if d in ("today",):
            date_iso = ref_dt.strftime("%Y-%m-%d")
        elif d in ("tomorrow", "tmrw", "tomo"):
            date_iso = (ref_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        elif d[:3] in _DOW:
            # Next weekday
            target = _DOW.index(d[:3])
            delta = (target - ref_dt.weekday()) % 7
            delta = 7 if delta == 0 else delta  # "this wed" (today) usually means next week
            date_iso = (ref_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
        else:
            try:
                parsed_d = dateparser.parse(d, dayfirst=False, yearfirst=False, default=ref_dt)
                date_iso = parsed_d.strftime("%Y-%m-%d")
            except Exception:
                pass

    # If we only had time, keep today's date
    if date_iso is None and time_24 is not None:
        date_iso = ref_dt.strftime("%Y-%m-%d")

    return date_iso, time_24

# ---------------- Thread summarization ----------------
_SYSTEM_SUMMARY = (
    "You are an assistant that summarizes an email thread for a restaurant reservation desk. "
    "Write a concise bullet summary with who, what, when, party size, and any constraints."
)

def summarize_thread(thread_bundle: list[dict]) -> str:
    """
    thread_bundle is a list of {from, date, subject, body} (you pass last few messages).
    Returns a short natural-language summary string.
    """
    # Build a compact transcript
    lines = []
    for m in thread_bundle:
        frm = m.get("from","")
        sub = m.get("subject","")
        body = (m.get("body","") or "").strip()
        body = re.sub(r"\s+", " ", body)
        body = body[:800]
        lines.append(f"From: {frm}\nSubject: {sub}\nBody: {body}\n")
    transcript = "\n---\n".join(lines)

    prompt = (
        f"{_SYSTEM_SUMMARY}\n\n"
        f"Thread (most recent last):\n{transcript}\n\n"
        "Now write a compact 3-6 bullet summary."
    )
    text = _gen(prompt)
    # Heuristic: return only the part after the prompt
    return text.split("summary", 1)[-1].strip() if "summary" in text.lower() else text.strip()

# ---------------- Extraction ----------------
_SYSTEM_EXTRACT = (
    "Extract reservation intent from the conversation.\n"
    "Return STRICT JSON ONLY (no prose), matching this schema:\n"
    "{"
    "  \"intent\": \"reservation\"|\"review\"|\"other\","
    "  \"confidence\": number (0..1),"
    "  \"name\": string|null,"
    "  \"party_size\": integer|null,"
    "  \"date_text\": string|null,"
    "  \"time_text\": string|null,"
    "  \"notes\": string|null"
    "}\n"
    "If uncertain, set fields to null and reduce confidence."
)

# Simple fallback regexes if model fails
_FALLBACK_PARTY = re.compile(r"(?:for|party of|group of|we are|we're|total of)\s*(\d{1,2})", re.I)
_FALLBACK_DATE  = re.compile(r"(today|tomorrow|mon|tue|wed|thu|fri|sat|sun|\d{1,2}[/-]\d{1,2}|"
                             r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)",
                             re.I)
_FALLBACK_TIME  = re.compile(r"(?:in\s+(?:an|\d+)\s*hour(?:s)?)|"
                             r"(?:\b\d{1,2}:\d{2}\s*(?:am|pm)?)|"
                             r"(?:\b\d{1,2}\s*(?:am|pm)\b)|"
                             r"(?:\bat\s*\d{1,2}(?::\d{2})?\b)", re.I)

def _fallback_extract_simple(text: str) -> dict:
    t = text or ""
    party = None
    m = _FALLBACK_PARTY.search(t)
    if m:
        try: party = int(m.group(1))
        except: party = None
    d = None
    md = _FALLBACK_DATE.search(t)
    if md: d = md.group(1)
    tm = None
    mt = _FALLBACK_TIME.search(t)
    if mt:
        raw = mt.group(0).lower().replace("at", "").strip()
        tm = raw
    name = None
    # crude name heuristic from a leading "Hi <Name>" or signature
    n1 = re.search(r"\b(?:hi|hello)\s+([A-Z][a-z]+)\b", text, re.I)
    if n1: name = n1.group(1)
    return {
        "intent": "reservation" if (party or d or tm) else "other",
        "confidence": 0.5 if (party or d or tm) else 0.3,
        "name": name,
        "party_size": party,
        "date_text": d,
        "time_text": tm,
        "notes": None
    }

def _extract_json_block(s: str) -> str | None:
    """Find the first plausible JSON object in a string."""
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return None

def call_llm_extract(thread_text: str) -> dict | None:
    """
    Use local LLM to extract structured fields. Falls back to regex if JSON parsing fails.
    """
    prompt = (
        f"{_SYSTEM_EXTRACT}\n\n"
        f"Conversation summary + key lines:\n{thread_text}\n\n"
        "JSON:"
    )
    try:
        out = _gen(prompt)
        # Try to isolate JSON
        blk = _extract_json_block(out) or out.strip()
        data = json.loads(blk)
        # Guardrails
        for k in ["intent","confidence","name","party_size","date_text","time_text","notes"]:
            if k not in data:
                raise ValueError("schema missing field")
        # Normalize intent
        data["intent"] = str(data["intent"]).lower()
        if data["intent"] not in ("reservation","review","other"):
            data["intent"] = "other"
        # Clamp confidence
        try:
            c = float(data["confidence"])
            data["confidence"] = max(0.0, min(1.0, c))
        except Exception:
            data["confidence"] = 0.5
        # Coerce party
        if data["party_size"] is not None:
            try:
                data["party_size"] = int(data["party_size"])
            except Exception:
                data["party_size"] = None
        return data
    except Exception:
        # Fallback extraction so we still work
        return _fallback_extract_simple(thread_text)

# ---------------- Decision policy ----------------
def decide_action(extract: dict, ref_dt: datetime) -> dict:
    """
    Returns a plan dict:
      {
        "action": "confirm" | "ask_missing" | "draft" | "other",
        "confidence": float,
        "date_iso": str|None,
        "time_24": str|None,
        "party_size": int|None
      }
    """
    intent = (extract.get("intent") or "other").lower()
    conf = float(extract.get("confidence") or 0.5)
    party = extract.get("party_size")
    date_text = extract.get("date_text")
    time_text = extract.get("time_text")

    date_iso, time_24 = _normalize_date_time(date_text, time_text, ref_dt)

    # Slightly adjust confidence if we have all three fields
    have_all = bool(party and date_iso and time_24)
    if have_all:
        conf = min(1.0, conf + 0.1)

    threshold = float(os.getenv("CONFIRM_THRESHOLD", "0.75"))

    if intent == "reservation":
        if have_all and conf >= threshold:
            action = "confirm"
        else:
            action = "ask_missing"
    elif intent == "review":
        action = "draft"
    else:
        action = "other"

    return {
        "action": action,
        "confidence": conf,
        "date_iso": date_iso,
        "time_24": time_24,
        "party_size": party
    }


