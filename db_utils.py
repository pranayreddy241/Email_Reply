import sqlite3
from datetime import datetime
from typing import Optional

OPEN_HOUR = 18  # 6pm
CLOSE_HOUR = 24 # 12am (midnight boundary)
SLOT_MINUTES = 30
DEFAULT_CAPACITY = 10

def ensure_schema(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS reservations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            confirmation_code TEXT UNIQUE,
            name TEXT,
            email TEXT,
            phone TEXT,
            party_size INTEGER,
            slot_datetime TEXT,
            status TEXT DEFAULT 'confirmed',
            source TEXT DEFAULT 'email',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS google_reviews(
            review_id TEXT PRIMARY KEY,
            rating INTEGER,
            comment TEXT,
            author_name TEXT,
            created_at TEXT,
            replied INTEGER DEFAULT 0,
            last_checked_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS claims(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id TEXT,
            name TEXT,
            email TEXT,
            phone TEXT,
            visit_date TEXT,
            details TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()

def _slot_key(date_iso: str, time_24: str) -> str:
    # stored as 'YYYY-MM-DD HH:MM'
    return f"{date_iso} {time_24}"

def slot_within_hours(time_24: str) -> bool:
    try:
        hh, mm = map(int, time_24.split(":"))
    except Exception:
        return False

    # allowed: 18:00 .. 23:30
    if hh < OPEN_HOUR or hh > 23:
        return False
    if hh == 23 and mm > 30:
        return False
    return mm in (0, 30)

def list_slots_for_date(date_iso: str):
    # 18:00 -> 23:30 inclusive
    slots = []
    for hh in range(OPEN_HOUR, 24):
        for mm in (0, 30):
            if hh == 23 and mm > 30:
                continue
            slots.append(f"{hh:02d}:{mm:02d}")
    return [s for s in slots if slot_within_hours(s)]

def count_booked(conn: sqlite3.Connection, date_iso: str, time_24: str) -> int:
    c = conn.cursor()
    slot = _slot_key(date_iso, time_24)
    row = c.execute(
        "SELECT COUNT(1) FROM reservations WHERE slot_datetime=? AND status='confirmed'",
        (slot,)
    ).fetchone()
    return int(row[0] or 0)

def reserve(
    conn: sqlite3.Connection,
    *,
    name: str,
    email: str,
    phone: Optional[str],
    party_size: Optional[int],
    date_iso: str,
    time_24: str,
    source: str = "email",
    capacity: int = DEFAULT_CAPACITY
):
    """
    Attempt to reserve a slot.
    Returns (ok: bool, confirmation_code: Optional[str], reason: Optional[str])
    """
    ensure_schema(conn)

    if not slot_within_hours(time_24):
        return False, None, "Requested time is outside operating hours or not on a 30-min boundary."

    booked = count_booked(conn, date_iso, time_24)
    if booked >= capacity:
        return False, None, "Slot is full."

    import hashlib
    base = f"{email}|{date_iso}|{time_24}|{datetime.utcnow().timestamp()}"
    code = "RES-" + hashlib.sha256(base.encode()).hexdigest()[:8].upper()

    slot = _slot_key(date_iso, time_24)
    c = conn.cursor()
    c.execute(
        """INSERT INTO reservations(confirmation_code, name, email, phone, party_size, slot_datetime, source)
           VALUES (?,?,?,?,?,?,?)""",
        (code, name, email, phone, party_size, slot, source)
    )
    conn.commit()
    return True, code, None

def next_available_slots(conn: sqlite3.Connection, date_iso: str, *, capacity: int = DEFAULT_CAPACITY, limit: int = 3):
    ensure_schema(conn)
    out = []
    for t in list_slots_for_date(date_iso):
        if count_booked(conn, date_iso, t) < capacity:
            out.append(t)
        if len(out) >= limit:
            break
    return out
