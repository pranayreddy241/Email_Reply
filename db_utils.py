from datetime import datetime
from typing import Optional, Tuple, List, Dict

OPEN_HOUR = 18
DEFAULT_CAPACITY = 10

def slot_within_hours(time_24: str) -> bool:
    try:
        hh, mm = map(int, time_24.split(":"))
    except Exception:
        return False
    if hh < 18 or hh > 23:
        return False
    if hh == 23 and mm > 30:
        return False
    return mm in (0, 30)

def list_slots_for_date(_date_iso: str) -> List[str]:
    slots = []
    for hh in range(18, 24):
        for mm in (0, 30):
            if hh == 23 and mm > 30:
                continue
            t = f"{hh:02d}:{mm:02d}"
            if slot_within_hours(t):
                slots.append(t)
    return slots

def _slot_dt(date_iso: str, time_24: str) -> datetime:
    return datetime.fromisoformat(f"{date_iso} {time_24}")

def count_booked(conn, date_iso: str, time_24: str) -> int:
    slot_dt = _slot_dt(date_iso, time_24)
    with conn.cursor() as c:
        c.execute(
            "SELECT COUNT(1) FROM reservations WHERE slot_datetime=%s AND status='confirmed'",
            (slot_dt,)
        )
        return int(c.fetchone()[0] or 0)

def get_latest_active_reservation(conn, email: str) -> Optional[Dict]:
    """
    Most recent confirmed reservation for this email (future or latest overall).
    """
    with conn.cursor() as c:
        c.execute(
            """
            SELECT id, confirmation_code, name, email, phone, party_size, slot_datetime, status, source, created_at
            FROM reservations
            WHERE email=%s AND status='confirmed'
            ORDER BY slot_datetime DESC
            LIMIT 1
            """,
            (email,)
        )
        row = c.fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "confirmation_code": row[1],
        "name": row[2],
        "email": row[3],
        "phone": row[4],
        "party_size": row[5],
        "slot_datetime": row[6],
        "status": row[7],
        "source": row[8],
        "created_at": row[9],
    }

def cancel_reservation_by_id(conn, res_id: int, reason: str = "customer_update") -> bool:
    with conn.cursor() as c:
        c.execute(
            """
            UPDATE reservations
            SET status='cancelled'
            WHERE id=%s AND status='confirmed'
            """,
            (res_id,)
        )
        updated = c.rowcount
    conn.commit()
    return updated > 0

def cancel_latest_reservation(conn, email: str) -> Optional[Dict]:
    old = get_latest_active_reservation(conn, email)
    if not old:
        return None
    cancel_reservation_by_id(conn, old["id"])
    return old

def reserve(
    conn,
    *,
    name: str,
    email: str,
    phone: Optional[str],
    party_size: Optional[int],
    date_iso: str,
    time_24: str,
    source: str = "email",
    capacity: int = DEFAULT_CAPACITY
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Returns: (ok, confirmation_code, reason)
    """
    if not date_iso or not time_24:
        return False, None, "Missing date or time."

    if not slot_within_hours(time_24):
        return False, None, "Requested time is outside operating hours or not on a 30-min boundary."

    booked = count_booked(conn, date_iso, time_24)
    if booked >= capacity:
        return False, None, "Slot is full."

    import hashlib
    base = f"{email}|{date_iso}|{time_24}|{datetime.utcnow().timestamp()}"
    code = "RES-" + hashlib.sha256(base.encode()).hexdigest()[:8].upper()

    slot_dt = _slot_dt(date_iso, time_24)
    with conn.cursor() as c:
        c.execute(
            """
            INSERT INTO reservations(confirmation_code, name, email, phone, party_size, slot_datetime, status, source)
            VALUES (%s,%s,%s,%s,%s,%s,'confirmed',%s)
            """,
            (code, name, email, phone, party_size, slot_dt, source)
        )
    conn.commit()
    return True, code, None

def next_available_slots(conn, date_iso: str, *, capacity: int = DEFAULT_CAPACITY, limit: int = 3) -> List[str]:
    out = []
    for t in list_slots_for_date(date_iso):
        if count_booked(conn, date_iso, t) < capacity:
            out.append(t)
        if len(out) >= limit:
            break
    return out
