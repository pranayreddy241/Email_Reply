# dashboard.py — “Superorder-style” Ops Dashboard (Streamlit + Postgres)
# Copy-paste this entire file.

import os
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from db_pg import get_pg_conn, ensure_schema_pg
from db_utils import (
    list_slots_for_date,
    count_booked,
    reserve,
    DEFAULT_CAPACITY,
    cancel_reservation_by_id,
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Restaurant Ops",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant")
CAPACITY = int(os.getenv("DEFAULT_CAPACITY", str(DEFAULT_CAPACITY)))
DASH_BG_URL = os.getenv(
    "DASH_BG_URL",
    # Change this to your own hosted image later
    "https://images.unsplash.com/photo-1529692236671-f1dc1c2d5f9b?auto=format&fit=crop&w=2400&q=80",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    f"""
<style>
/* App background */
.stApp {{
  background:
    linear-gradient(rgba(8,10,14,0.75), rgba(8,10,14,0.88)),
    url("{DASH_BG_URL}");
  background-size: cover;
  background-attachment: fixed;
  background-position: center;
}}

/* Containers */
.card {{
  background: rgba(18, 20, 26, 0.72);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 16px;
  backdrop-filter: blur(10px);
}}
.hero {{
  background: rgba(10, 12, 16, 0.62);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 22px;
  padding: 18px 18px;
  backdrop-filter: blur(12px);
}}
.subtle {{
  opacity: 0.78;
}}
.badge {{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.06);
}}
.badge-ok {{ background: rgba(34, 197, 94, 0.16); border-color: rgba(34,197,94,0.28); }}
.badge-warn {{ background: rgba(245, 158, 11, 0.16); border-color: rgba(245,158,11,0.28); }}
.badge-bad {{ background: rgba(239, 68, 68, 0.16); border-color: rgba(239,68,68,0.28); }}

hr {{
  border: none;
  border-top: 1px solid rgba(255,255,255,0.10);
}}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# DB helpers
# -----------------------------
def get_connection():
    conn = get_pg_conn()
    ensure_schema_pg(conn)
    return conn

def read_df(query: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

def exec_sql(query: str, params=None) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as c:
            c.execute(query, params or ())
            affected = c.rowcount
        conn.commit()
        return affected
    finally:
        conn.close()

@st.cache_data(ttl=8)
def load_status() -> pd.DataFrame:
    return read_df("SELECT key, value, updated_at FROM system_status ORDER BY updated_at DESC")

@st.cache_data(ttl=8)
def load_today_reservations(day: date) -> pd.DataFrame:
    start = datetime.combine(day, datetime.min.time())
    end = start + timedelta(days=1)
    return read_df(
        """
        SELECT id, confirmation_code, name, email, phone, party_size, slot_datetime, status, source, created_at
        FROM reservations
        WHERE slot_datetime >= %s AND slot_datetime < %s
        ORDER BY slot_datetime ASC
        """,
        params=(start, end),
    )

@st.cache_data(ttl=8)
def load_range_reservations(start_day: date, end_day: date) -> pd.DataFrame:
    start = datetime.combine(start_day, datetime.min.time())
    end = datetime.combine(end_day + timedelta(days=1), datetime.min.time())
    return read_df(
        """
        SELECT id, confirmation_code, name, email, phone, party_size, slot_datetime, status, source, created_at
        FROM reservations
        WHERE slot_datetime >= %s AND slot_datetime < %s
        ORDER BY slot_datetime ASC
        """,
        params=(start, end),
    )

@st.cache_data(ttl=8)
def load_feedback(limit=200) -> pd.DataFrame:
    return read_df("SELECT * FROM feedback_log ORDER BY created_at DESC LIMIT %s", params=(limit,))

@st.cache_data(ttl=8)
def load_coupons(limit=200) -> pd.DataFrame:
    return read_df("SELECT * FROM coupons ORDER BY created_at DESC LIMIT %s", params=(limit,))

@st.cache_data(ttl=8)
def load_processed(limit=200) -> pd.DataFrame:
    return read_df("SELECT message_id, action, processed_at FROM processed ORDER BY processed_at DESC LIMIT %s", params=(limit,))

def compute_slot_table(day: date) -> pd.DataFrame:
    date_iso = day.isoformat()
    conn = get_connection()
    try:
        rows = []
        for t in list_slots_for_date(date_iso):
            booked = count_booked(conn, date_iso, t)
            available = max(0, CAPACITY - booked)
            pct = int((booked / CAPACITY) * 100) if CAPACITY else 0
            status = "FULL" if booked >= CAPACITY else "OPEN"
            rows.append(
                {
                    "time": t,
                    "booked": booked,
                    "capacity": CAPACITY,
                    "available": available,
                    "fill_%": pct,
                    "status": status,
                }
            )
        return pd.DataFrame(rows)
    finally:
        conn.close()

def kpi_counts_last_24h():
    since = datetime.utcnow() - timedelta(hours=24)
    fb = read_df("SELECT COUNT(1) AS n FROM feedback_log WHERE created_at >= %s", params=(since,))
    cp = read_df("SELECT COUNT(1) AS n FROM coupons WHERE created_at >= %s", params=(since,))
    rs = read_df("SELECT COUNT(1) AS n FROM reservations WHERE created_at >= %s", params=(since,))
    pr = read_df("SELECT COUNT(1) AS n FROM processed WHERE processed_at >= %s", params=(since,))
    return (
        int(fb.iloc[0]["n"]) if not fb.empty else 0,
        int(cp.iloc[0]["n"]) if not cp.empty else 0,
        int(rs.iloc[0]["n"]) if not rs.empty else 0,
        int(pr.iloc[0]["n"]) if not pr.empty else 0,
    )

def format_badge(label: str, kind: str = "ok"):
    cls = "badge-ok" if kind == "ok" else "badge-warn" if kind == "warn" else "badge-bad"
    st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)

def clear_caches():
    st.cache_data.clear()

# -----------------------------
# Header / Hero
# -----------------------------
left, right = st.columns([0.72, 0.28])
with left:
    st.markdown(
        f"""
        <div class="hero">
          <div style="font-size:32px;font-weight:800;">🍽️ {RESTAURANT_NAME} — Ops</div>
          <div class="subtle">Reservations • Inbox automation • Feedback • Coupons • Live status</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.write("")
    if st.button("🔄 Refresh", use_container_width=True):
        clear_caches()
        st.rerun()

st.write("")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown(f"### {RESTAURANT_NAME}")
nav = st.sidebar.radio(
    "Navigate",
    ["Overview", "Reservations", "Feedback", "Coupons", "System"],
    index=0,
)

st.sidebar.markdown("---")
day_focus = st.sidebar.date_input("Focus date", value=date.today())
show_cancelled = st.sidebar.checkbox("Show cancelled in lists", value=False)
st.sidebar.caption("Tip: set DASH_BG_URL env var to your own HD background image.")

# -----------------------------
# OVERVIEW
# -----------------------------
if nav == "Overview":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Tonight at a glance")

    slot_df = compute_slot_table(day_focus)
    total_booked = int(slot_df["booked"].sum()) if not slot_df.empty else 0
    total_capacity = int(slot_df["capacity"].sum()) if not slot_df.empty else 0
    full_slots = int((slot_df["status"] == "FULL").sum()) if not slot_df.empty else 0

    fb24, cp24, rs24, pr24 = kpi_counts_last_24h()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Booked seats (slot-count)", total_booked)
    c2.metric("Slots full", full_slots)
    c3.metric("Reservations (24h)", rs24)
    c4.metric("Feedback (24h)", fb24)
    c5.metric("Coupons (24h)", cp24)

    st.write("")

    # “Timeline” / fill bars
    st.markdown("#### Timeline")
    # Create a progress-like visual with bars
    for _, row in slot_df.iterrows():
        t = row["time"]
        booked = int(row["booked"])
        cap = int(row["capacity"])
        pct = int(row["fill_%"])
        status = row["status"]

        cols = st.columns([0.12, 0.68, 0.20])
        with cols[0]:
            st.markdown(f"**{t}**")
        with cols[1]:
            st.progress(min(100, max(0, pct)))
        with cols[2]:
            if status == "FULL":
                format_badge(f"{booked}/{cap} FULL", "bad")
            elif pct >= 70:
                format_badge(f"{booked}/{cap} Busy", "warn")
            else:
                format_badge(f"{booked}/{cap} Open", "ok")

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live system status")

    status_df = load_status()
    if status_df.empty:
        st.info("No status yet. Run the agent worker once to populate status.")
    else:
        st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# RESERVATIONS
# -----------------------------
elif nav == "Reservations":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Reservations")

    # Search
    q = st.text_input("Search by email / phone / confirmation code", value="").strip()

    colA, colB = st.columns([0.62, 0.38])
    with colA:
        st.markdown("#### Today / focus date list")
        df_today = load_today_reservations(day_focus)

        if not show_cancelled and not df_today.empty:
            df_today = df_today[df_today["status"] == "confirmed"]

        if q and not df_today.empty:
            mask = (
                df_today["email"].fillna("").str.contains(q, case=False)
                | df_today["phone"].fillna("").str.contains(q, case=False)
                | df_today["confirmation_code"].fillna("").str.contains(q, case=False)
                | df_today["name"].fillna("").str.contains(q, case=False)
            )
            df_today = df_today[mask]

        if df_today.empty:
            st.info("No reservations found for this date (or your search/filter removed them).")
        else:
            st.dataframe(df_today, use_container_width=True, hide_index=True)

        st.write("")
        st.markdown("#### Slot availability")
        st.dataframe(compute_slot_table(day_focus), use_container_width=True, hide_index=True)

    with colB:
        st.markdown("#### Create reservation (manual)")
        with st.form("manual_reservation", clear_on_submit=True):
            name = st.text_input("Name*")
            email_addr = st.text_input("Email*")
            phone = st.text_input("Phone (optional)")
            party = st.number_input("Party size", min_value=1, max_value=50, value=2)
            time_24 = st.selectbox("Time", list_slots_for_date(day_focus.isoformat()))
            submit = st.form_submit_button("Confirm", use_container_width=True)
            if submit:
                if not name.strip() or not email_addr.strip():
                    st.error("Name and email are required.")
                else:
                    conn = get_connection()
                    try:
                        ok, code, reason = reserve(
                            conn,
                            name=name.strip(),
                            email=email_addr.strip(),
                            phone=phone.strip() if phone.strip() else None,
                            party_size=int(party),
                            date_iso=day_focus.isoformat(),
                            time_24=time_24,
                            source="web",
                            capacity=CAPACITY,
                        )
                    finally:
                        conn.close()
                    if ok:
                        st.success(f"Confirmed: {code}")
                        clear_caches()
                    else:
                        st.error(reason or "Could not book that slot.")

        st.write("")
        st.markdown("#### Quick actions (by reservation id)")
        st.caption("Copy the `id` from the table, then cancel it here.")
        cancel_id = st.number_input("Reservation id", min_value=0, step=1, value=0)
        if st.button("Cancel reservation", use_container_width=True, disabled=(cancel_id <= 0)):
            conn = get_connection()
            try:
                ok = cancel_reservation_by_id(conn, int(cancel_id))
            finally:
                conn.close()
            if ok:
                st.success("Cancelled.")
                clear_caches()
            else:
                st.warning("No confirmed reservation found with that id (or it’s already cancelled).")

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Range view")
    c1, c2 = st.columns(2)
    with c1:
        start_day = st.date_input("From", value=date.today(), key="range_from")
    with c2:
        end_day = st.date_input("To", value=date.today() + timedelta(days=7), key="range_to")

    df_range = load_range_reservations(start_day, end_day)
    if not show_cancelled and not df_range.empty:
        df_range = df_range[df_range["status"] == "confirmed"]

    if q and not df_range.empty:
        mask = (
            df_range["email"].fillna("").str.contains(q, case=False)
            | df_range["phone"].fillna("").str.contains(q, case=False)
            | df_range["confirmation_code"].fillna("").str.contains(q, case=False)
            | df_range["name"].fillna("").str.contains(q, case=False)
        )
        df_range = df_range[mask]

    if df_range.empty:
        st.info("No reservations in this range.")
    else:
        st.dataframe(df_range, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# FEEDBACK
# -----------------------------
elif nav == "Feedback":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feedback & sentiment")

    df = load_feedback()
    if df.empty:
        st.info("No feedback logs yet.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Sentiment mix")
            by_sent = df["sentiment"].fillna("unknown").value_counts().reset_index()
            by_sent.columns = ["sentiment", "count"]
            st.bar_chart(by_sent.set_index("sentiment"))
        with c2:
            st.markdown("#### Discounts issued")
            by_disc = df["discount"].fillna(0).value_counts().sort_index().reset_index()
            by_disc.columns = ["discount", "count"]
            st.bar_chart(by_disc.set_index("discount"))

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# COUPONS
# -----------------------------
elif nav == "Coupons":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Coupons issued")

    df = load_coupons()
    if df.empty:
        st.info("No coupons yet.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="coupons.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# SYSTEM
# -----------------------------
elif nav == "System":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("System & agent activity")

    st.markdown("#### Recent processed emails (dedupe table)")
    df = load_processed()
    if df.empty:
        st.info("No processed rows yet (run agent).")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.write("")
    st.markdown("#### Health checklist")
    status_df = load_status()
    keys = set(status_df["key"].tolist()) if not status_df.empty else set()

    col1, col2, col3 = st.columns(3)
    with col1:
        format_badge("Agent heartbeat" if "agent_last_run" in keys else "Agent not reporting", "ok" if "agent_last_run" in keys else "bad")
    with col2:
        format_badge("DB connected" , "ok")
    with col3:
        format_badge("Gmail token set (worker)" , "warn")

    st.caption("If the agent isn’t reporting: check worker logs and env vars (DATABASE_URL, GMAIL_TOKEN_JSON).")
    st.markdown("</div>", unsafe_allow_html=True)
