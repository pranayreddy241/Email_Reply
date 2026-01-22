#!/usr/bin/env python3
"""
Restaurant Email Agent Dashboard (Postgres + Streamlit)

Shows:
- System status (last run, last action, etc.)
- Reservations
- Feedback log
- Coupons
- Processed message log

Run locally:
  streamlit run dashboard.py

On Render:
- Create a Web Service
- Start command: streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0
- Set env var DATABASE_URL (Render Postgres provides it automatically if attached)
"""

import os
import pandas as pd
import streamlit as st

from db_pg import get_pg_conn, ensure_schema_pg


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Email Agent Dashboard",
    page_icon="📬",
    layout="wide",
)

st.title("📬 Restaurant Email Agent Dashboard")
st.caption("Postgres-backed dashboard for reservations + feedback + coupons + processed emails")


# ---------------------------
# DB helpers
# ---------------------------
@st.cache_resource
def _conn():
    conn = get_pg_conn()
    # Ensure tables exist (safe to call repeatedly)
    ensure_schema_pg(conn)
    return conn


def qdf(sql: str, params=None) -> pd.DataFrame:
    """Query -> DataFrame (safe)."""
    conn = _conn()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        st.error(f"DB query failed: {e}")
        st.code(sql)
        return pd.DataFrame()


def exec_sql(sql: str, params=None) -> bool:
    """Execute non-select SQL."""
    conn = _conn()
    try:
        with conn.cursor() as c:
            c.execute(sql, params or ())
        conn.commit()
        return True
    except Exception as e:
        st.error(f"DB execute failed: {e}")
        st.code(sql)
        return False


# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")
refresh = st.sidebar.button("🔄 Refresh data")
limit = st.sidebar.slider("Rows per table", min_value=25, max_value=500, value=100, step=25)

if refresh:
    st.cache_data.clear()

st.sidebar.divider()
st.sidebar.subheader("Quick health checks")
st.sidebar.write("DATABASE_URL set:", "✅" if bool(os.getenv("DATABASE_URL")) else "❌")


# ---------------------------
# Top: Status cards
# ---------------------------
status_df = qdf(
    """
    SELECT key, value, updated_at
    FROM system_status
    ORDER BY updated_at DESC
    LIMIT 50
    """
)

col1, col2, col3, col4 = st.columns(4)

def _get_status(k: str):
    if status_df.empty:
        return None
    row = status_df[status_df["key"] == k]
    if row.empty:
        return None
    return row.iloc[0]["value"]

with col1:
    st.metric("Last Start", _get_status("agent_last_start") or "—")

with col2:
    st.metric("Last Run", _get_status("agent_last_run") or "—")

with col3:
    st.metric("Last Fetched", _get_status("agent_last_fetched") or "—")

with col4:
    st.metric("Last Action", _get_status("agent_last_seen_action") or "—")


# ---------------------------
# Tabs
# ---------------------------
tab_status, tab_res, tab_feedback, tab_coupons, tab_processed = st.tabs(
    ["✅ Status", "📅 Reservations", "💬 Feedback", "🏷️ Coupons", "🧾 Processed"]
)

# ---- Status tab ----
with tab_status:
    st.subheader("System Status (latest)")
    if status_df.empty:
        st.info("No system_status entries yet. Agent will populate this after it runs.")
    else:
        st.dataframe(status_df, use_container_width=True, hide_index=True)

# ---- Reservations tab ----
with tab_res:
    st.subheader("Reservations")

    # If your reservations table name differs, change it here:
    # Common names: reservations, reservation_requests, bookings
    # Your db_utils typically uses "reservations"
    res_df = qdf(
        f"""
        SELECT *
        FROM reservations
        ORDER BY created_at DESC
        LIMIT {int(limit)}
        """
    )

    if res_df.empty:
        st.info("No reservations yet (or table name differs).")
        st.write("If your table is not named `reservations`, tell me the actual name and I’ll adjust.")
    else:
        # Make it nicer if columns exist
        preferred_cols = [c for c in ["created_at", "email", "name", "phone", "date_iso", "time_24", "party_size", "status", "source", "confirmation_code"] if c in res_df.columns]
        if preferred_cols:
            st.dataframe(res_df[preferred_cols], use_container_width=True, hide_index=True)
        else:
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Cancel a reservation (by reservation id)")

        if "id" in res_df.columns:
            rid = st.number_input("Reservation ID to cancel", min_value=1, step=1)
            reason = st.text_input("Reason (optional)", value="dashboard_cancel")
            if st.button("❌ Cancel reservation"):
                ok = exec_sql(
                    """
                    UPDATE reservations
                    SET status = 'cancelled', cancelled_at = NOW(), cancel_reason = %s
                    WHERE id = %s
                    """,
                    (reason, int(rid)),
                )
                if ok:
                    st.success("Cancelled.")
                    st.cache_data.clear()
        else:
            st.warning("No `id` column found in reservations table, so cancel-by-id is disabled.")

# ---- Feedback tab ----
with tab_feedback:
    st.subheader("Feedback Log")

    fb_df = qdf(
        f"""
        SELECT *
        FROM feedback_log
        ORDER BY created_at DESC
        LIMIT {int(limit)}
        """
    )

    if fb_df.empty:
        st.info("No feedback entries yet.")
    else:
        preferred_cols = [c for c in ["created_at", "email", "sentiment", "score", "discount", "code"] if c in fb_df.columns]
        if preferred_cols:
            st.dataframe(fb_df[preferred_cols], use_container_width=True, hide_index=True)
        else:
            st.dataframe(fb_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Open a feedback entry")
        if "id" in fb_df.columns:
            fid = st.number_input("Feedback ID", min_value=1, step=1, key="fid")
            if st.button("🔎 View details"):
                row = qdf("SELECT * FROM feedback_log WHERE id = %s", (int(fid),))
                if row.empty:
                    st.warning("No such feedback id.")
                else:
                    r = row.iloc[0].to_dict()
                    st.write("**Email:**", r.get("email"))
                    st.write("**Sentiment / Score:**", f"{r.get('sentiment')} / {r.get('score')}")
                    st.write("**Discount / Code:**", f"{r.get('discount')}% / {r.get('code')}")
                    st.text_area("Original text", value=r.get("original_text", "") or "", height=200)
                    st.text_area("Reply text", value=r.get("reply_text", "") or "", height=200)
        else:
            st.info("No `id` column found in feedback_log.")

# ---- Coupons tab ----
with tab_coupons:
    st.subheader("Coupons")

    cp_df = qdf(
        f"""
        SELECT *
        FROM coupons
        ORDER BY created_at DESC
        LIMIT {int(limit)}
        """
    )

    if cp_df.empty:
        st.info("No coupons generated yet.")
    else:
        preferred_cols = [c for c in ["created_at", "email", "code", "discount", "sentiment", "score"] if c in cp_df.columns]
        if preferred_cols:
            st.dataframe(cp_df[preferred_cols], use_container_width=True, hide_index=True)
        else:
            st.dataframe(cp_df, use_container_width=True, hide_index=True)

# ---- Processed tab ----
with tab_processed:
    st.subheader("Processed Emails (dedupe log)")

    pr_df = qdf(
        f"""
        SELECT *
        FROM processed
        ORDER BY processed_at DESC
        LIMIT {int(limit)}
        """
    )

    if pr_df.empty:
        st.info("No processed messages logged yet.")
    else:
        st.dataframe(pr_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Danger zone: clear processed log")
        st.caption("Only use this if you *want* emails to be re-processed again.")
        if st.button("🧨 Clear processed table"):
            ok = exec_sql("TRUNCATE processed")
            if ok:
                st.success("Cleared processed table.")
                st.cache_data.clear()
