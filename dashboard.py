#!/usr/bin/env python3
"""
Restaurant Email Agent – Owner Dashboard

Run locally:
  streamlit run dashboard.py

Reads from SQLite:
  - coupons      (created by the agent)
  - feedback_log (original emails + replies, if you've added that table)

Env:
  AGENT_DB_PATH
  RESTAURANT_NAME
  RESERVATION_LINK
  EMAIL_ADDRESS
  LOGO_URL (optional)
"""

import os
import sqlite3
from datetime import datetime, date
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("AGENT_DB_PATH", "email_agent.sqlite")
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant").strip('"')
RESERVATION_LINK = os.getenv("RESERVATION_LINK", "")
OWNER_EMAIL = os.getenv("EMAIL_ADDRESS", "")
LOGO_URL = os.getenv("LOGO_URL", "")


# ---------- Data helpers ----------

def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner=False)
def load_coupons() -> pd.DataFrame:
    try:
        conn = _connect()
        df = pd.read_sql_query(
            "SELECT email, code, discount, sentiment, score, created_at "
            "FROM coupons ORDER BY datetime(created_at) DESC",
            conn,
        )
        conn.close()
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"])
        return df
    except Exception as e:
        st.error(f"Could not load coupons from DB ({DB_PATH}): {e}")
        return pd.DataFrame(
            columns=["email", "code", "discount", "sentiment", "score", "created_at"]
        )


@st.cache_data(show_spinner=False)
def load_feedback_log() -> pd.DataFrame:
    """Original feedback emails + our replies (if table exists)."""
    try:
        conn = _connect()
        df = pd.read_sql_query(
            "SELECT email, sentiment, score, discount, code, original_text, "
            "reply_text, created_at "
            "FROM feedback_log ORDER BY datetime(created_at) DESC",
            conn,
        )
        conn.close()
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"])
        return df
    except Exception:
        # table might not exist yet – that's fine
        return pd.DataFrame(
            columns=[
                "email", "sentiment", "score", "discount", "code",
                "original_text", "reply_text", "created_at",
            ]
        )


# ---------- UI helpers ----------

def _nice_date(dt: datetime) -> str:
    try:
        return dt.strftime("%b %d, %Y %H:%M")
    except Exception:
        return str(dt)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### 🔍 Filters")

    sentiments = ["all"] + sorted(df["sentiment"].dropna().unique().tolist())
    sentiment_choice = st.sidebar.selectbox("Sentiment", sentiments, index=0)

    if not df.empty:
        min_date = df["created_at"].min().date()
        max_date = df["created_at"].max().date()
    else:
        today = date.today()
        min_date = max_date = today

    st.sidebar.markdown("#### Date range")
    start = st.sidebar.date_input("From", min_date)
    end = st.sidebar.date_input("To", max_date)

    email_search = st.sidebar.text_input("Email contains", "")

    filtered = df.copy()
    if sentiment_choice != "all":
        filtered = filtered[filtered["sentiment"] == sentiment_choice]

    filtered = filtered[(filtered["created_at"].dt.date >= start) &
                        (filtered["created_at"].dt.date <= end)]

    if email_search.strip():
        s = email_search.strip().lower()
        filtered = filtered[filtered["email"].str.lower().str.contains(s)]

    return filtered


def add_global_style():
    st.set_page_config(
        page_title=f"{RESTAURANT_NAME} – Feedback Dashboard",
        page_icon="🎟️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left,#0f172a 0,#020617 45%,#020617 100%);
            color: #e5e7eb;
        }
        .main > div {
            padding-top: 1.0rem;
        }
        .soft-card {
            background: rgba(15,23,42,0.96);
            border-radius: 1.1rem;
            padding: 1.0rem 1.2rem;
            box-shadow: 0 18px 40px rgba(0,0,0,0.5);
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
        .big-title {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            display: flex;
            align-items: center;
            gap: 0.8rem;
            color: #f9fafb;
        }
        .logo-circle {
            width: 44px;
            height: 44px;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 30%, #fecaca, #fb7185);
            display:flex;
            align-items:center;
            justify-content:center;
            font-size: 1.4rem;
            box-shadow: 0 10px 25px rgba(248,113,113,0.6);
        }
        .accent-badge {
            background: rgba(34,197,94,0.18);
            color: #bbf7d0;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            border: 1px solid rgba(34,197,94,0.5);
        }
        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #9ca3af;
            letter-spacing: 0.09em;
            margin-bottom: 0.15rem;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 700;
            color: #f9fafb;
        }
        .coupon-pill {
            font-family: "SF Mono","Menlo",monospace;
            font-weight: 700;
            font-size: 1.0rem;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            border: 1px dashed rgba(248,250,252,0.6);
            background: linear-gradient(135deg,#f97316,#ec4899);
            color: #f9fafb;
        }
        .tag-badge {
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .tag-pos { background:#022c22; color:#6ee7b7; }
        .tag-neg { background:#450a0a; color:#fecaca; }
        .tag-neu { background:#020617; color:#bfdbfe; border:1px solid rgba(148,163,184,0.7); }
        .email-chip {
            font-size: 0.8rem;
            color:#e5e7eb;
            background: rgba(15,23,42,0.9);
            padding:0.25rem 0.55rem;
            border-radius:999px;
            border:1px solid rgba(148,163,184,0.6);
        }
        .section-title {
            font-size:1.1rem;
            font-weight:600;
            margin-bottom:0.4rem;
            color:#e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Main layout ----------

def main():
    add_global_style()

    coupons_df = load_coupons()
    feedback_df = load_feedback_log()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Control panel")
        st.markdown(
            f"**Restaurant:** `{RESTAURANT_NAME}`  \n"
            f"**DB:** `{os.path.basename(DB_PATH)}`"
        )
        filtered = sidebar_filters(coupons_df)

        st.markdown("---")
        st.markdown("### 🔗 Quick links")
        if RESERVATION_LINK:
            st.markdown(f"• [Reservation page]({RESERVATION_LINK})")
        if OWNER_EMAIL:
            st.markdown(f"• [Open Gmail](https://mail.google.com/mail/u/0/#inbox)")

        st.markdown("---")
        st.markdown(
            "### ☁️ Share with owner\n"
            "Host this on **Streamlit Cloud** or **Render**, set the same env vars, "
            "and send them the URL."
        )

    # Header with logo
    logo_html = ""
    if LOGO_URL:
        logo_html = f'<img src="{LOGO_URL}" alt="logo" style="width:42px;height:42px;border-radius:999px;object-fit:cover;box-shadow:0 12px 30px rgba(0,0,0,0.55);" />'
    else:
        initials = (RESTAURANT_NAME[:2] or "R").upper()
        logo_html = f'<div class="logo-circle">{initials}</div>'

    st.markdown(
        f"""
        <div class="soft-card" style="margin-bottom:1.0rem; background:linear-gradient(135deg,#020617,#111827);">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:1.2rem;">
            <div style="display:flex;align-items:center;gap:0.9rem;">
              {logo_html}
              <div>
                <div class="big-title">
                  <span>{RESTAURANT_NAME}</span>
                </div>
                <div style="font-size:0.9rem;color:#9ca3af;margin-top:0.1rem;">
                  Live command center for feedback, coupons and replies.
                </div>
              </div>
            </div>
            <div>
              <span class="accent-badge">Feedback engine online</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if coupons_df.empty:
        st.warning("No coupons found yet. Once the agent replies to feedback emails, they’ll show up here.")
        return

    # Metrics row
    total = len(filtered)
    avg_disc = round(filtered["discount"].mean(), 2) if total else 0.0
    neg_count = int((filtered["sentiment"] == "negative").sum())
    pos_count = int((filtered["sentiment"] == "positive").sum())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-label">Total coupons</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-label">Avg discount %</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_disc}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-label">Negative feedback</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{neg_count}</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-label">Raving fans</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{pos_count}</div>', unsafe_allow_html=True)

    st.markdown("")

    # Tabs: Coupons vs Emails
    tab1, tab2 = st.tabs(["🎟 Coupons overview", "✉️ Emails & replies"])

    with tab1:
        c1, c2 = st.columns([1.1, 1])

        with c1:
            st.markdown('<div class="section-title">Sentiment & discount overview</div>', unsafe_allow_html=True)

            if not filtered.empty:
                sentiment_counts = (
                    filtered.groupby("sentiment")
                    .size()
                    .reset_index(name="count")
                )
                bar = (
                    alt.Chart(sentiment_counts)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("sentiment:N", title="Sentiment"),
                        y=alt.Y("count:Q", title="Number of coupons"),
                        tooltip=["sentiment", "count"],
                        color=alt.Color("sentiment:N", legend=None),
                    )
                    .properties(height=260)
                )
                st.altair_chart(bar, use_container_width=True)

            daily = (
                filtered.copy()
                .assign(day=lambda d: d["created_at"].dt.date)
                .groupby("day")
                .size()
                .reset_index(name="coupons")
            )
            if not daily.empty:
                line = (
                    alt.Chart(daily)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("day:T", title="Day"),
                        y=alt.Y("coupons:Q", title="Coupons issued"),
                        tooltip=["day", "coupons"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(line, use_container_width=True)

        with c2:
            st.markdown('<div class="section-title">Latest coupons</div>', unsafe_allow_html=True)
            latest = filtered.head(6)
            for _, row in latest.iterrows():
                sentiment = row["sentiment"] or "neutral"
                tag_class = (
                    "tag-pos" if sentiment == "positive"
                    else "tag-neg" if sentiment == "negative"
                    else "tag-neu"
                )
                st.markdown(
                    f"""
                    <div class="soft-card" style="margin-bottom:0.7rem;">
                      <div style="display:flex;justify-content:space-between;align-items:center;gap:0.6rem;">
                        <div style="flex:1;">
                          <div class="email-chip">{row['email']}</div>
                          <div class="coupon-pill" style="margin-top:0.35rem;">{row['code']}</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-size:0.78rem;color:#9ca3af;">Discount</div>
                          <div style="font-size:1.4rem;font-weight:700;color:#f97316;">{int(row['discount'])}%</div>
                          <span class="tag-badge {tag_class}">{sentiment}</span>
                        </div>
                      </div>
                      <div style="margin-top:0.4rem;font-size:0.75rem;color:#9ca3af;">
                        Issued at {_nice_date(row['created_at'])}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown('<div class="section-title">Full coupons log</div>', unsafe_allow_html=True)
        st.dataframe(
            filtered.assign(created_at=filtered["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")),
            use_container_width=True,
            hide_index=True,
        )
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download as CSV",
            csv,
            file_name="restaurant_coupons.csv",
            mime="text/csv",
        )

    with tab2:
        st.markdown('<div class="section-title">Feedback emails & our replies</div>', unsafe_allow_html=True)
        if feedback_df.empty:
            st.info(
                "No feedback emails logged yet. "
                "Make sure `feedback_log` table exists and `handle_feedback()` inserts into it."
            )
        else:
            # optional sentiment filter specifically for emails
            email_sentiments = ["all"] + sorted(feedback_df["sentiment"].dropna().unique().tolist())
            esel = st.selectbox("Filter by sentiment", email_sentiments, index=0)
            emails_view = feedback_df.copy()
            if esel != "all":
                emails_view = emails_view[emails_view["sentiment"] == esel]

            for _, row in emails_view.head(15).iterrows():
                sentiment = row["sentiment"] or "neutral"
                tag_class = (
                    "tag-pos" if sentiment == "positive"
                    else "tag-neg" if sentiment == "negative"
                    else "tag-neu"
                )
                st.markdown(
                    f"""
                    <div class="soft-card" style="margin-bottom:0.75rem;">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;">
                        <div style="flex:1;">
                          <div class="email-chip">{row['email']}</div>
                          <div style="margin-top:0.25rem;font-size:0.8rem;color:#9ca3af;">
                            Code <span style="font-family:'SF Mono','Menlo',monospace;">{row['code']}</span>
                            &nbsp;·&nbsp; <strong>{int(row['discount'])}%</strong> off
                          </div>
                        </div>
                        <div style="text-align:right;">
                          <span class="tag-badge {tag_class}">{sentiment}</span>
                          <div style="font-size:0.7rem;color:#9ca3af;margin-top:0.18rem;">
                            {_nice_date(row['created_at'])}
                          </div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("View customer email"):
                    st.write(row["original_text"] or "(empty)")
                with st.expander("View our reply"):
                    st.write(row["reply_text"] or "(empty)")

    # How it works
    st.markdown("---")
    with st.expander("ℹ️ How this system works & setup checklist", expanded=False):
        st.markdown(
            """
            1. **Agent (`agent.py`)** reads new emails from Gmail, decides what to do, and:
               - Replies automatically to feedback emails with a coupon  
               - Stores coupons in the `coupons` table  
               - (With the snippet we added) stores original email + reply in `feedback_log`

            2. **This dashboard (`dashboard.py`)**:
               - Connects to the same SQLite DB (`AGENT_DB_PATH`)  
               - Renders charts, metrics, and coupon cards  
               - Shows full email + reply text under *Emails & replies* tab  

            3. **To run everything:**
               - `.env` with `OPENAI_API_KEY`, Gmail `EMAIL_ADDRESS` + app password,  
                 `RESTAURANT_NAME`, `AGENT_DB_PATH`, etc.  
               - `client_secret.json` + `token.json` for the Gmail API agent.  
               - Run `python agent.py` on a schedule (cron / PM2 / background service).  
               - Run `streamlit run dashboard.py` locally or host it on Streamlit Cloud.
            """
        )


if __name__ == "__main__":
    main()
