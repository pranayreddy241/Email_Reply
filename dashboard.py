#!/usr/bin/env python3
"""
Restaurant Email Agent – Owner Dashboard
----------------------------------------

Run locally:
  streamlit run dashboard.py

Reads from the same SQLite DB used by the agent:
  - coupons table (email, code, discount, sentiment, score, created_at)

Uses:
  AGENT_DB_PATH   -> path to email_agent.sqlite (same as agent)
  RESTAURANT_NAME -> branding
  RESERVATION_LINK, EMAIL_ADDRESS (optional) for quick links
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
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant")
RESERVATION_LINK = os.getenv("RESERVATION_LINK", "")
OWNER_EMAIL = os.getenv("EMAIL_ADDRESS", "")


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


# ---------- UI helpers ----------

def _nice_date(dt: datetime) -> str:
    try:
        return dt.strftime("%b %d, %Y %H:%M")
    except Exception:
        return str(dt)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### 🔍 Filters")

    # sentiment filter
    sentiments = ["all"] + sorted(df["sentiment"].dropna().unique().tolist())
    sentiment_choice = st.sidebar.selectbox("Sentiment", sentiments, index=0)

    # date range filter
    if not df.empty:
        min_date = df["created_at"].min().date()
        max_date = df["created_at"].max().date()
    else:
        today = date.today()
        min_date = max_date = today

    st.sidebar.markdown("#### Date range")
    start = st.sidebar.date_input("From", min_date)
    end = st.sidebar.date_input("To", max_date)

    # email search
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
        page_title=f"{RESTAURANT_NAME} – Coupon Dashboard",
        page_icon="🎟️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 1.2rem;
        }
        .big-title {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .accent-badge {
            background: linear-gradient(135deg,#ff4b6e,#ff9f43);
            color: white;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .soft-card {
            background: #ffffff;
            border-radius: 1rem;
            padding: 1.0rem 1.1rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.25);
        }
        .coupon-pill {
            font-family: "SF Mono","Menlo",monospace;
            font-weight: 700;
            font-size: 1.1rem;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            border: 1px dashed rgba(148,163,184,0.9);
            background: linear-gradient(135deg,#fef3c7,#fee2e2);
        }
        .tag-badge {
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .tag-pos { background:#ecfdf5; color:#15803d; }
        .tag-neg { background:#fef2f2; color:#b91c1c; }
        .tag-neu { background:#eff6ff; color:#1d4ed8; }
        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            color: #6b7280;
            letter-spacing: 0.06em;
            margin-bottom: 0.1rem;
        }
        .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Main layout ----------

def main():
    add_global_style()

    df = load_coupons()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Control panel")
        st.markdown(
            f"**Restaurant:** `{RESTAURANT_NAME}`  \n"
            f"**DB:** `{os.path.basename(DB_PATH)}`"
        )
        filtered = sidebar_filters(df)

        st.markdown("---")
        st.markdown("### 🔗 Quick links")
        if RESERVATION_LINK:
            st.markdown(f"• [Reservation page]({RESERVATION_LINK})")
        if OWNER_EMAIL:
            st.markdown(f"• [Open Gmail](https://mail.google.com/mail/u/0/#inbox)")

        st.markdown("---")
        st.markdown(
            "### 🙋 Need to share?\n"
            "Deploy this dashboard to **Streamlit Cloud** or **Render** "
            "and log in with your restaurant Gmail/OpenAI keys."
        )

    # Header
    st.markdown(
        f"""
        <div class="soft-card" style="margin-bottom:1.2rem; background:linear-gradient(135deg,#eef2ff,#fef3c7);">
          <div class="big-title">
            <span class="accent-badge">Live</span>
            <span>Restaurant Feedback Coupons Dashboard</span>
          </div>
          <div style="margin-top:0.4rem; font-size:0.95rem; color:#4b5563;">
            {RESTAURANT_NAME}’s control room for angry customers, happy regulars, and all the coupons in between.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("No coupons found yet. Once the agent replies to feedback emails, they’ll show up here.")
        return

    # Top metrics row
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

    # Main body: two columns – charts + latest coupons list
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.subheader("📊 Sentiment & discount overview")

        # Bar chart – count by sentiment
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

        # Line chart – coupons over time
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
        st.subheader("🎟️ Latest coupons")

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
                      <div style="font-size:0.85rem;color:#6b7280;">{row['email']}</div>
                      <div class="coupon-pill" style="margin-top:0.25rem;">{row['code']}</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:0.8rem;color:#6b7280;">Discount</div>
                      <div style="font-size:1.4rem;font-weight:700;">{int(row['discount'])}%</div>
                      <span class="tag-badge {tag_class}">{sentiment}</span>
                    </div>
                  </div>
                  <div style="margin-top:0.4rem;font-size:0.78rem;color:#9ca3af;">
                    Issued at {_nice_date(row['created_at'])}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Full table + download
    st.subheader("📋 Full coupons log")
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

    # How it works / requirements
    with st.expander("ℹ️ How this system works & requirements", expanded=False):
        st.markdown(
            """
            **What this dashboard is showing**

            - Every time the email agent spots a feedback email, it:
              - Analyzes the sentiment & how upset the guest is
              - Chooses a discount (5–40%) and generates a unique coupon code
              - Sends a personalized reply and stores the coupon in SQLite

            - This page reads from that same database and turns it into:
              - Live metrics and charts
              - A log of all coupons issued
              - Filters by sentiment, date, and email

            **To run this end-to-end you need**

            1. A `.env` file with:
               - `OPENAI_API_KEY` – your paid OpenAI key  
               - `EMAIL_ADDRESS` + Gmail **App Password**  
               - `RESTAURANT_NAME`, `RESERVATION_LINK` (optional but nice)  
               - `AGENT_DB_PATH` – same DB path for both the agent & this dashboard  

            2. Gmail OAuth credentials (`client_secret.json`) for the agent version
               that uses the Gmail API.

            3. The agent script (`agent.py`) running periodically (cron, GitHub
               Actions, or a small server) to read emails and write to the DB.

            **Hosting this dashboard publicly**

            - Push your project to GitHub  
            - Create a free app on **Streamlit Cloud** or **Render**  
            - Set the environment variables there (same as your local `.env`)  
            - Point the app to `dashboard.py`  

            Then your restaurant owner can simply visit a URL like  
            `https://your-app-name.streamlit.app` and see this dashboard.
            """
        )


if __name__ == "__main__":
    main()
