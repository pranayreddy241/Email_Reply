#!/usr/bin/env python3
"""
Restaurant Email Agent – Owner Dashboard (Premium)

Run:
  streamlit run dashboard.py

Reads from SQLite:
  - coupons
  - email_log (recommended; gives thread viewer + replies)
  - feedback_log (fallback if email_log doesn't exist)

Env:
  AGENT_DB_PATH
  RESTAURANT_NAME
  RESERVATION_LINK
  EMAIL_ADDRESS
  LOGO_URL (optional)
"""

import os
import sqlite3
from datetime import datetime, date, timedelta

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

# ---------- DB helpers ----------

def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(q, (name,)).fetchone() is not None

@st.cache_data(show_spinner=False)
def load_coupons() -> pd.DataFrame:
    conn = _connect()
    try:
        df = pd.read_sql_query(
            "SELECT email, code, discount, sentiment, score, created_at "
            "FROM coupons ORDER BY datetime(created_at) DESC",
            conn,
        )
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Could not load coupons from DB ({DB_PATH}): {e}")
        return pd.DataFrame(columns=["email","code","discount","sentiment","score","created_at"])
    finally:
        conn.close()

@st.cache_data(show_spinner=False)
def load_email_log() -> pd.DataFrame:
    """
    Preferred: email_log gives inbox/thread UX.
    Expected columns (recommended by agent):
      gmail_thread_id, gmail_msg_id, from_email, subject, intent, action,
      received_at, original_text, reply_text, coupon_code, discount, sentiment, score, created_at
    """
    conn = _connect()
    try:
        if not _table_exists(conn, "email_log"):
            return pd.DataFrame()
        df = pd.read_sql_query(
            "SELECT gmail_thread_id, gmail_msg_id, message_id, from_email, subject, intent, action, "
            "received_at, original_text, reply_text, coupon_code, discount, sentiment, score, created_at "
            "FROM email_log ORDER BY datetime(created_at) DESC",
            conn,
        )
        if not df.empty:
            for col in ["created_at", "received_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(show_spinner=False)
def load_feedback_log_fallback() -> pd.DataFrame:
    conn = _connect()
    try:
        if not _table_exists(conn, "feedback_log"):
            return pd.DataFrame()
        df = pd.read_sql_query(
            "SELECT email, sentiment, score, discount, code, original_text, reply_text, created_at "
            "FROM feedback_log ORDER BY datetime(created_at) DESC",
            conn,
        )
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

# ---------- UI helpers ----------

def _nice_dt(x) -> str:
    try:
        if pd.isna(x):
            return ""
        if isinstance(x, pd.Timestamp):
            return x.strftime("%b %d, %Y %I:%M %p")
        if isinstance(x, datetime):
            return x.strftime("%b %d, %Y %I:%M %p")
        return str(x)
    except Exception:
        return str(x)

def _sentiment_badge(sentiment: str) -> str:
    s = (sentiment or "neutral").lower()
    if s == "positive":
        return "tag-pos"
    if s == "negative":
        return "tag-neg"
    return "tag-neu"

def _anger_label(score: int) -> str:
    if score >= 5:
        return "🔥 Furious"
    if score == 4:
        return "😡 Angry"
    if score == 3:
        return "😐 Neutral"
    if score == 2:
        return "🙂 Happy"
    return "😍 Raving"

def _anger_color(score: int) -> str:
    if score >= 5:
        return "#ff2d2d"
    if score == 4:
        return "#ff6b2d"
    if score == 3:
        return "#fbbf24"
    if score == 2:
        return "#34d399"
    return "#22c55e"

def add_global_style():
    st.set_page_config(
        page_title=f"{RESTAURANT_NAME} – Owner Dashboard",
        page_icon="🍽️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left,#0b1220 0,#050914 40%,#020617 100%);
            color: #e5e7eb;
        }
        .main > div { padding-top: 0.9rem; }

        .soft-card {
            background: rgba(15,23,42,0.88);
            border-radius: 1.25rem;
            padding: 1.0rem 1.2rem;
            box-shadow: 0 18px 45px rgba(0,0,0,0.6);
            border: 1px solid rgba(148, 163, 184, 0.32);
        }
        .hero {
            background: linear-gradient(135deg, rgba(2,6,23,1), rgba(17,24,39,1));
            border-radius: 1.35rem;
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.28);
            box-shadow: 0 22px 55px rgba(0,0,0,0.65);
        }
        .big-title {
            font-size: 2.2rem;
            font-weight: 850;
            letter-spacing: 0.02em;
            color: #f9fafb;
            margin: 0;
        }
        .sub {
            font-size:0.92rem;
            color:#9ca3af;
            margin-top: 0.15rem;
        }
        .accent-badge {
            background: rgba(34,197,94,0.15);
            color: #bbf7d0;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 650;
            text-transform: uppercase;
            border: 1px solid rgba(34,197,94,0.45);
        }
        .metric-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            color: #9ca3af;
            letter-spacing: 0.1em;
            margin-bottom: 0.15rem;
        }
        .metric-value {
            font-size: 1.95rem;
            font-weight: 780;
            color: #f9fafb;
        }
        .tag-badge {
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 650;
            display:inline-block;
        }
        .tag-pos { background:#022c22; color:#6ee7b7; }
        .tag-neg { background:#450a0a; color:#fecaca; }
        .tag-neu { background:#0b1220; color:#bfdbfe; border:1px solid rgba(148,163,184,0.55); }

        .email-chip {
            font-size: 0.82rem;
            color:#e5e7eb;
            background: rgba(2,6,23,0.7);
            padding:0.28rem 0.62rem;
            border-radius:999px;
            border:1px solid rgba(148,163,184,0.55);
            display:inline-block;
        }
        .coupon-ticket {
            border-radius: 1.1rem;
            padding: 0.85rem 1rem;
            background: linear-gradient(135deg,#f97316,#ec4899);
            color: white;
            border: 1px dashed rgba(255,255,255,0.7);
            font-family: "SF Mono","Menlo",monospace;
            font-weight: 800;
            font-size: 1.05rem;
            letter-spacing: 0.03em;
        }
        .thread-bubble-user {
            background: rgba(148,163,184,0.12);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 1rem;
            padding: 0.9rem 1rem;
        }
        .thread-bubble-agent {
            background: rgba(34,197,94,0.10);
            border: 1px solid rgba(34,197,94,0.25);
            border-radius: 1rem;
            padding: 0.9rem 1rem;
        }
        .small-muted { color:#9ca3af; font-size:0.82rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def sidebar_filters_default_30_days():
    st.sidebar.markdown("### 🔍 Filters")
    today = date.today()
    default_from = today - timedelta(days=30)

    start = st.sidebar.date_input("From", default_from)
    end = st.sidebar.date_input("To", today)
    sentiment = st.sidebar.selectbox("Sentiment", ["all","positive","neutral","negative"], index=0)
    q = st.sidebar.text_input("Search email/subject contains", "")
    return start, end, sentiment, q

def apply_date_sentiment_search(df: pd.DataFrame, start: date, end: date, sentiment: str, q: str,
                                date_col: str = "created_at",
                                email_col: str = "email",
                                subject_col: str = "subject") -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if date_col in out.columns:
        out = out[out[date_col].notna()]
        out = out[(out[date_col].dt.date >= start) & (out[date_col].dt.date <= end)]

    if sentiment != "all" and "sentiment" in out.columns:
        out = out[out["sentiment"] == sentiment]

    if q.strip():
        s = q.strip().lower()
        if email_col in out.columns:
            out = out[out[email_col].fillna("").str.lower().str.contains(s) |
                      out.get(subject_col, pd.Series([""]*len(out))).fillna("").astype(str).str.lower().str.contains(s)]
    return out

# ---------- Main ----------

def main():
    add_global_style()

    coupons_df = load_coupons()
    email_log_df = load_email_log()
    feedback_fallback_df = load_feedback_log_fallback()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Control panel")
        st.markdown(f"**Restaurant:** `{RESTAURANT_NAME}`  \n**DB:** `{os.path.basename(DB_PATH)}`")
        start, end, sentiment, q = sidebar_filters_default_30_days()

        st.markdown("---")
        st.markdown("### 🔗 Quick links")
        if RESERVATION_LINK:
            st.markdown(f"• [Reservation page]({RESERVATION_LINK})")
        if OWNER_EMAIL:
            st.markdown("• [Open Gmail](https://mail.google.com/mail/u/0/#inbox)")

        st.markdown("---")
        st.caption("Tip: Keep the agent running on a schedule (cron) so this stays live.")

    # Hero
    if LOGO_URL:
        logo = f'<img src="{LOGO_URL}" style="width:48px;height:48px;border-radius:999px;object-fit:cover;border:1px solid rgba(148,163,184,0.35);" />'
    else:
        initials = (RESTAURANT_NAME[:2] or "R").upper()
        logo = f'<div style="width:48px;height:48px;border-radius:999px;background:linear-gradient(135deg,#fb7185,#7c5cff);display:flex;align-items:center;justify-content:center;font-weight:900;color:white;">{initials}</div>'

    st.markdown(
        f"""
        <div class="hero">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;">
            <div style="display:flex;align-items:center;gap:0.9rem;">
              {logo}
              <div>
                <div class="big-title">{RESTAURANT_NAME}</div>
                <div class="sub">Owner command center • feedback • coupons • replies • threads</div>
              </div>
            </div>
            <div><span class="accent-badge">engine online</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Filtered views
    filtered_coupons = apply_date_sentiment_search(
        coupons_df, start, end, sentiment, q,
        date_col="created_at", email_col="email", subject_col="email"
    )

    # For inbox, prefer email_log; fallback to feedback_log
    using_email_log = not email_log_df.empty
    if using_email_log:
        inbox_df = email_log_df.copy()
        inbox_df = inbox_df.rename(columns={"from_email":"email"})
        filtered_inbox = apply_date_sentiment_search(
            inbox_df, start, end, sentiment, q,
            date_col="created_at", email_col="email", subject_col="subject"
        )
    else:
        # fallback: no subject/thread fields
        fb = feedback_fallback_df.copy()
        filtered_inbox = apply_date_sentiment_search(
            fb, start, end, sentiment, q,
            date_col="created_at", email_col="email", subject_col="email"
        )

    # Metrics
    total = int(len(filtered_coupons)) if not filtered_coupons.empty else 0
    avg_disc = float(round(filtered_coupons["discount"].mean(), 2)) if total else 0.0
    neg = int((filtered_coupons["sentiment"] == "negative").sum()) if total else 0
    pos = int((filtered_coupons["sentiment"] == "positive").sum()) if total else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="metric-label">Total coupons</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total}</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-label">Avg discount %</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_disc}</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-label">Negative feedback</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{neg}</div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-label">Raving fans</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{pos}</div>', unsafe_allow_html=True)

    st.markdown("")

    tab1, tab2, tab3 = st.tabs(["🧠 Command Center", "📬 Inbox (Threads)", "🎟 Coupons"])

    # ---------------- Tab 1: Command Center ----------------
    with tab1:
        left, right = st.columns([1.1, 0.9])

        with left:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("#### Sentiment overview")
            if filtered_coupons.empty:
                st.info("No coupons in this date range yet.")
            else:
                sentiment_counts = (
                    filtered_coupons.groupby("sentiment")
                    .size().reset_index(name="count")
                )
                bar = (
                    alt.Chart(sentiment_counts)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("sentiment:N", title="Sentiment"),
                        y=alt.Y("count:Q", title="Coupons"),
                        tooltip=["sentiment","count"],
                        color=alt.Color("sentiment:N", legend=None),
                    )
                    .properties(height=240)
                )
                st.altair_chart(bar, use_container_width=True)

                daily = (
                    filtered_coupons.assign(day=lambda d: d["created_at"].dt.date)
                    .groupby("day").size().reset_index(name="coupons")
                )
                line = (
                    alt.Chart(daily)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("day:T", title="Day"),
                        y=alt.Y("coupons:Q", title="Coupons issued"),
                        tooltip=["day","coupons"],
                    )
                    .properties(height=240)
                )
                st.altair_chart(line, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("#### Latest coupons")
            if filtered_coupons.empty:
                st.info("No coupons yet.")
            else:
                for _, row in filtered_coupons.head(6).iterrows():
                    s = (row.get("sentiment") or "neutral")
                    badge = _sentiment_badge(s)
                    code = row.get("code", "")
                    st.markdown(
                        f"""
                        <div style="margin-bottom:0.85rem;">
                          <div class="email-chip">{row.get('email','')}</div>
                          <div style="display:flex;justify-content:space-between;align-items:center;gap:0.8rem;margin-top:0.55rem;">
                            <div class="coupon-ticket" style="flex:1;">{code}</div>
                            <div style="text-align:right;min-width:90px;">
                              <div class="small-muted">Discount</div>
                              <div style="font-size:1.45rem;font-weight:850;color:#f97316;">{int(row.get('discount',0))}%</div>
                              <span class="tag-badge {badge}">{s}</span>
                            </div>
                          </div>
                          <div class="small-muted" style="margin-top:0.35rem;">Issued {_nice_dt(row.get('created_at'))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    cols = st.columns([0.5, 0.5])
                    with cols[0]:
                        st.code(code, language="text")
                    with cols[1]:
                        st.caption("Copy the code from above.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Tab 2: Inbox (Threads) ----------------
    with tab2:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)

        if filtered_inbox.empty:
            if using_email_log:
                st.info("No inbox items in this range yet. Run the agent so email_log gets populated.")
            else:
                st.info("No feedback logged yet. Your agent must insert into feedback_log or (better) email_log.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("#### Inbox viewer")
            st.caption("Select an item on the left to view the customer email + your reply on the right.")

            left, right = st.columns([0.44, 0.56], gap="large")

            with left:
                # Build list items
                view = filtered_inbox.copy()

                if using_email_log:
                    # show only feedback/reservation first if you want; keep all now
                    view["score"] = pd.to_numeric(view.get("score"), errors="coerce").fillna(3).astype(int)
                    view["label"] = view.apply(
                        lambda r: f"{r.get('email','')} • {r.get('subject','(no subject)')[:34]} • {_anger_label(int(r.get('score',3)))}",
                        axis=1
                    )
                    options = view["label"].tolist()
                else:
                    # fallback feedback_log has no subject
                    view["score"] = pd.to_numeric(view.get("score"), errors="coerce").fillna(3).astype(int)
                    view["label"] = view.apply(
                        lambda r: f"{r.get('email','')} • {_anger_label(int(r.get('score',3)))} • {_nice_dt(r.get('created_at'))}",
                        axis=1
                    )
                    options = view["label"].tolist()

                sel = st.radio("Inbox", options, index=0, label_visibility="collapsed")
                idx = options.index(sel)
                row = view.iloc[idx]

            with right:
                # Header block
                sentiment = (row.get("sentiment") or "neutral")
                badge = _sentiment_badge(sentiment)
                score = int(row.get("score", 3) or 3)
                anger = _anger_label(score)

                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
                      <div>
                        <div class="email-chip">{row.get('email','')}</div>
                        <div style="margin-top:0.35rem;font-size:1.05rem;font-weight:800;color:#f9fafb;">
                          {row.get('subject','(no subject)') if using_email_log else "Feedback"}
                        </div>
                        <div class="small-muted" style="margin-top:0.15rem;">
                          {_nice_dt(row.get('created_at'))}
                        </div>
                      </div>
                      <div style="text-align:right;">
                        <span class="tag-badge {badge}">{sentiment}</span>
                        <div style="margin-top:0.45rem;font-weight:850;color:{_anger_color(score)};">
                          {anger} (score {score}/5)
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("")
                if row.get("coupon_code") or row.get("code"):
                    code = row.get("coupon_code") or row.get("code")
                    disc = row.get("discount")
                    st.markdown("##### Coupon")
                    st.markdown(f'<div class="coupon-ticket">{code}</div>', unsafe_allow_html=True)
                    cols = st.columns([0.55, 0.45])
                    with cols[0]:
                        st.code(code, language="text")
                    with cols[1]:
                        try:
                            st.metric("Discount", f"{int(disc)}%")
                        except Exception:
                            st.metric("Discount", f"{disc}%")

                # Anger meter (simple progress)
                st.markdown("##### Anger meter")
                st.progress(score / 5.0)

                st.markdown("##### Customer email")
                st.markdown(f'<div class="thread-bubble-user">{(row.get("original_text") or "(empty)").replace("\\n","<br>")}</div>', unsafe_allow_html=True)

                st.markdown("##### Our reply")
                reply = row.get("reply_text") or "(not logged yet)"
                st.markdown(f'<div class="thread-bubble-agent">{reply.replace("\\n","<br>")}</div>', unsafe_allow_html=True)

                st.markdown("")
                st.markdown("##### Quick actions")
                a1, a2, a3 = st.columns(3)
                with a1:
                    if RESERVATION_LINK:
                        st.link_button("Open reservation page", RESERVATION_LINK)
                with a2:
                    st.link_button("Open Gmail inbox", "https://mail.google.com/mail/u/0/#inbox")
                with a3:
                    st.caption("Easter egg: type `konami` in search 😄")

            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Tab 3: Coupons ----------------
    with tab3:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("#### Coupons table")
        if filtered_coupons.empty:
            st.info("No coupons in this range.")
        else:
            show = filtered_coupons.copy()
            show["created_at"] = show["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(show, use_container_width=True, hide_index=True)

            csv = filtered_coupons.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                csv,
                file_name="restaurant_coupons.csv",
                mime="text/csv",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("ℹ️ Setup checklist (agent + dashboard)", expanded=False):
        st.markdown(
            """
            **To get the best Inbox (Threads) experience**, your agent should write to `email_log`.

            Recommended columns to log in agent:
            - gmail_thread_id, gmail_msg_id, from_email, subject, intent, action, received_at
            - original_text, reply_text
            - coupon_code, discount, sentiment, score
            - created_at

            Dashboard defaults to **last 30 days**.
            """
        )

if __name__ == "__main__":
    main()
