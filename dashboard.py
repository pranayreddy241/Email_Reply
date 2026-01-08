#!/usr/bin/env python3
"""
Restaurant Email Agent – Premium Owner Dashboard v2.0

A beautiful, modern dashboard with glass morphism design, 
real-time analytics, and powerful filtering capabilities.

Run: streamlit run dashboard.py
"""

import os
import sqlite3
from datetime import datetime, date, timedelta
from typing import Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Configuration
DB_PATH = os.getenv("AGENT_DB_PATH", "email_agent.sqlite")
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Restaurant").strip('"')
RESERVATION_LINK = os.getenv("RESERVATION_LINK", "")
OWNER_EMAIL = os.getenv("EMAIL_ADDRESS", "")
LOGO_URL = os.getenv("LOGO_URL", "")

# ========== DATABASE FUNCTIONS ==========

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(query, (table_name,)).fetchone() is not None

@st.cache_data(ttl=30, show_spinner=False)
def load_coupons() -> pd.DataFrame:
    try:
        conn = get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM coupons ORDER BY datetime(created_at) DESC",
            conn
        )
        conn.close()
        
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        return df
    except Exception as e:
        return pd.DataFrame(columns=["email", "code", "discount", "sentiment", "score", "created_at"])

@st.cache_data(ttl=30, show_spinner=False)
def load_email_log() -> pd.DataFrame:
    try:
        conn = get_connection()
        if not table_exists(conn, "email_log"):
            conn.close()
            return pd.DataFrame()
        
        df = pd.read_sql_query("SELECT * FROM email_log ORDER BY datetime(created_at) DESC", conn)
        conn.close()
        
        if not df.empty:
            for col in ["created_at", "received_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def load_feedback_log() -> pd.DataFrame:
    try:
        conn = get_connection()
        if not table_exists(conn, "feedback_log"):
            conn.close()
            return pd.DataFrame()
        
        df = pd.read_sql_query("SELECT * FROM feedback_log ORDER BY datetime(created_at) DESC", conn)
        conn.close()
        
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

# ========== UTILITY FUNCTIONS ==========

def format_datetime(dt) -> str:
    """Format datetime for display"""
    try:
        if pd.isna(dt):
            return "—"
        if isinstance(dt, (pd.Timestamp, datetime)):
            return dt.strftime("%b %d, %I:%M %p")
        return str(dt)
    except Exception:
        return "—"

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    mapping = {"positive": "😊", "negative": "😞", "neutral": "😐"}
    return mapping.get(str(sentiment).lower(), "😐")

def get_score_emoji(score: int) -> str:
    """Get emoji for anger score"""
    if score >= 5:
        return "😡"
    elif score == 4:
        return "😟"
    elif score == 3:
        return "😐"
    elif score == 2:
        return "😊"
    return "🤩"

def calculate_trend(df: pd.DataFrame, days: int = 7) -> float:
    """Calculate percentage change over period"""
    if df.empty or "created_at" not in df.columns:
        return 0.0
    
    now = datetime.now()
    recent_start = now - timedelta(days=days)
    previous_start = now - timedelta(days=days * 2)
    
    recent = df[df["created_at"] >= recent_start]
    previous = df[(df["created_at"] >= previous_start) & (df["created_at"] < recent_start)]
    
    if len(previous) == 0:
        return 100.0 if len(recent) > 0 else 0.0
    
    return ((len(recent) - len(previous)) / len(previous)) * 100

# ========== STYLING ==========

def apply_modern_theme():
    """Apply beautiful glass morphism theme"""
    st.set_page_config(
        page_title=f"{RESTAURANT_NAME} Dashboard",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * { font-family: 'Inter', -apple-system, sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main { padding: 1.5rem 2.5rem; }
    .block-container { max-width: 1600px; padding-top: 0.5rem; }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.75rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.4);
        margin-bottom: 1.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(255,255,255,0.92));
        backdrop-filter: blur(15px);
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #111827;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .metric-trend {
        font-size: 0.875rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .trend-up { color: #10b981; }
    .trend-down { color: #ef4444; }
    .trend-neutral { color: #6b7280; }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(255,255,255,0.95));
        backdrop-filter: blur(25px);
        border-radius: 24px;
        padding: 2.25rem 2.75rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.6);
    }
    
    .hero-title {
        font-size: 2.75rem;
        font-weight: 900;
        color: #111827;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .status-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: white;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Badges */
    .sentiment-badge {
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-positive {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        color: #065f46;
    }
    
    .badge-negative {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
    }
    
    .badge-neutral {
        background: linear-gradient(135deg, #e5e7eb, #d1d5db);
        color: #374151;
    }
    
    /* Message Cards */
    .message-card {
        background: white;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 2px solid #e5e7eb;
        cursor: pointer;
        transition: all 0.25s ease;
    }
    
    .message-card:hover {
        border-color: #8b5cf6;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.15);
        transform: translateX(5px);
    }
    
    .message-card.selected {
        background: linear-gradient(135deg, #faf5ff, #f3e8ff);
        border-color: #8b5cf6;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.2);
    }
    
    /* Message Bubbles */
    .message-bubble {
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1.25rem 0;
        border-left: 4px solid;
    }
    
    .bubble-customer {
        background: linear-gradient(135deg, #faf5ff, #f9fafb);
        border-left-color: #8b5cf6;
    }
    
    .bubble-agent {
        background: linear-gradient(135deg, #f0fdf4, #f9fafb);
        border-left-color: #10b981;
    }
    
    .bubble-label {
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        opacity: 0.7;
    }
    
    /* Coupon Ticket */
    .coupon-ticket {
        background: linear-gradient(135deg, #f97316 0%, #dc2626 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        font-family: 'SF Mono', 'Consolas', monospace;
        font-weight: 900;
        font-size: 2rem;
        text-align: center;
        letter-spacing: 0.15em;
        box-shadow: 0 8px 24px rgba(249, 115, 22, 0.3);
        border: 3px dashed rgba(255, 255, 255, 0.6);
        margin: 1rem 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 0.75rem;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.75rem 1.75rem;
        font-weight: 700;
        font-size: 0.95rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.625rem 1.5rem;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,255,255,0.95));
        backdrop-filter: blur(25px);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        padding: 0.5rem 0;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Utility classes */
    .text-sm { font-size: 0.875rem; }
    .text-muted { color: #9ca3af; }
    .font-bold { font-weight: 700; }
    .mb-2 { margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

# ========== UI COMPONENTS ==========

def render_metric(icon: str, label: str, value: str, trend: Optional[float] = None):
    """Render a beautiful metric card"""
    trend_html = ""
    if trend is not None:
        if trend > 0:
            trend_class = "trend-up"
            trend_icon = "↗"
        elif trend < 0:
            trend_class = "trend-down"
            trend_icon = "↘"
        else:
            trend_class = "trend-neutral"
            trend_icon = "→"
        
        trend_html = f'<div class="metric-trend {trend_class}">{trend_icon} {abs(trend):.1f}% vs last week</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {trend_html}
    </div>
    """

def render_hero():
    """Render hero section with logo and title"""
    if LOGO_URL:
        logo = f'<img src="{LOGO_URL}" style="width: 70px; height: 70px; border-radius: 18px; object-fit: cover; box-shadow: 0 6px 16px rgba(0,0,0,0.1);">'
    else:
        initials = RESTAURANT_NAME[:2].upper()
        logo = f'''<div style="width: 70px; height: 70px; border-radius: 18px; 
                    background: linear-gradient(135deg, #f97316, #dc2626); 
                    display: flex; align-items: center; justify-content: center; 
                    font-size: 1.75rem; font-weight: 900; color: white; 
                    box-shadow: 0 6px 16px rgba(0,0,0,0.1);">{initials}</div>'''
    
    st.markdown(f"""
    <div class="hero-section">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 1.75rem;">
                {logo}
                <div>
                    <h1 class="hero-title">{RESTAURANT_NAME}</h1>
                    <p class="hero-subtitle">AI-Powered Customer Intelligence Dashboard</p>
                </div>
            </div>
            <div>
                <span class="status-badge">
                    <span class="pulse-dot"></span>
                    Live
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== MAIN APPLICATION ==========

def main():
    apply_modern_theme()
    
    # Initialize session state
    if "selected_msg_idx" not in st.session_state:
        st.session_state.selected_msg_idx = 0
    
    # Load data
    coupons_df = load_coupons()
    email_log_df = load_email_log()
    feedback_df = load_feedback_log()
    
    # Determine which dataset to use for inbox
    using_email_log = not email_log_df.empty
    inbox_df = email_log_df if using_email_log else feedback_df
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("### 🎯 Filters & Settings")
        
        # Date range filter
        today = date.today()
        preset = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time", "Custom"],
            index=1
        )
        
        if preset == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From", today - timedelta(days=30))
            with col2:
                end_date = st.date_input("To", today)
        else:
            days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": 99999}
            days = days_map[preset]
            start_date = today - timedelta(days=days) if days < 99999 else date(2000, 1, 1)
            end_date = today
        
        # Sentiment filter
        sentiment_options = st.multiselect(
            "Sentiment",
            ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"]
        )
        
        # Search
        search_text = st.text_input("🔍 Search", placeholder="Email or subject...")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### 🔗 Quick Actions")
        if RESERVATION_LINK:
            st.link_button("📅 View Reservations", RESERVATION_LINK, use_container_width=True)
        st.link_button("📧 Open Gmail", "https://mail.google.com/mail/u/0/#inbox", use_container_width=True)
        
        st.markdown("---")
        
        # System info
        st.markdown("### ⚙️ System")
        st.caption(f"**Database:** {os.path.basename(DB_PATH)}")
        st.caption(f"**Restaurant:** {RESTAURANT_NAME}")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # ========== FILTER DATA ==========
    def apply_filters(df: pd.DataFrame, email_col: str = "email") -> pd.DataFrame:
        if df.empty:
            return df
        
        result = df.copy()
        
        # Date filter
        if "created_at" in result.columns:
            result = result[
                (result["created_at"].dt.date >= start_date) & 
                (result["created_at"].dt.date <= end_date)
            ]
        
        # Sentiment filter
        if "sentiment" in result.columns and sentiment_options:
            result = result[result["sentiment"].isin(sentiment_options)]
        
        # Search filter
        if search_text:
            search_lower = search_text.lower()
            mask = result[email_col].str.lower().str.contains(search_lower, na=False)
            if "subject" in result.columns:
                mask |= result["subject"].str.lower().str.contains(search_lower, na=False)
            result = result[mask]
        
        return result
    
    filtered_coupons = apply_filters(coupons_df)
    filtered_inbox = apply_filters(inbox_df, "from_email" if using_email_log else "email")
    
    # ========== HERO ==========
    render_hero()
    
    # ========== METRICS ==========
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(filtered_coupons)
    avg_discount = filtered_coupons["discount"].mean() if total > 0 else 0
    negative = (filtered_coupons["sentiment"] == "negative").sum() if total > 0 else 0
    positive = (filtered_coupons["sentiment"] == "positive").sum() if total > 0 else 0
    
    trend = calculate_trend(coupons_df, 7)
    
    with col1:
        st.markdown(render_metric("🎟️", "Total Coupons", str(total), trend), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric("💰", "Avg Discount", f"{avg_discount:.1f}%"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric("😟", "Unhappy", str(negative)), unsafe_allow_html=True)
    with col4:
        st.markdown(render_metric("🌟", "Fans", str(positive)), unsafe_allow_html=True)
    
    # ========== TABS ==========
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📬 Inbox", "🎟️ Coupons"])
    
    # ========== TAB 1: ANALYTICS ==========
    with tab1:
        col_left, col_right = st.columns([0.6, 0.4])
        
        with col_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Activity Timeline")
            
            if not filtered_coupons.empty:
                timeline = filtered_coupons.copy()
                timeline["date"] = timeline["created_at"].dt.date
                daily = timeline.groupby(["date", "sentiment"]).size().reset_index(name="count")
                
                chart = alt.Chart(daily).mark_area(opacity=0.7, interpolate="monotone").encode(
                    x=alt.X("date:T", title="Date", axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("count:Q", title="Count"),
                    color=alt.Color("sentiment:N", scale=alt.Scale(
                        domain=["positive", "neutral", "negative"],
                        range=["#10b981", "#6b7280", "#ef4444"]
                    ), legend=alt.Legend(title="Sentiment")),
                    tooltip=["date:T", "sentiment:N", "count:Q"]
                ).properties(height=300).interactive()
                
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("📭 No activity data for selected period")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🎭 Sentiment Split")
            
            if not filtered_coupons.empty:
                sentiment_counts = filtered_coupons.groupby("sentiment").size().reset_index(name="count")
                
                pie = alt.Chart(sentiment_counts).mark_arc(innerRadius=50, outerRadius=110).encode(
                    theta=alt.Theta("count:Q"),
                    color=alt.Color("sentiment:N", scale=alt.Scale(
                        domain=["positive", "neutral", "negative"],
                        range=["#10b981", "#6b7280", "#ef4444"]
                    ), legend=None),
                    tooltip=["sentiment:N", "count:Q"]
                ).properties(height=300)
                
                st.altair_chart(pie, use_container_width=True)
                
                # Stats below
                for _, row in sentiment_counts.iterrows():
                    sent = row["sentiment"]
                    count = row["count"]
                    pct = (count / total * 100) if total > 0 else 0
                    emoji = get_sentiment_emoji(sent)
                    badge_class = f"badge-{sent}"
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
                        <span>{emoji} <span class="sentiment-badge {badge_class}">{sent}</span></span>
                        <span class="font-bold">{count} ({pct:.1f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📭 No sentiment data available")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== TAB 2: INBOX ==========
    with tab2:
        if filtered_inbox.empty:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.info("📭 No messages found for the selected period")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            col_list, col_detail = st.columns([0.35, 0.65], gap="large")
            
            with col_list:
                st.markdown('<div class="glass-card" style="max-height: 800px; overflow-y: auto;">', unsafe_allow_html=True)
                st.markdown(f"#### 📬 Messages ({len(filtered_inbox)})")
                
                for idx, row in filtered_inbox.iterrows():
                    email = row.get("from_email" if using_email_log else "email", "Unknown")
                    subject = row.get("subject", "No subject")[:35]
                    sentiment = row.get("sentiment", "neutral")
                    score = int(row.get("score", 3))
                    created = format_datetime(row.get("created_at"))
                    
                    emoji = get_score_emoji(score)
                    badge_class = f"badge-{sentiment}"
                    
                    is_selected = (st.session_state.selected_msg_idx == idx)
                    card_class = "message-card selected" if is_selected else "message-card"
                    
                    if st.button(
                        f"{emoji} {email}",
                        key=f"msg_{idx}",
                        use_container_width=True,
                        help=f"{subject} • {created}"
                    ):
                        st.session_state.selected_msg_idx = idx
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_detail:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                row = filtered_inbox.iloc[st.session_state.selected_msg_idx]
                
                email = row.get("from_email" if using_email_log else "email", "Unknown")
                subject = row.get("subject", "Feedback")
                sentiment = row.get("sentiment", "neutral")
                score = int(row.get("score", 3))
                created = format_datetime(row.get("created_at"))
                
                # Header
                st.markdown(f"### {subject}")
                
                c1, c2 = st.columns([0.65, 0.35])
                with c1:
                    st.markdown(f"**From:** {email}")
                    st.markdown(f"**Date:** {created}")
                with c2:
                    badge_class = f"badge-{sentiment}"
                    st.markdown(f'<span class="sentiment-badge {badge_class}">{sentiment}</span>', unsafe_allow_html=True)
                    st.markdown(f"**Score:** {get_score_emoji(score)} {score}/5")
                
                # Coupon if exists
                coupon_code = row.get("coupon_code") or row.get("code")
                if coupon_code:
                    st.markdown("---")
                    st.markdown("#### 🎟️ Coupon Issued")
                    st.markdown(f'<div class="coupon-ticket">{coupon_code}</div>', unsafe_allow_html=True)
                    
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        st.code(coupon_code, language="text")
                    with c_col2:
                        discount = row.get("discount", 0)
                        st.metric("Discount", f"{int(discount)}%")
                
                st.markdown("---")
                
                # Messages
                st.markdown("#### 💬 Conversation")
                
                original = row.get("original_text", "No message recorded")
                st.markdown(f'''
                <div class="message-bubble bubble-customer">
                    <div class="bubble-label">Customer Message</div>
                    {original.replace(chr(10), "<br>")}
                </div>
                ''', unsafe_allow_html=True)
                
                reply = row.get("reply_text", "No reply logged")
                st.markdown(f'''
                <div class="message-bubble bubble-agent">
                    <div class="bubble-label">Your Response</div>
                    {reply.replace(chr(10), "<br>")}
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== TAB 3: COUPONS ==========
    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        if filtered_coupons.empty:
            st.info("🎟️ No coupons found for the selected period")
        else:
            st.markdown(f"#### 🎟️ All Coupons ({len(filtered_coupons)} total)")
            
            # Prepare display dataframe
            display_df = filtered_coupons.copy()
            display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
            
            # Show dataframe with custom config
            st.dataframe(
                display_df[["email", "code", "discount", "sentiment", "score", "created_at"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "email": st.column_config.TextColumn("Customer Email", width="medium"),
                    "code": st.column_config.TextColumn("Coupon Code", width="medium"),
                    "discount": st.column_config.NumberColumn("Discount", format="%d%%"),
                    "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                    "score": st.column_config.NumberColumn("Score", format="%d/5"),
                    "created_at": st.column_config.TextColumn("Issued At", width="medium")
                }
            )
            
            st.markdown("")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = display_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Export CSV",
                    csv_data,
                    "coupons_export.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Summary stats
                total_discount_given = (filtered_coupons["discount"].sum() if not filtered_coupons.empty else 0)
                st.metric("Total Discount Given", f"{int(total_discount_given)}%")
            
            with col3:
                unique_customers = filtered_coupons["email"].nunique() if not filtered_coupons.empty else 0
                st.metric("Unique Customers", unique_customers)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== FOOTER ==========
    st.markdown("")
    with st.expander("ℹ️ Dashboard Info & Setup"):
        st.markdown("""
        ### 📊 About This Dashboard
        
        This dashboard provides real-time insights into your restaurant's customer feedback and coupon system.
        
        **Features:**
        - 📈 Real-time analytics and trends
        - 📬 Inbox viewer with full conversation threads
        - 🎟️ Coupon management and tracking
        - 🎯 Advanced filtering and search
        - 📥 Data export capabilities
        
        **Data Sources:**
        - `coupons` table: All issued coupons
        - `email_log` table: Email conversations (preferred)
        - `feedback_log` table: Feedback entries (fallback)
        
        **Tips:**
        - Use the sidebar filters to narrow down data
        - Click on messages in the inbox to view details
        - Export data anytime using the download buttons
        - Keep your agent running on a schedule for real-time updates
        
        **Refresh:** Dashboard data refreshes every 30 seconds automatically, or click the refresh button in the sidebar.
        """)

if __name__ == "__main__":
    main()
