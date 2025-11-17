import sqlite3
import pandas as pd
import streamlit as st
import os

DB_PATH = os.getenv("AGENT_DB_PATH", "email_agent.sqlite")

st.set_page_config(page_title="Restaurant Email Agent – Coupons", layout="wide")

st.title("🎟️ Restaurant Feedback Coupons Dashboard")

@st.cache_data
def load_coupons(db_path: str):
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["email","code","discount","sentiment","score","created_at"])
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT email, code, discount, sentiment, score, created_at FROM coupons ORDER BY created_at DESC",
        conn
    )
    conn.close()
    return df

df = load_coupons(DB_PATH)

if df.empty:
    st.info("No coupons have been generated yet. Once feedback emails are processed, they will appear here.")
else:
    # Sidebar filters
    st.sidebar.header("Filters")
    sentiments = ["all"] + sorted(df["sentiment"].dropna().unique().tolist())
    selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)

    if selected_sentiment != "all":
        df = df[df["sentiment"] == selected_sentiment]

    st.subheader("Coupons log")
    st.dataframe(df, use_container_width=True, height=400)

    # Simple stats
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total coupons", len(df))
    with col2:
        if len(df) > 0:
            st.metric("Avg discount %", round(df["discount"].mean(), 2))
        else:
            st.metric("Avg discount %", "–")
    with col3:
        if len(df) > 0:
            angry = (df["sentiment"] == "negative").sum()
            st.metric("Negative feedback count", angry)
        else:
            st.metric("Negative feedback count", "–")

    st.download_button(
        "⬇️ Download as CSV",
        data=df.to_csv(index=False),
        file_name="coupons_export.csv",
        mime="text/csv"
    )
