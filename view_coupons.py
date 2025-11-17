import sqlite3
from tabulate import tabulate

DB_PATH = "email_agent.sqlite"

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT email, code, discount, sentiment, score, created_at FROM coupons ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    if not rows:
        print("No coupons found yet.")
        return

    headers = ["Email", "Code", "Discount %", "Sentiment", "Score", "Created At"]
    print(tabulate(rows, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    try:
        from tabulate import tabulate  # ensure installed
    except ImportError:
        print("Install first: pip install tabulate")
        raise
    main()
