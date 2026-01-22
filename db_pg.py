import os
import psycopg2

def _normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url

def get_pg_conn():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL env var not set")
    url = _normalize_db_url(url)
    return psycopg2.connect(url)

def ensure_schema_pg(conn):
    with conn.cursor() as c:
        # processed emails
        c.execute("""
            CREATE TABLE IF NOT EXISTS processed(
                message_id TEXT PRIMARY KEY,
                processed_at TIMESTAMP DEFAULT NOW(),
                action TEXT
            )
        """)

        # drafts (optional)
        c.execute("""
            CREATE TABLE IF NOT EXISTS drafts(
                id SERIAL PRIMARY KEY,
                to_email TEXT,
                subject TEXT,
                body TEXT,
                in_reply_to TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                sent_at TIMESTAMP
            )
        """)

        # coupons
        c.execute("""
            CREATE TABLE IF NOT EXISTS coupons(
                id SERIAL PRIMARY KEY,
                email TEXT,
                code TEXT UNIQUE,
                discount INT,
                sentiment TEXT,
                score INT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # feedback log
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback_log(
                id SERIAL PRIMARY KEY,
                email TEXT,
                sentiment TEXT,
                score INT,
                discount INT,
                code TEXT,
                original_text TEXT,
                reply_text TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # reservations
        c.execute("""
            CREATE TABLE IF NOT EXISTS reservations(
                id SERIAL PRIMARY KEY,
                confirmation_code TEXT UNIQUE,
                name TEXT,
                email TEXT,
                phone TEXT,
                party_size INT,
                slot_datetime TIMESTAMP,
                status TEXT DEFAULT 'confirmed',
                source TEXT DEFAULT 'email',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # In case you created an older version without some columns:
        c.execute("ALTER TABLE reservations ADD COLUMN IF NOT EXISTS phone TEXT")
        c.execute("ALTER TABLE reservations ADD COLUMN IF NOT EXISTS status TEXT")
        c.execute("ALTER TABLE reservations ADD COLUMN IF NOT EXISTS source TEXT")
        c.execute("ALTER TABLE reservations ADD COLUMN IF NOT EXISTS created_at TIMESTAMP")

        # helpful indexes for lookups
        c.execute("CREATE INDEX IF NOT EXISTS idx_res_email ON reservations(email)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_res_slot ON reservations(slot_datetime)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_res_status ON reservations(status)")

        # optional: reviews scaffold
        c.execute("""
            CREATE TABLE IF NOT EXISTS google_reviews(
                review_id TEXT PRIMARY KEY,
                rating INT,
                comment TEXT,
                author_name TEXT,
                created_at TIMESTAMP,
                replied INT DEFAULT 0,
                last_checked_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # optional: claims scaffold
        c.execute("""
            CREATE TABLE IF NOT EXISTS claims(
                id SERIAL PRIMARY KEY,
                review_id TEXT,
                name TEXT,
                email TEXT,
                phone TEXT,
                visit_date DATE,
                details TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # status
        c.execute("""
            CREATE TABLE IF NOT EXISTS system_status(
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)

    conn.commit()
