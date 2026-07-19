<!-- Put this in the Email_Reply repo as README.md. Written from the actual code. -->

# Restaurant Email Agent
An autonomous agent that reads a restaurant's Gmail inbox and handles **reservations and customer
feedback end-to-end**  booking, rescheduling, capacity checks, sentiment-scored coupons, and
personalized replies — sent automatically through the Gmail API, with every action logged to
Postgres.

**Stack:** Python · OpenAI (`gpt-4o-mini`) · Gmail API (OAuth 2.0) · PostgreSQL · Streamlit

-----

## What it does

For every unread email, the agent extracts intent with an LLM and takes the right action:

- **Reservations (capacity-aware):** books a table, issues a confirmation code, and stores it in
  Postgres  only if capacity allows. If the slot is full, it replies with the next available
  times. If details are missing (date / time / party size), it asks for exactly what's missing.
- **Reschedules / updates:** detects change requests, cancels the latest confirmed reservation, and
  rebooks confirming the update in one reply.
- **Feedback → coupons:** scores sentiment 1–5 (OpenAI, with a heuristic fallback if the API fails),
  picks a tiered discount (5–40% by sentiment/severity), generates a unique coupon code, writes a
  warm **personalized** reply, and logs the whole exchange.
- **Contact capture:** pulls the customer's phone from the LLM extraction or a regex fallback.

## Why it's reliable (the engineering, not just the demo)

- **Idempotent:** a Postgres `processed` table keyed on `Message-ID` guarantees no email is ever
  answered twice  critical for an agent that sends real replies in a loop.
- **Resilient Gmail I/O:** exponential-backoff retries on transient Gmail API errors (500/502/503/504)
  and multiple unread-fetch strategies.
- **Graceful degradation:** if OpenAI is unavailable, sentiment and replies fall back to heuristics
  and templates  the agent keeps working.
- **Deploy-ready:** reads `DATABASE_URL` and can bootstrap the Gmail token from an env var
  (`GMAIL_TOKEN_JSON`) for headless deployment (e.g. Render).

## Architecture

```
unread Gmail messages
        │  (filter no-reply/marketing)
        ▼
[summarize thread] → [LLM extract + decide]  →  action ∈ {confirm, ask_missing, feedback, skip}
                                                     │
        ┌────────────────────────────┬──────────────┴───────────────┐
        ▼                            ▼                              ▼
  reservation logic            ask for missing               sentiment → coupon
  (capacity, codes,            details                        → personalized reply
   reschedule, slots)
        │                            │                              │
        └──────────────► send reply via Gmail API ◄────────────────┘
                                     │
                         log to Postgres + mark READ
                         (idempotency via processed table)
```

A **Streamlit dashboard** (`dashboard.py`) provides a UI over the reservations, feedback log, and
issued coupons.

## Repository structure
```
agent.py              # Main agent: reservations + feedback, Gmail send, Postgres logging
agent_plus.py         # Enhanced agent variant
llm_agent.py          # LLM layer: thread summary, detail extraction, action decision
db_pg.py              # Postgres connection + schema bootstrap
db_utils.py           # Reservation/business logic: reserve(), capacity, slots, cancel
gmail_drafts.py       # Gmail draft-creation helper
dashboard.py          # Streamlit dashboard (index.html for the web view)
notifier.py           # Notification helper
view_coupons.py       # View issued coupons
train_classifier.py   # Intent-classifier training script
sample_training_data.jsonl   # Labeled sample intents
test_feedback.py      # Feedback-path test
requirements.txt
```

## Setup & run

**Requirements:** Python 3.10+, a PostgreSQL database, a Google Cloud project with the Gmail API
enabled (OAuth client credentials).

```bash
git clone https://github.com/Rainsongit/Email_Reply.git
cd Email_Reply
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` (do **not** commit it):
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=postgresql://user:pass@host:5432/dbname
RESTAURANT_NAME=My Restaurant
RESERVATION_PHONE=...
RESERVATION_LINK=...
MAX_PROCESS=10
```

Add your Gmail OAuth client as `client_secret.json` (a `token.json` is generated on first run).

```bash
python agent.py            # process unread inbox once (schedule via cron for continuous operation)
streamlit run dashboard.py # launch the dashboard
```

## Security
`client_secret.json`, `token.json`, and `.env` contain secrets — they must **never** be committed.
Add them to `.gitignore` and ship a `.env.example` with placeholders. (See the cleanup guide for a
secret scan.)
