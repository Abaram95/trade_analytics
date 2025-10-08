# core/database/engine2.py

import os, time
from contextlib import suppress
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import make_url

load_dotenv(override=True)

DB_URL = os.getenv("NEON_DATABASE_URL")  # ou DATABASE_URL
if not DB_URL:
    raise RuntimeError("Defina NEON_DATABASE_URL ou DATABASE_URL")

APP_NAME = os.getenv("APP_NAME", "trade_analytics")

url = make_url(DB_URL)
is_pgbouncer = (url.port == 6543) or (os.getenv("USE_PGBOUNCER", "0") == "1")

# ------------ CONNECT_ARGS sem 'options' ------------
CONNECT_ARGS = {
    "connect_timeout": 10,
    # keepalive TCP
    "keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 5,
}
# psycopg3 atrás do PgBouncer: desabilita server-side prepare
# só setar se o driver for psycopg3
if is_pgbouncer and DB_URL.startswith("postgresql+psycopg://"):
    CONNECT_ARGS["prepare_threshold"] = 0

POOL = dict(pool_size=5, max_overflow=5, pool_pre_ping=True, pool_recycle=1800, pool_timeout=20)

engine = create_engine(DB_URL, connect_args=CONNECT_ARGS, **POOL)

# ------------ SETs pós-conexão (aceitos pelo pooler) ------------
@event.listens_for(engine, "connect")
def _apply_session_settings(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    # cuidado com aspas; APP_NAME vem do env
    cur.execute(f"SET application_name = '{APP_NAME}'")
    cur.execute("SET statement_timeout = 60000")                   # 60s
    cur.execute("SET idle_in_transaction_session_timeout = 15000") # 15s
    cur.close()

def wait_for_db(max_wait_seconds: int = 60, sleep_seconds: float = 2.0) -> None:
    deadline = time.time() + max_wait_seconds
    last_err = None
    while time.time() < deadline:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return
        except Exception as e:
            last_err = e
            time.sleep(sleep_seconds)
    raise RuntimeError(f"DB não ficou pronto em {max_wait_seconds}s: {last_err}")

def init_db_or_die():
    wait_for_db(int(os.getenv("DB_WAIT_MAX_SECONDS", "60")))
    with suppress(Exception):
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements"))
