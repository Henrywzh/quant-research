from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from db.core import DirectEngine
from db.models import Base

def init_db():
    if DirectEngine is None:
        raise RuntimeError("DIRECT_URL is not set. Add it to .env (port 5432)")

    # --- 1) Extensions in AUTOCOMMIT (so a failure doesn't poison the txn) ---
    with DirectEngine.connect() as c:
        ac = c.execution_options(isolation_level="AUTOCOMMIT")
        # pgcrypto should work on Supabase
        ac.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
        # Timescale may not be available on Supabase; try and ignore
        try:
            ac.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
            timescale_enabled = True
        except DBAPIError:
            print("[INFO] TimescaleDB not available on this server. Continuing without it.")
            timescale_enabled = False

    # --- 2) Schema, indexes, views in a clean transaction ---
    with DirectEngine.begin() as conn:
        # Create tables
        Base.metadata.create_all(bind=conn)

        # If Timescale exists, optionally hypertable-ize; otherwise just index
        if timescale_enabled:
            conn.execute(text(
                "SELECT create_hypertable('macro_observations', by_range('ts'), if_not_exists => TRUE);"
            ))

        # Always ensure useful index
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_macro_obs_id_ts ON macro_observations(macro_id, ts DESC);"
        ))

        # Enforce (code,country) uniqueness (idempotent)
        conn.execute(text("ALTER TABLE macro_series DROP CONSTRAINT IF EXISTS macro_series_code_key;"))
        # Ensure (code, country) is unique — idempotent
        conn.execute(text("""
        DO $$
        BEGIN
          -- drop the old single-column unique if it exists (harmless if not present)
          IF EXISTS (
            SELECT 1 FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            WHERE t.relname = 'macro_series' AND c.conname = 'macro_series_code_key'
          ) THEN
            ALTER TABLE macro_series DROP CONSTRAINT macro_series_code_key;
          END IF;

          -- add the new composite unique only if missing
          IF NOT EXISTS (
            SELECT 1 FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            WHERE t.relname = 'macro_series' AND c.conname = 'uq_macro_series_code_country'
          ) THEN
            ALTER TABLE macro_series ADD CONSTRAINT uq_macro_series_code_country UNIQUE (code, country);
          END IF;
        END$$;
        """))

        # Latest snapshot view
        conn.execute(text("""
        CREATE OR REPLACE VIEW latest_macro AS
        SELECT m.macro_id, m.code, m.country, m.category, o.ts, o.value
        FROM macro_series m
        JOIN LATERAL (
            SELECT ts, value
            FROM macro_observations o
            WHERE o.macro_id = m.macro_id
            ORDER BY ts DESC, revision DESC
            LIMIT 1
        ) o ON TRUE;
        """))

if __name__ == "__main__":
    init_db()
    print("DB initialized ✔ (Timescale optional)")
