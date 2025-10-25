import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")  # pooler (6543) for ETL
DIRECT_URL   = os.getenv("DIRECT_URL")    # direct (5432) for DDL/extensions

def make_engine(url: str):
    if not url:
        raise RuntimeError("Missing database URL")
    return create_engine(url, pool_pre_ping=True, future=True)

Engine = make_engine(DATABASE_URL)      # used by ETL
DirectEngine = make_engine(DIRECT_URL)  # used by init/migrations

SessionLocal = sessionmaker(bind=Engine, autoflush=False, autocommit=False, future=True)
