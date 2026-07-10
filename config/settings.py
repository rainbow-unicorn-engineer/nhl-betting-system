"""
nhl-betting-system/config/settings.py
Central configuration and database connection management.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Logging ──
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nhl")

# ── Database ──
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "nhl_betting")
DB_USER = os.getenv("POSTGRES_USER", "nhl")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "nhl_dev_2026")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10, echo=False)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    """Get a database session (use as context manager)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def execute_sql(sql: str, params: dict = None):
    """Execute raw SQL and return results."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        conn.commit()
        return result

def check_db_connection() -> bool:
    """Verify database is reachable and schema exists."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = 'raw'"))
            count = result.scalar()
            if count == 0:
                logger.warning("Schema 'raw' not found. Run schema.sql first.")
                return False
            logger.info("Database connection OK. Schemas verified.")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

# ── API Keys ──
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ── Paths ──
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── NHL Constants ──
CURRENT_SEASON = 20252026
BACKFILL_SEASONS = [20202021, 20212022, 20222023, 20232024, 20242025, 20252026]

# Team abbreviation mapping (handles historical changes)
TEAM_ABBREV_MAP = {
    "PHX": "ARI", "ARI": "UTA",  # Arizona -> Utah Hockey Club (2024)
    "ATL": "WPG",                  # Atlanta -> Winnipeg (2011)
}
