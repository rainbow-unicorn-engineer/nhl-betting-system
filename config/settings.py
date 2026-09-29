"""
nhl-betting-system/config/settings.py
Central configuration and database connection management.
"""
import os
import re
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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

# The driver is named explicitly: SQLAlchemy 2.1 maps a bare postgresql://
# URL to psycopg (v3), which this project does not install.
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

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
# A relative DATA_DIR (.env.example sets ./data) is relative to the repo,
# not to the folder a command happens to start in.
DATA_DIR = Path(os.getenv("DATA_DIR", "").strip() or PROJECT_ROOT / "data").expanduser()
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_ROOT / DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Local time ──
# "Today" (slate date, refresh window, season) is the USER's calendar day,
# not the machine clock's: a Docker container or cloud box runs on UTC, where
# 7pm Central is already tomorrow. LOCAL_TIMEZONE is an IANA name such as
# America/Chicago; unset (or unknown) = this machine's local zone.
LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "").strip()


def _load_local_tz() -> Optional[ZoneInfo]:
    if not LOCAL_TIMEZONE:
        return None
    try:
        return ZoneInfo(LOCAL_TIMEZONE)
    # OSError: a tzdata folder name such as "America" is a directory, which
    # raises IsADirectoryError (PermissionError on Windows) instead
    except (ZoneInfoNotFoundError, ValueError, OSError):
        logger.warning(f"LOCAL_TIMEZONE={LOCAL_TIMEZONE!r} is not an IANA time "
                       f"zone (e.g. America/Chicago) — using this machine's local zone")
        return None


LOCAL_TZ = _load_local_tz()


def to_local(dt: datetime) -> datetime:
    """An aware datetime converted to the user's local zone."""
    return dt.astimezone(LOCAL_TZ) if LOCAL_TZ is not None else dt.astimezone()


def local_now() -> datetime:
    """Now, as an aware datetime in the user's local zone."""
    return to_local(datetime.now().astimezone())


def local_today() -> date:
    """The user's calendar date right now."""
    return local_now().date()


def local_tz_name() -> str:
    """Human label for the zone in use (setup output, logs)."""
    if LOCAL_TZ is not None:
        return LOCAL_TIMEZONE
    return f"{local_now().tzname()} (this machine's zone; set LOCAL_TIMEZONE to override)"


# ── NHL Constants ──
def season_for(d: date) -> int:
    """The NHL season a date belongs to, as YYYYYYYY. The league year turns
    over on July 1: 2026-10-07 -> 20262027, 2027-03-01 -> 20262027,
    2026-06-30 -> 20252026."""
    if d.month >= 7:
        return d.year * 10000 + d.year + 1
    return (d.year - 1) * 10000 + d.year


def _seasons_between(first: int, last: int) -> list:
    return [y * 10000 + y + 1 for y in range(first // 10000, last // 10000 + 1)]


def _season_setting(name: str, default: int) -> int:
    """A season from the environment, such as 20252026: eight digits, the
    second year one after the first. Unset = default; anything else logs an
    error and uses default."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    if re.fullmatch(r"[0-9]{8}", raw) and int(raw[4:]) == int(raw[:4]) + 1:
        return int(raw)
    logger.error(f"{name}={raw!r} is not a season like 20252026 (eight digits, "
                 f"the second year one after the first) — using {default}")
    return default


def _backfill_seasons(first: int, current: int) -> list:
    """Every season from first through current. A first season later than
    current (BACKFILL_FIRST_SEASON after the season in use, or NHL_SEASON
    pinned before it) would make that list empty, and backfill would
    report success with nothing loaded: log an error and load current
    alone instead."""
    if first > current:
        logger.error(f"BACKFILL_FIRST_SEASON ({first}) is later than the "
                     f"season in use ({current}, from NHL_SEASON or today's "
                     f"date) — backfill loads {current} only")
        first = current
    return _seasons_between(first, current)


# NHL_SEASON pins the season (e.g. to finish a playoff run after July 1);
# unset = the season containing today's local date.
CURRENT_SEASON = _season_setting("NHL_SEASON", season_for(local_today()))
# Every season from BACKFILL_FIRST_SEASON through CURRENT_SEASON; never empty
BACKFILL_SEASONS = _backfill_seasons(
    _season_setting("BACKFILL_FIRST_SEASON", 20202021), CURRENT_SEASON)

# Team abbreviation mapping (handles historical changes)
TEAM_ABBREV_MAP = {
    "PHX": "ARI", "ARI": "UTA",  # Arizona -> Utah Hockey Club (2024)
    "ATL": "WPG",                  # Atlanta -> Winnipeg (2011)
}
