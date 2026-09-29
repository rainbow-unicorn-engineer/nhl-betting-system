"""
config/migrate.py
In-place, idempotent upgrades for databases created from an older
db/schema.sql (schema.sql only runs when the Docker volume is first
created, so existing databases never see its new columns).

ensure_schema() adds any missing column and is a no-op afterwards. It runs
once per process: pipeline.py calls it after every successful
check_db_connection(), and the entry points that need the new columns
(ingest_schedule, snapshot_odds, generate_recommendations, settle_paper,
clv_report) call it too, so `python -m ...` module CLIs upgrade an old
database as well.

seed_venues() applies db/seed_venues.sql (arena coordinates and time
zones for the travel features) through SQLAlchemy, so the seed is one
command that reads the same on macOS, Linux and Windows:
`python -m config.migrate --seed-venues`. `python pipeline.py setup` runs
it on its own when raw.teams has no coordinates yet.
"""
import argparse
import logging
from pathlib import Path

from sqlalchemy import text

from config.settings import PROJECT_ROOT, engine

logger = logging.getLogger("nhl.migrate")

SEED_VENUES_SQL = PROJECT_ROOT / "db" / "seed_venues.sql"

# (schema, table, column, type/default). Keep in sync with db/schema.sql.
COLUMNS = (
    # Puck drop as an instant (NHL startTimeUTC). Odds matching and the
    # closing-line window key on it; raw.games.date is the league's local
    # (Eastern) schedule date.
    ("raw", "games", "start_time_utc", "TIMESTAMPTZ"),
    # NHL gameScheduleState: OK, PPD (postponed), SUSP (suspended), CNCL
    # (cancelled). The live slate skips the last three; settlement voids
    # picks on PPD/CNCL games.
    ("raw", "games", "schedule_state", "VARCHAR(10)"),
    # captured_at (naive UTC, like raw.odds_snapshots.captured_at) of the
    # snapshot a pick's price came from; NULL when priced from
    # raw.historical_odds (simulation).
    ("betting", "recommendations", "priced_at", "TIMESTAMP"),
    # The game's start_time_utc when the pick was written. Settlement voids
    # the pick when the game starts more than 3 hours away from it (moved);
    # NULL (a pick from before this column) falls back to the 36-hour rule.
    ("betting", "recommendations", "scheduled_start", "TIMESTAMPTZ"),
    # Paper trail until a human records a real bet (was betting/settle.py _DDL)
    ("betting", "placed_bets", "is_paper", "BOOLEAN NOT NULL DEFAULT TRUE"),
)

_done = False


def ensure_schema() -> None:
    """Add any missing COLUMNS. Checks information_schema first so an
    up-to-date database takes no table locks."""
    global _done
    if _done:
        return
    with engine.begin() as conn:
        present = {tuple(r) for r in conn.execute(text("""
            SELECT table_schema, table_name, column_name
            FROM information_schema.columns
            WHERE table_schema IN ('raw', 'betting')
        """))}
        for schema, table, column, ddl_type in COLUMNS:
            if (schema, table, column) in present:
                continue
            logger.info(f"Schema upgrade: adding {schema}.{table}.{column}")
            conn.execute(text(f"ALTER TABLE {schema}.{table} "
                              f"ADD COLUMN IF NOT EXISTS {column} {ddl_type}"))
    _done = True


# ── Venue seed ─────────────────────────────────────────────────────

def split_sql(script: str) -> list:
    """The statements of a plain SQL script, comments removed. Splits on
    semicolons outside 'string literals', "quoted identifiers", -- line
    comments and /* block comments */. Dollar-quoted bodies ($$ ... $$)
    are not supported; the seed file has none."""
    statements, buf = [], []
    i, n = 0, len(script)
    while i < n:
        c = script[i]
        if c in ("'", '"'):
            j = i + 1
            while j < n:
                if script[j] == c:
                    if j + 1 < n and script[j + 1] == c:   # '' or "" escape
                        j += 2
                        continue
                    break
                j += 1
            buf.append(script[i:j + 1])
            i = j + 1
        elif script.startswith("--", i):
            j = script.find("\n", i)
            i = n if j == -1 else j           # keep the newline itself
        elif script.startswith("/*", i):
            j = script.find("*/", i + 2)
            i = n if j == -1 else j + 2
            buf.append(" ")
        elif c == ";":
            statements.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(c)
            i += 1
    statements.append("".join(buf))
    return [s.strip() for s in statements if s.strip()]


def seed_venues(path: Path = SEED_VENUES_SQL, db=None) -> int:
    """Apply db/seed_venues.sql in one transaction; returns the number of
    statements run. The file is idempotent (INSERT ... ON CONFLICT DO
    UPDATE), so re-running it is safe. Read as UTF-8: it spells Montréal
    with an accent, which the Windows default code page would garble."""
    statements = split_sql(Path(path).read_text(encoding="utf-8"))
    with (db or engine).begin() as conn:
        for stmt in statements:
            # exec_driver_sql: no bind-parameter parsing of the SQL text
            conn.exec_driver_sql(stmt)
    logger.info(f"Venue seed applied from {Path(path).name} "
                f"({len(statements)} statement(s))")
    return len(statements)


def venues_missing() -> bool:
    """True when no team in raw.teams has arena coordinates yet."""
    with engine.connect() as conn:
        return not conn.execute(text(
            "SELECT COUNT(*) FROM raw.teams WHERE latitude IS NOT NULL")).scalar()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m config.migrate",
        description="Add any columns a database made from an older "
                    "db/schema.sql is missing (pipeline commands do this on "
                    "their own). Safe to re-run.")
    parser.add_argument("--seed-venues", action="store_true",
                        help="Also apply db/seed_venues.sql: arena coordinates "
                             "and time zones for the travel features. Safe to "
                             "re-run")
    args = parser.parse_args(argv)
    ensure_schema()
    print("Schema is up to date.")
    if args.seed_venues:
        seed_venues()
        print("Venue seed applied.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
