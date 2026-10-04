"""
config/migrate.py
In-place, idempotent upgrades for databases created from an older
db/schema.sql (schema.sql only runs when the Docker volume is first
created, so existing databases never see its new tables and columns).

ensure_schema() creates any missing table in TABLES, with its indexes,
then adds any missing column in COLUMNS, and is a no-op afterwards. It
reads information_schema first, so an up-to-date database runs no DDL and
takes no table locks. It runs once per process: pipeline.py calls it after
every successful check_db_connection(), and the entry points that need the
new columns (ingest_schedule, snapshot_odds, generate_recommendations,
settle_paper, clv_report, the NHL feed snapshot) call it too, so
`python -m ...` module CLIs upgrade an old database as well. The modules
that write a new table also apply its DDL themselves on first use, so a
module run before any pipeline command still works.

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
    # ESPN pickcenter prices beyond the closing line (2026-09-29): opening
    # moneylines and total, and the over/under and puck-line prices, closing
    # and opening. ingestion/espn_odds.HISTORICAL_ODDS_COLUMNS lists the same
    # columns; `python -m ingestion.espn_odds --refresh` fills older rows.
    ("raw", "historical_odds", "home_ml_open", "INTEGER"),
    ("raw", "historical_odds", "away_ml_open", "INTEGER"),
    ("raw", "historical_odds", "total_open", "NUMERIC(4,1)"),
    ("raw", "historical_odds", "over_price", "INTEGER"),
    ("raw", "historical_odds", "under_price", "INTEGER"),
    ("raw", "historical_odds", "over_price_open", "INTEGER"),
    ("raw", "historical_odds", "under_price_open", "INTEGER"),
    ("raw", "historical_odds", "spread_home_price", "INTEGER"),
    ("raw", "historical_odds", "spread_away_price", "INTEGER"),
    ("raw", "historical_odds", "spread_open", "NUMERIC(4,1)"),
    ("raw", "historical_odds", "spread_home_price_open", "INTEGER"),
    ("raw", "historical_odds", "spread_away_price_open", "INTEGER"),
    ("raw", "historical_odds", "espn_event_id", "VARCHAR(12)"),
    ("raw", "historical_odds", "prices_fetched_at", "TIMESTAMP"),
    # When ingestion/nhl_stats last filled the row's power-play, penalty-kill
    # and faceoff columns (naive UTC). NULL = never, so those zeros are
    # defaults, not data
    ("raw", "skater_games", "stats_filled_at", "TIMESTAMP"),
    # A slip whose result was set by hand (betting/ledger.settle_by_hand):
    # a later leg correction must not overwrite it
    ("betting", "slips", "settled_by_hand", "BOOLEAN NOT NULL DEFAULT FALSE"),
    # After Daily Faceoff refuses a request (429/403): no request for the
    # team before this time (ingestion/dailyfaceoff_lines.py)
    ("raw", "lineup_fetches", "next_fetch_after", "TIMESTAMPTZ"),
)

# (schema, table, statements): tables added after db/schema.sql first
# shipped, created with their indexes when missing. Keep in sync with
# db/schema.sql. The module that writes each table holds the same DDL and
# applies it on first use; tests/test_migrate.py checks all three copies
# declare the same columns.
TABLES = (
    # The NHL's free odds feed (ingestion/nhl_odds.py), kept apart from The
    # Odds API's raw.odds_snapshots so the two can be compared
    ("raw", "nhl_feed_snapshots", (
        """
        CREATE TABLE IF NOT EXISTS raw.nhl_feed_snapshots (
            id                BIGSERIAL PRIMARY KEY,
            captured_at       TIMESTAMP NOT NULL,
            game_id           BIGINT NOT NULL REFERENCES raw.games(game_id),
            source            VARCHAR(12) NOT NULL,
            book              VARCHAR(40) NOT NULL,
            market            VARCHAR(6) NOT NULL,
            home_price        INTEGER,
            away_price        INTEGER,
            over_price        INTEGER,
            under_price       INTEGER,
            draw_price        INTEGER,
            line              NUMERIC(4,1),
            feed_updated_utc  TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_nhl_feed_game ON raw.nhl_feed_snapshots(game_id)",
        "CREATE INDEX IF NOT EXISTS idx_nhl_feed_time ON raw.nhl_feed_snapshots(captured_at)",
    )),
    # ESPN's injury list, one snapshot a day (ingestion/espn_injuries.py)
    ("raw", "injuries", (
        """
        CREATE TABLE IF NOT EXISTS raw.injuries (
            snapshot_date    DATE NOT NULL,
            espn_athlete_id  BIGINT NOT NULL,
            player_name      VARCHAR(80) NOT NULL,
            team_abbrev      VARCHAR(3),
            position         VARCHAR(2),
            status           VARCHAR(30),
            injury_type      VARCHAR(60),
            injury_detail    VARCHAR(60),
            return_date      DATE,
            short_comment    TEXT,
            long_comment     TEXT,
            reported_at      TIMESTAMPTZ,
            player_id        INTEGER,
            fetched_at       TIMESTAMP NOT NULL DEFAULT now(),
            PRIMARY KEY (snapshot_date, espn_athlete_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_injuries_player ON raw.injuries(player_id, snapshot_date)",
    )),
    # Past player-prop prices from ESPN (ingestion/espn_props.py)
    ("raw", "prop_odds_hist", (
        """
        CREATE TABLE IF NOT EXISTS raw.prop_odds_hist (
            game_id          BIGINT NOT NULL REFERENCES raw.games(game_id),
            espn_event_id    VARCHAR(12) NOT NULL,
            book             VARCHAR(40) NOT NULL,
            market           VARCHAR(40) NOT NULL,
            player_name      VARCHAR(80),
            espn_athlete_id  BIGINT NOT NULL,
            player_id        INTEGER,
            line             NUMERIC(4,1) NOT NULL,
            over_price       INTEGER,
            under_price      INTEGER,
            over_price_open  INTEGER,
            under_price_open INTEGER,
            last_updated     TIMESTAMPTZ,
            event_start      TIMESTAMPTZ,
            fetched_at       TIMESTAMP NOT NULL DEFAULT now(),
            UNIQUE (game_id, book, market, espn_athlete_id, line)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_prop_odds_hist_player "
        "ON raw.prop_odds_hist(player_id, market)",
    )),
    # The games the ESPN props backfill has fetched, so it can resume
    ("raw", "prop_odds_fetches", (
        """
        CREATE TABLE IF NOT EXISTS raw.prop_odds_fetches (
            game_id        BIGINT PRIMARY KEY REFERENCES raw.games(game_id),
            espn_event_id  VARCHAR(12),
            books          VARCHAR(120),
            n_rows         INTEGER NOT NULL DEFAULT 0,
            fetched_at     TIMESTAMP NOT NULL DEFAULT now()
        )""",
    )),
    # Live player-prop lines from The Odds API (ingestion/props_odds.py,
    # the props machine's job)
    ("raw", "prop_snapshots", (
        """
        CREATE TABLE IF NOT EXISTS raw.prop_snapshots (
            snapshot_id     BIGSERIAL PRIMARY KEY,
            captured_at     TIMESTAMP NOT NULL,
            game_id         BIGINT NOT NULL REFERENCES raw.games(game_id),
            event_id        VARCHAR(64) NOT NULL,
            book            VARCHAR(40) NOT NULL,
            market          VARCHAR(60) NOT NULL,
            player_name     VARCHAR(80) NOT NULL,
            player_id       INTEGER,
            line            NUMERIC(5,1),
            over_price      INTEGER,
            under_price     INTEGER,
            book_updated_at TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_prop_snapshots_game ON raw.prop_snapshots(game_id)",
        "CREATE INDEX IF NOT EXISTS idx_prop_snapshots_event "
        "ON raw.prop_snapshots(event_id, captured_at)",
        "CREATE INDEX IF NOT EXISTS idx_prop_snapshots_player "
        "ON raw.prop_snapshots(player_id, market)",
    )),
    # Daily Faceoff line combinations, a snapshot per change
    # (ingestion/dailyfaceoff_lines.py), and its last fetch per team
    ("raw", "lineups", (
        """
        CREATE TABLE IF NOT EXISTS raw.lineups (
            snapshot_ts         TIMESTAMPTZ NOT NULL,
            team                VARCHAR(3) NOT NULL,
            game_date           DATE NOT NULL,
            unit                VARCHAR(6) NOT NULL,
            slot                VARCHAR(6) NOT NULL,
            player_name         VARCHAR(80) NOT NULL,
            player_id           INTEGER,
            df_player_id        INTEGER,
            position            VARCHAR(2),
            injury_status       VARCHAR(12),
            game_time_decision  BOOLEAN NOT NULL DEFAULT FALSE,
            source_updated_at   TIMESTAMPTZ,
            PRIMARY KEY (snapshot_ts, team, unit, slot)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_lineups_team ON raw.lineups(team, snapshot_ts)",
        "CREATE INDEX IF NOT EXISTS idx_lineups_player ON raw.lineups(player_id, game_date)",
    )),
    ("raw", "lineup_fetches", (
        """
        CREATE TABLE IF NOT EXISTS raw.lineup_fetches (
            team                VARCHAR(3) PRIMARY KEY,
            fetched_at          TIMESTAMPTZ NOT NULL,
            source_updated_at   TIMESTAMPTZ,
            lines_hash          VARCHAR(64),
            status              VARCHAR(12) NOT NULL,
            next_fetch_after    TIMESTAMPTZ
        )""",
    )),
    # The news monitor (betting/news.py): what changed, the last state seen
    # per source and team, and one row per run
    ("raw", "news_events", (
        """
        CREATE TABLE IF NOT EXISTS raw.news_events (
            event_id        BIGSERIAL PRIMARY KEY,
            ts              TIMESTAMPTZ NOT NULL,
            game_id         BIGINT REFERENCES raw.games(game_id),
            game_date       DATE,
            team            VARCHAR(3) NOT NULL,
            kind            VARCHAR(20) NOT NULL CHECK (kind IN ('STARTER_CONFIRMED', 'STARTER_CHANGED', 'PLAYER_OUT', 'PLAYER_IN', 'LINE_CHANGE', 'PP_UNIT_CHANGE')),
            source          VARCHAR(20) NOT NULL,
            player_name     VARCHAR(80),
            player_id       INTEGER,
            detail          TEXT,
            previous        TEXT,
            current         TEXT,
            rescored        BOOLEAN,
            new_pick        BOOLEAN,
            market_moved    BOOLEAN,
            market_note     TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_news_events_ts ON raw.news_events(ts)",
        "CREATE INDEX IF NOT EXISTS idx_news_events_game ON raw.news_events(game_id)",
    )),
    ("raw", "news_state", (
        """
        CREATE TABLE IF NOT EXISTS raw.news_state (
            source          VARCHAR(20) NOT NULL,
            team            VARCHAR(3) NOT NULL,
            state           JSONB NOT NULL,
            updated_at      TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (source, team)
        )""",
    )),
    ("raw", "news_runs", (
        """
        CREATE TABLE IF NOT EXISTS raw.news_runs (
            run_id          BIGSERIAL PRIMARY KEY,
            started_at      TIMESTAMPTZ NOT NULL,
            finished_at     TIMESTAMPTZ,
            events          INTEGER,
            notes           TEXT
        )""",
    )),
    # Finished pipeline jobs, one row per job and local date (config/runs.py).
    # The news monitor makes no pick before today's 'daily' row exists
    ("raw", "pipeline_runs", (
        """
        CREATE TABLE IF NOT EXISTS raw.pipeline_runs (
            job             VARCHAR(20) NOT NULL,
            run_date        DATE NOT NULL,
            finished_at     TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (job, run_date)
        )""",
    )),
    # The bet ledger (betting/ledger.py): real bets as slips (a single bet
    # or a parlay) with their legs, and each bettor's deposits,
    # withdrawals and bonuses per platform. slips comes before slip_legs,
    # which references it
    ("betting", "slips", (
        """
        CREATE TABLE IF NOT EXISTS betting.slips (
            slip_id         BIGSERIAL PRIMARY KEY,
            bettor          VARCHAR(40) NOT NULL,
            platform        VARCHAR(40) NOT NULL,
            placed_at       TIMESTAMP NOT NULL,
            stake           NUMERIC(10,2) NOT NULL CHECK (stake > 0),
            price_american  INTEGER NOT NULL,
            is_parlay       BOOLEAN NOT NULL DEFAULT FALSE,
            is_bonus_bet    BOOLEAN NOT NULL DEFAULT FALSE,
            status          VARCHAR(10) NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'WON', 'LOST', 'PUSH', 'VOID', 'CASHED_OUT')),
            payout          NUMERIC(10,2),
            settled_at      TIMESTAMP,
            notes           TEXT,
            is_paper        BOOLEAN NOT NULL DEFAULT FALSE,
            settled_by_hand BOOLEAN NOT NULL DEFAULT FALSE,
            created_at      TIMESTAMP NOT NULL DEFAULT NOW()
        )""",
        "CREATE INDEX IF NOT EXISTS idx_slips_who ON betting.slips(bettor, platform)",
        "CREATE INDEX IF NOT EXISTS idx_slips_status ON betting.slips(status)",
    )),
    ("betting", "slip_legs", (
        """
        CREATE TABLE IF NOT EXISTS betting.slip_legs (
            slip_id         BIGINT NOT NULL REFERENCES betting.slips(slip_id) ON DELETE CASCADE,
            leg_no          SMALLINT NOT NULL,
            game_id         BIGINT REFERENCES raw.games(game_id),
            market          VARCHAR(10) NOT NULL CHECK (market IN ('ml', 'pl', 'total', 'prop_sog', 'other')),
            side            VARCHAR(80) NOT NULL,
            line            NUMERIC(5,1),
            price_american  INTEGER,
            player_id       INTEGER,
            rec_id          BIGINT REFERENCES betting.recommendations(rec_id),
            result          VARCHAR(5) CHECK (result IN ('WIN', 'LOSS', 'PUSH', 'VOID')),
            settled_at      TIMESTAMP,
            PRIMARY KEY (slip_id, leg_no)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_slip_legs_game ON betting.slip_legs(game_id)",
    )),
    ("betting", "bankroll_txns", (
        """
        CREATE TABLE IF NOT EXISTS betting.bankroll_txns (
            txn_id          BIGSERIAL PRIMARY KEY,
            bettor          VARCHAR(40) NOT NULL,
            platform        VARCHAR(40) NOT NULL,
            ts              TIMESTAMP NOT NULL,
            kind            VARCHAR(10) NOT NULL CHECK (kind IN ('DEPOSIT', 'WITHDRAWAL', 'BONUS', 'ADJUSTMENT')),
            amount          NUMERIC(10,2) NOT NULL,
            note            TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_bankroll_txns_who "
        "ON betting.bankroll_txns(bettor, platform)",
    )),
)

_done = False


def ensure_schema() -> None:
    """Create any missing TABLES (with their indexes), then add any missing
    COLUMNS. Checks information_schema first, so an up-to-date database
    runs no DDL and takes no table locks."""
    global _done
    if _done:
        return
    with engine.begin() as conn:
        tables = {tuple(r) for r in conn.execute(text("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema IN ('raw', 'betting')
        """))}
        for schema, table, statements in TABLES:
            if (schema, table) in tables:
                continue
            logger.info(f"Schema upgrade: creating {schema}.{table}")
            for stmt in statements:
                conn.execute(text(stmt))
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
        description="Create any tables and add any columns a database "
                    "made from an older db/schema.sql is missing (pipeline "
                    "commands do this on their own). Safe to re-run.")
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
