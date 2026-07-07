"""
pipeline.py
Master orchestrator for the NHL Betting System data pipeline.

Usage:
    python pipeline.py setup             # First-time setup verification
    python pipeline.py status            # Check database status
    python pipeline.py backfill          # Full historical backfill (5 seasons)
    python pipeline.py daily             # Daily refresh (run via cron)
"""
import sys
from datetime import date

from config.settings import check_db_connection, engine, BACKFILL_SEASONS, logger
from sqlalchemy import text


def db_status():
    """Print current database population status."""
    if not check_db_connection():
        return

    queries = {
        "Teams": "SELECT COUNT(*) FROM raw.teams",
        "Games": "SELECT COUNT(*) FROM raw.games",
        "Games (FINAL)": "SELECT COUNT(*) FROM raw.games WHERE game_state IN ('FINAL', 'OFF')",
        "Games (SCHEDULED)": "SELECT COUNT(*) FROM raw.games WHERE game_state = 'SCHEDULED'",
        "Shots": "SELECT COUNT(*) FROM raw.shots",
        "Skater game logs": "SELECT COUNT(*) FROM raw.skater_games",
        "Goalie game logs": "SELECT COUNT(*) FROM raw.goalie_games",
        "Odds snapshots": "SELECT COUNT(*) FROM raw.odds_snapshots",
        "Players": "SELECT COUNT(*) FROM raw.players",
    }

    print("\n" + "=" * 50)
    print("  NHL BETTING SYSTEM — DATABASE STATUS")
    print("=" * 50)

    with engine.connect() as conn:
        for label, sql in queries.items():
            try:
                count = conn.execute(text(sql)).scalar()
                print(f"  {label:.<35} {count:>10,}")
            except Exception:
                print(f"  {label:.<35} {'ERROR':>10}")

        print("\n  --- Games by Season ---")
        result = conn.execute(text("""
            SELECT season, COUNT(*) as games,
                   SUM(CASE WHEN game_state IN ('FINAL','OFF') THEN 1 ELSE 0 END) as completed
            FROM raw.games GROUP BY season ORDER BY season
        """))
        for row in result.fetchall():
            print(f"  {row[0]}:  {row[1]:>5} games ({row[2]:>5} completed)")

    print("=" * 50 + "\n")


def backfill():
    """Full historical backfill of all configured seasons."""
    from ingestion.nhl_api import ingest_teams, ingest_season

    logger.info("=" * 60)
    logger.info("STARTING FULL BACKFILL")
    logger.info(f"Seasons: {BACKFILL_SEASONS}")
    logger.info("=" * 60)

    ingest_teams()
    for season in BACKFILL_SEASONS:
        ingest_season(season)

    logger.info("BACKFILL COMPLETE")
    db_status()


def daily():
    """Daily refresh pipeline. Call via cron."""
    from ingestion.nhl_api import daily_refresh
    from ingestion.odds_api import snapshot_odds

    logger.info(f"DAILY REFRESH — {date.today()}")
    daily_refresh()
    snapshot_odds()
    logger.info("DAILY REFRESH COMPLETE")


def setup_check():
    """Verify all prerequisites for first-time setup."""
    from config.settings import ODDS_API_KEY, DATA_DIR

    print("\n" + "=" * 50)
    print("  NHL BETTING SYSTEM — SETUP CHECK")
    print("=" * 50)

    db_ok = check_db_connection()
    print(f"  Database connection ........ {'OK' if db_ok else 'FAIL'}")

    try:
        from nhlpy import NHLClient
        NHLClient()
        print(f"  nhl-api-py (nhlpy) ......... OK")
    except ImportError:
        print(f"  nhl-api-py (nhlpy) ......... FAIL (pip install nhl-api-py)")

    has_key = bool(ODDS_API_KEY and ODDS_API_KEY != "your_key_here")
    print(f"  Odds API key ............... {'OK' if has_key else 'NOT SET (optional for now)'}")
    print(f"  Data directory .............. {DATA_DIR}")
    print("=" * 50)

    if db_ok:
        print("\n  Ready to run: python pipeline.py backfill")
    else:
        print("\n  Start PostgreSQL first: docker compose up -d")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
NHL Betting System Pipeline
============================
Usage:
    python pipeline.py setup       Check prerequisites
    python pipeline.py status      Database population status
    python pipeline.py backfill    Full 5-season historical backfill
    python pipeline.py daily       Daily refresh (schedule + boxscores + odds)
        """)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "setup":
        setup_check()
    elif cmd == "status":
        db_status()
    elif cmd == "backfill":
        if not check_db_connection():
            print("ERROR: Database not reachable. Run: docker compose up -d")
            sys.exit(1)
        backfill()
    elif cmd == "daily":
        if not check_db_connection():
            logger.error("Database not reachable")
            sys.exit(1)
        daily()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
