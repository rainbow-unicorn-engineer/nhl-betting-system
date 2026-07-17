"""
pipeline.py
Master orchestrator for the NHL Betting System data pipeline.

Usage:
    python pipeline.py setup                        # First-time setup verification
    python pipeline.py status                       # Check database status
    python pipeline.py backfill                     # Full historical backfill (6 seasons)
    python pipeline.py features [--season YYYYYYYY] # Build feature store (all seasons by default)
    python pipeline.py daily                        # Daily refresh (run via cron)
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
    from ingestion.moneypuck import ingest_season_shots

    logger.info("=" * 60)
    logger.info("STARTING FULL BACKFILL")
    logger.info(f"Seasons: {BACKFILL_SEASONS}")
    logger.info("=" * 60)

    ingest_teams()
    for season in BACKFILL_SEASONS:
        ingest_season(season)
        # MoneyPuck shots must load after the season's games exist (FK)
        try:
            ingest_season_shots(season)
        except Exception as e:
            logger.error(f"MoneyPuck shots ingestion failed for {season}: {e}")

    logger.info("BACKFILL COMPLETE")
    db_status()


def features(season=None):
    """Build the feature store (Layer 2) for one season or all seasons."""
    from features.build_all import build_features

    build_features(season)


def _wait_for_network(timeout_s: int = 180) -> bool:
    """Wake-race guard: launchd fires catch-up jobs the moment the Mac
    wakes, often seconds before Wi-Fi is back. Block until DNS resolves
    (or time out) so the ingestion calls don't crash on a dead network."""
    import socket
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            socket.getaddrinfo("api-web.nhle.com", 443)
            return True
        except OSError:
            time.sleep(5)
    logger.error(f"Network unavailable after {timeout_s}s — aborting run")
    return False


def starters():
    """Confirmed starting goalies from Daily Faceoff (non-fatal)."""
    try:
        from ingestion.dailyfaceoff import ingest_starting_goalies
        ingest_starting_goalies()
    except Exception as e:
        logger.error(f"Daily Faceoff ingestion failed (non-fatal): {e}")


def settle():
    """Settle finished paper bets + rebuild the bankroll/CLV ledger."""
    try:
        from betting.settle import settle_paper
        settle_paper()
    except Exception as e:
        logger.error(f"Paper settlement failed (non-fatal): {e}")


def recommend():
    """Score today's slate through the betting engine and write
    betting.recommendations (no-op when there are no games)."""
    from betting.recommend import generate_recommendations

    try:
        generate_recommendations()
    except Exception as e:
        logger.error(f"Recommendation job failed (non-fatal): {e}")


def odds():
    """Odds snapshot only — cheap enough to run near game time for CLV.
    Recommendations + arb/middle alerts refresh right after: this is the
    freshest-lines moment."""
    from ingestion.odds_api import snapshot_odds

    if not _wait_for_network():
        return
    snapshot_odds()
    starters()      # confirmations roll in through gameday
    recommend()
    try:
        from betting.alerts import run_alerts
        run_alerts()
    except Exception as e:
        logger.error(f"Alert scan failed (non-fatal): {e}")


def daily():
    """Daily refresh pipeline. Call via cron."""
    from ingestion.nhl_api import daily_refresh
    from ingestion.odds_api import snapshot_odds
    from config.settings import CURRENT_SEASON

    logger.info(f"DAILY REFRESH — {date.today()}")
    if not _wait_for_network():
        return
    daily_refresh()
    snapshot_odds()
    # ESPN reference-line top-up for newly-final games (no-op when current)
    try:
        from ingestion.espn_odds import backfill_historical_odds
        backfill_historical_odds(CURRENT_SEASON)
    except Exception as e:
        logger.error(f"ESPN odds top-up failed (non-fatal): {e}")
    # Feature refresh after ingestion: current season only (Elo is always
    # full-history inside the build)
    features(season=CURRENT_SEASON)
    settle()        # yesterday's finals + closing snapshots are in
    starters()
    recommend()
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
    python pipeline.py backfill    Full 6-season historical backfill
    python pipeline.py features    Build feature store [--season YYYYYYYY]
    python pipeline.py daily       Daily refresh (schedule + boxscores + odds + features + recs)
    python pipeline.py odds        Odds snapshot + starters + recommendation refresh
    python pipeline.py recommend   Score today's slate -> betting.recommendations
    python pipeline.py starters    Ingest Daily Faceoff confirmed goalies
    python pipeline.py settle      Settle paper bets + rebuild bankroll/CLV ledger
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
    elif cmd == "features":
        if not check_db_connection():
            print("ERROR: Database not reachable. Run: docker compose up -d")
            sys.exit(1)
        season = None
        if "--season" in sys.argv:
            try:
                season = int(sys.argv[sys.argv.index("--season") + 1])
            except (IndexError, ValueError):
                print("ERROR: --season requires a value like 20242025")
                sys.exit(1)
        features(season)
    elif cmd == "odds":
        if not check_db_connection():
            logger.error("Database not reachable")
            sys.exit(1)
        odds()
    elif cmd == "recommend":
        if not check_db_connection():
            logger.error("Database not reachable")
            sys.exit(1)
        recommend()
    elif cmd == "starters":
        if not check_db_connection():
            logger.error("Database not reachable")
            sys.exit(1)
        starters()
    elif cmd == "settle":
        if not check_db_connection():
            logger.error("Database not reachable")
            sys.exit(1)
        settle()
    elif cmd == "daily":
        if not check_db_connection():
            logger.error("Database not reachable")
            sys.exit(1)
        daily()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
