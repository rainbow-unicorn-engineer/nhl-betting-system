"""
pipeline.py
Master orchestrator for the NHL Betting System data pipeline.

Usage (`python pipeline.py --help` lists every command; `<command> --help`
prints that command's options and runs nothing):
    python pipeline.py setup                        # First-time setup verification
    python pipeline.py status                       # Check database status
    python pipeline.py backfill                     # Full historical backfill (BACKFILL_SEASONS)
    python pipeline.py features [--season YYYYYYYY] # Build feature store (all seasons by default)
    python pipeline.py daily                        # Daily refresh + picks (the picks machine)
    python pipeline.py odds                         # Midday snapshot (3 credits) + picks + alerts
    python pipeline.py close [--due]                # Pre-puck-drop moneyline snapshot (1 credit)
    python pipeline.py refresh                      # Free data only: no odds, no picks (props)
    python pipeline.py props [--due]                # Player-props snapshot (the props machine)
    python pipeline.py nhl-odds                     # Free snapshot of the NHL's own odds feed
    python pipeline.py compare-feeds [--date D]     # NHL feed vs The Odds API, stored prices
    python pipeline.py settle                       # Settle paper picks + the bettors' recorded bets
    python pipeline.py injuries                     # ESPN injury list snapshot (free)
    python pipeline.py news [--due]                 # Team news: starters, lines, injuries (free)
    python pipeline.py nhl-stats [--season S]       # Power-play, penalty-kill, faceoff stats (free)

Machine roles: each machine has its own .env, Odds API key and database.
The picks jobs are daily, odds, close and news --due; the props jobs are
props and props --due (plus refresh on a machine without daily). The owner runs
every job on both the Mac and the Windows PC. ops/launchd/ and
ops/windows/ (-Role all) schedule them.

Steps marked non-fatal log an error and let the chain continue.
"""
import argparse
import sys
from datetime import date

from config.settings import (check_db_connection, engine, BACKFILL_SEASONS,
                             local_today, logger)
from sqlalchemy import text


def db_ready() -> bool:
    """check_db_connection() + the idempotent schema upgrade (config/migrate)."""
    if not check_db_connection():
        return False
    from config.migrate import ensure_schema
    ensure_schema()
    return True


def db_status():
    """Print current database population status."""
    if not db_ready():
        return

    queries = {
        "Teams": "SELECT COUNT(*) FROM raw.teams",
        "Games": "SELECT COUNT(*) FROM raw.games",
        "Games (FINAL)": "SELECT COUNT(*) FROM raw.games WHERE game_state IN ('FINAL', 'OFF')",
        # The NHL API marks upcoming games FUT or PRE, never SCHEDULED
        "Games (upcoming)": "SELECT COUNT(*) FROM raw.games WHERE game_state NOT IN ('FINAL', 'OFF')",
        "Shots": "SELECT COUNT(*) FROM raw.shots",
        "Skater game logs": "SELECT COUNT(*) FROM raw.skater_games",
        "Goalie game logs": "SELECT COUNT(*) FROM raw.goalie_games",
        "Odds snapshots": "SELECT COUNT(*) FROM raw.odds_snapshots",
        "NHL feed snapshots (free)": "SELECT COUNT(*) FROM raw.nhl_feed_snapshots",
        "ESPN lines": "SELECT COUNT(*) FROM raw.historical_odds",
        "ESPN lines with O/U prices": "SELECT COUNT(*) FROM raw.historical_odds "
                                      "WHERE over_price IS NOT NULL",
        "Skater games with PP stats": "SELECT COUNT(*) FROM raw.skater_games "
                                      "WHERE stats_filled_at IS NOT NULL",
        "Injury list rows": "SELECT COUNT(*) FROM raw.injuries",
        "Lineup rows (Daily Faceoff)": "SELECT COUNT(*) FROM raw.lineups",
        "News events": "SELECT COUNT(*) FROM raw.news_events",
        "Prop lines (live, Odds API)": "SELECT COUNT(*) FROM raw.prop_snapshots",
        "Prop lines (history, ESPN)": "SELECT COUNT(*) FROM raw.prop_odds_hist",
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


def settle_ledger():
    """Settle the bet ledger's open slips (real bets recorded on the
    dashboard's My bets tab) from final scores and box scores
    (non-fatal)."""
    try:
        from betting.ledger import settle_slips
        settle_slips()
    except Exception as e:
        logger.error(f"Bet-ledger settlement failed (non-fatal): {e}")


def recommend():
    """Score today's slate through the betting engine and write
    betting.recommendations (no-op when there are no games)."""
    from betting.recommend import generate_recommendations

    try:
        generate_recommendations()
    except Exception as e:
        logger.error(f"Recommendation job failed (non-fatal): {e}")


def nhl_feed(skip_when_idle: bool = True):
    """Free snapshot of the NHL's own odds feed (DraftKings, FanDuel Canada
    and the schedule's partner books) into raw.nhl_feed_snapshots
    (non-fatal, no credits: three requests to api-web.nhle.com). The odds
    and close chains take one right after each Odds API snapshot, so
    `compare-feeds` has a free price within minutes of every paid one.
    skip_when_idle: no request when no game starts in the next 24 hours."""
    try:
        from ingestion.nhl_odds import snapshot as nhl_snapshot
        nhl_snapshot(skip_when_idle=skip_when_idle)
    except Exception as e:
        logger.error(f"NHL feed snapshot failed (non-fatal): {e}")


def injuries():
    """Today's ESPN injury list into raw.injuries (non-fatal, one free
    request). ESPN keeps no history, so a day without a run has no list."""
    try:
        from ingestion.espn_injuries import ingest_injuries
        ingest_injuries()
    except Exception as e:
        logger.error(f"ESPN injury snapshot failed (non-fatal): {e}")


def news(due: bool = False):
    """The news monitor (betting/news.py, non-fatal, no Odds API request):
    Daily Faceoff starters and line combinations and ESPN's injury list,
    compared with the previous run; changes go to raw.news_events, and
    starter news on a game without a pick re-scores that game's date.
    due=True (`news --due`, every 15 minutes): only on a game day from
    NEWS_START_HOUR (8:00) local until the last puck drop. It waits for the
    network only after deciding the run is due, so the runs outside the
    window never block or log a network error."""
    try:
        from betting.news import run_news
        run_news(due=due, network_ready=_wait_for_network)
    except Exception as e:
        logger.error(f"News monitor failed (non-fatal): {e}")


def nhl_stats(season=None):
    """Power-play and penalty-kill ice time, power-play goals and assists,
    and faceoffs for raw.skater_games from the free NHL stats API
    (non-fatal). With no season: only the current season's dates whose
    rows are still unfilled (3 requests on a normal day, none when every
    row is filled). With a season: that whole season (about 27 requests)."""
    try:
        from ingestion import nhl_stats as stats
        summary = stats.fill_season(season) if season else stats.fill_missing()
        if summary.get("failed_windows"):
            logger.error(f"NHL stats: {summary['failed_windows']} window(s) failed "
                         f"(non-fatal; the next run retries them)")
    except Exception as e:
        logger.error(f"NHL stats fill failed (non-fatal): {e}")


def props(due: bool = False, markets=None):
    """Player-props snapshot from The Odds API into raw.prop_snapshots
    (non-fatal). The props machine's job, never part of `daily`. Morning:
    every game starting in the next 24 hours, 1 credit a game per market
    returned. due=True (`props --due`, every 15 minutes): only games
    starting within PROPS_CLOSE_LEAD_MINUTES (16) with no prop snapshot in
    the last PROPS_CLOSE_MIN_GAP_MINUTES (16). See ingestion/props_odds.py."""
    if not _wait_for_network():
        return
    try:
        from ingestion.props_odds import snapshot_props
        snapshot_props(markets=markets, due=due)
    except Exception as e:
        logger.error(f"Props snapshot failed (non-fatal): {e}")


def refresh():
    """The props machine's daily run: free data only, no Odds API request
    and no picks. Refreshes the schedule (which props matching needs) and
    box scores, fills power-play stats for newly finished games, and saves
    the ESPN injury list."""
    from ingestion.nhl_api import daily_refresh

    logger.info(f"DATA REFRESH (no odds, no picks) — {local_today()}")
    if not _wait_for_network():
        return
    daily_refresh()
    nhl_stats()
    injuries()
    logger.info("DATA REFRESH COMPLETE")


def odds():
    """Full odds snapshot (3 credits with the default ODDS_BOOKMAKERS) +
    a free NHL-feed snapshot + starters, then recommendations for games
    that have no pick yet (issued picks stay frozen) + arb/middle alerts:
    this is the freshest-lines moment. For the closing line alone, `close`
    is cheaper (1 credit)."""
    from ingestion.odds_api import snapshot_odds

    if not _wait_for_network():
        return
    snapshot_odds()
    nhl_feed()      # free, paired with the paid snapshot for compare-feeds
    starters()      # confirmations roll in through gameday
    recommend()
    try:
        from betting.alerts import run_alerts
        run_alerts()
    except Exception as e:
        logger.error(f"Alert scan failed (non-fatal): {e}")


def close(due: bool = False):
    """Closing-line snapshot: moneyline only (markets=h2h, 1 credit with
    the default ODDS_BOOKMAKERS), then a free NHL-feed snapshot; no
    recommendations, no alerts. It becomes the close that settlement
    grades each pick against.

    due=True (`close --due`, the scheduled job every 15 minutes): snapshot
    only when some game starts within CLOSE_LEAD_MINUTES (default 16) and
    no moneyline snapshot was taken in the last CLOSE_MIN_GAP_MINUTES
    (default 16); otherwise log why and return. Plain `close` always
    snapshots (while any game starts within 24 hours)."""
    from ingestion import odds_api

    if due:
        is_due, why = odds_api.close_is_due()
        if not is_due:
            logger.info(f"close --due: no snapshot, {why}")
            return
        logger.info(f"close --due: taking the closing snapshot, {why}")
    if not _wait_for_network():
        return
    odds_api.snapshot_odds(markets="h2h")
    nhl_feed()      # free, paired with the close for compare-feeds


def daily():
    """Daily refresh pipeline for a machine that makes picks. Run via
    launchd (ops/launchd/) or Task Scheduler (ops/windows/, -Role all or
    picks)."""
    from ingestion.nhl_api import daily_refresh
    from ingestion.odds_api import snapshot_odds
    from config.settings import CURRENT_SEASON

    run_date = local_today()
    logger.info(f"DAILY REFRESH — {run_date} (season {CURRENT_SEASON})")
    if not _wait_for_network():
        return
    daily_refresh()
    snapshot_odds()
    nhl_feed()      # free, paired with the paid snapshot for compare-feeds
    # ESPN reference-line top-up for newly-final games (no-op when current);
    # it also stores their opening and over/under prices
    try:
        from ingestion.espn_odds import backfill_historical_odds
        backfill_historical_odds(CURRENT_SEASON)
    except Exception as e:
        logger.error(f"ESPN odds top-up failed (non-fatal): {e}")
    # Power-play / penalty-kill time and faceoffs for yesterday's box scores
    # (before the feature rebuild, which reads them)
    nhl_stats()
    # MoneyPuck shots (xG) for the season so far — shot-based features go
    # stale in season without this
    try:
        from ingestion.moneypuck import refresh_season
        refresh_season(CURRENT_SEASON)
    except Exception as e:
        logger.error(f"MoneyPuck refresh failed (non-fatal): {e}")
    # Feature refresh after ingestion: current season only (Elo is always
    # full-history inside the build)
    features(season=CURRENT_SEASON)
    settle()        # yesterday's finals + closing snapshots are in
    settle_ledger() # the bettors' recorded bets, from the same finals and box scores
    starters()
    injuries()      # ESPN keeps no history: save today's list before picks
    recommend()
    # The marker the news monitor waits for: before it, today's data is not
    # loaded, so news makes no pick (betting/news.py, config/runs.py)
    try:
        from config.runs import mark_finished
        mark_finished("daily", run_date)
    except Exception as e:
        logger.error(f"Could not record the finished daily run (non-fatal; news "
                     f"makes no pick today until it is recorded): {e}")
    logger.info("DAILY REFRESH COMPLETE")


def seed_venues_if_missing() -> str:
    """Apply db/seed_venues.sql when raw.teams has no arena coordinates
    (a fresh database); returns a status word for the setup check."""
    from config.migrate import seed_venues, venues_missing
    try:
        if not venues_missing():
            return "OK"
        seed_venues()
        return "SEEDED (db/seed_venues.sql applied)"
    except Exception as e:
        logger.error(f"Venue seed failed: {e}")
        return "FAIL (run: python -m config.migrate --seed-venues)"


def setup_check():
    """Verify all prerequisites for first-time setup."""
    from config.settings import (ODDS_API_KEY, DATA_DIR, CURRENT_SEASON,
                                 local_tz_name)

    print("\n" + "=" * 50)
    print("  NHL BETTING SYSTEM — SETUP CHECK")
    print("=" * 50)

    db_ok = db_ready()
    print(f"  Database connection ........ {'OK' if db_ok else 'FAIL'}")
    if db_ok:
        print(f"  Venue coordinates .......... {seed_venues_if_missing()}")

    try:
        from nhlpy import NHLClient
        NHLClient()
        print(f"  nhl-api-py (nhlpy) ......... OK")
    except ImportError:
        print(f"  nhl-api-py (nhlpy) ......... FAIL (pip install nhl-api-py)")

    has_key = bool(ODDS_API_KEY and ODDS_API_KEY != "your_key_here")
    print(f"  Odds API key ............... {'OK' if has_key else 'NOT SET (optional for now)'}")
    print(f"  Season ..................... {CURRENT_SEASON}")
    print(f"  Local time zone ............ {local_tz_name()}")
    print(f"  Today (local) .............. {local_today()}")
    print(f"  Data directory .............. {DATA_DIR}")
    print("=" * 50)

    if db_ok:
        print("\n  Ready to run: python pipeline.py backfill")
    else:
        print("\n  Start PostgreSQL first: docker compose up -d")
    print()


# ── Command line ───────────────────────────────────────────────────

def _season_value(value: str) -> int:
    v = value.strip()
    if len(v) == 8 and v.isdigit() and int(v[4:]) == int(v[:4]) + 1:
        return int(v)
    raise argparse.ArgumentTypeError(f"{value!r} is not a season like 20252026")


def _date_value(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date like 2026-09-29")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python pipeline.py",
        description="NHL Betting System pipeline. `<command> --help` prints a "
                    "command's options and runs nothing.",
        epilog="Machine roles: the picks jobs are daily, odds, close and news; the "
               "props jobs are props and props --due (plus refresh where daily "
               "doesn't run). The owner runs every job on both machines. Each "
               "machine has its own .env, Odds API key and database.")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name, text):
        return sub.add_parser(name, help=text, description=text)

    add("setup", "Check prerequisites: the database (adding any missing tables "
                 "and columns), nhlpy, the Odds API key; seed the venues")
    add("status", "Database population status")
    add("backfill", "Full historical backfill (BACKFILL_FIRST_SEASON through the "
                    "current season)")
    p = add("features", "Build the feature store")
    p.add_argument("--season", type=int, default=None, metavar="YYYYYYYY",
                   help="one season, such as 20252026 (default: every season)")
    add("daily", "The picks machine's daily run: schedule and box scores, odds "
                 "snapshot (3 credits), free NHL-feed snapshot, ESPN lines, "
                 "power-play stats, shots, features, settlement, starters, "
                 "injuries, picks")
    add("odds", "Odds snapshot (3 credits) + free NHL-feed snapshot + starters + "
                "picks for games without one + arbitrage/middle alerts")
    p = add("close", "Closing-line snapshot before puck drop: moneyline only "
                     "(1 credit), then a free NHL-feed snapshot")
    p.add_argument("--due", action="store_true",
                   help="only if a game starts within CLOSE_LEAD_MINUTES (16) and no "
                        "moneyline snapshot is under CLOSE_MIN_GAP_MINUTES (16) old")
    add("recommend", "Score today's slate into betting.recommendations")
    add("starters", "Starting goalies from Daily Faceoff")
    add("settle", "Settle paper bets and rebuild the bankroll and CLV ledger, "
                  "then settle the bettors' recorded bets (the bet ledger)")
    add("refresh", "The props machine's daily run: schedule and box scores, "
                   "power-play stats, ESPN injuries. No odds request, no picks")
    p = add("props", "Player-props snapshot from The Odds API into "
                     "raw.prop_snapshots (the props machine; 1 credit a game per "
                     "market returned)")
    p.add_argument("--due", action="store_true",
                   help="pre-game form, every 15 minutes: only games starting within "
                        "PROPS_CLOSE_LEAD_MINUTES (16) with no prop snapshot in the "
                        "last PROPS_CLOSE_MIN_GAP_MINUTES (16)")
    p.add_argument("--markets", default=None,
                   help="comma-separated Odds API prop markets (default: "
                        "PROPS_MARKETS, else player_shots_on_goal)")
    add("nhl-odds", "Free snapshot of the NHL's own odds feed into "
                    "raw.nhl_feed_snapshots (no credits)")
    p = add("compare-feeds", "Compare stored NHL-feed prices with The Odds API's "
                             "for one date (reads only)")
    p.add_argument("--date", type=_date_value, default=None, metavar="YYYY-MM-DD",
                   help="schedule date (default: today's local date)")
    p.add_argument("--detail", action="store_true", help="also list every paired price")
    add("injuries", "Save today's ESPN injury list into raw.injuries (free)")
    p = add("news", "Team news: refresh Daily Faceoff starters and lines and ESPN "
                    "injuries, record what changed in raw.news_events, re-score "
                    "games with starter news and no pick yet (free, no Odds API "
                    "request)")
    p.add_argument("--due", action="store_true",
                   help="scheduled form, every 15 minutes: only on a game day from "
                        "NEWS_START_HOUR (8:00) local until the last puck drop")
    p = add("nhl-stats", "Fill power-play / penalty-kill ice time, power-play "
                         "points and faceoffs in raw.skater_games (free)")
    p.add_argument("--season", type=_season_value, default=None, metavar="YYYYYYYY",
                   help="fill this whole season (about 27 requests; default: only "
                        "the current season's dates with unfilled rows)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cmd = args.command
    if cmd is None:
        parser.print_help()
        return 0
    if cmd == "setup":
        setup_check()
        return 0
    if cmd == "status":
        db_status()
        return 0
    if not db_ready():
        if cmd in ("backfill", "features"):
            print("ERROR: Database not reachable. Run: docker compose up -d")
        else:
            logger.error("Database not reachable")
        return 1

    if cmd == "backfill":
        backfill()
    elif cmd == "features":
        features(args.season)
    elif cmd == "daily":
        daily()
    elif cmd == "odds":
        odds()
    elif cmd == "close":
        close(due=args.due)
    elif cmd == "recommend":
        recommend()
    elif cmd == "starters":
        starters()
    elif cmd == "settle":
        settle()
        settle_ledger()
    elif cmd == "refresh":
        refresh()
    elif cmd == "props":
        props(due=args.due, markets=args.markets)
    elif cmd == "nhl-odds":
        if _wait_for_network():
            nhl_feed(skip_when_idle=False)
    elif cmd == "compare-feeds":
        from ingestion.nhl_odds import compare_feeds, format_report
        print(format_report(compare_feeds(args.date), detail=args.detail))
    elif cmd == "injuries":
        injuries()
    elif cmd == "news":
        news(due=args.due)
    elif cmd == "nhl-stats":
        nhl_stats(args.season)
    return 0


if __name__ == "__main__":
    sys.exit(main())
