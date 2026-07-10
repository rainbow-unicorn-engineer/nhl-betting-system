"""
ingestion/nhl_api.py
NHL data ingestion via coreyjs/nhl-api-py (RQI 4.03, our primary data dependency).
Populates: raw.games, raw.teams, raw.players, raw.rosters, raw.skater_games, raw.goalie_games

CRITICAL: the pip package "nhl-api-py" imports as `nhlpy`, NOT `nhl_api_py`.
"""
import time
import logging
from datetime import date, timedelta, datetime
from typing import Optional

import pandas as pd
import requests
from nhlpy import NHLClient
from sqlalchemy import text

from config.settings import engine, BACKFILL_SEASONS

# nhlpy 3.3.0 does not wrap the right-rail endpoint (team game stats)
RIGHT_RAIL_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail"

logger = logging.getLogger("nhl.ingestion.nhl_api")

client = NHLClient()


# ─────────────────────────────────────────────
# TEAMS
# ─────────────────────────────────────────────
def ingest_teams():
    """Pull all current NHL teams and upsert into raw.teams."""
    logger.info("Ingesting teams...")
    # nhlpy 3.x renamed get_standings() -> league_standings()
    standings = client.standings.league_standings(date="now")
    records = []
    for entry in standings.get("standings", []):
        records.append({
            "team_abbrev": entry.get("teamAbbrev", {}).get("default", ""),
            "team_name": entry.get("teamName", {}).get("default", ""),
            "conference": entry.get("conferenceName", ""),
            "division": entry.get("divisionName", ""),
        })

    if not records:
        logger.warning("No team data returned from standings API")
        return 0

    df = pd.DataFrame(records).drop_duplicates(subset=["team_abbrev"])

    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO raw.teams (team_abbrev, team_name, conference, division)
                VALUES (:team_abbrev, :team_name, :conference, :division)
                ON CONFLICT (team_abbrev) DO UPDATE SET
                    team_name = EXCLUDED.team_name,
                    conference = EXCLUDED.conference,
                    division = EXCLUDED.division,
                    updated_at = NOW()
            """), dict(row))

    logger.info(f"Upserted {len(df)} teams")
    return len(df)


# ─────────────────────────────────────────────
# SCHEDULE / GAMES
# ─────────────────────────────────────────────
def ingest_schedule(start_date: str, end_date: str):
    """
    Pull NHL schedule for a date range and upsert into raw.games.
    Dates in YYYY-MM-DD format.
    """
    logger.info(f"Ingesting schedule: {start_date} to {end_date}")
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    total_inserted = 0

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            # nhlpy 3.x renamed get_schedule() -> weekly_schedule() (returns a 7-day gameWeek)
            schedule = client.schedule.weekly_schedule(date=date_str)
        except Exception as e:
            logger.warning(f"Schedule fetch failed for {date_str}: {e}")
            current += timedelta(days=1)
            continue

        game_week = schedule.get("gameWeek", [])
        for day_data in game_week:
            day_date = day_data.get("date", date_str)
            games = day_data.get("games", [])

            for g in games:
                game_id = g.get("id")
                game_type = g.get("gameType", 0)
                if game_type not in (2, 3):  # regular season and playoffs only
                    continue

                season_start = int(str(game_id)[:4])
                season = season_start * 10000 + (season_start + 1)

                home = g.get("homeTeam", {})
                away = g.get("awayTeam", {})
                state = g.get("gameState", "SCHEDULED")

                record = {
                    "game_id": game_id,
                    "season": season,
                    "game_type": game_type,
                    "date": day_date,
                    "home_team": home.get("abbrev", ""),
                    "away_team": away.get("abbrev", ""),
                    "home_score": home.get("score") if state in ("FINAL", "OFF") else None,
                    "away_score": away.get("score") if state in ("FINAL", "OFF") else None,
                    "game_state": state,
                    "venue": g.get("venue", {}).get("default", ""),
                    "is_ot": None,
                    "is_so": None,
                }

                if state in ("FINAL", "OFF"):
                    # gameOutcome.lastPeriodType is the authoritative source (REG/OT/SO);
                    # fall back to periodDescriptor for older payloads
                    last_period = g.get("gameOutcome", {}).get("lastPeriodType")
                    if last_period is None:
                        period = g.get("periodDescriptor", {})
                        period_num = period.get("number", 3)
                        last_period = period.get("periodType", "REG")
                        if period_num > 3 and last_period == "REG":
                            last_period = "OT"
                    record["is_ot"] = last_period in ("OT", "SO")
                    record["is_so"] = last_period == "SO"

                _upsert_game(record)
                total_inserted += 1

        current += timedelta(days=7)  # schedule API returns a week at a time
        time.sleep(0.5)  # polite rate limiting

    logger.info(f"Upserted {total_inserted} games from {start_date} to {end_date}")
    return total_inserted


def _upsert_game(record: dict):
    """Upsert a single game into raw.games."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw.games (game_id, season, game_type, date, home_team, away_team,
                                   home_score, away_score, game_state, venue, is_ot, is_so)
            VALUES (:game_id, :season, :game_type, :date, :home_team, :away_team,
                    :home_score, :away_score, :game_state, :venue,
                    COALESCE(:is_ot, FALSE), COALESCE(:is_so, FALSE))
            ON CONFLICT (game_id) DO UPDATE SET
                home_score = COALESCE(EXCLUDED.home_score, raw.games.home_score),
                away_score = COALESCE(EXCLUDED.away_score, raw.games.away_score),
                game_state = EXCLUDED.game_state,
                is_ot = COALESCE(EXCLUDED.is_ot, raw.games.is_ot),
                is_so = COALESCE(EXCLUDED.is_so, raw.games.is_so),
                updated_at = NOW()
        """), record)


# ─────────────────────────────────────────────
# BOXSCORES (skater + goalie game logs)
# ─────────────────────────────────────────────
def ingest_boxscore(game_id: int):
    """
    Pull boxscore for a single game and populate raw.skater_games + raw.goalie_games.
    Also populates raw.players for any new player encountered.
    """
    try:
        # nhlpy 3.x renamed get_boxscore() -> boxscore() and expects a string id
        box = client.game_center.boxscore(game_id=str(game_id))
    except Exception as e:
        logger.warning(f"Boxscore fetch failed for game {game_id}: {e}")
        return False

    player_by_position = box.get("playerByGameStats", {})

    for side in ("homeTeam", "awayTeam"):
        team_data = box.get(side, {})
        team_abbrev = team_data.get("abbrev", "")
        side_stats = player_by_position.get(side, {})

        for pos_group in ("forwards", "defense"):
            for player in side_stats.get(pos_group, []):
                _upsert_player_from_boxscore(player)
                _upsert_skater_game(player, game_id, team_abbrev)

        for player in side_stats.get("goalies", []):
            _upsert_player_from_boxscore(player)
            _upsert_goalie_game(player, game_id, team_abbrev)

    return True


def _upsert_player_from_boxscore(player: dict):
    """Ensure player exists in raw.players."""
    pid = player.get("playerId")
    if not pid:
        return
    name_obj = player.get("name", {})
    full_name = f"{name_obj.get('default', '')}".strip()
    if not full_name:
        full_name = str(pid)

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw.players (player_id, full_name, position)
            VALUES (:pid, :name, :pos)
            ON CONFLICT (player_id) DO UPDATE SET
                full_name = EXCLUDED.full_name,
                updated_at = NOW()
        """), {"pid": pid, "name": full_name, "pos": player.get("position", "")})


def _upsert_skater_game(player: dict, game_id: int, team: str):
    """Upsert a skater's game stats."""
    pid = player.get("playerId")
    if not pid:
        return

    record = {
        "player_id": pid,
        "game_id": game_id,
        "team": team,
        "position": player.get("position", ""),
        "toi_seconds": _toi_to_seconds(player.get("toi", "0:00")),
        "goals": player.get("goals", 0),
        "assists": player.get("assists", 0),
        "points": player.get("points", 0),
        # NHL API renamed shots -> sog (shots on goal)
        "shots": player.get("sog", player.get("shots", 0)),
        "hits": player.get("hits", 0),
        "blocks": player.get("blockedShots", 0) or player.get("blocks", 0),
        "pim": player.get("pim", 0),
        "plus_minus": player.get("plusMinus", 0),
    }

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw.skater_games (player_id, game_id, team, position, toi_seconds,
                goals, assists, points, shots, hits, blocks, pim, plus_minus)
            VALUES (:player_id, :game_id, :team, :position, :toi_seconds,
                    :goals, :assists, :points, :shots, :hits, :blocks, :pim, :plus_minus)
            ON CONFLICT (player_id, game_id) DO UPDATE SET
                goals = EXCLUDED.goals,
                assists = EXCLUDED.assists,
                points = EXCLUDED.points,
                shots = EXCLUDED.shots,
                hits = EXCLUDED.hits,
                blocks = EXCLUDED.blocks,
                pim = EXCLUDED.pim,
                plus_minus = EXCLUDED.plus_minus,
                toi_seconds = EXCLUDED.toi_seconds
        """), record)


def _upsert_goalie_game(player: dict, game_id: int, team: str):
    """Upsert a goalie's game stats."""
    pid = player.get("playerId")
    if not pid:
        return

    sa = player.get("shotsAgainst", 0) or 0
    sv = player.get("saves", 0) or 0
    ga = player.get("goalsAgainst", 0) or 0
    sv_pct = sv / sa if sa > 0 else None
    toi = _toi_to_seconds(player.get("toi", "0:00"))
    # NHL API now provides an explicit starter flag; TOI heuristic is the fallback
    is_starter = player.get("starter")
    if is_starter is None:
        is_starter = toi > 1800

    decision_raw = player.get("decision", None)
    decision = decision_raw if decision_raw in ("W", "L", "O") else None
    if decision == "O":
        decision = "OTL"

    # Strength-split shots come as "saves/shots" strings (e.g. "19/21")
    even_sv, even_sa = _parse_saves_shots(player.get("evenStrengthShotsAgainst"))
    pp_sv, pp_sa = _parse_saves_shots(player.get("powerPlayShotsAgainst"))
    sh_sv, sh_sa = _parse_saves_shots(player.get("shorthandedShotsAgainst"))

    record = {
        "player_id": pid,
        "game_id": game_id,
        "team": team,
        "decision": decision,
        "is_starter": is_starter,
        "shots_against": sa,
        "saves": sv,
        "goals_against": ga,
        "sv_pct": sv_pct,
        "toi_seconds": toi,
        "even_shots": even_sa,
        "even_saves": even_sv,
        "pp_shots": pp_sa,
        "pp_saves": pp_sv,
        "sh_shots": sh_sa,
        "sh_saves": sh_sv,
    }

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw.goalie_games (player_id, game_id, team, decision, is_starter,
                shots_against, saves, goals_against, sv_pct, toi_seconds,
                even_shots, even_saves, pp_shots, pp_saves, sh_shots, sh_saves)
            VALUES (:player_id, :game_id, :team, :decision, :is_starter,
                    :shots_against, :saves, :goals_against, :sv_pct, :toi_seconds,
                    :even_shots, :even_saves, :pp_shots, :pp_saves, :sh_shots, :sh_saves)
            ON CONFLICT (player_id, game_id) DO UPDATE SET
                decision = EXCLUDED.decision,
                is_starter = EXCLUDED.is_starter,
                shots_against = EXCLUDED.shots_against,
                saves = EXCLUDED.saves,
                goals_against = EXCLUDED.goals_against,
                sv_pct = EXCLUDED.sv_pct,
                toi_seconds = EXCLUDED.toi_seconds,
                even_shots = EXCLUDED.even_shots,
                even_saves = EXCLUDED.even_saves,
                pp_shots = EXCLUDED.pp_shots,
                pp_saves = EXCLUDED.pp_saves,
                sh_shots = EXCLUDED.sh_shots,
                sh_saves = EXCLUDED.sh_saves
        """), record)


# ─────────────────────────────────────────────
# TEAM GAME STATS (right-rail endpoint)
# ─────────────────────────────────────────────
def ingest_team_stats(game_id: int, home_team: str, away_team: str) -> bool:
    """
    Pull team-level game stats (PP conversions, faceoffs, hits, blocks...)
    from the gamecenter right-rail endpoint into raw.team_games.
    Only meaningful for completed games.
    """
    try:
        resp = requests.get(RIGHT_RAIL_URL.format(game_id=game_id), timeout=30)
        resp.raise_for_status()
        tgs = resp.json().get("teamGameStats")
    except Exception as e:
        logger.warning(f"Right-rail fetch failed for game {game_id}: {e}")
        return False

    if not tgs:
        logger.warning(f"No teamGameStats for game {game_id}")
        return False

    stats = {c.get("category"): (c.get("homeValue"), c.get("awayValue")) for c in tgs}

    for team, is_home, idx in ((home_team, True, 0), (away_team, False, 1)):
        pp_goals, pp_opps = _parse_saves_shots(stats.get("powerPlay", (None, None))[idx])
        fo_wins, fo_total = _parse_saves_shots(stats.get("faceoffWins", (None, None))[idx])

        def _num(cat):
            val = stats.get(cat, (None, None))[idx]
            return int(val) if val is not None else None

        record = {
            "game_id": game_id,
            "team": team,
            "is_home": is_home,
            "sog": _num("sog"),
            "faceoff_wins": fo_wins,
            "faceoff_total": fo_total,
            "pp_goals": pp_goals,
            "pp_opps": pp_opps,
            "pim": _num("pim"),
            "hits": _num("hits"),
            "blocked_shots": _num("blockedShots"),
            "giveaways": _num("giveaways"),
            "takeaways": _num("takeaways"),
        }

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw.team_games (game_id, team, is_home, sog,
                    faceoff_wins, faceoff_total, pp_goals, pp_opps,
                    pim, hits, blocked_shots, giveaways, takeaways)
                VALUES (:game_id, :team, :is_home, :sog,
                        :faceoff_wins, :faceoff_total, :pp_goals, :pp_opps,
                        :pim, :hits, :blocked_shots, :giveaways, :takeaways)
                ON CONFLICT (game_id, team) DO UPDATE SET
                    sog = EXCLUDED.sog,
                    faceoff_wins = EXCLUDED.faceoff_wins,
                    faceoff_total = EXCLUDED.faceoff_total,
                    pp_goals = EXCLUDED.pp_goals,
                    pp_opps = EXCLUDED.pp_opps,
                    pim = EXCLUDED.pim,
                    hits = EXCLUDED.hits,
                    blocked_shots = EXCLUDED.blocked_shots,
                    giveaways = EXCLUDED.giveaways,
                    takeaways = EXCLUDED.takeaways
            """), record)

    return True


def backfill_team_stats(season: Optional[int] = None):
    """Fetch right-rail team stats for all FINAL games missing team_games rows."""
    season_filter = f"AND g.season = {season}" if season else ""

    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT g.game_id, g.home_team, g.away_team FROM raw.games g
            WHERE g.game_state IN ('FINAL', 'OFF')
            {season_filter}
            AND NOT EXISTS (
                SELECT 1 FROM raw.team_games tg WHERE tg.game_id = g.game_id
            )
            ORDER BY g.date
        """))
        games = result.fetchall()

    logger.info(f"Backfilling team stats for {len(games)} games...")
    success = 0
    for i, (gid, home, away) in enumerate(games):
        if ingest_team_stats(gid, home, away):
            success += 1
        if (i + 1) % 50 == 0:
            logger.info(f"  ... {i+1}/{len(games)} processed ({success} succeeded)")
        time.sleep(0.3)

    logger.info(f"Team stats backfill complete: {success}/{len(games)} succeeded")
    return success


# ─────────────────────────────────────────────
# BATCH OPERATIONS
# ─────────────────────────────────────────────
def backfill_boxscores(season: Optional[int] = None):
    """Fetch boxscores for all FINAL games missing skater_games entries."""
    season_filter = f"AND g.season = {season}" if season else ""

    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT g.game_id FROM raw.games g
            WHERE g.game_state IN ('FINAL', 'OFF')
            {season_filter}
            AND NOT EXISTS (
                SELECT 1 FROM raw.skater_games sg WHERE sg.game_id = g.game_id
            )
            ORDER BY g.date
        """))
        game_ids = [row[0] for row in result.fetchall()]

    logger.info(f"Backfilling boxscores for {len(game_ids)} games...")
    success = 0
    for i, gid in enumerate(game_ids):
        if ingest_boxscore(gid):
            success += 1
        if (i + 1) % 50 == 0:
            logger.info(f"  ... {i+1}/{len(game_ids)} processed ({success} succeeded)")
        time.sleep(0.3)

    logger.info(f"Boxscore backfill complete: {success}/{len(game_ids)} succeeded")
    return success


def ingest_season(season: int):
    """Full ingestion pipeline for a single season (e.g., 20242025)."""
    start_year = season // 10000
    start_date = f"{start_year}-10-01"
    # July 31, not June 30: the COVID-delayed 2020-21 Cup Final ran to July 7
    end_date = f"{start_year + 1}-07-31"

    logger.info(f"=== Ingesting season {season} ({start_date} to {end_date}) ===")
    ingest_schedule(start_date, end_date)
    backfill_boxscores(season=season)
    backfill_team_stats(season=season)
    logger.info(f"=== Season {season} ingestion complete ===")


def daily_refresh():
    """Daily update: refresh recent schedule + backfill new boxscores. Run via cron."""
    logger.info("=== Starting daily refresh ===")
    ingest_teams()

    today = date.today()
    start = (today - timedelta(days=3)).strftime("%Y-%m-%d")
    end = (today + timedelta(days=7)).strftime("%Y-%m-%d")
    ingest_schedule(start, end)

    backfill_boxscores()
    backfill_team_stats()
    logger.info("=== Daily refresh complete ===")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _parse_saves_shots(val) -> tuple:
    """
    Parse the NHL API's strength-split format "saves/shots" (e.g. "19/21")
    into (saves, shots) ints. Returns (0, 0) for missing/malformed values.
    """
    if not val:
        return 0, 0
    try:
        saves_str, shots_str = str(val).split("/")
        return int(saves_str), int(shots_str)
    except (ValueError, AttributeError):
        return 0, 0


def _toi_to_seconds(toi_str: str) -> int:
    """Convert 'MM:SS' time-on-ice string to integer seconds."""
    if not toi_str or toi_str == "--:--":
        return 0
    try:
        parts = str(toi_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return 0


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from config.settings import check_db_connection

    if not check_db_connection():
        print("ERROR: Database not reachable. Start PostgreSQL first.")
        sys.exit(1)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "teams":
            ingest_teams()
        elif cmd == "daily":
            daily_refresh()
        elif cmd == "season" and len(sys.argv) > 2:
            ingest_season(int(sys.argv[2]))
        elif cmd == "backfill-all":
            for s in BACKFILL_SEASONS:
                ingest_season(s)
        else:
            print("Usage: python -m ingestion.nhl_api [teams|daily|season <YYYYYYYY>|backfill-all]")
    else:
        print("Usage: python -m ingestion.nhl_api [teams|daily|season <YYYYYYYY>|backfill-all]")
