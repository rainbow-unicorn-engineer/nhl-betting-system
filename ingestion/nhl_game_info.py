"""
ingestion/nhl_game_info.py
Scratches, officials and head coaches for every game, from the NHL's
gamecenter "right-rail" page, into raw.game_scratches, raw.game_officials
and raw.game_info (which doubles as the resumable fetch log).

Terms:
- Scratch (healthy scratch) → a player on the team's roster who did not
  dress for the game. With the box score (who did dress), this rebuilds
  "who was missing" for every past game: a free stand-in for injury lists
  that nobody keeps for past seasons. The list mixes injured players with
  players left out by choice; it does not say which.
- Referees and linesmen → the officials. The two referees call penalties
  (so power plays); linesmen call offside and icing.
- Head coach → the coach listed for each side; a change shows up here.

Source: https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail (free,
no key, undocumented), block gameInfo:
  referees / linesmen          [{fullName: {default}, sweaterNumber}]
  homeTeam / awayTeam
    headCoach                  {default}
    scratches                  [{id, firstName: {default}, lastName: {default}}]
ingestion/nhl_api.ingest_team_stats already downloads this page for the
team stats of each newly finished game; it hands the same response to
store_payload(), so the daily run costs no extra request. backfill()
fetches the page for finished games that have no raw.game_info row yet
(about 3 requests a second, resumable).

When it was known (point-in-time): scratches are confirmed at the
pre-game warm-up, roughly 30-60 minutes before puck drop, so they are
known before the closing line but NOT in the morning. Referee assignments
are public on the morning of the game. Playoff scratch lists are long:
they include the extra players a team carries ("black aces").

raw.game_info.status: ok (gameInfo was present), empty (the page had no
gameInfo; re-fetched while the game is recent), error (the request failed;
re-fetched every run). Each store replaces the game's rows, so a re-fetch
never duplicates.

Acceptance checks (written 2026-10-04, before the backfill ran; each miss
is reported):
  G1  at least 99% of finished regular-season and playoff games in each
      season end with status 'ok';
  G2  at least 98% of 'ok' games list exactly 2 referees;
  G3  no scratched player also appears in that game's box score
      (scratched_but_played = 0; any found are listed as data errors).

`python -m ingestion.nhl_game_info --help`.
"""
import argparse
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from config.settings import engine, local_today
from ingestion.polite import PoliteClient

logger = logging.getLogger("nhl.ingestion.nhl_game_info")

RIGHT_RAIL_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail"
MIN_INTERVAL_S = 0.34     # at most ~3 requests a second
RECENT_DAYS = 14          # 'empty' games this recent are re-fetched

# The same DDL is in config/migrate.py (TABLES) and db/schema.sql.
DDL = [
    """
        CREATE TABLE IF NOT EXISTS raw.game_info (
            game_id             BIGINT PRIMARY KEY REFERENCES raw.games(game_id),
            status              VARCHAR(10) NOT NULL,          -- ok, empty, error
            home_coach          VARCHAR(80),
            away_coach          VARCHAR(80),
            n_scratches_home    SMALLINT NOT NULL DEFAULT 0,
            n_scratches_away    SMALLINT NOT NULL DEFAULT 0,
            n_referees          SMALLINT NOT NULL DEFAULT 0,
            n_linesmen          SMALLINT NOT NULL DEFAULT 0,
            attempts            SMALLINT NOT NULL DEFAULT 1,
            problem             TEXT,
            fetched_at          TIMESTAMP NOT NULL
        )""",
    """
        CREATE TABLE IF NOT EXISTS raw.game_scratches (
            game_id         BIGINT NOT NULL REFERENCES raw.games(game_id),
            team            VARCHAR(3) NOT NULL,
            player_id       INTEGER NOT NULL,
            player_name     VARCHAR(80),
            PRIMARY KEY (game_id, player_id)
        )""",
    "CREATE INDEX IF NOT EXISTS idx_game_scratches_player "
    "ON raw.game_scratches(player_id, game_id)",
    """
        CREATE TABLE IF NOT EXISTS raw.game_officials (
            game_id         BIGINT NOT NULL REFERENCES raw.games(game_id),
            role            VARCHAR(10) NOT NULL,          -- referee, linesman
            official_name   VARCHAR(80) NOT NULL,
            sweater_number  SMALLINT,
            PRIMARY KEY (game_id, role, official_name)
        )""",
    "CREATE INDEX IF NOT EXISTS idx_game_officials_name "
    "ON raw.game_officials(official_name, game_id)",
]

_ensured = False


def ensure_tables(db=None) -> None:
    """Apply DDL when a table is missing (information_schema first, so an
    up-to-date database runs no DDL). Once per process."""
    global _ensured
    if _ensured and db is None:
        return
    with (db or engine).begin() as conn:
        n = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'raw'
              AND table_name IN ('game_info', 'game_scratches', 'game_officials')
        """)).scalar()
        if n != 3:
            logger.info("Schema upgrade: raw.game_info / game_scratches / game_officials")
            for stmt in DDL:
                conn.execute(text(stmt))
    if db is None:
        _ensured = True


# ── Pure parsing ──────────────────────────────────────────────────

def _text(obj) -> Optional[str]:
    """{'default': 'x', ...} or 'x' → 'x' (stripped); None when empty."""
    if isinstance(obj, dict):
        obj = obj.get("default")
    if obj is None:
        return None
    s = str(obj).strip()
    return s or None


def _int(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_game_info(body: dict, game_id: int, home_team: str, away_team: str) -> dict:
    """{'status', 'info', 'scratches', 'officials'} from a right-rail body.
    status is 'empty' when the page has no gameInfo block."""
    gi = (body or {}).get("gameInfo") if isinstance(body, dict) else None
    if not isinstance(gi, dict) or not gi:
        return {"status": "empty", "info": {}, "scratches": [], "officials": []}

    scratches: Dict[int, dict] = {}
    counts = {"home": 0, "away": 0}
    for side, team in (("homeTeam", home_team), ("awayTeam", away_team)):
        block = gi.get(side) or {}
        for p in block.get("scratches") or []:
            pid = _int(p.get("id"))
            if pid is None or pid in scratches:
                continue
            name = " ".join(x for x in (_text(p.get("firstName")),
                                        _text(p.get("lastName"))) if x) or None
            scratches[pid] = {"game_id": int(game_id), "team": team,
                              "player_id": pid, "player_name": name}
            counts["home" if side == "homeTeam" else "away"] += 1

    officials: Dict[Tuple[str, str], dict] = {}
    for key, role in (("referees", "referee"), ("linesmen", "linesman")):
        for o in gi.get(key) or []:
            name = _text(o.get("fullName"))
            if not name or (role, name) in officials:
                continue
            officials[(role, name)] = {"game_id": int(game_id), "role": role,
                                       "official_name": name,
                                       "sweater_number": _int(o.get("sweaterNumber"))}

    info = {
        "home_coach": _text((gi.get("homeTeam") or {}).get("headCoach")),
        "away_coach": _text((gi.get("awayTeam") or {}).get("headCoach")),
        "n_scratches_home": counts["home"], "n_scratches_away": counts["away"],
        "n_referees": sum(1 for r, _ in officials if r == "referee"),
        "n_linesmen": sum(1 for r, _ in officials if r == "linesman"),
    }
    return {"status": "ok", "info": info, "scratches": list(scratches.values()),
            "officials": list(officials.values())}


# ── Database ──────────────────────────────────────────────────────

UPSERT_INFO = text("""
    INSERT INTO raw.game_info (game_id, status, home_coach, away_coach,
        n_scratches_home, n_scratches_away, n_referees, n_linesmen,
        attempts, problem, fetched_at)
    VALUES (:game_id, :status, :home_coach, :away_coach,
        :n_scratches_home, :n_scratches_away, :n_referees, :n_linesmen,
        1, :problem, (now() AT TIME ZONE 'UTC'))
    ON CONFLICT (game_id) DO UPDATE SET
        status = CASE WHEN EXCLUDED.status = 'ok' OR raw.game_info.status <> 'ok'
                      THEN EXCLUDED.status ELSE raw.game_info.status END,
        home_coach = COALESCE(EXCLUDED.home_coach, raw.game_info.home_coach),
        away_coach = COALESCE(EXCLUDED.away_coach, raw.game_info.away_coach),
        n_scratches_home = CASE WHEN EXCLUDED.status = 'ok' THEN EXCLUDED.n_scratches_home
                                ELSE raw.game_info.n_scratches_home END,
        n_scratches_away = CASE WHEN EXCLUDED.status = 'ok' THEN EXCLUDED.n_scratches_away
                                ELSE raw.game_info.n_scratches_away END,
        n_referees = CASE WHEN EXCLUDED.status = 'ok' THEN EXCLUDED.n_referees
                          ELSE raw.game_info.n_referees END,
        n_linesmen = CASE WHEN EXCLUDED.status = 'ok' THEN EXCLUDED.n_linesmen
                          ELSE raw.game_info.n_linesmen END,
        attempts = LEAST(raw.game_info.attempts + 1, 32000),
        problem = EXCLUDED.problem,
        fetched_at = EXCLUDED.fetched_at
""")


def store(game_id: int, parsed: dict, problem: Optional[str] = None, db=None) -> None:
    """Write one game's parse in one transaction. An 'ok' parse replaces the
    game's scratches and officials; 'empty'/'error' touch only the log (an
    earlier 'ok' keeps its status and rows)."""
    if db is None:
        ensure_tables()
    info = {"home_coach": None, "away_coach": None, "n_scratches_home": 0,
            "n_scratches_away": 0, "n_referees": 0, "n_linesmen": 0,
            **(parsed.get("info") or {})}
    with (db or engine).begin() as conn:
        if parsed["status"] == "ok":
            conn.execute(text("DELETE FROM raw.game_scratches WHERE game_id = :g"),
                         {"g": game_id})
            conn.execute(text("DELETE FROM raw.game_officials WHERE game_id = :g"),
                         {"g": game_id})
            if parsed["scratches"]:
                conn.execute(text("""
                    INSERT INTO raw.game_scratches (game_id, team, player_id, player_name)
                    VALUES (:game_id, :team, :player_id, :player_name)
                """), parsed["scratches"])
            if parsed["officials"]:
                conn.execute(text("""
                    INSERT INTO raw.game_officials (game_id, role, official_name, sweater_number)
                    VALUES (:game_id, :role, :official_name, :sweater_number)
                """), parsed["officials"])
        conn.execute(UPSERT_INFO, {"game_id": game_id, "status": parsed["status"],
                                   "problem": problem, **info})


def store_payload(game_id: int, home_team: str, away_team: str, body: dict, db=None) -> str:
    """Parse and store a right-rail body already downloaded elsewhere
    (ingestion/nhl_api.ingest_team_stats). Returns the status."""
    parsed = parse_game_info(body, game_id, home_team, away_team)
    store(game_id, parsed, db=db)
    return parsed["status"]


def games_to_fetch(season: Optional[int] = None, retry_empty: bool = False,
                   limit: Optional[int] = None, today: Optional[date] = None,
                   db=None) -> List[tuple]:
    """(game_id, home_team, away_team) of finished games due a fetch."""
    today = today or local_today()
    sql = """
        SELECT g.game_id, g.home_team, g.away_team FROM raw.games g
        LEFT JOIN raw.game_info i ON i.game_id = g.game_id
        WHERE g.game_state IN ('FINAL', 'OFF') AND g.game_type IN (2, 3)
          AND (:season IS NULL OR g.season = CAST(:season AS integer))
          AND (i.game_id IS NULL OR i.status = 'error'
               OR (i.status = 'empty'
                   AND (CAST(:retry_all AS boolean) OR g.date >= :recent)))
        ORDER BY g.date, g.game_id
    """
    params = {"season": season, "retry_all": bool(retry_empty),
              "recent": today - timedelta(days=RECENT_DAYS)}
    if limit:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)
    with (db or engine).connect() as conn:
        return [tuple(r) for r in conn.execute(text(sql), params)]


def fetch_games(games: Sequence[tuple], client: Optional[PoliteClient] = None,
                db=None, max_consecutive_errors: int = 25) -> Dict[str, int]:
    ensure_tables(db)
    client = client or PoliteClient("NHL right-rail", min_interval_s=MIN_INTERVAL_S)
    counts = {"games": len(games), "ok": 0, "empty": 0, "error": 0,
              "scratches": 0, "officials": 0, "stopped_early": 0}
    streak = 0
    for i, (gid, home, away) in enumerate(games, 1):
        reply = client.get_json(RIGHT_RAIL_URL.format(game_id=int(gid)))
        if reply.status == "ok" and isinstance(reply.body, dict):
            parsed = parse_game_info(reply.body, gid, home, away)
            store(gid, parsed, db=db)
            counts[parsed["status"]] += 1
            counts["scratches"] += len(parsed["scratches"])
            counts["officials"] += len(parsed["officials"])
            streak = 0
        else:
            store(gid, {"status": "error"}, problem=reply.problem or "not a JSON object",
                  db=db)
            counts["error"] += 1
            streak += 1
        if i % 200 == 0 or i == len(games):
            logger.info(f"Game info {i}/{len(games)}: {counts['ok']} ok, "
                        f"{counts['empty']} empty, {counts['error']} error, "
                        f"{counts['scratches']:,} scratches")
        if streak >= max_consecutive_errors:
            logger.error(f"Game info: {streak} failures in a row; stopping "
                         f"(a re-run resumes from here)")
            counts["stopped_early"] = 1
            break
    return counts


def backfill(season: Optional[int] = None, retry_empty: bool = False,
             limit: Optional[int] = None) -> Dict[str, int]:
    """The resumable backfill and the daily form: every game due a fetch.
    The daily run finds none for games whose team stats were just loaded
    (that load stores their game info from the same response)."""
    ensure_tables()
    games = games_to_fetch(season=season, retry_empty=retry_empty, limit=limit)
    if not games:
        logger.info("Game info: every finished game is already fetched")
        return {"games": 0, "ok": 0, "empty": 0, "error": 0, "scratches": 0,
                "officials": 0, "stopped_early": 0}
    logger.info(f"Game info: {len(games)} game(s) to fetch")
    return fetch_games(games)


def coverage(db=None) -> List[dict]:
    """Per season: finished games, fetch outcomes, rows, and scratched
    players who also appear in the box score (should be none)."""
    ensure_tables(db)
    with (db or engine).connect() as conn:
        rows = conn.execute(text("""
            WITH g AS (SELECT game_id, season FROM raw.games
                       WHERE game_state IN ('FINAL', 'OFF') AND game_type IN (2, 3)),
            sc AS (SELECT game_id, COUNT(*) AS n FROM raw.game_scratches GROUP BY game_id),
            played AS (
                SELECT s.game_id, COUNT(*) AS n FROM raw.game_scratches s
                WHERE EXISTS (SELECT 1 FROM raw.skater_games k
                              WHERE k.game_id = s.game_id AND k.player_id = s.player_id)
                   OR EXISTS (SELECT 1 FROM raw.goalie_games q
                              WHERE q.game_id = s.game_id AND q.player_id = s.player_id)
                GROUP BY s.game_id)
            SELECT g.season, COUNT(*) AS finished,
                   COUNT(*) FILTER (WHERE i.status = 'ok') AS ok,
                   COUNT(*) FILTER (WHERE i.status = 'empty') AS empty,
                   COUNT(*) FILTER (WHERE i.status = 'error') AS error,
                   COUNT(*) FILTER (WHERE i.game_id IS NULL) AS not_fetched,
                   COALESCE(SUM(sc.n), 0) AS scratches,
                   COALESCE(SUM(i.n_referees + i.n_linesmen), 0) AS officials,
                   COUNT(*) FILTER (WHERE i.status = 'ok' AND i.n_referees <> 2) AS games_without_2_refs,
                   COALESCE(SUM(p.n), 0) AS scratched_but_played
            FROM g LEFT JOIN raw.game_info i ON i.game_id = g.game_id
            LEFT JOIN sc ON sc.game_id = g.game_id
            LEFT JOIN played p ON p.game_id = g.game_id
            GROUP BY g.season ORDER BY g.season
        """)).mappings().all()
    return [dict(r) for r in rows]


def _season_arg(value: str) -> int:
    v = value.strip()
    if len(v) == 8 and v.isdigit() and int(v[4:]) == int(v[:4]) + 1:
        return int(v)
    raise argparse.ArgumentTypeError(f"{value!r} is not a season like 20252026")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.nhl_game_info",
        description="Load scratched players, referees, linesmen and head coaches "
                    "from the NHL right-rail page for every finished game not "
                    "fetched yet. Free; about 3 requests a second; safe to re-run.")
    parser.add_argument("--season", type=_season_arg, default=None,
                        help="only this season, such as 20252026 (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="fetch at most this many games this run")
    parser.add_argument("--retry-empty", action="store_true",
                        help="also re-fetch every 'empty' game, however old")
    parser.add_argument("--report", action="store_true",
                        help="print coverage by season and fetch nothing")
    args = parser.parse_args(argv)
    if args.report:
        for r in coverage():
            print(r)
        return 0
    counts = backfill(args.season, args.retry_empty, args.limit)
    print(counts)
    return 1 if counts["stopped_early"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
