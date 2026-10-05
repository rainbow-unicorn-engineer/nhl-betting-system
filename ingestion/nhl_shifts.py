"""
ingestion/nhl_shifts.py
NHL shift charts → raw.shifts, with a resumable fetch log in raw.shift_fetches.

Terms:
- Shift → one stretch a player spends on the ice before changing. A
  forward's shift lasts about 40-50 seconds; a game has 700-900 of them.
- Line → the forwards (usually three) and defence pair (two) who go on the
  ice together. Who actually played with whom is only visible in shifts.
- Power-play unit (PP1, PP2) → the group a team sends out when the other
  side has a player in the penalty box. PP1 gets most of that time, and it
  is where a player's points and shots jump.
- Point-in-time → what was known before a game. A game's shifts exist only
  after the game, so a feature may use the shifts of EARLIER games only.

Source: https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=<id>
(free, no key, undocumented; checked 2026-10-04 on games from 2020-21 to
2026-27). One request returns the whole game as {"data": [...], "total": n}.
Each row is either
  typeCode 517 → a shift: playerId, teamAbbrev, period, startTime and
                 endTime ("MM:SS" into the period), duration ("MM:SS"),
                 shiftNumber (the player's 1st, 2nd, ... shift), and the
                 NHL's own row id;
  typeCode 505 → a goal marker (EVG, PPG, SHG, EN) with no duration.
Only shifts are stored; goal markers are counted in the fetch log
(n_goal_events), since raw.shots already holds every goal.

Writes, per game, in one transaction: the game's old raw.shifts rows are
replaced by the new ones (so a re-fetch never duplicates), and the game's
raw.shift_fetches row records the outcome:
  ok       → a full game (at least MIN_SHIFTS shifts and MIN_PLAYERS players)
  partial  → fewer than that; stored anyway, and re-fetched while the game
             is recent (the NHL sometimes posts shifts in pieces)
  empty    → the endpoint had no shifts for the game; nothing is deleted
  error    → the request failed after its retries; nothing is deleted
A run fetches every finished game (game_state OFF, regular season and
playoffs, puck drop at least SETTLE_HOURS ago) with no fetch-log row, every
'error', and every 'empty' or 'partial' game from the last RECENT_DAYS days
(all of them with --retry-empty). So a stopped run resumes where it left
off, and the daily run fetches only last night's games.

Pace: at most one request every MIN_INTERVAL_S (0.34 s, about 3 a second;
ingestion/polite.py), retried on timeouts, 429 and 5xx.

Acceptance checks (written 2026-10-04, before the backfill ran; the
backfill passes only if all hold, and each miss is reported):
  S1  at least 99% of finished regular-season and playoff games in each
      season 2020-21 to 2025-26 end with status 'ok';
  S2  in 'ok' games, box-score players (raw.skater_games, raw.goalie_games)
      with no shift are at most 0.5% of box-score player-games;
  S3  in 'ok' games, the median absolute gap between a skater's summed
      shift durations and raw.skater_games.toi_seconds is at most 5 seconds.

`python -m ingestion.nhl_shifts --help`; `--report` prints coverage by
season without fetching anything.
"""
import argparse
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from config.settings import engine, local_today
from ingestion.polite import PoliteClient, Reply

logger = logging.getLogger("nhl.ingestion.nhl_shifts")

URL = "https://api.nhle.com/stats/rest/en/shiftcharts"
SHIFT_TYPE = 517
GOAL_TYPE = 505
MIN_INTERVAL_S = 0.34     # at most ~3 requests a second
MIN_SHIFTS = 400          # a full 60-minute game has 700-900
MIN_PLAYERS = 30          # two dressed teams are 38-40 players
RECENT_DAYS = 14          # empty/partial games this recent are re-fetched
SETTLE_HOURS = 6          # never fetch a game that started less than this ago

# The same DDL is in config/migrate.py (COLUMNS, TABLES) and db/schema.sql.
DDL = [
    "ALTER TABLE raw.shifts ADD COLUMN IF NOT EXISTS nhl_shift_id BIGINT",
    "ALTER TABLE raw.shifts ADD COLUMN IF NOT EXISTS shift_number SMALLINT",
    """
        CREATE TABLE IF NOT EXISTS raw.shift_fetches (
            game_id         BIGINT PRIMARY KEY REFERENCES raw.games(game_id),
            status          VARCHAR(10) NOT NULL,          -- ok, partial, empty, error
            n_shifts        INTEGER NOT NULL DEFAULT 0,
            n_players       SMALLINT NOT NULL DEFAULT 0,
            n_goal_events   SMALLINT NOT NULL DEFAULT 0,
            attempts        SMALLINT NOT NULL DEFAULT 1,
            problem         TEXT,
            fetched_at      TIMESTAMP NOT NULL
        )""",
]

_ensured = False


def ensure_tables(db=None) -> None:
    """Apply DDL when anything is missing. Checks information_schema first,
    so an up-to-date database runs no DDL and takes no table lock. Once per
    process."""
    global _ensured
    if _ensured and db is None:
        return
    with (db or engine).begin() as conn:
        have = conn.execute(text("""
            SELECT
              (SELECT COUNT(*) FROM information_schema.columns
               WHERE table_schema = 'raw' AND table_name = 'shifts'
                 AND column_name IN ('nhl_shift_id', 'shift_number')),
              (SELECT COUNT(*) FROM information_schema.tables
               WHERE table_schema = 'raw' AND table_name = 'shift_fetches')
        """)).one()
        if tuple(have) != (2, 1):
            logger.info("Schema upgrade: raw.shifts columns / raw.shift_fetches")
            for stmt in DDL:
                conn.execute(text(stmt))
    if db is None:
        _ensured = True


# ── Pure parsing ──────────────────────────────────────────────────

def mmss(value) -> Optional[int]:
    """'MM:SS' → seconds; None when missing or malformed."""
    if value is None:
        return None
    try:
        m, s = str(value).strip().split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


def _int(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_shifts(body: dict, game_id: int) -> Tuple[List[dict], int]:
    """(shift rows, goal-marker count) from one shiftcharts response.
    Rows for another game, rows missing a player, period or start time, and
    repeated NHL row ids are dropped. A missing duration is end - start."""
    rows: Dict[object, dict] = {}
    goals = 0
    for r in (body or {}).get("data") or []:
        if _int(r.get("gameId")) != int(game_id):
            continue
        type_code = _int(r.get("typeCode"))
        if type_code == GOAL_TYPE:
            goals += 1
            continue
        if type_code != SHIFT_TYPE:
            continue
        pid, period, start = _int(r.get("playerId")), _int(r.get("period")), mmss(r.get("startTime"))
        if pid is None or period is None or start is None:
            continue
        end = mmss(r.get("endTime"))
        duration = mmss(r.get("duration"))
        if duration is None and end is not None and end >= start:
            duration = end - start
        nhl_id = _int(r.get("id"))
        key = nhl_id if nhl_id is not None else (pid, period, start)
        rows[key] = {
            "game_id": int(game_id), "player_id": pid, "period": period,
            "start_time": start, "end_time": end, "duration": duration,
            "team": (r.get("teamAbbrev") or None), "nhl_shift_id": nhl_id,
            "shift_number": _int(r.get("shiftNumber")),
        }
    out = sorted(rows.values(), key=lambda x: (x["period"], x["start_time"], x["player_id"]))
    return out, goals


def classify(rows: Sequence[dict]) -> str:
    """ok / partial / empty for a parsed game."""
    if not rows:
        return "empty"
    players = {r["player_id"] for r in rows}
    if len(rows) < MIN_SHIFTS or len(players) < MIN_PLAYERS:
        return "partial"
    return "ok"


# ── Database ──────────────────────────────────────────────────────

INSERT_SHIFTS = text("""
    INSERT INTO raw.shifts (game_id, player_id, period, start_time, end_time,
                            duration, team, nhl_shift_id, shift_number)
    SELECT * FROM unnest(
        CAST(:game_id AS bigint[]), CAST(:player_id AS integer[]),
        CAST(:period AS smallint[]), CAST(:start_time AS integer[]),
        CAST(:end_time AS integer[]), CAST(:duration AS integer[]),
        CAST(:team AS varchar[]), CAST(:nhl_shift_id AS bigint[]),
        CAST(:shift_number AS smallint[]))
""")

UPSERT_FETCH = text("""
    INSERT INTO raw.shift_fetches (game_id, status, n_shifts, n_players,
                                   n_goal_events, attempts, problem, fetched_at)
    VALUES (:game_id, :status, :n_shifts, :n_players, :n_goal_events, 1, :problem,
            (now() AT TIME ZONE 'UTC'))
    ON CONFLICT (game_id) DO UPDATE SET
        status = EXCLUDED.status,
        n_shifts = CASE WHEN EXCLUDED.status IN ('ok', 'partial')
                        THEN EXCLUDED.n_shifts ELSE raw.shift_fetches.n_shifts END,
        n_players = CASE WHEN EXCLUDED.status IN ('ok', 'partial')
                         THEN EXCLUDED.n_players ELSE raw.shift_fetches.n_players END,
        n_goal_events = CASE WHEN EXCLUDED.status IN ('ok', 'partial')
                             THEN EXCLUDED.n_goal_events
                             ELSE raw.shift_fetches.n_goal_events END,
        attempts = LEAST(raw.shift_fetches.attempts + 1, 32000),
        problem = EXCLUDED.problem,
        fetched_at = EXCLUDED.fetched_at
""")


def store_game(game_id: int, rows: List[dict], goals: int, status: str,
               problem: Optional[str] = None, db=None) -> None:
    """Replace the game's shifts (only when rows were parsed) and record
    the fetch, in one transaction."""
    with (db or engine).begin() as conn:
        if rows:
            conn.execute(text("DELETE FROM raw.shifts WHERE game_id = :g"), {"g": game_id})
            cols = ["game_id", "player_id", "period", "start_time", "end_time",
                    "duration", "team", "nhl_shift_id", "shift_number"]
            conn.execute(INSERT_SHIFTS, {c: [r[c] for r in rows] for c in cols})
        conn.execute(UPSERT_FETCH, {
            "game_id": game_id, "status": status, "n_shifts": len(rows),
            "n_players": len({r["player_id"] for r in rows}),
            "n_goal_events": goals, "problem": problem})


def games_to_fetch(season: Optional[int] = None, retry_empty: bool = False,
                   limit: Optional[int] = None, today: Optional[date] = None,
                   now_utc: Optional[datetime] = None, db=None) -> List[int]:
    """Game ids due a fetch (see the module docstring), oldest first."""
    today = today or local_today()
    now_utc = now_utc or datetime.now(timezone.utc)
    sql = """
        SELECT g.game_id FROM raw.games g
        LEFT JOIN raw.shift_fetches f ON f.game_id = g.game_id
        WHERE g.game_state = 'OFF' AND g.game_type IN (2, 3)
          AND (g.start_time_utc IS NULL OR g.start_time_utc <= :settled)
          AND (:season IS NULL OR g.season = CAST(:season AS integer))
          AND (f.game_id IS NULL OR f.status = 'error'
               OR (f.status IN ('empty', 'partial')
                   AND (CAST(:retry_all AS boolean) OR g.date >= :recent)))
        ORDER BY g.date, g.game_id
    """
    params = {"settled": now_utc - timedelta(hours=SETTLE_HOURS), "season": season,
              "retry_all": bool(retry_empty), "recent": today - timedelta(days=RECENT_DAYS)}
    if limit:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)
    with (db or engine).connect() as conn:
        return [r[0] for r in conn.execute(text(sql), params)]


# ── Fetching ──────────────────────────────────────────────────────

def fetch_game(client: PoliteClient, game_id: int) -> Tuple[str, List[dict], int, Optional[str]]:
    """(status, rows, goal markers, problem) for one game."""
    reply: Reply = client.get_json(URL, {"cayenneExp": f"gameId={int(game_id)}"})
    if reply.status != "ok":
        return "error", [], 0, reply.problem
    if not isinstance(reply.body, dict) or not isinstance(reply.body.get("data"), list):
        return "error", [], 0, "response has no data list"
    rows, goals = parse_shifts(reply.body, game_id)
    status = classify(rows)
    problem = None
    if status == "partial":
        problem = (f"{len(rows)} shifts, {len({r['player_id'] for r in rows})} players")
    total = _int(reply.body.get("total"))
    if total is not None and total > len(reply.body["data"]):
        status, problem = "partial", f"response held {len(reply.body['data'])} of {total} rows"
    return status, rows, goals, problem


def fetch_games(game_ids: Sequence[int], client: Optional[PoliteClient] = None,
                db=None, max_consecutive_errors: int = 25) -> Dict[str, int]:
    """Fetch and store each game; returns counts by status. Stops early
    after max_consecutive_errors failures in a row (the API is down)."""
    ensure_tables(db)
    client = client or PoliteClient("NHL shift charts", min_interval_s=MIN_INTERVAL_S)
    counts = {"games": len(game_ids), "ok": 0, "partial": 0, "empty": 0, "error": 0,
              "shifts": 0, "stopped_early": 0}
    streak = 0
    started = time.monotonic()
    for i, gid in enumerate(game_ids, 1):
        status, rows, goals, problem = fetch_game(client, gid)
        store_game(gid, rows, goals, status, problem, db=db)
        counts[status] += 1
        counts["shifts"] += len(rows)
        streak = streak + 1 if status == "error" else 0
        if i % 100 == 0 or i == len(game_ids):
            rate = i / max(time.monotonic() - started, 1e-9)
            logger.info(f"Shift charts {i}/{len(game_ids)} ({rate:.1f} games/s): "
                        f"{counts['ok']} ok, {counts['partial']} partial, "
                        f"{counts['empty']} empty, {counts['error']} error, "
                        f"{counts['shifts']:,} shifts")
        if streak >= max_consecutive_errors:
            logger.error(f"Shift charts: {streak} failures in a row; stopping "
                         f"(a re-run resumes from here)")
            counts["stopped_early"] = 1
            break
    return counts


def fetch_missing(season: Optional[int] = None, retry_empty: bool = False,
                  limit: Optional[int] = None) -> Dict[str, int]:
    """The resumable backfill and the daily form: every game due a fetch."""
    ensure_tables()
    ids = games_to_fetch(season=season, retry_empty=retry_empty, limit=limit)
    if not ids:
        logger.info("Shift charts: every finished game is already fetched")
        return {"games": 0, "ok": 0, "partial": 0, "empty": 0, "error": 0,
                "shifts": 0, "stopped_early": 0}
    logger.info(f"Shift charts: {len(ids)} game(s) to fetch"
                + (f" (season {season})" if season else ""))
    return fetch_games(ids)


def coverage(db=None) -> List[dict]:
    """Per season: finished games, fetch outcomes, stored shifts, and
    box-score players (raw.skater_games + raw.goalie_games) of fetched
    games who have no shift."""
    ensure_tables(db)
    with (db or engine).connect() as conn:
        rows = conn.execute(text("""
            WITH g AS (
                SELECT game_id, season FROM raw.games
                WHERE game_state = 'OFF' AND game_type IN (2, 3)
            ),
            box AS (
                SELECT game_id, player_id FROM raw.skater_games
                UNION SELECT game_id, player_id FROM raw.goalie_games
            ),
            shifted AS (SELECT DISTINCT game_id, player_id FROM raw.shifts),
            missing AS (
                SELECT b.game_id, COUNT(*) AS n
                FROM box b
                JOIN raw.shift_fetches f2 ON f2.game_id = b.game_id AND f2.status = 'ok'
                LEFT JOIN shifted s ON s.game_id = b.game_id AND s.player_id = b.player_id
                WHERE s.game_id IS NULL
                GROUP BY b.game_id
            )
            SELECT g.season,
                   COUNT(*) AS finished,
                   COUNT(*) FILTER (WHERE f.status = 'ok') AS ok,
                   COUNT(*) FILTER (WHERE f.status = 'partial') AS partial,
                   COUNT(*) FILTER (WHERE f.status = 'empty') AS empty,
                   COUNT(*) FILTER (WHERE f.status = 'error') AS error,
                   COUNT(*) FILTER (WHERE f.game_id IS NULL) AS not_fetched,
                   COALESCE(SUM(f.n_shifts), 0) AS shifts,
                   COALESCE(SUM(m.n), 0) AS box_players_without_shifts
            FROM g
            LEFT JOIN raw.shift_fetches f ON f.game_id = g.game_id
            LEFT JOIN missing m ON m.game_id = g.game_id
            GROUP BY g.season ORDER BY g.season
        """)).mappings().all()
    return [dict(r) for r in rows]


# ── CLI ───────────────────────────────────────────────────────────

def _season_arg(value: str) -> int:
    v = value.strip()
    if len(v) == 8 and v.isdigit() and int(v[4:]) == int(v[:4]) + 1:
        return int(v)
    raise argparse.ArgumentTypeError(f"{value!r} is not a season like 20252026")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.nhl_shifts",
        description="Load NHL shift charts (who was on the ice, shift by shift) "
                    "into raw.shifts for every finished game not fetched yet. "
                    "Free; about 3 requests a second; safe to stop and re-run.")
    parser.add_argument("--season", type=_season_arg, default=None,
                        help="only this season, such as 20252026 (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="fetch at most this many games this run")
    parser.add_argument("--retry-empty", action="store_true",
                        help="also re-fetch every 'empty' or 'partial' game, however old")
    parser.add_argument("--report", action="store_true",
                        help="print coverage by season and fetch nothing")
    args = parser.parse_args(argv)
    if args.report:
        for r in coverage():
            print(r)
        return 0
    counts = fetch_missing(args.season, args.retry_empty, args.limit)
    print(counts)
    return 1 if counts["stopped_early"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
