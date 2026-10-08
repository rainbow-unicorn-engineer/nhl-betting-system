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

Fallback (added 2026-10-05, after the backfill found the endpoint empty
for 57 games of 2024-25, 2025-04-08 to 2025-04-15): when the endpoint has
no shifts for a game, the NHL's official HTML time-on-ice reports
(https://www.nhl.com/scores/htmlreports/<season>/TH<nnnnnn>.HTM for the
home team, TV... for the visitors) carry the same shifts: per player, the
shift number, period, start and end ("elapsed / game" clock) and duration.
Those rows name players by sweater number, so the game's boxscore
(api-web.nhle.com/v1/gamecenter/<id>/boxscore) maps number to player id.
Three requests per game; rows stored this way have no NHL row id, and the
fetch log says source = 'html' (otherwise 'api').

Bad source rows (found 2026-10-05, after the backfill; see clean_rows):
the API sometimes returns the same shift twice (same player, period and
start time under two NHL row ids), and for a few games it returns another
game's shifts, all looking normal. So every game's rows are cleaned
before they are stored: rows of a team that is not the game's home or
away team are dropped, and one row is kept per (player, period, start
time). Then the QA check (qa_problem) compares each skater's summed
shift durations with his box-score ice time (raw.skater_games,
toi_seconds): a full game where any skater is more than QA_TOLERANCE_S
(60) seconds off, or that has no box score yet, is 'suspect'. A suspect
game tries the HTML reports, which replace the API's shifts only when
they pass the check.

Writes, per game, in one transaction: the game's old raw.shifts rows are
replaced by the new ones (so a re-fetch never duplicates), and the game's
raw.shift_fetches row records the outcome:
  ok       → a full game (at least MIN_SHIFTS shifts and MIN_PLAYERS
             players) that passes the QA check
  suspect  → a full game that fails it; stored anyway (a feature should
             read 'ok' games only), and re-fetched while the game is recent
  partial  → fewer than that; stored anyway, and re-fetched while the game
             is recent (the NHL sometimes posts shifts in pieces)
  empty    → the endpoint had no shifts for the game (none of its own
             teams'); nothing is deleted
  error    → the request failed after its retries; nothing is deleted
A run fetches every finished game (game_state FINAL or OFF, as everywhere
else in the repo; regular season and playoffs; puck drop at least
SETTLE_HOURS ago) with no fetch-log row, every 'error', and every 'empty',
'partial' or 'suspect' game from the last RECENT_DAYS days (all of them
with --retry-empty). So a stopped run resumes where it left off, and the
daily run fetches only last night's games. `--recheck` cleans and
re-checks the stored shifts without fetching anything (recheck_stored);
`--game ID` fetches one game again whatever its status.

Pace: at most one request every MIN_INTERVAL_S (0.34 s, about 3 a second;
ingestion/polite.py), retried on timeouts, 429 and 5xx.

Acceptance checks (written 2026-10-04, before the backfill ran; the
backfill passes only if all hold, and each miss is reported):
  S1  at least 99% of finished regular-season and playoff games in each
      season 2020-21 to 2025-26 end with status 'ok';
  S2  in 'ok' games, box-score players who played (raw.skater_games,
      raw.goalie_games, toi_seconds > 0) with no shift are at most 0.5% of
      those player-games. (Amended before the backfill, after a 3-game
      trial: the first wording counted backup goalies, who dress with 0
      ice time and never take a shift.)
  S3  in 'ok' games, the median absolute gap between a skater's summed
      shift durations and raw.skater_games.toi_seconds is at most 5 seconds.

STATUS (backfill finished 2026-10-05, live database): 7,984 of 7,984
finished games 2020-21 to 2026-10-05 fetched 'ok', 6,098,180 shifts.
S1 pass (100% in every season; 57 games of April 2025 came from the HTML
fallback, without which 2024-25 was 95.9%). S2 pass (1 of 304,309
player-games who played has no shift: 1 second of ice time). S3 pass
(median gap 0 seconds in every season). Per-season table:
docs/data_sources.md, section 2.13.

But S3's median hid a bad tail, found by an independent check on
2026-10-06 (the backfill stored the source faithfully; the source was
wrong): 19,203 repeated shift rows in 1,537 games, 1,371 rows of other
teams in 2 games (2021020513: a third copy of its own shifts under the
codes STL and MIN; 2025020565: game 2024020565's shifts, the same game
number a season earlier), and 923 games with a skater more than 60 s
from his box-score ice time (2.1% of skater-games more than 5 s off).
clean_rows, the QA check and 'suspect' came from that.

REPAIR (2026-10-07; rehearsed on a full copy of the live tables, and the
numbers are the copy's; NOT YET RUN on the live database, which needs
`--recheck` and then `--retry-empty`; backup taken first,
data/backups/shifts_before_repair_20261007.dump):
  --recheck      deletes 18,499 repeated and 1,371 wrong-team rows (both
                 cross-game games keep 704 and 746 shifts of their own
                 teams and pass the QA check; a re-fetch gives the same
                 rows); 7,908 ok, 76 suspect.
  --retry-empty  the 76 suspect games: 72 pass from the HTML reports, 1
                 from the API, 3 stay suspect (2020020124, 2020020252,
                 2021020326: the HTML reports disagree with the box score
                 too).
  After: 7,981 ok, 3 suspect, 6,078,021 shifts, 0 repeated or wrong-team
  rows; in ok games 0 skater-games more than 60 s off and 889 of 287,225
  (0.31%) more than 5 s off. S1 still passes (lowest 2020-21, 950 of 952);
  S2 0 of 304,309; S3 median 0 s. `--report` prints these per season.

`python -m ingestion.nhl_shifts --help`; `--report` prints coverage by
season without fetching anything.
"""
import argparse
import html as html_lib
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from config.settings import engine, local_today
from ingestion.polite import PoliteClient, Reply

logger = logging.getLogger("nhl.ingestion.nhl_shifts")

URL = "https://api.nhle.com/stats/rest/en/shiftcharts"
HTML_URL = "https://www.nhl.com/scores/htmlreports/{season}/T{side}{suffix}.HTM"
BOXSCORE_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
SHIFT_TYPE = 517
GOAL_TYPE = 505
MIN_INTERVAL_S = 0.34     # at most ~3 requests a second
MIN_SHIFTS = 400          # a full 60-minute game has 700-900
MIN_PLAYERS = 30          # two dressed teams are 38-40 players
RECENT_DAYS = 14          # empty/partial/suspect games this recent are re-fetched
SETTLE_HOURS = 6          # never fetch a game that started less than this ago
QA_TOLERANCE_S = 60       # per-skater summed shifts vs box-score ice time
# A finished game is game_state FINAL or OFF, the definition the rest of
# the repo uses (nhl_game_info, settle, features); OFF alone missed a game
# the NHL had not closed yet.
FINISHED_SQL = "g.game_state IN ('FINAL', 'OFF')"

# The same DDL is in config/migrate.py (COLUMNS, TABLES) and db/schema.sql.
DDL = [
    "ALTER TABLE raw.shifts ADD COLUMN IF NOT EXISTS nhl_shift_id BIGINT",
    "ALTER TABLE raw.shifts ADD COLUMN IF NOT EXISTS shift_number SMALLINT",
    """
        CREATE TABLE IF NOT EXISTS raw.shift_fetches (
            game_id         BIGINT PRIMARY KEY REFERENCES raw.games(game_id),
            status          VARCHAR(10) NOT NULL,          -- ok, suspect, partial, empty, error
            n_shifts        INTEGER NOT NULL DEFAULT 0,
            n_players       SMALLINT NOT NULL DEFAULT 0,
            n_goal_events   SMALLINT NOT NULL DEFAULT 0,
            attempts        SMALLINT NOT NULL DEFAULT 1,
            problem         TEXT,
            fetched_at      TIMESTAMP NOT NULL,
            source          VARCHAR(8)                     -- api, html
        )""",
    "ALTER TABLE raw.shift_fetches ADD COLUMN IF NOT EXISTS source VARCHAR(8)",
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
              (SELECT COUNT(*) FROM information_schema.columns
               WHERE table_schema = 'raw' AND table_name = 'shift_fetches'
                 AND column_name = 'source')
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


def clean_rows(rows: Sequence[dict], teams: Optional[Sequence[str]] = None
               ) -> Tuple[List[dict], Dict[str, int]]:
    """(kept rows, {"wrong_team": n, "duplicates": n}).

    The shift-chart API sometimes returns bad rows, and they all look
    normal (found 2026-10-05, after the backfill):
    - another game's shifts: game 2021020513 (NYI-WSH) also held STL and
      MIN shifts, and 2025020565 (NJD-BUF) held only VGK and SJS shifts.
      With `teams` (the game's home and away codes) every row whose team
      is not one of them is dropped.
    - the same shift twice: same player, period and start time, under two
      NHL row ids and shift numbers (19,203 extra rows in 1,537 games).
      One row per (player, period, start time) is kept: the lowest NHL row
      id (the first one posted), or the first seen when there is no id.
      In 251 cases the two copies end at different times; neither choice
      matches the box score better, and the QA check (qa_problem) flags a
      game whose totals are still off.
    Output is sorted by period, start time and player."""
    allowed = {t for t in (teams or []) if t}
    kept: Dict[tuple, dict] = {}
    counts = {"wrong_team": 0, "duplicates": 0}
    for r in rows:
        if allowed and r.get("team") not in allowed:
            counts["wrong_team"] += 1
            continue
        key = (r["player_id"], r["period"], r["start_time"])
        old = kept.get(key)
        if old is not None:
            counts["duplicates"] += 1
            new_id, old_id = r.get("nhl_shift_id"), old.get("nhl_shift_id")
            if new_id is None or (old_id is not None and old_id <= new_id):
                continue
        kept[key] = r
    out = sorted(kept.values(), key=lambda x: (x["period"], x["start_time"], x["player_id"]))
    return out, counts


def dropped_note(counts: Dict[str, int]) -> Optional[str]:
    """'dropped 3 rows of another team, 2 repeated shifts' or None."""
    parts = []
    if counts.get("wrong_team"):
        parts.append(f"{counts['wrong_team']} rows of another team")
    if counts.get("duplicates"):
        parts.append(f"{counts['duplicates']} repeated shifts")
    return ("dropped " + ", ".join(parts)) if parts else None


def join_problems(*parts: Optional[str]) -> Optional[str]:
    """The non-empty parts joined with '; ', or None."""
    kept = [p for p in parts if p]
    return "; ".join(kept) if kept else None


def parse_shifts(body: dict, game_id: int, teams: Optional[Sequence[str]] = None
                 ) -> Tuple[List[dict], int, Dict[str, int]]:
    """(shift rows, goal-marker count, dropped counts) from one shiftcharts
    response. Rows for another game id, rows missing a player, period or
    start time, and repeated NHL row ids are skipped; then clean_rows drops
    rows of another team (when `teams` is given) and repeated shifts. A
    missing duration is end - start."""
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
    out, dropped = clean_rows(list(rows.values()), teams)
    return out, goals, dropped


_BLOCK_RE = re.compile(r'class="playerHeading[^"]*"[^>]*>\s*(\d+)\s+([^<]*)<', re.I)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.I | re.S)
_CLOCK_RE = re.compile(r"^(\d{1,2}:\d{2})\s*/\s*\d{1,2}:\d{2}$")


def _cell_text(raw: str) -> str:
    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", raw)).split())


def parse_toi_report(page: str) -> List[dict]:
    """Shift rows from one NHL HTML time-on-ice report (one team):
    [{sweater, shift_number, period, start_time, end_time, duration}].
    Only rows whose start cell reads "elapsed / game" are shifts; the
    per-period summary tables under each player are skipped. Period "OT"
    is 4; a shootout row (no clock) never matches."""
    out: List[dict] = []
    page = page or ""
    blocks = list(_BLOCK_RE.finditer(page))
    for i, b in enumerate(blocks):
        sweater = int(b.group(1))
        chunk = page[b.end(): blocks[i + 1].start() if i + 1 < len(blocks) else len(page)]
        for row in _ROW_RE.findall(chunk):
            cells = [_cell_text(c) for c in _CELL_RE.findall(row)]
            if len(cells) < 5:
                continue
            start_m, end_m = _CLOCK_RE.match(cells[2]), _CLOCK_RE.match(cells[3])
            if not start_m:
                continue
            per = cells[1].upper()
            period = 4 if per == "OT" else _int(per)
            if period is None:
                continue
            start = mmss(start_m.group(1))
            end = mmss(end_m.group(1)) if end_m else None
            duration = mmss(cells[4])
            if duration is None and end is not None and end >= start:
                duration = end - start
            out.append({"sweater": sweater, "shift_number": _int(cells[0]),
                        "period": period, "start_time": start, "end_time": end,
                        "duration": duration})
    return out


def sweater_map(boxscore: dict) -> Dict[Tuple[str, int], Tuple[int, Optional[str]]]:
    """{(side 'H' or 'V', sweater number): (player_id, team)} from a boxscore."""
    out: Dict[Tuple[str, int], Tuple[int, Optional[str]]] = {}
    boxscore = boxscore or {}
    stats = boxscore.get("playerByGameStats") or {}
    for side, key in (("H", "homeTeam"), ("V", "awayTeam")):
        team = (boxscore.get(key) or {}).get("abbrev") or None
        for group in ("forwards", "defense", "goalies"):
            for p in (stats.get(key) or {}).get(group) or []:
                pid, num = _int(p.get("playerId")), _int(p.get("sweaterNumber"))
                if pid is not None and num is not None:
                    out[(side, num)] = (pid, team)
    return out


def html_rows(game_id: int, pages: Dict[str, str], boxscore: dict) -> Tuple[List[dict], int]:
    """raw.shifts rows from the two reports ({'H': page, 'V': page}) and
    the boxscore; returns (rows, shift rows whose sweater was not found)."""
    numbers = sweater_map(boxscore)
    rows: Dict[tuple, dict] = {}
    unmatched = 0
    for side, page in pages.items():
        for r in parse_toi_report(page):
            hit = numbers.get((side, r["sweater"]))
            if hit is None:
                unmatched += 1
                continue
            pid, team = hit
            rows[(pid, r["period"], r["start_time"])] = {
                "game_id": int(game_id), "player_id": pid, "period": r["period"],
                "start_time": r["start_time"], "end_time": r["end_time"],
                "duration": r["duration"], "team": team, "nhl_shift_id": None,
                "shift_number": r["shift_number"]}
    out = sorted(rows.values(), key=lambda x: (x["period"], x["start_time"], x["player_id"]))
    return out, unmatched


def classify(rows: Sequence[dict]) -> str:
    """ok / partial / empty for a parsed game."""
    if not rows:
        return "empty"
    players = {r["player_id"] for r in rows}
    if len(rows) < MIN_SHIFTS or len(players) < MIN_PLAYERS:
        return "partial"
    return "ok"


def qa_problem(rows: Sequence[dict], box_toi: Dict[int, int],
               tolerance_s: int = QA_TOLERANCE_S) -> Optional[str]:
    """The per-game QA check: None when it passes, else why it failed.

    box_toi is {player_id: box-score ice time in seconds} for the game's
    skaters who played (raw.skater_games, toi_seconds > 0). Each of them
    must have shifts whose durations add up to within tolerance_s of that
    ice time; a skater with no shift at all is off by his whole ice time.
    Goalies are not checked (raw.skater_games has none). With no box score
    to compare against, the check fails: the game cannot be confirmed."""
    if not box_toi:
        return "no box-score ice time to check against"
    summed: Dict[int, int] = {}
    for r in rows:
        summed[r["player_id"]] = summed.get(r["player_id"], 0) + (r.get("duration") or 0)
    off = sorted(((abs(summed.get(pid, 0) - toi), pid, summed.get(pid, 0), toi)
                  for pid, toi in box_toi.items()
                  if abs(summed.get(pid, 0) - toi) > tolerance_s), reverse=True)
    if not off:
        return None
    _, pid, secs, toi = off[0]
    return (f"{len(off)} skater(s) more than {tolerance_s} s from box-score ice time "
            f"(worst: player {pid}, shifts {secs} s against {toi} s)")


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
                                   n_goal_events, attempts, problem, fetched_at, source)
    VALUES (:game_id, :status, :n_shifts, :n_players, :n_goal_events, 1, :problem,
            (now() AT TIME ZONE 'UTC'), :source)
    ON CONFLICT (game_id) DO UPDATE SET
        status = EXCLUDED.status,
        n_shifts = CASE WHEN EXCLUDED.status IN ('ok', 'partial', 'suspect')
                        THEN EXCLUDED.n_shifts ELSE raw.shift_fetches.n_shifts END,
        n_players = CASE WHEN EXCLUDED.status IN ('ok', 'partial', 'suspect')
                         THEN EXCLUDED.n_players ELSE raw.shift_fetches.n_players END,
        n_goal_events = CASE WHEN EXCLUDED.status IN ('ok', 'partial', 'suspect')
                             THEN EXCLUDED.n_goal_events
                             ELSE raw.shift_fetches.n_goal_events END,
        attempts = LEAST(raw.shift_fetches.attempts + 1, 32000),
        problem = EXCLUDED.problem,
        fetched_at = EXCLUDED.fetched_at,
        source = CASE WHEN EXCLUDED.status IN ('ok', 'partial', 'suspect')
                      THEN EXCLUDED.source ELSE raw.shift_fetches.source END
""")


GAME_TEAMS = text("SELECT home_team, away_team FROM raw.games WHERE game_id = :g")
BOX_TOI = text("""
    SELECT player_id, toi_seconds FROM raw.skater_games
    WHERE game_id = :g AND toi_seconds > 0
""")


def _teams(conn, game_id: int) -> List[str]:
    row = conn.execute(GAME_TEAMS, {"g": game_id}).first()
    return [t for t in (row or ()) if t]


def game_context(game_id: int, db=None) -> Tuple[List[str], Dict[int, int]]:
    """(the game's home and away team codes, box-score ice time per skater
    who played) from raw.games and raw.skater_games."""
    with (db or engine).connect() as conn:
        teams = _teams(conn, game_id)
        box = {int(p): int(t) for p, t in conn.execute(BOX_TOI, {"g": game_id})}
    return teams, box


def store_game(game_id: int, rows: List[dict], goals: int, status: str,
               problem: Optional[str] = None, db=None, source: str = "api") -> None:
    """Replace the game's shifts (only when rows were parsed) and record
    the fetch, in one transaction. As a guard, the rows go through
    clean_rows with the game's teams from raw.games first, so another
    team's shifts or a repeated shift can never be stored. When the guard
    drops rows, the status is checked again (classify), so a game it
    empties is logged 'empty', not 'ok'."""
    with (db or engine).begin() as conn:
        if rows:
            n_in = len(rows)
            rows, _ = clean_rows(rows, _teams(conn, game_id))
            if len(rows) < n_in and status in ("ok", "partial", "suspect"):
                new = classify(rows)
                status = status if new == "ok" else new
        if rows:
            conn.execute(text("DELETE FROM raw.shifts WHERE game_id = :g"), {"g": game_id})
            cols = ["game_id", "player_id", "period", "start_time", "end_time",
                    "duration", "team", "nhl_shift_id", "shift_number"]
            conn.execute(INSERT_SHIFTS, {c: [r[c] for r in rows] for c in cols})
        conn.execute(UPSERT_FETCH, {
            "game_id": game_id, "status": status, "n_shifts": len(rows),
            "n_players": len({r["player_id"] for r in rows}),
            "n_goal_events": goals, "problem": problem, "source": source})


def games_to_fetch(season: Optional[int] = None, retry_empty: bool = False,
                   limit: Optional[int] = None, today: Optional[date] = None,
                   now_utc: Optional[datetime] = None, db=None) -> List[int]:
    """Game ids due a fetch (see the module docstring), oldest first."""
    today = today or local_today()
    now_utc = now_utc or datetime.now(timezone.utc)
    sql = f"""
        SELECT g.game_id FROM raw.games g
        LEFT JOIN raw.shift_fetches f ON f.game_id = g.game_id
        WHERE {FINISHED_SQL} AND g.game_type IN (2, 3)
          AND (g.start_time_utc IS NULL OR g.start_time_utc <= :settled)
          AND (:season IS NULL OR g.season = CAST(:season AS integer))
          AND (f.game_id IS NULL OR f.status = 'error'
               OR (f.status IN ('empty', 'partial', 'suspect')
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

def fetch_game(client: PoliteClient, game_id: int, teams: Optional[Sequence[str]] = None
               ) -> Tuple[str, List[dict], int, Optional[str]]:
    """(status, rows, goal markers, problem) for one game. With `teams`
    (the game's two team codes), rows of any other team are dropped; a
    game left with none of its own shifts is 'empty'."""
    reply: Reply = client.get_json(URL, {"cayenneExp": f"gameId={int(game_id)}"})
    if reply.status != "ok":
        return "error", [], 0, reply.problem
    if not isinstance(reply.body, dict) or not isinstance(reply.body.get("data"), list):
        return "error", [], 0, "response has no data list"
    rows, goals, dropped = parse_shifts(reply.body, game_id, teams)
    status = classify(rows)
    problem = None
    if status == "partial":
        problem = (f"{len(rows)} shifts, {len({r['player_id'] for r in rows})} players")
    total = _int(reply.body.get("total"))
    if total is not None and total > len(reply.body["data"]):
        status, problem = "partial", f"response held {len(reply.body['data'])} of {total} rows"
    return status, rows, goals, join_problems(problem, dropped_note(dropped))


def fetch_game_html(client: PoliteClient, game_id: int,
                    teams: Optional[Sequence[str]] = None,
                    reason: str = "the shift-chart API had none"
                    ) -> Tuple[str, List[dict], Optional[str]]:
    """(status, rows, problem) from the HTML time-on-ice reports; status is
    ok, partial or empty, or error when a page or the boxscore failed."""
    gid = str(int(game_id))
    season = f"{gid[:4]}{int(gid[:4]) + 1}"
    box = client.get_json(BOXSCORE_URL.format(game_id=gid))
    if box.status != "ok" or not isinstance(box.body, dict):
        return "error", [], f"boxscore: {box.problem or box.status}"
    pages = {}
    for side in ("H", "V"):
        page = client.get_text(HTML_URL.format(season=season, side=side, suffix=gid[4:]))
        if page.status != "ok" or not isinstance(page.body, str):
            return "error", [], f"T{side} report: {page.problem or page.status}"
        pages[side] = page.body
    rows, unmatched = html_rows(game_id, pages, box.body)
    rows, dropped = clean_rows(rows, teams)
    status = classify(rows)
    problem = f"from the HTML time-on-ice reports ({reason})"
    if unmatched:
        problem += f"; {unmatched} shift rows had a sweater number not in the boxscore"
    return status, rows, join_problems(problem, dropped_note(dropped))


def _empty_counts(n: int = 0) -> Dict[str, int]:
    return {"games": n, "ok": 0, "partial": 0, "empty": 0, "error": 0, "suspect": 0,
            "shifts": 0, "stopped_early": 0, "from_html": 0}


def fetch_games(game_ids: Sequence[int], client: Optional[PoliteClient] = None,
                db=None, max_consecutive_errors: int = 25,
                html_fallback: bool = True) -> Dict[str, int]:
    """Fetch and store each game; returns counts by status. Stops early
    after max_consecutive_errors failures in a row (the API is down).

    Per game: the API's shifts, minus other teams' rows and repeats; a
    full game then goes through the QA check (qa_problem) against the box
    score and becomes 'suspect' when it fails. An empty API answer, or a
    suspect one for a game that has a box score, tries the HTML reports;
    the HTML shifts are used for an empty game, and replace suspect API
    shifts only when they pass the QA check themselves."""
    ensure_tables(db)
    client = client or PoliteClient("NHL shift charts", min_interval_s=MIN_INTERVAL_S)
    counts = _empty_counts(len(game_ids))
    streak = 0
    started = time.monotonic()
    for i, gid in enumerate(game_ids, 1):
        teams, box = game_context(gid, db)
        status, rows, goals, problem = fetch_game(client, gid, teams)
        source = "api"
        qa = qa_problem(rows, box) if status == "ok" else None
        if qa:
            status, problem = "suspect", join_problems(problem, f"QA: {qa}")
        if html_fallback and (status == "empty" or (status == "suspect" and box)):
            reason = ("the shift-chart API had none" if status == "empty"
                      else "the shift-chart API's shifts failed the QA check")
            h_status, h_rows, h_problem = fetch_game_html(client, gid, teams, reason)
            h_qa = qa_problem(h_rows, box) if h_status == "ok" else None
            if h_qa:
                h_status, h_problem = "suspect", join_problems(h_problem, f"QA: {h_qa}")
            if (status == "empty" and h_rows) or (status == "suspect" and h_status == "ok"):
                status, rows, problem, source = h_status, h_rows, h_problem, "html"
                counts["from_html"] += 1
            elif h_status == "error":
                problem = join_problems(
                    problem if status == "suspect" else "shift-chart API empty", h_problem)
            elif status == "suspect":
                problem = join_problems(problem, "the HTML reports did not pass either")
        store_game(gid, rows, goals, status, problem, db=db, source=source)
        counts[status] += 1
        counts["shifts"] += len(rows)
        streak = streak + 1 if status == "error" else 0
        if i % 100 == 0 or i == len(game_ids):
            rate = i / max(time.monotonic() - started, 1e-9)
            logger.info(f"Shift charts {i}/{len(game_ids)} ({rate:.1f} games/s): "
                        f"{counts['ok']} ok, {counts['suspect']} suspect, "
                        f"{counts['partial']} partial, {counts['empty']} empty, "
                        f"{counts['error']} error, {counts['shifts']:,} shifts")
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
        return _empty_counts()
    logger.info(f"Shift charts: {len(ids)} game(s) to fetch"
                + (f" (season {season})" if season else ""))
    return fetch_games(ids)


STORED_ROWS = text("""
    SELECT shift_id, player_id, period, start_time, duration, team, nhl_shift_id
    FROM raw.shifts WHERE game_id = :g
""")

UPDATE_FETCH_CHECKED = text("""
    UPDATE raw.shift_fetches
    SET status = :status, n_shifts = :n_shifts, n_players = :n_players, problem = :problem
    WHERE game_id = :g
""")


def _base_problem(problem: Optional[str]) -> Optional[str]:
    """A stored problem without its old QA verdict, which a recheck writes
    afresh, so checking a game twice gives the same text. Notes of rows
    dropped earlier stay: they record what was removed."""
    parts = [p for p in (problem or "").split("; ") if p and not p.startswith("QA: ")]
    return join_problems(*parts)


def recheck_stored(season: Optional[int] = None, game_ids: Optional[Sequence[int]] = None,
                   db=None) -> Dict[str, int]:
    """Clean and QA-check the shifts already stored, fetching nothing: the
    repair for games loaded before clean_rows and the QA check existed,
    and for games whose box score arrived after their shifts.

    For every game whose fetch is ok, partial or suspect, in one
    transaction per game: deletes the rows clean_rows drops (another
    team's, repeated shifts), then sets the status from the rows left:
    classify(), and 'suspect' when an otherwise full game fails qa_problem
    (or back to 'ok' when a suspect one now passes). A game left with no
    rows becomes 'empty', so a `--retry-empty` run re-fetches it.
    attempts, fetched_at and source are kept. Returns counts: games
    checked, rows deleted, and games per new status."""
    ensure_tables(db)
    sql = """
        SELECT f.game_id, f.status, f.problem FROM raw.shift_fetches f
        JOIN raw.games g ON g.game_id = f.game_id
        WHERE f.status IN ('ok', 'partial', 'suspect')
          AND (:season IS NULL OR g.season = CAST(:season AS integer))
    """
    params: dict = {"season": season}
    if game_ids is not None:
        sql += " AND f.game_id = ANY(CAST(:ids AS bigint[]))"
        params["ids"] = [int(g) for g in game_ids]
    sql += " ORDER BY f.game_id"
    with (db or engine).connect() as conn:
        todo = [tuple(r) for r in conn.execute(text(sql), params)]
    counts = {"games": len(todo), "wrong_team_deleted": 0, "duplicates_deleted": 0,
              "ok": 0, "suspect": 0, "partial": 0, "empty": 0, "changed": 0}
    for i, (gid, old_status, old_problem) in enumerate(todo, 1):
        with (db or engine).begin() as conn:
            teams = _teams(conn, gid)
            stored = [dict(r) for r in conn.execute(STORED_ROWS, {"g": gid}).mappings()]
            kept, dropped = clean_rows(stored, teams)
            gone = {r["shift_id"] for r in stored} - {r["shift_id"] for r in kept}
            if gone:
                conn.execute(text("DELETE FROM raw.shifts WHERE shift_id = ANY(:ids)"),
                             {"ids": sorted(gone)})
            box = {int(p): int(t) for p, t in conn.execute(BOX_TOI, {"g": gid})}
            status = classify(kept)
            qa = qa_problem(kept, box) if status == "ok" else None
            if qa:
                status = "suspect"
            problem = join_problems(_base_problem(old_problem), dropped_note(dropped),
                                    f"QA: {qa}" if qa else None)
            conn.execute(UPDATE_FETCH_CHECKED, {
                "g": gid, "status": status, "n_shifts": len(kept),
                "n_players": len({r["player_id"] for r in kept}), "problem": problem})
        counts["wrong_team_deleted"] += dropped["wrong_team"]
        counts["duplicates_deleted"] += dropped["duplicates"]
        counts[status] += 1
        counts["changed"] += int(status != old_status)
        if i % 500 == 0 or i == len(todo):
            logger.info(f"Shift recheck {i}/{len(todo)}: {counts['ok']} ok, "
                        f"{counts['suspect']} suspect, {counts['partial']} partial, "
                        f"{counts['empty']} empty; deleted "
                        f"{counts['duplicates_deleted']:,} repeated and "
                        f"{counts['wrong_team_deleted']:,} wrong-team rows")
    return counts


def coverage(db=None) -> List[dict]:
    """Per season: finished games, fetch outcomes, stored shifts,
    box-score players who played (raw.skater_games + raw.goalie_games,
    toi_seconds > 0) in 'ok' games but have no shift (check S2), and the
    median absolute gap in seconds between a skater's summed shifts and his
    box-score ice time in 'ok' games (check S3).

    Diagnostics added 2026-10-06 (the median in S3 cannot see a bad tail):
    skater-games in 'ok' games more than 5 s and more than QA_TOLERANCE_S
    off their box-score ice time, and, over every stored row, repeated
    shifts (same player, period and start) and rows of a team not in the
    game. After the repair the last two are 0."""
    ensure_tables(db)
    with (db or engine).connect() as conn:
        rows = conn.execute(text(f"""
            WITH g AS (
                SELECT g.game_id, g.season, g.home_team, g.away_team FROM raw.games g
                WHERE {FINISHED_SQL} AND g.game_type IN (2, 3)
            ),
            box AS (
                SELECT game_id, player_id FROM raw.skater_games WHERE toi_seconds > 0
                UNION SELECT game_id, player_id FROM raw.goalie_games WHERE toi_seconds > 0
            ),
            shifted AS (SELECT DISTINCT game_id, player_id FROM raw.shifts),
            missing AS (
                SELECT b.game_id, COUNT(*) AS n
                FROM box b
                JOIN raw.shift_fetches f2 ON f2.game_id = b.game_id AND f2.status = 'ok'
                LEFT JOIN shifted s ON s.game_id = b.game_id AND s.player_id = b.player_id
                WHERE s.game_id IS NULL
                GROUP BY b.game_id
            ),
            toi_gap AS (
                SELECT g3.season, ABS(st.secs - sg.toi_seconds) AS gap
                FROM (SELECT s.game_id, s.player_id, SUM(s.duration) AS secs
                      FROM raw.shifts s
                      JOIN raw.shift_fetches f3 ON f3.game_id = s.game_id AND f3.status = 'ok'
                      GROUP BY s.game_id, s.player_id) st
                JOIN raw.skater_games sg ON sg.game_id = st.game_id AND sg.player_id = st.player_id
                JOIN raw.games g3 ON g3.game_id = st.game_id
                WHERE sg.toi_seconds > 0
            ),
            toi_stats AS (
                SELECT season, percentile_cont(0.5) WITHIN GROUP (ORDER BY gap) AS med,
                       COUNT(*) AS n,
                       COUNT(*) FILTER (WHERE gap > 5) AS over_5,
                       COUNT(*) FILTER (WHERE gap > :tol) AS over_tol
                FROM toi_gap GROUP BY season
            ),
            repeats AS (
                SELECT game_id, SUM(n - 1) AS n FROM (
                    SELECT game_id, COUNT(*) AS n FROM raw.shifts
                    GROUP BY game_id, player_id, period, start_time HAVING COUNT(*) > 1) x
                GROUP BY game_id
            ),
            wrong_team AS (
                SELECT s.game_id, COUNT(*) AS n FROM raw.shifts s
                JOIN g ON g.game_id = s.game_id
                WHERE s.team IS DISTINCT FROM g.home_team AND s.team IS DISTINCT FROM g.away_team
                GROUP BY s.game_id
            )
            SELECT g.season,
                   COUNT(*) AS finished,
                   COUNT(*) FILTER (WHERE f.status = 'ok') AS ok,
                   COUNT(*) FILTER (WHERE f.status = 'ok' AND f.source = 'html') AS ok_from_html,
                   COUNT(*) FILTER (WHERE f.status = 'suspect') AS suspect,
                   COUNT(*) FILTER (WHERE f.status = 'partial') AS partial,
                   COUNT(*) FILTER (WHERE f.status = 'empty') AS empty,
                   COUNT(*) FILTER (WHERE f.status = 'error') AS error,
                   COUNT(*) FILTER (WHERE f.game_id IS NULL) AS not_fetched,
                   COALESCE(SUM(f.n_shifts), 0) AS shifts,
                   COALESCE(SUM(m.n), 0) AS box_players_without_shifts,
                   (SELECT med FROM toi_stats t WHERE t.season = g.season)
                       AS median_skater_toi_gap_s,
                   (SELECT n FROM toi_stats t WHERE t.season = g.season)
                       AS skater_games_checked,
                   (SELECT over_5 FROM toi_stats t WHERE t.season = g.season)
                       AS skater_games_over_5s,
                   (SELECT over_tol FROM toi_stats t WHERE t.season = g.season)
                       AS skater_games_over_tolerance,
                   COALESCE(SUM(r.n), 0) AS repeated_shift_rows,
                   COALESCE(SUM(w.n), 0) AS wrong_team_rows
            FROM g
            LEFT JOIN raw.shift_fetches f ON f.game_id = g.game_id
            LEFT JOIN missing m ON m.game_id = g.game_id
            LEFT JOIN repeats r ON r.game_id = g.game_id
            LEFT JOIN wrong_team w ON w.game_id = g.game_id
            GROUP BY g.season ORDER BY g.season
        """), {"tol": QA_TOLERANCE_S}).mappings().all()
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
                        help="also re-fetch every 'empty', 'partial' or 'suspect' game, "
                             "however old")
    parser.add_argument("--report", action="store_true",
                        help="print coverage by season and fetch nothing")
    parser.add_argument("--recheck", action="store_true",
                        help="clean the stored shifts (drop repeated shifts and other "
                             "teams' rows) and re-run the QA check against the box "
                             "scores; fetches nothing")
    parser.add_argument("--game", type=int, action="append", default=None, metavar="GAME_ID",
                        help="fetch this game again whatever its status (repeatable), "
                             "such as 2021020513")
    args = parser.parse_args(argv)
    if args.report:
        for r in coverage():
            print(r)
        return 0
    if args.recheck:
        print(recheck_stored(args.season))
        return 0
    if args.game:
        counts = fetch_games(args.game)
        print(counts)
        return 1 if counts["stopped_early"] else 0
    counts = fetch_missing(args.season, args.retry_empty, args.limit)
    print(counts)
    return 1 if counts["stopped_early"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
