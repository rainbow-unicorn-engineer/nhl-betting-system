"""
ingestion/nhl_stats.py
Power-play, penalty-kill and faceoff columns of raw.skater_games, from the
NHL stats API (api.nhle.com/stats/rest, free, no key, undocumented).

Terms:
- Power play (PP): a team plays with more skaters because the other side
  has a player in the penalty box.
- Penalty kill (PK, "shorthanded", sh in column names): the side that is a
  player short.
- Time on ice (TOI): how long a player was on the ice, here in seconds.
- Faceoff: the puck drop that restarts play; one centre wins it.

Why: the boxscore that ingestion/nhl_api.py loads has no special-teams
split, so raw.skater_games.pp_toi_seconds, sh_toi_seconds, pp_goals,
pp_assists, fow and fol always kept their default, 0. That matters beyond
props: features/team_features.py divides power-play expected goals by the
team's summed pp_toi_seconds (and penalty-kill xG against by sh_toi_seconds),
so with every value 0, pp_xgf_per60 and pk_xga_per60 are the constant 99
(the clip) in every features.team_rolling row that isn't NULL (measured on
a copy of the live data, 2026-09-29). Filling these columns and
rebuilding features turns those two into real features, which changes
what the moneyline and totals models see: re-run their walk-forward
evaluations before trusting new numbers, and fill every season first so
the feature means the same thing in every season.

Source: three per-game skater reports, one row per (game, player):
  skater/timeonice    ppTimeOnIce, shTimeOnIce  -> pp_toi_seconds, sh_toi_seconds
  skater/powerplay    ppGoals, ppAssists        -> pp_goals, pp_assists
  skater/faceoffwins  totalFaceoffWins, ...Losses -> fow, fol
Each report lists every skater who played, so the three row sets are the
same. Rows are filtered with cayenneExp: a gameDate range, gameTypeId>=2
(regular season and playoffs; the few other types never match a stored
row), and seasonId for --season.

Paging (measured 2026-09-29): limit=-1 returns every matching row in one
response; an explicit limit above 100 is silently cut to 100. So the date
range is split into windows of WINDOW_DAYS (30) days, about 8,000 rows or
4 MB per report, one request per report per window. If a response ever
holds fewer rows than its `total`, the rest is paged PAGE_SIZE (100) at a
time with start=.

Writes: UPDATE only, keyed by (game_id, player_id). Rows come from the
boxscore load; a stats row with no raw.skater_games row (game not loaded
yet) is counted and skipped. stats_filled_at records when this module
last wrote a row, so a 0 in these columns is real data only where it is
set. A row is written only when a value changed or it was never filled,
so re-running is safe and cheap. A report that fails for a window (after
RETRIES attempts) skips that whole window, so a row is never half filled;
it is logged, counted, and the run goes on.

Cost: free. A season is about 9 windows x 3 reports = 27 requests with a
PAUSE_S (1 s) pause between them. `python -m ingestion.nhl_stats --help`.
"""
import argparse
import logging
import time
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from sqlalchemy import text

from config.settings import CURRENT_SEASON, engine, local_today

logger = logging.getLogger("nhl.ingestion.nhl_stats")

BASE_URL = "https://api.nhle.com/stats/rest/en/skater"

# report -> {stats API field: raw.skater_games column}
REPORTS: Dict[str, Dict[str, str]] = {
    "timeonice": {"ppTimeOnIce": "pp_toi_seconds", "shTimeOnIce": "sh_toi_seconds"},
    "powerplay": {"ppGoals": "pp_goals", "ppAssists": "pp_assists"},
    "faceoffwins": {"totalFaceoffWins": "fow", "totalFaceoffLosses": "fol"},
}
COLUMNS: List[str] = [c for fields in REPORTS.values() for c in fields.values()]

WINDOW_DAYS = 30      # days per request; a month is ~8k rows, ~4 MB
PAGE_SIZE = 100       # the API's largest explicit page; bigger limits become 100
PAUSE_S = 1.0         # between requests: a free, undocumented API
TIMEOUT_S = 60
RETRIES = 3
BATCH = 5000          # rows per UPDATE statement
SORT = '[{"property":"gameId","direction":"ASC"},{"property":"playerId","direction":"ASC"}]'

# The same DDL is in config/migrate.py (COLUMNS) and db/schema.sql.
# stats_filled_at: when this module last wrote the row's special-teams and
# faceoff columns (naive UTC); NULL = never, so those zeros are defaults.
DDL = [
    "ALTER TABLE raw.skater_games ADD COLUMN IF NOT EXISTS stats_filled_at TIMESTAMP",
]

_ensured = False


def ensure_columns() -> None:
    """Apply DDL when the column is missing. Checks information_schema
    first, so an up-to-date database takes no table lock. Once per process."""
    global _ensured
    if _ensured:
        return
    with engine.begin() as conn:
        present = conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'raw' AND table_name = 'skater_games'
              AND column_name = 'stats_filled_at'
        """)).scalar()
        if not present:
            logger.info("Schema upgrade: adding raw.skater_games.stats_filled_at")
            for stmt in DDL:
                conn.execute(text(stmt))
    _ensured = True


# ── Pure helpers ──────────────────────────────────────────────────

def cayenne(date_from: date, date_to: date, season: Optional[int] = None) -> str:
    """The cayenneExp filter for one window: both dates inclusive,
    regular season and playoffs, optionally one season."""
    exp = (f'gameDate>="{date_from.isoformat()}" and '
           f'gameDate<="{date_to.isoformat()}" and gameTypeId>=2')
    if season is not None:
        exp += f" and seasonId={int(season)}"
    return exp


def windows(date_from: date, date_to: date, days: int = WINDOW_DAYS) -> List[Tuple[date, date]]:
    """[date_from, date_to] cut into consecutive inclusive windows of at
    most `days` days, with no gap and no overlap."""
    out, start = [], date_from
    while start <= date_to:
        end = min(start + timedelta(days=days - 1), date_to)
        out.append((start, end))
        start = end + timedelta(days=1)
    return out


def windows_for_dates(dates: Iterable[date], days: int = WINDOW_DAYS) -> List[Tuple[date, date]]:
    """Windows of at most `days` days that cover every date given, starting
    a new one at the first date that doesn't fit. Dates in between that
    weren't asked for are fetched too; re-filling them changes nothing."""
    out: List[Tuple[date, date]] = []
    for d in sorted(set(dates)):
        if out and (d - out[-1][0]).days < days:
            out[-1] = (out[-1][0], d)
        else:
            out.append((d, d))
    return out


def _to_int(value) -> Optional[int]:
    """A count from the API as int; None when missing or not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def merge_reports(results: Dict[str, list]) -> List[dict]:
    """One dict per (game_id, player_id) with every COLUMNS value; a column
    whose report has no row for that player is None (left as stored).
    Rows without a gameId or playerId are dropped."""
    merged: Dict[Tuple[int, int], dict] = {}
    for report, rows in results.items():
        fields = REPORTS[report]
        for r in rows or []:
            gid, pid = _to_int(r.get("gameId")), _to_int(r.get("playerId"))
            if gid is None or pid is None:
                continue
            row = merged.setdefault((gid, pid), {"game_id": gid, "player_id": pid,
                                                 **{c: None for c in COLUMNS}})
            for field, column in fields.items():
                row[column] = _to_int(r.get(field))
    return [merged[k] for k in sorted(merged)]


# ── Network ───────────────────────────────────────────────────────

def _get_json(url: str, params: dict) -> Optional[dict]:
    """GET with RETRIES attempts; None after the last failure (logged)."""
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_S)
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body, dict) and isinstance(body.get("data"), list):
                return body
            problem = "response has no data list"
        except requests.HTTPError as e:
            r = e.response
            problem = f"HTTP {getattr(r, 'status_code', '?')} {getattr(r, 'reason', '')}".strip()
        except requests.Timeout:
            problem = f"timed out after {TIMEOUT_S}s"
        except requests.ConnectionError as e:
            problem = f"could not reach api.nhle.com ({type(e).__name__})"
        except ValueError:
            problem = "response is not JSON"
        if attempt < RETRIES:
            logger.warning(f"NHL stats request failed ({problem}); retrying "
                           f"({attempt}/{RETRIES})")
            time.sleep(PAUSE_S * 2 * attempt)
        else:
            logger.error(f"NHL stats request failed after {RETRIES} attempts: {problem}")
    return None


def fetch_report(report: str, exp: str) -> Optional[list]:
    """Every row of one per-game skater report for a cayenneExp filter;
    None when a request failed (the window must then be skipped)."""
    url = f"{BASE_URL}/{report}"
    base = {"isAggregate": "false", "isGame": "true", "sort": SORT, "cayenneExp": exp}
    body = _get_json(url, {**base, "start": 0, "limit": -1})
    if body is None:
        return None
    rows = list(body["data"])
    total = _to_int(body.get("total"))
    # limit=-1 has always returned everything; page the rest if it ever stops
    while total is not None and len(rows) < total:
        time.sleep(PAUSE_S)
        page = _get_json(url, {**base, "start": len(rows), "limit": PAGE_SIZE})
        if page is None:
            return None
        if not page["data"]:
            logger.warning(f"NHL stats {report}: {len(rows)} of {total} rows "
                           f"returned, then an empty page; keeping what came back")
            break
        rows.extend(page["data"])
    return rows


# ── Database ──────────────────────────────────────────────────────

def _update_sql() -> str:
    arrays = ",\n            ".join(
        ["CAST(:game_id AS bigint[])", "CAST(:player_id AS integer[])"]
        + [f"CAST(:{c} AS integer[])" for c in COLUMNS])
    names = ", ".join(["game_id", "player_id"] + COLUMNS)
    sets = ",\n            ".join(f"{c} = COALESCE(v.{c}, sg.{c})" for c in COLUMNS)
    changed = "\n           OR ".join(
        f"COALESCE(v.{c}, sg.{c}) IS DISTINCT FROM sg.{c}" for c in COLUMNS)
    return f"""
    WITH v AS (
        SELECT * FROM unnest(
            {arrays}
        ) AS v({names})
    ),
    matched AS (
        SELECT COUNT(*) AS n FROM v
        JOIN raw.skater_games sg ON sg.game_id = v.game_id AND sg.player_id = v.player_id
    ),
    changed AS (
        UPDATE raw.skater_games sg SET
            {sets},
            stats_filled_at = (now() AT TIME ZONE 'UTC')
        FROM v
        WHERE sg.game_id = v.game_id AND sg.player_id = v.player_id
          AND (sg.stats_filled_at IS NULL
           OR {changed})
        RETURNING 1
    )
    SELECT (SELECT n FROM matched) AS matched, (SELECT COUNT(*) FROM changed) AS changed
    """


UPDATE_SQL = _update_sql()


def apply_updates(rows: List[dict]) -> Tuple[int, int]:
    """Write merged rows to raw.skater_games. Returns (matched, changed):
    rows that have a raw.skater_games row, and rows actually written."""
    matched = changed = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        params = {k: [r[k] for r in chunk] for k in ["game_id", "player_id"] + COLUMNS}
        with engine.begin() as conn:
            m, c = conn.execute(text(UPDATE_SQL), params).one()
        matched += int(m)
        changed += int(c)
    return matched, changed


def fill_windows(wins: List[Tuple[date, date]], season: Optional[int] = None) -> dict:
    """Fetch all REPORTS for each window and update raw.skater_games.
    Returns totals: fetched, matched, changed, unmatched, failed_windows."""
    ensure_columns()
    summary = {"windows": len(wins), "requests": 0, "fetched": 0, "matched": 0,
               "changed": 0, "unmatched": 0, "failed_windows": 0}
    for n, (d0, d1) in enumerate(wins, 1):
        exp = cayenne(d0, d1, season)
        results, failed = {}, None
        for report in REPORTS:
            if summary["requests"]:
                time.sleep(PAUSE_S)
            summary["requests"] += 1
            rows = fetch_report(report, exp)
            if rows is None:
                failed = report
                break
            results[report] = rows
        if failed:
            summary["failed_windows"] += 1
            logger.error(f"NHL stats: {d0} to {d1} skipped, the {failed} report "
                         f"failed; nothing written for that window")
            continue
        merged = merge_reports(results)
        matched, changed = apply_updates(merged) if merged else (0, 0)
        summary["fetched"] += len(merged)
        summary["matched"] += matched
        summary["changed"] += changed
        summary["unmatched"] += len(merged) - matched
        logger.info(f"NHL stats {n}/{len(wins)}: {d0} to {d1}: {len(merged)} player-games, "
                    f"{matched} in raw.skater_games, {changed} written")
    if summary["unmatched"]:
        logger.info(f"{summary['unmatched']} stats row(s) had no raw.skater_games row "
                    f"(box score not loaded yet); load it, then run this again")
    return summary


def fill_range(date_from: date, date_to: date, season: Optional[int] = None) -> dict:
    """Fill every game dated date_from..date_to (inclusive)."""
    if date_from > date_to:
        raise ValueError(f"--from {date_from} is after --to {date_to}")
    logger.info(f"NHL stats: filling special-teams and faceoff columns, "
                f"{date_from} to {date_to}" + (f", season {season}" if season else ""))
    return fill_windows(windows(date_from, date_to), season)


def season_bounds(season: int) -> Tuple[date, date]:
    """First and last date of the season's stored regular-season and
    playoff games; without any, Sept 1 to Sept 30 of the next year (the
    window ingest_season uses). Never past today."""
    with engine.connect() as conn:
        lo, hi = conn.execute(text("""
            SELECT MIN(date), MAX(date) FROM raw.games
            WHERE season = :s AND game_type IN (2, 3)
        """), {"s": season}).one()
    start_year = season // 10000
    lo = lo or date(start_year, 9, 1)
    hi = min(hi or date(start_year + 1, 9, 30), local_today())
    return lo, hi


def fill_season(season: int) -> dict:
    lo, hi = season_bounds(season)
    return fill_range(lo, hi, season=season)


def fill_missing(season: Optional[int] = None) -> dict:
    """Fill only the dates of the season's finished games that still have
    unfilled raw.skater_games rows: the daily form, one window (3 requests)
    on a normal day, nothing when every row is filled. Rows the stats API
    doesn't have yet stay unfilled and are tried again next run."""
    season = season or CURRENT_SEASON
    ensure_columns()
    with engine.connect() as conn:
        dates = [r[0] for r in conn.execute(text("""
            SELECT DISTINCT g.date FROM raw.games g
            JOIN raw.skater_games sg ON sg.game_id = g.game_id
            WHERE g.season = :s AND g.game_type IN (2, 3)
              AND g.game_state IN ('FINAL', 'OFF')
              AND sg.stats_filled_at IS NULL
            ORDER BY g.date
        """), {"s": season})]
    if not dates:
        logger.info(f"NHL stats: every finished {season} game is already filled")
        return {"windows": 0, "requests": 0, "fetched": 0, "matched": 0,
                "changed": 0, "unmatched": 0, "failed_windows": 0}
    logger.info(f"NHL stats: {len(dates)} date(s) of season {season} have unfilled rows")
    return fill_windows(windows_for_dates(dates), season)


def fill_rates(date_from: date, date_to: date) -> dict:
    """How much of raw.skater_games in a date range is filled: rows, rows
    with stats_filled_at, and rows with a non-zero value per column."""
    ensure_columns()
    nonzero = ", ".join(f"COUNT(*) FILTER (WHERE sg.{c} <> 0) AS {c}" for c in COLUMNS)
    with engine.connect() as conn:
        row = conn.execute(text(f"""
            SELECT COUNT(*) AS rows, COUNT(sg.stats_filled_at) AS filled, {nonzero}
            FROM raw.skater_games sg JOIN raw.games g ON g.game_id = sg.game_id
            WHERE g.date BETWEEN :lo AND :hi
        """), {"lo": date_from, "hi": date_to}).mappings().one()
    return dict(row)


# ── CLI ───────────────────────────────────────────────────────────

def _season_arg(value: str) -> int:
    v = value.strip()
    if len(v) == 8 and v.isdigit() and int(v[4:]) == int(v[:4]) + 1:
        return int(v)
    raise argparse.ArgumentTypeError(f"{value!r} is not a season like 20252026")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.nhl_stats",
        description="Fill raw.skater_games power-play and penalty-kill ice time, "
                    "power-play goals and assists, and faceoffs won and lost from "
                    "the free NHL stats API (api.nhle.com). Updates rows the box "
                    "score load already stored; safe to re-run.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--season", type=_season_arg,
                       help="one season, such as 20252026 (about 27 requests)")
    group.add_argument("--from", dest="date_from", type=date.fromisoformat,
                       help="first game date, YYYY-MM-DD (with --to)")
    group.add_argument("--missing", action="store_true",
                       help="only dates of the current season with unfilled rows "
                            "(the daily form)")
    parser.add_argument("--to", dest="date_to", type=date.fromisoformat,
                        help="last game date, YYYY-MM-DD (default: today)")
    args = parser.parse_args(argv)
    if args.date_to and not args.date_from:
        parser.error("--to needs --from")

    started = time.monotonic()
    if args.season:
        summary = fill_season(args.season)
    elif args.missing:
        summary = fill_missing()
    else:
        try:
            summary = fill_range(args.date_from, args.date_to or local_today())
        except ValueError as e:
            parser.error(str(e))
    print(f"{summary['requests']} request(s) over {summary['windows']} window(s) in "
          f"{time.monotonic() - started:.0f}s: {summary['fetched']} player-games fetched, "
          f"{summary['matched']} in raw.skater_games, {summary['changed']} written, "
          f"{summary['failed_windows']} window(s) failed")
    return 1 if summary["failed_windows"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
