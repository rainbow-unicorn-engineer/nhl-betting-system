"""
ingestion/espn_injuries.py
Daily snapshot of ESPN's NHL injury list into raw.injuries.

Why: the NHL API has no injury endpoint, and ESPN's list is current-state
only (it keeps no history). Saving one copy a day builds the history that
later features and the starter logic need (e.g. ruling out an injured
goalie), with no look-ahead: a snapshot only ever holds what was known on
its date.

Source: https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries
(free, no key). One request per run. Each entry has a status (Day-To-Day,
Out, Injured Reserve, Suspension), the injury type ("Upper Body"), an
expected return date, a short and a long news comment, and the athlete
(ESPN id, name, position, team).

Storage: one row per (snapshot_date, espn_athlete_id). snapshot_date is the
user's local date (config.settings.local_today). Re-running on the same day
replaces that day's snapshot: rows are upserted, and players no longer on
ESPN's list are removed from that day only. An empty or failed download
writes nothing, so a glitch can never wipe a day's list. Other days are
never touched.

Players: ESPN publishes full names ("Jeremy Swayman"); raw.players stores
abbreviated ones ("J. Swayman"), the trap PROJECT_CONTEXT §9 records for
Daily Faceoff. PlayerIndex matches the accent-stripped exact name first,
then first initial + surname, then the surname alone among players last
seen on the same team, and breaks ties by team and then by position group
(goalie, defence, forward). It never matches a goalie to a skater. An
unresolved player keeps player_id NULL and is logged, never dropped.
PlayerIndex is shared with ingestion/espn_props.py.

Teams: ESPN uses its own abbreviations for four teams (LA, NJ, SJ, TB);
espn_team_abbrev() maps names through the shared Odds API table first,
then fixes those abbreviations.

The table is created on first use (DDL, applied by ensure_table());
db/schema.sql should hold it too.
"""
import argparse
import logging
import re
from collections import Counter
from datetime import date as date_cls, datetime, timezone
from typing import Iterable, List, Optional

import requests
from sqlalchemy import text

from config.settings import engine, local_today
from ingestion.dailyfaceoff import _initial_key, _normalize
from ingestion.espn_odds import _espn_name_to_abbrev
from ingestion.odds_api import _TEAM_NAME_TO_ABBREV

logger = logging.getLogger("nhl.ingestion.espn_injuries")

INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
TIMEOUT_S = 30

DDL = [
    """
    CREATE TABLE IF NOT EXISTS raw.injuries (
        snapshot_date    DATE NOT NULL,            -- local date the list was saved
        espn_athlete_id  BIGINT NOT NULL,
        player_name      VARCHAR(80) NOT NULL,     -- as ESPN publishes it
        team_abbrev      VARCHAR(3),               -- NHL abbreviation (LAK, not LA)
        position         VARCHAR(2),               -- C, L, R, D, G
        status           VARCHAR(30),              -- Day-To-Day, Out, Injured Reserve, Suspension
        injury_type      VARCHAR(60),              -- e.g. Upper Body, Hip
        injury_detail    VARCHAR(60),              -- e.g. Surgery, Strain; often NULL
        return_date      DATE,                     -- ESPN's expected return
        short_comment    TEXT,
        long_comment     TEXT,
        reported_at      TIMESTAMPTZ,              -- ESPN's date on the entry
        player_id        INTEGER,                  -- raw.players id; NULL when unresolved
        fetched_at       TIMESTAMP NOT NULL DEFAULT now(),
        PRIMARY KEY (snapshot_date, espn_athlete_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_injuries_player ON raw.injuries(player_id, snapshot_date)",
]

COLUMNS = ("espn_athlete_id", "player_name", "team_abbrev", "position", "status",
           "injury_type", "injury_detail", "return_date", "short_comment",
           "long_comment", "reported_at", "player_id")


def ensure_table() -> None:
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))


# ── Teams and positions (pure) ─────────────────────────────────────

# ESPN abbreviations that differ from the NHL's
_ESPN_ABBREV_FIX = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "UTAH": "UTA"}
_NHL_ABBREVS = frozenset(_TEAM_NAME_TO_ABBREV.values())

# ESPN position abbreviations -> raw.players.position
POSITION_MAP = {"C": "C", "LW": "L", "RW": "R", "L": "L", "R": "R", "W": "F",
                "F": "F", "D": "D", "G": "G"}


def espn_team_abbrev(display_name: Optional[str] = None,
                     abbreviation: Optional[str] = None) -> Optional[str]:
    """NHL abbreviation for an ESPN team, by full name first, then by
    ESPN's abbreviation (LA -> LAK, NJ -> NJD, SJ -> SJS, TB -> TBL)."""
    by_name = _espn_name_to_abbrev(display_name or "")
    if by_name:
        return by_name
    ab = (abbreviation or "").strip().upper()
    ab = _ESPN_ABBREV_FIX.get(ab, ab)
    return ab if ab in _NHL_ABBREVS else None


def position_group(position: Optional[str]) -> Optional[str]:
    """G, D or F (C, L, R, F), or None when unknown."""
    if not position:
        return None
    p = POSITION_MAP.get(position.strip().upper(), position.strip().upper())
    if p == "G":
        return "G"
    if p == "D":
        return "D"
    return "F" if p in ("C", "L", "R", "F") else None


# ── Player resolution ──────────────────────────────────────────────

class PlayerIndex:
    """Name -> raw.players id lookup. Pure once built.

    players: (player_id, full_name, position) rows; last_team: player_id ->
    the team he most recently played for."""

    def __init__(self, players: Iterable[tuple], last_team: Optional[dict] = None):
        self.by_name: dict = {}
        self.by_key: dict = {}
        self.by_surname: dict = {}
        self.group: dict = {}
        self.last_team = dict(last_team or {})
        for pid, full_name, pos in players:
            if not full_name:
                continue
            self.group[pid] = position_group(pos)
            key = _initial_key(full_name)
            self.by_name.setdefault(_normalize(full_name), []).append(pid)
            self.by_key.setdefault(key, []).append(pid)
            self.by_surname.setdefault(key[1], []).append(pid)

    def _pick(self, cands: list, team: Optional[str], group: Optional[str]):
        """(player_id or None, candidates left). Never crosses goalie and
        skater; ties are broken by team, then by position group."""
        cands = list(dict.fromkeys(cands))
        if group:
            cands = [p for p in cands
                     if self.group.get(p) is None or (self.group[p] == "G") == (group == "G")]
        if len(cands) == 1:
            return cands[0], cands
        if team:
            on_team = [p for p in cands if self.last_team.get(p) == team]
            if len(on_team) == 1:
                return on_team[0], cands
            if on_team:
                cands = on_team
        if group:
            same = [p for p in cands if self.group.get(p) == group]
            if len(same) == 1:
                return same[0], cands
        return None, cands

    def match(self, name: Optional[str], team: Optional[str] = None,
              position: Optional[str] = None) -> tuple:
        """(player_id or None, how): how is exact, initial, surname,
        ambiguous or unknown."""
        if not name or not name.strip():
            return None, "unknown"
        group = position_group(position)
        key = _initial_key(name)
        for how, cands in (("exact", self.by_name.get(_normalize(name))),
                           ("initial", self.by_key.get(key))):
            if not cands:
                continue
            pid, left = self._pick(cands, team, group)
            if pid is not None:
                return pid, how
            if left:
                return None, "ambiguous"
        if team:
            same_team = [p for p in self.by_surname.get(key[1], [])
                         if self.last_team.get(p) == team]
            if same_team:
                pid, left = self._pick(same_team, team, group)
                if pid is not None:
                    return pid, "surname"
                if left:
                    return None, "ambiguous"
        return None, "unknown"


def load_player_index() -> PlayerIndex:
    """Every raw.players row, with the team each last played for."""
    with engine.connect() as conn:
        players = conn.execute(text(
            "SELECT player_id, full_name, position FROM raw.players")).fetchall()
        recent = conn.execute(text("""
            SELECT DISTINCT ON (x.player_id) x.player_id, x.team
            FROM (SELECT player_id, team, game_id FROM raw.skater_games
                  UNION ALL
                  SELECT player_id, team, game_id FROM raw.goalie_games) x
            JOIN raw.games g USING (game_id)
            ORDER BY x.player_id, g.date DESC, g.game_id DESC
        """)).fetchall()
    return PlayerIndex(players, dict(recent))


def resolve_player_ids(rows: List[dict], index: Optional[PlayerIndex] = None,
                       name_key: str = "player_name", team_key: str = "team_abbrev",
                       position_key: str = "position") -> List[dict]:
    """Set row["player_id"] (None when unresolved) and log one summary
    line for the unresolved ones."""
    if not rows:
        return rows
    index = index or load_player_index()
    missed = []
    for r in rows:
        pid, how = index.match(r.get(name_key), r.get(team_key), r.get(position_key))
        r["player_id"] = pid
        if pid is None:
            missed.append(f"{r.get(name_key)} ({r.get(team_key)}, {how})")
    if missed:
        logger.warning(f"{len(missed)} of {len(rows)} ESPN players not matched to "
                       f"raw.players (stored with player_id NULL): "
                       + "; ".join(missed[:10]) + (" ..." if len(missed) > 10 else ""))
    return rows


# ── Parsing (pure) ─────────────────────────────────────────────────

_ID_PATTERNS = (re.compile(r"/athletes/(\d+)"), re.compile(r"/id/(\d+)"),
                re.compile(r"/players/full/(\d+)\."), re.compile(r"~a:(\d+)"))


def athlete_id(athlete: dict) -> Optional[int]:
    """ESPN's athlete id. The injuries feed gives no `id` field on the
    athlete (the entry's own `id` is the injury note's), so it is read from
    the athlete's links, news note, headshot or uid."""
    if not isinstance(athlete, dict):
        return None
    raw = athlete.get("id")
    if raw is not None and str(raw).isdigit():
        return int(raw)
    candidates = [athlete.get("uid"), athlete.get("$ref"),
                  (athlete.get("headshot") or {}).get("href")]
    candidates += [(link or {}).get("href") for link in athlete.get("links") or []]
    candidates += [((n or {}).get("injury") or {}).get("$ref")
                   for n in (athlete.get("notes") or {}).get("items") or []]
    for value in candidates:
        if not value:
            continue
        for pattern in _ID_PATTERNS:
            m = pattern.search(str(value))
            if m:
                return int(m.group(1))
    return None


def _date(value) -> Optional[date_cls]:
    if not value:
        return None
    try:
        return date_cls.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _timestamp(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _clip(value, n: int) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s[:n] if s else None


def parse_injuries(payload: dict) -> List[dict]:
    """One row per injured player (player_id not yet resolved). Entries
    without an athlete id are logged and skipped; a player listed twice
    keeps his most recently reported entry."""
    teams = payload.get("injuries")
    if not isinstance(teams, list):
        raise ValueError("ESPN injuries payload has no 'injuries' list — the "
                         "feed changed; parser needs updating")
    by_athlete: dict = {}
    skipped = 0
    for team in teams:
        for entry in (team or {}).get("injuries") or []:
            athlete = entry.get("athlete") or {}
            aid = athlete_id(athlete)
            name = athlete.get("displayName") or " ".join(
                p for p in (athlete.get("firstName"), athlete.get("lastName")) if p)
            if aid is None or not name:
                skipped += 1
                continue
            team_info = athlete.get("team") or {}
            details = entry.get("details") or {}
            row = {
                "espn_athlete_id": aid,
                "player_name": _clip(name, 80),
                "team_abbrev": espn_team_abbrev(team_info.get("displayName"),
                                                team_info.get("abbreviation"))
                or espn_team_abbrev(team.get("displayName"), team.get("abbreviation")),
                "position": POSITION_MAP.get(
                    str((athlete.get("position") or {}).get("abbreviation") or "").upper()),
                "status": _clip(entry.get("status"), 30),
                "injury_type": _clip(details.get("type"), 60),
                "injury_detail": _clip(details.get("detail"), 60),
                "return_date": _date(details.get("returnDate")),
                "short_comment": _clip(entry.get("shortComment"), 10_000),
                "long_comment": _clip(entry.get("longComment"), 10_000),
                "reported_at": _timestamp(entry.get("date")),
            }
            old = by_athlete.get(aid)
            if old is None or (row["reported_at"] or datetime.min.replace(tzinfo=timezone.utc)) \
                    > (old["reported_at"] or datetime.min.replace(tzinfo=timezone.utc)):
                by_athlete[aid] = row
    if skipped:
        logger.warning(f"ESPN injuries: {skipped} entries had no athlete id or name "
                       f"and were skipped")
    return list(by_athlete.values())


# ── Fetch and store ────────────────────────────────────────────────

def fetch_injuries() -> dict:
    resp = requests.get(INJURIES_URL, timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


_UPSERT_SQL = text(f"""
    INSERT INTO raw.injuries (snapshot_date, {', '.join(COLUMNS)})
    VALUES (:snapshot_date, {', '.join(':' + c for c in COLUMNS)})
    ON CONFLICT (snapshot_date, espn_athlete_id) DO UPDATE SET
        {', '.join(f'{c} = EXCLUDED.{c}' for c in COLUMNS if c != 'espn_athlete_id')},
        fetched_at = now()
""")


def write_injuries(rows: List[dict], snapshot_date: date_cls) -> int:
    """Replace one day's snapshot with `rows` (upsert, then remove players
    no longer listed that day), in one transaction. An empty list writes
    nothing and leaves the day as it was."""
    if not rows:
        return 0
    ensure_table()
    params = [{**{c: r.get(c) for c in COLUMNS}, "snapshot_date": snapshot_date}
              for r in rows]
    with engine.begin() as conn:
        conn.execute(_UPSERT_SQL, params)
        conn.execute(text("""
            DELETE FROM raw.injuries
            WHERE snapshot_date = :d AND NOT (espn_athlete_id = ANY(:ids))
        """), {"d": snapshot_date, "ids": [int(r["espn_athlete_id"]) for r in rows]})
    return len(rows)


def ingest_injuries(snapshot_date: Optional[date_cls] = None) -> int:
    """Fetch, parse, resolve and store today's (or `snapshot_date`'s)
    snapshot. Returns rows written."""
    snapshot_date = snapshot_date or local_today()
    rows = parse_injuries(fetch_injuries())
    if not rows:
        logger.warning("ESPN injuries: the list came back empty; nothing written "
                       f"(the {snapshot_date} snapshot, if any, is unchanged)")
        return 0
    resolve_player_ids(rows)
    n = write_injuries(rows, snapshot_date)
    resolved = sum(1 for r in rows if r.get("player_id"))
    statuses = Counter(r.get("status") for r in rows)
    logger.info(f"ESPN injuries {snapshot_date}: {n} players "
                f"({resolved} matched to raw.players); "
                + ", ".join(f"{k} {v}" for k, v in statuses.most_common()))
    return n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.espn_injuries",
        description="Save today's ESPN NHL injury list into raw.injuries (free, "
                    "no key, one request). Re-running the same day replaces "
                    "that day's snapshot.")
    parser.add_argument("--date", type=date_cls.fromisoformat, default=None,
                        metavar="YYYY-MM-DD",
                        help="the snapshot date to store the list under (default: "
                             "today's local date). ESPN serves only the current "
                             "list, so only today, yesterday or tomorrow is "
                             "accepted: an older date would plant today's news "
                             "in the past")
    args = parser.parse_args(argv)
    if args.date is not None and abs((args.date - local_today()).days) > 1:
        parser.error(f"--date {args.date} is more than a day from today "
                     f"({local_today()}); ESPN has no injury history")
    if args.date is not None and args.date != local_today():
        logger.warning(f"Storing the current ESPN list under {args.date} "
                       f"(today is {local_today()})")
    return ingest_injuries(args.date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
