"""
ingestion/dailyfaceoff_lines.py
Projected line combinations from Daily Faceoff into raw.lineups.

Terms:
  line             three forwards (left wing, centre, right wing) who play
                   their shifts together; F1 is the top line → the one that
                   plays the most minutes
  defence pair     two defencemen who play together (D1 is the top pair)
  power play (PP)  a man advantage after an opponent's penalty; PP1 is the
                   first unit of five → it gets most of the power-play time
  penalty kill     the four skaters who play while their team is a man down
  (PK)
  scratch          a healthy player left out of the game's lineup
  game-time        the team will decide only at warm-ups whether he plays
  decision (GTD)

Why: player props (shots on goal) and team scoring depend on who plays
with whom and who runs PP1. Daily Faceoff publishes each team's current
lines, updated from morning skate and warm-ups, before box scores or the
NHL's own lineup exist. Saving a snapshot every time they change builds
the history a model needs, with no look-ahead: a snapshot holds only what
was published at its time.

Source: https://www.dailyfaceoff.com/teams/<team-slug>/line-combinations
— a Next.js page whose __NEXT_DATA__ JSON holds pageProps.combinations:
the team, updatedAt (when Daily Faceoff last changed the lines) and one
entry per player per unit: categoryIdentifier (ev = even strength, pp =
power play, pk = penalty kill, oi = other information such as injured
reserve), groupIdentifier (f1-f4, d1-d3, g, pp1, pp2, pk1, pk2, ir) and
positionIdentifier (lw, c, rw, ld, rd, g1 = the projected starter, g2 =
the backup, sk1-sk5, ir1...), plus injuryStatus (out, dtd, ir) and
gameTimeDecision.

Being polite (checked 2026-10-04): robots.txt allows /teams/ (it disallows
only /api/ and /cms/); no terms-of-use page was found on the site (its
footer links only a privacy policy). This module:
  - makes at most one request per team per run, for teams with a game
    today only, LINEUPS_PAUSE_SECONDS (2) apart;
  - names itself in the User-Agent header;
  - skips a team fetched less than LINEUPS_MIN_GAP_MINUTES (29) ago, or
    LINEUPS_FAR_GAP_MINUTES (55) ago while its puck drop is more than
    LINEUPS_NEAR_HOURS (3) away, so with a run every 15 minutes a slate of
    16 games costs about 32 requests an hour in the morning and 64 in the
    last hours;
  - stops at once when the site refuses a request (HTTP 429 → too many
    requests, or 403 → forbidden): no other team is asked that run, and
    no team is asked again for LINEUPS_REFUSED_BACKOFF_MINUTES (60), or
    for as long as the site's Retry-After header says if that is longer;
  - stores a new snapshot only when the lines changed (the fetch itself
    is recorded in raw.lineup_fetches).
The site serves pages from a cache: one fetch on 2026-10-04 was an hour
old (Age: 3593), so a change can reach us up to about an hour late, and
fetching a team more often than every half hour mostly gets the same copy.

Storage:
  raw.lineups         one row per (snapshot_ts, team, unit, slot): unit F1-F4,
                      D1-D4, G, PP1, PP2, PK1, PK2, IR (other groups as their
                      own name upper-cased); slot lw/c/rw/ld/rd, g1/g2,
                      sk1-sk5 or ir1...; the player's name as published, his
                      raw.players id (ingestion/espn_injuries.PlayerIndex,
                      NULL when unresolved), his position (C, L, R, D, G;
                      for PP, PK and IR rows the position from his
                      even-strength slot, else NULL), injury status and the
                      game-time-decision flag.
  raw.lineup_fetches  one row per team: when it was last fetched, Daily
                      Faceoff's updatedAt, a hash of the lines, how the
                      fetch went (new, same, broken, failed, refused) and,
                      after a refusal, the time before which it is not
                      fetched again (next_fetch_after). It drives the
                      politeness gap and the "store only on change" rule.

CLI: python -m ingestion.dailyfaceoff_lines [--team BOS ...] [--force]
     (default: the teams with a game today that has not started).
"""
import argparse
import hashlib
import json
import logging
import os
import re
import time
from datetime import date as date_cls, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Dict, Iterable, List, Optional

import requests
from sqlalchemy import text

from config.settings import engine, local_today

logger = logging.getLogger("nhl.ingestion.dailyfaceoff_lines")

PAGE_URL = "https://www.dailyfaceoff.com/teams/{slug}/line-combinations"
USER_AGENT = ("nhl-betting-system/1.0 (personal, non-commercial; one request "
              "per team per run)")
TIMEOUT_S = 30

# NHL abbreviation -> Daily Faceoff's team slug (its sortedTeams list;
# tests check every slug against a saved page)
DF_SLUGS = {
    "ANA": "anaheim-ducks", "BOS": "boston-bruins", "BUF": "buffalo-sabres",
    "CGY": "calgary-flames", "CAR": "carolina-hurricanes",
    "CHI": "chicago-blackhawks", "COL": "colorado-avalanche",
    "CBJ": "columbus-blue-jackets", "DAL": "dallas-stars",
    "DET": "detroit-red-wings", "EDM": "edmonton-oilers",
    "FLA": "florida-panthers", "LAK": "los-angeles-kings",
    "MIN": "minnesota-wild", "MTL": "montreal-canadiens",
    "NSH": "nashville-predators", "NJD": "new-jersey-devils",
    "NYI": "new-york-islanders", "NYR": "new-york-rangers",
    "OTT": "ottawa-senators", "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins", "SJS": "san-jose-sharks",
    "SEA": "seattle-kraken", "STL": "st-louis-blues",
    "TBL": "tampa-bay-lightning", "TOR": "toronto-maple-leafs",
    "UTA": "utah-mammoth", "VAN": "vancouver-canucks",
    "VGK": "vegas-golden-knights", "WSH": "washington-capitals",
    "WPG": "winnipeg-jets",
}

DDL = [
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
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lineups_team ON raw.lineups(team, snapshot_ts)",
    "CREATE INDEX IF NOT EXISTS idx_lineups_player ON raw.lineups(player_id, game_date)",
    """
    CREATE TABLE IF NOT EXISTS raw.lineup_fetches (
        team                VARCHAR(3) PRIMARY KEY,
        fetched_at          TIMESTAMPTZ NOT NULL,
        source_updated_at   TIMESTAMPTZ,
        lines_hash          VARCHAR(64),
        status              VARCHAR(12) NOT NULL,
        next_fetch_after    TIMESTAMPTZ
    )
    """,
]

_table_ready = False


def ensure_table() -> None:
    """Create raw.lineups and raw.lineup_fetches if missing, and add any
    column a table created by an older version lacks (config/migrate;
    once per process)."""
    global _table_ready
    if _table_ready:
        return
    from config.migrate import ensure_schema
    ensure_schema()
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
    _table_ready = True


def _number(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        if value >= 0:
            return value
    except ValueError:
        pass
    logger.error(f"{name}={raw!r} is not a number of 0 or more; using {default:g}")
    return default


# ── Parsing (pure) ─────────────────────────────────────────────────

_POSITIONS = {"lw": "L", "c": "C", "rw": "R", "ld": "D", "rd": "D",
              "g1": "G", "g2": "G"}


def unit_name(group: Optional[str]) -> Optional[str]:
    """Daily Faceoff's group -> our unit: f1 -> F1, d2 -> D2, g -> G, pp1 ->
    PP1, pk2 -> PK2, ir -> IR. Anything else keeps its group name
    upper-cased (at most 6 characters); None when the group is missing."""
    group = (group or "").strip().lower()
    if not group:
        return None
    if group == "g":
        return "G"
    return group.upper()[:6]


def _next_data(html: str) -> dict:
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                  html, re.S)
    if not m:
        raise ValueError("No __NEXT_DATA__ payload: the Daily Faceoff page layout "
                         "changed; the parser needs updating")
    return json.loads(m.group(1))


def _timestamp(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_lineup(html: str) -> dict:
    """One page -> {"team_name", "updated_at", "rows": [...]}. Each row:
    unit, slot, player_name, df_player_id, position, injury_status,
    game_time_decision. PP, PK and IR rows take the position of the
    player's even-strength slot when he has one. Raises ValueError when
    the page has no line combinations."""
    combos = (_next_data(html).get("props", {}).get("pageProps", {})
              .get("combinations"))
    if not isinstance(combos, dict) or not isinstance(combos.get("players"), list):
        raise ValueError("Daily Faceoff page has no line combinations: the page "
                         "layout changed; the parser needs updating")
    rows, ev_position, seen = [], {}, set()
    for p in combos["players"]:
        name = (p.get("name") or "").strip()
        unit = unit_name(p.get("groupIdentifier"))
        slot = (p.get("positionIdentifier") or "").strip().lower()[:6]
        if not name or not unit or not slot or (unit, slot) in seen:
            continue
        seen.add((unit, slot))
        position = _POSITIONS.get(slot) if p.get("categoryIdentifier") == "ev" else None
        if position:
            ev_position[name] = position
        status = (p.get("injuryStatus") or "").strip().lower() or None
        rows.append({"unit": unit, "slot": slot, "player_name": name[:80],
                     "df_player_id": p.get("playerId"), "position": position,
                     "injury_status": status[:12] if status else None,
                     "game_time_decision": bool(p.get("gameTimeDecision"))})
    for r in rows:
        if r["position"] is None:
            r["position"] = ev_position.get(r["player_name"])
    return {"team_name": combos.get("teamName"),
            "updated_at": _timestamp(combos.get("updatedAt")),
            "rows": rows}


def lines_hash(rows: Iterable[dict]) -> str:
    """A fingerprint of the lines (who is where, injury status, game-time
    decisions), independent of row order."""
    key = sorted((r["unit"], r["slot"], r["player_name"], r.get("injury_status") or "",
                  bool(r.get("game_time_decision"))) for r in rows)
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()


def dressed_count(rows: Iterable[dict]) -> int:
    """Forwards and defencemen in the even-strength lines (18 on a full
    lineup). A page with very few is treated as broken, not as news."""
    return sum(1 for r in rows if re.fullmatch(r"[FD]\d", r["unit"]))


MIN_DRESSED = 12


# ── Politeness ─────────────────────────────────────────────────────

class SiteRefused(Exception):
    """Daily Faceoff answered 429 (too many requests) or 403 (forbidden):
    stop asking. retry_after: seconds from its Retry-After header, if any."""

    def __init__(self, status: int, retry_after: Optional[float] = None):
        self.status, self.retry_after = status, retry_after
        super().__init__(f"Daily Faceoff refused the request (HTTP {status})"
                         + (f", Retry-After {retry_after:.0f} s" if retry_after else ""))


def retry_after_seconds(value, now: Optional[datetime] = None) -> Optional[float]:
    """A Retry-After header (seconds, or an HTTP date) -> seconds from now;
    None when missing or unreadable."""
    if value is None or not str(value).strip():
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - (now or datetime.now(timezone.utc))).total_seconds())


def refused_until(now: datetime, retry_after: Optional[float],
                  backoff_minutes: float = None) -> datetime:
    """When the next request may go out after a refusal: the longer of
    LINEUPS_REFUSED_BACKOFF_MINUTES (60) and the site's Retry-After."""
    backoff = (_number("LINEUPS_REFUSED_BACKOFF_MINUTES", 60)
               if backoff_minutes is None else backoff_minutes)
    return now + max(timedelta(minutes=backoff), timedelta(seconds=retry_after or 0))


def fetch_gap(start_utc: Optional[datetime], now: datetime,
              near_gap: float = None, far_gap: float = None,
              near_hours: float = None) -> timedelta:
    """How long since the team's last fetch before it may be fetched again:
    LINEUPS_FAR_GAP_MINUTES while its puck drop is more than
    LINEUPS_NEAR_HOURS away, else LINEUPS_MIN_GAP_MINUTES."""
    near_gap = _number("LINEUPS_MIN_GAP_MINUTES", 29) if near_gap is None else near_gap
    far_gap = _number("LINEUPS_FAR_GAP_MINUTES", 55) if far_gap is None else far_gap
    near_hours = (_number("LINEUPS_NEAR_HOURS", 3) if near_hours is None
                  else near_hours)
    if start_utc is not None and start_utc - now > timedelta(hours=near_hours):
        return timedelta(minutes=far_gap)
    return timedelta(minutes=near_gap)


def teams_to_fetch(starts: Dict[str, Optional[datetime]],
                   last_fetch: Dict[str, datetime], now: datetime,
                   force: bool = False, not_before: Dict[str, datetime] = None,
                   **gaps) -> List[str]:
    """Pure: the teams (from {team: its puck drop}) due a fetch now.
    not_before: {team: time} after a refusal; force ignores it too."""
    due = []
    for team in sorted(starts):
        if team not in DF_SLUGS:
            logger.warning(f"No Daily Faceoff page known for {team!r}; extend DF_SLUGS")
            continue
        hold = (not_before or {}).get(team)
        if not force and hold is not None and now < hold:
            continue
        last = last_fetch.get(team)
        if force or last is None or now - last >= fetch_gap(starts[team], now, **gaps):
            due.append(team)
    return due


# ── Fetch and store ────────────────────────────────────────────────

def fetch_page(team: str) -> str:
    resp = requests.get(PAGE_URL.format(slug=DF_SLUGS[team]),
                        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    if resp.status_code in (403, 429):
        raise SiteRefused(resp.status_code,
                          retry_after_seconds(resp.headers.get("Retry-After")))
    resp.raise_for_status()
    return resp.text


def todays_teams(on_date: Optional[date_cls] = None) -> Dict[str, Optional[datetime]]:
    """{team: puck drop (aware UTC)} for every team with a game on
    `on_date` (default today, local) that has not started and is not
    postponed, suspended or cancelled."""
    on_date = on_date or local_today()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT home_team, away_team, start_time_utc FROM raw.games
            WHERE date = :d AND game_type IN (2, 3)
              AND game_state NOT IN ('FINAL', 'OFF', 'LIVE', 'CRIT')
              AND COALESCE(schedule_state, 'OK') NOT IN ('PPD', 'SUSP', 'CNCL')
              AND (start_time_utc IS NULL OR start_time_utc > NOW())
        """), {"d": on_date}).fetchall()
    out = {}
    for home, away, start in rows:
        for team in (home, away):
            out[team] = start
    return out


def _last_fetches() -> Dict[str, dict]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT team, fetched_at, lines_hash, next_fetch_after
            FROM raw.lineup_fetches
        """)).fetchall()
    return {t: {"fetched_at": f, "lines_hash": h, "next_fetch_after": n}
            for t, f, h, n in rows}


def _record_fetch(conn, team: str, now: datetime, updated_at, digest, status: str,
                  not_before: Optional[datetime] = None):
    conn.execute(text("""
        INSERT INTO raw.lineup_fetches (team, fetched_at, source_updated_at,
                                        lines_hash, status, next_fetch_after)
        VALUES (:t, :now, :u, :h, :s, :nb)
        ON CONFLICT (team) DO UPDATE SET
            fetched_at = EXCLUDED.fetched_at,
            source_updated_at = COALESCE(EXCLUDED.source_updated_at,
                                         raw.lineup_fetches.source_updated_at),
            lines_hash = COALESCE(EXCLUDED.lines_hash, raw.lineup_fetches.lines_hash),
            status = EXCLUDED.status,
            next_fetch_after = EXCLUDED.next_fetch_after
    """), {"t": team, "now": now, "u": updated_at, "h": digest, "s": status,
           "nb": not_before})


def _hold(conn, team: str, now: datetime, until: datetime):
    """After a refusal: no request for `team` before `until`. A team with
    no fetch yet gets a 'refused' row; an existing row keeps its last
    fetch and only gets the hold."""
    conn.execute(text("""
        INSERT INTO raw.lineup_fetches (team, fetched_at, status, next_fetch_after)
        VALUES (:t, :now, 'refused', :nb)
        ON CONFLICT (team) DO UPDATE SET next_fetch_after = EXCLUDED.next_fetch_after
    """), {"t": team, "now": now, "nb": until})


def write_snapshot(team: str, game_date: date_cls, parsed: dict, now: datetime,
                   previous_hash: Optional[str]) -> str:
    """Store the parsed lines as a new snapshot when they differ from the
    last one, and record the fetch. Returns new, same or broken."""
    rows = parsed["rows"]
    digest = lines_hash(rows)
    status = ("broken" if dressed_count(rows) < MIN_DRESSED
              else "same" if digest == previous_hash else "new")
    with engine.begin() as conn:
        if status == "new":
            conn.execute(text("""
                INSERT INTO raw.lineups
                    (snapshot_ts, team, game_date, unit, slot, player_name,
                     player_id, df_player_id, position, injury_status,
                     game_time_decision, source_updated_at)
                VALUES (:snapshot_ts, :team, :game_date, :unit, :slot,
                        :player_name, :player_id, :df_player_id, :position,
                        :injury_status, :game_time_decision, :source_updated_at)
            """), [{**{k: r.get(k) for k in (
                "unit", "slot", "player_name", "player_id", "df_player_id",
                "position", "injury_status", "game_time_decision")},
                "snapshot_ts": now, "team": team, "game_date": game_date,
                "source_updated_at": parsed["updated_at"]} for r in rows])
        _record_fetch(conn, team, now, parsed["updated_at"],
                      digest if status != "broken" else None, status)
    return status


def ingest_lineups(teams: Optional[Dict[str, Optional[datetime]]] = None,
                   game_date: Optional[date_cls] = None, force: bool = False,
                   fetch: Optional[Callable[[str], str]] = None,
                   pause_s: Optional[float] = None, index=None) -> Dict[str, str]:
    """Fetch, parse and store the lines of each due team. teams: {team:
    puck drop} (default: today's teams whose game has not started).
    Returns {team: new | same | broken | failed | skipped}; a team that
    fails is logged and the others carry on, except when the site refuses
    a request (429 or 403): then no other team is asked (they are failed),
    and every team waits until refused_until()."""
    ensure_table()
    fetch = fetch or fetch_page
    game_date = game_date or local_today()
    teams = todays_teams(game_date) if teams is None else teams
    if not teams:
        logger.info("Lineups: no team has a game left today; nothing fetched")
        return {}
    if pause_s is None:
        pause_s = _number("LINEUPS_PAUSE_SECONDS", 2.0)
    now = datetime.now(timezone.utc)
    last = _last_fetches()
    due = teams_to_fetch(teams, {t: v["fetched_at"] for t, v in last.items()},
                         now, force=force,
                         not_before={t: v.get("next_fetch_after") for t, v in last.items()})
    held = [t for t, v in last.items() if t in teams and not force
            and v.get("next_fetch_after") and now < v["next_fetch_after"]]
    if held:
        logger.info(f"Lineups: Daily Faceoff refused a request earlier; no request "
                    f"before {max(last[t]['next_fetch_after'] for t in held):%H:%M} UTC "
                    f"({len(held)} team(s) waiting)")
    result = {t: "skipped" for t in teams}
    if due and index is None:
        from ingestion.espn_injuries import load_player_index
        index = load_player_index()
    for i, team in enumerate(due):
        if i and pause_s:
            time.sleep(pause_s)
        fetched_at = datetime.now(timezone.utc)
        try:
            parsed = parse_lineup(fetch(team))
            for r in parsed["rows"]:
                r["player_id"], _ = index.match(r["player_name"], team, r["position"])
            result[team] = write_snapshot(team, game_date, parsed, fetched_at,
                                          (last.get(team) or {}).get("lines_hash"))
            if result[team] == "broken":
                logger.warning(f"Lineups {team}: only {dressed_count(parsed['rows'])} "
                               f"forwards and defencemen on the page; not stored")
        except SiteRefused as e:
            until = refused_until(fetched_at, e.retry_after)
            rest = due[i + 1:]
            for t in [team] + rest:
                result[t] = "failed"
            logger.error(f"Lineups {team}: {e}. Stopping: {len(rest)} other team(s) not "
                         f"asked, and no request before {until:%H:%M} UTC")
            try:
                with engine.begin() as conn:
                    _record_fetch(conn, team, fetched_at, None, None, "refused", until)
                    for t in teams:
                        if t != team and t in DF_SLUGS:
                            _hold(conn, t, fetched_at, until)
            except Exception as db_error:
                logger.error(f"Lineups: could not record the refusal: {db_error}")
            break
        except Exception as e:
            result[team] = "failed"
            logger.error(f"Lineups {team}: fetch or parse failed (non-fatal): {e}")
            try:
                with engine.begin() as conn:
                    _record_fetch(conn, team, fetched_at, None, None, "failed")
            except Exception:
                pass
    counts: Dict[str, int] = {}
    for status in result.values():
        counts[status] = counts.get(status, 0) + 1
    logger.info(f"Lineups {game_date}: " + ", ".join(
        f"{v} {k}" for k, v in sorted(counts.items())) + f" ({len(due)} request(s))")
    return result


def latest_lineups(teams: Iterable[str]) -> Dict[str, List[dict]]:
    """{team: rows of its latest stored snapshot} for the given teams."""
    teams = list(teams)
    if not teams:
        return {}
    ensure_table()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT l.team, l.unit, l.slot, l.player_name, l.player_id, l.position,
                   l.injury_status, l.game_time_decision, l.snapshot_ts
            FROM raw.lineups l
            JOIN (SELECT team, MAX(snapshot_ts) AS ts FROM raw.lineups
                  WHERE team = ANY(:teams) GROUP BY team) m
              ON m.team = l.team AND m.ts = l.snapshot_ts
        """), {"teams": teams}).mappings().fetchall()
    out: Dict[str, List[dict]] = {}
    for r in rows:
        out.setdefault(r["team"], []).append(dict(r))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.dailyfaceoff_lines",
        description="Save Daily Faceoff's line combinations (lines, defence pairs, "
                    "power-play and penalty-kill units, goalies, injured reserve) "
                    "into raw.lineups. One request per team, only for teams with "
                    "a game today unless --team is given.")
    parser.add_argument("--team", action="append", default=None, metavar="ABBR",
                        help="an NHL team abbreviation such as BOS (repeatable); "
                             "default: today's teams whose game has not started")
    parser.add_argument("--force", action="store_true",
                        help="ignore the minimum gap since each team's last fetch")
    args = parser.parse_args(argv)
    teams = None
    if args.team:
        bad = [t for t in args.team if t.upper() not in DF_SLUGS]
        if bad:
            parser.error(f"unknown team(s): {', '.join(bad)}")
        teams = {t.upper(): None for t in args.team}
    ingest_lineups(teams, force=args.force)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
