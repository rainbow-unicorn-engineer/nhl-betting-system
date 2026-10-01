"""
ingestion/espn_props.py
Past player-prop prices from ESPN's free core API into raw.prop_odds_hist,
for a first props backtest at no cost.

Terms:
- Player prop: a bet on one player's own numbers, such as "over 2.5 shots
  on goal".
- Line: the number an over/under is set at (the 2.5 above).
- Price: American odds → -120 risks 120 to win 100, +110 risks 100 to win 110.
- Opening price: the book's first price; the "current" price is its last
  update, which for a finished game is at or after puck drop.

Source (free, no key, checked 2026-09-29):
  GET http://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/events/
      {event}/competitions/{event}/odds                       -> the books
  GET .../odds/{provider}/propBets?limit=1000                  -> the props
  GET site.api.espn.com/.../summary?event={event}              -> the names
The props list gives each player as an ESPN athlete id only; the game's
boxscore in the summary maps ids to names, teams and positions.

What ESPN still serves for 2025-26 (checked 2026-09-29 on every 7th game
date, all games, plus every game on 2026-01-15, 2026-02-05 and 2026-04-16;
then 22 games loaded into the clone):
- October to November 2025: ESPN BET (provider 58) on every game sampled
  (39 of 39), plus "ESPN Bet - Live Odds" (59), which is in-play and never
  stored. ESPN BET's last update came after puck drop on 99% of the lines
  loaded, so in practice only its OPENING prices are pre-game.
- December 2025 to mid-April 2026: DraftKings (100) on a few dates only:
  every game on 2026-01-15 and 2026-04-16, none on 2026-02-05 or the 18
  other dates sampled (0 of 97 games). Coverage is all-or-nothing per date.
  The 2026-01-15 lines were last updated 2.5 to 4.9 hours before puck
  drop; the 2026-04-16 ones after it.
- Playoffs (April to June 2026): DraftKings on every game sampled (15 of
  15). Of 8 loaded, 5 were last updated before puck drop (the latest update
  0.1 to 1.9 hours before it) and 3 after it (opening prices only).

Over or under? The two sides of a line come as two separate entries.
- ESPN BET labels each entry (`current.over` / `current.under`); verified
  on 216 goals-0.5 pairs: the labelled over is the long-odds side in 215.
  Its list order is random (over first in 124, second in 92).
- DraftKings entries carry no label. They come in adjacent pairs and the
  FIRST is the over: checked against 605 DraftKings "N+" milestone prices
  (an N+ milestone is the over at N-0.5; the first entry is the closer one
  in 570, the second in 16, 19 ties) and against outcomes on 642 resolved
  pairs (Brier score 0.227 with first = over, 0.307 with first = under).
  The stored rows pass the same outcome check for both books (Brier as
  stored vs swapped: DraftKings 0.228 vs 0.303, ESPN BET openings 0.183 vs
  0.418). Brier score: the mean squared error of a probability, lower is
  better.
- Unlabelled pairs from any other book are skipped (counted in the log):
  their order has not been verified.

Markets kept, named as The Odds API names them (so ingestion/props_odds.py
rows line up): player_points, player_assists, player_goals,
player_shots_on_goal, player_power_play_points, player_blocked_shots,
player_total_saves, player_hits (ESPN BET only), player_goal_scorer_anytime
(stored as the over 0.5 goals, no under), and DraftKings' one-sided "N+"
milestones as player_*_alternate at line N-0.5 (over only). Team, period,
first/last goalscorer and ESPN BET's unnamed "Hockey Player Prop" entries
are skipped.

In-play rule (as everywhere in this repo, prices after puck drop are never
stored): when an entry's last update came at or after puck drop, its
current price and current line are dropped, and the row keeps the opening
price at the opening line. last_updated is still stored, so a row with
last_updated >= event_start is recognisably opening-price only.

Quirks handled:
- ESPN keeps ONE entry per player, market and side. When a book showed
  several lines (DraftKings' 2+, 3+, 4+ shot milestones), only the last
  survives, and its "open" can be the open of another line ("3+" current,
  "2+" open). The opening price is kept only when the open line equals the
  line stored.
- A pair whose prices no two-way market could quote is stored as NULL,
  NULL (ingestion/espn_odds.sane_pair).

Players are resolved to raw.players ids first among the players who
actually played that game (the NHL boxscore, by team), then across all of
raw.players (ingestion/espn_injuries.PlayerIndex). Unresolved players are
stored with player_id NULL and counted in the log. In a 22-game sample
3,309 of 3,326 rows resolved; the other 17 belong to one athlete id that
is in no boxscore and that ESPN's athlete lookup does not know, so they
have no name either. A second, 30-game sample (2026-10-01: 20 games evenly
spread over the season plus all of 2026-01-15) resolved all 2,526 rows.

Resumable and polite: raw.prop_odds_fetches records every game looked at
(with the number of rows it gave, often 0), so a re-run skips them;
--refresh re-fetches them. A failed download is not recorded and is
retried on the next run. REQUEST_PAUSE_S between requests.

Tables are created on first use (DDL, applied by ensure_tables());
db/schema.sql should hold them too.
"""
import argparse
import logging
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import List, Optional

import requests
from sqlalchemy import text

from config.settings import engine
from ingestion.espn_injuries import (POSITION_MAP, PlayerIndex, espn_team_abbrev,
                                     load_player_index)
from ingestion.espn_odds import (fetch_scoreboard, match_events, parse_american,
                                 parse_line, sane_pair)

logger = logging.getLogger("nhl.ingestion.espn_props")

CORE_EVENT_URL = ("http://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/"
                  "events/{event}/competitions/{event}")
ATHLETE_URL = "http://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/athletes/{athlete}"
SUMMARY_URL = "http://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary"
PARAMS = {"lang": "en", "region": "us"}
PAGE_LIMIT = 1000
TIMEOUT_S = 30
REQUEST_PAUSE_S = 0.5   # an unauthenticated public API: one request at a time

# ESPN prop type name -> (market key, kind). kind: "ou" = over/under pair,
# "yes" = one-sided yes bet (stored as the over 0.5), "milestone" = one-sided
# "N+" (stored as the over N-0.5).
MARKETS = {
    "Total Points": ("player_points", "ou"),
    "Total Assists": ("player_assists", "ou"),
    "Total Goals": ("player_goals", "ou"),
    "Total Shots on Goal": ("player_shots_on_goal", "ou"),
    "Total Power Play Points": ("player_power_play_points", "ou"),
    "Total Blocked Shots": ("player_blocked_shots", "ou"),
    "Total Saves": ("player_total_saves", "ou"),
    "Total Hits": ("player_hits", "ou"),
    "Anytime Goalscorer": ("player_goal_scorer_anytime", "yes"),
    "Points Milestones": ("player_points_alternate", "milestone"),
    "Assists Milestones": ("player_assists_alternate", "milestone"),
    "Goals Milestones": ("player_goals_alternate", "milestone"),
    "Shots on Goal Milestones": ("player_shots_on_goal_alternate", "milestone"),
    "Blocked Shots Milestones": ("player_blocked_shots_alternate", "milestone"),
}

# Books whose unlabelled pairs list the over first (verified, see above)
OVER_FIRST_BOOKS = frozenset({"draftkings"})

DDL = [
    """
    CREATE TABLE IF NOT EXISTS raw.prop_odds_hist (
        game_id          BIGINT NOT NULL REFERENCES raw.games(game_id),
        espn_event_id    VARCHAR(12) NOT NULL,
        book             VARCHAR(40) NOT NULL,     -- ESPN's provider name: DraftKings, ESPN BET
        market           VARCHAR(40) NOT NULL,     -- Odds API style key, e.g. player_shots_on_goal
        player_name      VARCHAR(80),              -- as ESPN publishes it
        espn_athlete_id  BIGINT NOT NULL,
        player_id        INTEGER,                  -- raw.players id; NULL when unresolved
        line             NUMERIC(4,1) NOT NULL,
        over_price       INTEGER,                  -- last pre-game price; NULL if updated in play
        under_price      INTEGER,
        over_price_open  INTEGER,                  -- opening price at this line
        under_price_open INTEGER,
        last_updated     TIMESTAMPTZ,              -- ESPN's last update of the line
        event_start      TIMESTAMPTZ,              -- scheduled puck drop (ESPN)
        fetched_at       TIMESTAMP NOT NULL DEFAULT now(),
        UNIQUE (game_id, book, market, espn_athlete_id, line)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_prop_odds_hist_player ON raw.prop_odds_hist(player_id, market)",
    """
    CREATE TABLE IF NOT EXISTS raw.prop_odds_fetches (
        game_id        BIGINT PRIMARY KEY REFERENCES raw.games(game_id),
        espn_event_id  VARCHAR(12),                -- NULL: not on ESPN's scoreboard
        books          VARCHAR(120),               -- books that had props, comma-separated
        n_rows         INTEGER NOT NULL DEFAULT 0,
        fetched_at     TIMESTAMP NOT NULL DEFAULT now()
    )
    """,
]

ROW_COLUMNS = ("game_id", "espn_event_id", "book", "market", "player_name",
               "espn_athlete_id", "player_id", "line", "over_price", "under_price",
               "over_price_open", "under_price_open", "last_updated", "event_start")


def ensure_tables() -> None:
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))


# ── Parsing (pure) ─────────────────────────────────────────────────

_ATHLETE_RE = re.compile(r"/athletes/(\d+)")


def _athlete_ref_id(item: dict) -> Optional[int]:
    m = _ATHLETE_RE.search(str((item.get("athlete") or {}).get("$ref") or ""))
    return int(m.group(1)) if m else None


def _timestamp(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def milestone_line(value) -> Optional[float]:
    """'3+' (three or more) -> 2.5, the over line it equals. Also takes a
    bare number of the level (3 or 3.0)."""
    if value is None or isinstance(value, bool):
        return None
    s = str(value).strip().rstrip("+")
    try:
        level = float(s)
    except ValueError:
        return None
    return level - 0.5 if level >= 1 else None


def _side_label(item: dict) -> Optional[str]:
    for block in (item.get("current"), item.get("open")):
        if isinstance(block, dict):
            for side in ("over", "under"):
                if side in block:
                    return side
    return None


def _lines(item: dict, kind: str) -> tuple:
    """(current line, opening line) of one entry, as floats or None."""
    odds_total = (item.get("odds") or {}).get("total") or {}
    cur_target = ((item.get("current") or {}).get("target") or {})
    open_target = ((item.get("open") or {}).get("target") or {})
    if kind == "yes":
        return 0.5, 0.5
    if kind == "milestone":
        cur = milestone_line(odds_total.get("value") or cur_target.get("displayValue"))
        opn = milestone_line(odds_total.get("open") or open_target.get("displayValue"))
        return cur, opn
    cur = parse_line(odds_total.get("value"))
    opn = parse_line(odds_total.get("open"))
    if cur is None:
        cur = parse_line(cur_target.get("value"))
    if opn is None:
        opn = parse_line(open_target.get("value"))
    return cur, opn


def parse_prop_bets(items: List[dict], book: str,
                    event_start: Optional[datetime]) -> tuple:
    """(rows, counts) for one book's propBets items. Pure.

    rows: one dict per (espn_athlete_id, market, line) with over/under
    prices (current and opening) and last_updated; names and ids are added
    later. counts: why entries were skipped or prices dropped."""
    counts = Counter()
    over_first = (book or "").strip().lower() in OVER_FIRST_BOOKS
    entries = []           # (order, athlete, market, kind, side|None, item)
    for order, item in enumerate(items or []):
        name = (item.get("type") or {}).get("name")
        aid = _athlete_ref_id(item)
        if name not in MARKETS or aid is None:
            counts["skipped_market"] += 1
            continue
        market, kind = MARKETS[name]
        side = "over" if kind in ("yes", "milestone") else _side_label(item)
        entries.append([order, aid, market, kind, side, item])

    # Unlabelled over/under entries: pair adjacent entries on the same line
    unlabelled = defaultdict(list)
    for e in entries:
        if e[3] == "ou" and e[4] is None:
            unlabelled[(e[1], e[2])].append(e)
    for group in unlabelled.values():
        if not over_first:
            counts["skipped_unlabelled"] += len(group)
            continue
        by_line = defaultdict(list)
        for e in group:
            by_line[_lines(e[5], "ou")[0]].append(e)
        for pair in by_line.values():
            if len(pair) == 2:
                pair[0][4], pair[1][4] = "over", "under"
            else:
                counts["skipped_unpaired"] += len(pair)

    # Side records -> rows keyed (athlete, market, line)
    rows = {}
    records = []
    for order, aid, market, kind, side, item in entries:
        if side is None:
            continue
        cur_line, open_line = _lines(item, kind)
        american = (item.get("odds") or {}).get("american") or {}
        cur_price, open_price = parse_american(american.get("value")), parse_american(american.get("open"))
        updated = _timestamp(item.get("lastUpdated"))
        pregame = updated is not None and event_start is not None and updated < event_start
        if pregame:
            line = cur_line
            open_at_line = open_price if (open_line is not None and cur_line is not None
                                          and abs(open_line - cur_line) < 1e-9) else None
            price = cur_price
        else:
            counts["in_play_prices_dropped"] += 1
            line, price, open_at_line = open_line, None, open_price
        if line is None:
            counts["skipped_no_line"] += 1
            continue
        relined = not (cur_line is not None and open_line is not None
                       and abs(cur_line - open_line) < 1e-9)
        records.append((relined, order, aid, market, round(line, 1), side, price,
                        open_at_line, updated))

    # Entries whose line never moved win over re-lined ones on a clash
    for relined, order, aid, market, line, side, price, open_price, updated in sorted(records):
        row = rows.setdefault((aid, market, line), {
            "espn_athlete_id": aid, "market": market, "line": line, "book": book,
            "over_price": None, "under_price": None,
            "over_price_open": None, "under_price_open": None, "last_updated": None})
        for field, value in ((f"{side}_price", price), (f"{side}_price_open", open_price)):
            if value is None:
                continue
            if row[field] is None:
                row[field] = value
            elif row[field] != value:
                counts["conflicting_duplicates"] += 1
        if updated is not None and (row["last_updated"] is None or updated > row["last_updated"]):
            row["last_updated"] = updated

    out = []
    for row in rows.values():
        row["over_price"], row["under_price"] = sane_pair(row["over_price"], row["under_price"])
        row["over_price_open"], row["under_price_open"] = sane_pair(
            row["over_price_open"], row["under_price_open"])
        if all(row[c] is None for c in ("over_price", "under_price",
                                        "over_price_open", "under_price_open")):
            counts["skipped_no_price"] += 1
            continue
        out.append(row)
    return out, counts


def parse_roster(summary: dict) -> dict:
    """ESPN athlete id -> {name, short_name, team, position} from a game
    summary's boxscore (everyone who dressed)."""
    out = {}
    for team in ((summary or {}).get("boxscore") or {}).get("players") or []:
        info = team.get("team") or {}
        abbrev = espn_team_abbrev(info.get("displayName"), info.get("abbreviation"))
        for group in team.get("statistics") or []:
            for a in group.get("athletes") or []:
                ath = a.get("athlete") or {}
                if not str(ath.get("id") or "").isdigit():
                    continue
                out[int(ath["id"])] = {
                    "name": ath.get("displayName"),
                    "short_name": ath.get("shortName"),
                    "team": abbrev,
                    "position": POSITION_MAP.get(
                        str((ath.get("position") or {}).get("abbreviation") or "").upper()),
                }
    return out


# ── ESPN requests ──────────────────────────────────────────────────

def _get_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    """JSON, or None on a 404 (ESPN answers "No propBets found" that way).
    Any other failure raises, so the game is retried on the next run."""
    resp = requests.get(url, params={**PARAMS, **(params or {})}, timeout=TIMEOUT_S)
    time.sleep(REQUEST_PAUSE_S)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def fetch_books(event_id: str) -> list:
    """[(provider id, name)] ESPN lists for a game, live-odds feeds left out."""
    data = _get_json(CORE_EVENT_URL.format(event=event_id) + "/odds") or {}
    books = []
    for item in data.get("items") or []:
        prov = item.get("provider") or {}
        pid, name = prov.get("id"), prov.get("name") or ""
        if pid and "live" not in name.lower():
            books.append((str(pid), name))
    return books


def fetch_prop_bets(event_id: str, provider_id: str) -> list:
    url = CORE_EVENT_URL.format(event=event_id) + f"/odds/{provider_id}/propBets"
    data = _get_json(url, {"limit": PAGE_LIMIT}) or {}
    items = list(data.get("items") or [])
    for page in range(2, int(data.get("pageCount") or 1) + 1):
        more = _get_json(url, {"limit": PAGE_LIMIT, "page": page}) or {}
        items += more.get("items") or []
    return items


def fetch_summary(event_id: str) -> dict:
    return _get_json(SUMMARY_URL, {"event": event_id}) or {}


def fetch_athlete(athlete_id: int) -> dict:
    """{name, position} for an athlete missing from the boxscore (a
    scratched player whose props were posted)."""
    data = _get_json(ATHLETE_URL.format(athlete=athlete_id)) or {}
    return {"name": data.get("displayName") or data.get("fullName"),
            "short_name": data.get("shortName"), "team": None,
            "position": POSITION_MAP.get(
                str((data.get("position") or {}).get("abbreviation") or "").upper())}


# ── Names and ids ──────────────────────────────────────────────────

def load_game_index(game_id: int) -> PlayerIndex:
    """The players who played this game (NHL boxscore), with their team."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT x.player_id, p.full_name, p.position, x.team
            FROM (SELECT player_id, team FROM raw.skater_games WHERE game_id = :g
                  UNION ALL
                  SELECT player_id, team FROM raw.goalie_games WHERE game_id = :g) x
            JOIN raw.players p USING (player_id)
        """), {"g": game_id}).fetchall()
    return PlayerIndex([(r[0], r[1], r[2]) for r in rows], {r[0]: r[3] for r in rows})


def name_rows(rows: List[dict], roster: dict, game_index: PlayerIndex,
              global_index: Optional[PlayerIndex]) -> Counter:
    """Fill player_name and player_id in place. Pure given its inputs.
    Tries the game's own players first (short name, then full name, by
    team), then everyone in raw.players."""
    counts = Counter()
    for row in rows:
        info = roster.get(row["espn_athlete_id"]) or {}
        row["player_name"] = (info.get("name") or info.get("short_name") or "")[:80] or None
        pid = None
        for name in (info.get("short_name"), info.get("name")):
            if name and pid is None:
                pid, _ = game_index.match(name, info.get("team"), info.get("position"))
        if pid is None and global_index is not None and info.get("name"):
            pid, _ = global_index.match(info.get("name"), info.get("team"), info.get("position"))
        row["player_id"] = pid
        counts["resolved" if pid else "unresolved"] += 1
    return counts


# ── Store ──────────────────────────────────────────────────────────

_INSERT_SQL = text(f"""
    INSERT INTO raw.prop_odds_hist ({', '.join(ROW_COLUMNS)})
    VALUES ({', '.join(':' + c for c in ROW_COLUMNS)})
    ON CONFLICT (game_id, book, market, espn_athlete_id, line) DO UPDATE SET
        {', '.join(f'{c} = EXCLUDED.{c}' for c in ROW_COLUMNS
                   if c not in ('game_id', 'book', 'market', 'espn_athlete_id', 'line'))},
        fetched_at = now()
""")

_LOG_SQL = text("""
    INSERT INTO raw.prop_odds_fetches (game_id, espn_event_id, books, n_rows)
    VALUES (:game_id, :espn_event_id, :books, :n_rows)
    ON CONFLICT (game_id) DO UPDATE SET
        espn_event_id = EXCLUDED.espn_event_id, books = EXCLUDED.books,
        n_rows = EXCLUDED.n_rows, fetched_at = now()
""")


def write_game(game_id: int, event_id: Optional[str], rows: List[dict],
               books: List[str]) -> int:
    """Store one game's rows and record the fetch, in one transaction. A
    game that already has rows keeps them when a re-fetch finds none
    (ESPN dropping old props must not erase what was saved)."""
    with engine.begin() as conn:
        if rows:
            conn.execute(text("DELETE FROM raw.prop_odds_hist WHERE game_id = :g"),
                         {"g": game_id})
            conn.execute(_INSERT_SQL, [{c: r.get(c) for c in ROW_COLUMNS} for r in rows])
            n = len(rows)
        else:
            n = conn.execute(text("SELECT count(*) FROM raw.prop_odds_hist WHERE game_id = :g"),
                             {"g": game_id}).scalar()
            if n:
                logger.warning(f"game {game_id}: ESPN no longer lists props; "
                               f"kept the {n} rows stored earlier")
        conn.execute(_LOG_SQL, {"game_id": game_id, "espn_event_id": event_id,
                                "books": ",".join(books)[:120] or None, "n_rows": n})
    return len(rows)


# ── Backfill ───────────────────────────────────────────────────────

def fetch_game_rows(game_id: int, event_id: str, event_start: Optional[datetime],
                    global_index: Optional[PlayerIndex], totals: Counter) -> tuple:
    """(rows, books with props) for one game. Raises on a failed request."""
    rows, books = [], []
    for provider_id, book in fetch_books(event_id):
        items = fetch_prop_bets(event_id, provider_id)
        if not items:
            continue
        parsed, counts = parse_prop_bets(items, book, event_start)
        totals.update(counts)
        if parsed:
            rows += parsed
            books.append(book)
    if not rows:
        return rows, books
    roster = parse_roster(fetch_summary(event_id))
    for aid in {r["espn_athlete_id"] for r in rows} - set(roster):
        try:
            roster[aid] = fetch_athlete(aid)
        except Exception as e:    # a name is nice to have, not worth the game
            logger.debug(f"athlete {aid} lookup failed: {e}")
    totals.update(name_rows(rows, roster, load_game_index(game_id), global_index))
    for r in rows:
        r.update(game_id=game_id, espn_event_id=str(event_id), event_start=event_start)
    return rows, books


def backfill_props(season: int, limit: Optional[int] = None, refresh: bool = False,
                   game_ids: Optional[list] = None) -> int:
    """Fetch ESPN props for a season's finished games not fetched before
    (all of them with refresh=True), at most `limit` games per run.
    Returns the number of rows written."""
    ensure_tables()
    clauses = ["g.season = :season", "g.game_state IN ('FINAL', 'OFF')"]
    params = {"season": season}
    if not refresh:
        clauses.append("g.game_id NOT IN (SELECT game_id FROM raw.prop_odds_fetches)")
    if game_ids:
        clauses.append("g.game_id = ANY(:ids)")
        params["ids"] = [int(g) for g in game_ids]
    if limit:
        params["limit"] = limit
    with engine.connect() as conn:
        games = conn.execute(text(f"""
            SELECT g.game_id, g.date, g.home_team, g.start_time_utc
            FROM raw.games g WHERE {' AND '.join(clauses)}
            ORDER BY g.date, g.game_id {"LIMIT :limit" if limit else ""}
        """), params).mappings().all()
    by_date = defaultdict(list)
    for g in games:
        by_date[g["date"]].append(dict(g))
    logger.info(f"ESPN props: {len(games)} games over {len(by_date)} dates to fetch")
    if not games:
        return 0

    global_index = load_player_index()
    totals, written, with_props, failed = Counter(), 0, 0, 0
    for game_date, day_games in by_date.items():
        try:
            events = fetch_scoreboard(game_date.strftime("%Y%m%d"))
            time.sleep(REQUEST_PAUSE_S)
        except Exception as e:
            logger.error(f"ESPN scoreboard {game_date} failed (retried next run): {e}")
            failed += len(day_games)
            continue
        matched = match_events(events, day_games)
        for g in day_games:
            m = matched.get(g["game_id"])
            if not m:
                write_game(g["game_id"], None, [], [])
                totals["no_espn_event"] += 1
                continue
            start = m["start"] or g["start_time_utc"]
            try:
                rows, books = fetch_game_rows(g["game_id"], m["event_id"], start,
                                              global_index, totals)
            except Exception as e:
                logger.error(f"ESPN props for game {g['game_id']} failed "
                             f"(retried next run): {e}")
                failed += 1
                continue
            written += write_game(g["game_id"], m["event_id"], rows, books)
            with_props += bool(rows)

    logger.info(f"ESPN props complete: {written} rows from {with_props} of "
                f"{len(games) - failed} games; " + ", ".join(
                    f"{k} {v}" for k, v in sorted(totals.items())))
    if failed:
        logger.error(f"ESPN props: {failed} games failed and will be retried next run")
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.espn_props",
        description="Backfill past player-prop prices from ESPN's free API into "
                    "raw.prop_odds_hist for one season's finished games. "
                    "Resumable: games already fetched are skipped.")
    parser.add_argument("--season", type=int, required=True,
                        help="the season, such as 20252026")
    parser.add_argument("--limit", type=int, default=None,
                        help="at most this many games in this run")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch games already fetched as well")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be 1 or more")
    return backfill_props(args.season, limit=args.limit, refresh=args.refresh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
