"""
ingestion/props_odds.py
Live player-prop lines from The Odds API into raw.prop_snapshots.

Terms:
- Player prop: a bet on one player's own numbers, such as "over 2.5 shots
  on goal".
- Line: the number an over/under is set at (the 2.5 above).
- Price: what a side pays, in American odds → -120 means risk 120 to win
  100, +110 means risk 100 to win 110.
- Credits: The Odds API's billing unit.

Machine role: this is the PC's job. The Mac spends its key on moneyline
picks and closes; the PC's own ODDS_API_KEY (each machine has its own
.env, and each key its own 500 free credits a month) collects props lines.
Its database needs a current schedule (the daily refresh), because every
line is tied to a raw.games row.

How one run works:
1. GET /v4/sports/icehockey_nhl/events lists upcoming games. It is free.
2. Keep games that start in the next 24 hours (PROPS_HORIZON). With --due,
   keep only games that start in the next PROPS_CLOSE_LEAD_MINUTES (16) and
   have no prop snapshot from the last PROPS_CLOSE_MIN_GAP_MINUTES (16).
3. Tie each game to raw.games with ingestion/odds_api.match_event: same
   home team, start time within 6 hours, nearest first. A game that has
   started, or that matches no row, is skipped before any paid request.
4. For each game, GET /v4/sports/icehockey_nhl/events/{eventId}/odds with
   markets=PROPS_MARKETS, oddsFormat=american and bookmakers=PROPS_BOOKMAKERS.
   Props are only served one game at a time, from this endpoint.
5. Store one row per book, market, player and line, with the player
   resolved to raw.players where possible.

Cost (the v4 docs): "cost = [number of unique markets returned] x [number
of regions specified]", and "every group of 10 bookmakers is the
equivalent of 1 region". So with 10 or fewer books a game costs 1 credit
per market that comes back, per snapshot, and a game where no book has
posted props yet costs nothing ("Responses with empty data do not count
towards the usage quota"). On the 2026-27 schedule (1,344 games), a
morning snapshot plus one pre-game snapshot per game is 2 credits a game
per market (UTC calendar months):
  1 market (player_shots_on_goal): 2 credits x games a month, 308 to 464
    in October to March (Feb 308, Jan 464), 2,688 for the season; fits
    the free 500, narrowly in January.
  4 markets: 1,232 to 1,856 a month, 10,752 a season; needs the 20K plan.
Those are ceilings: morning requests for games without posted props yet
cost 0 (and collect nothing).

Books: PROPS_BOOKMAKERS, comma-separated keys. Unset = the same ten as the
moneyline default (ODDS_BOOKMAKERS): kalshi, polymarket, pinnacle,
draftkings, fanduel, betmgm, betrivers, espnbet, hardrockbet, novig. The
docs say props coverage "is mainly limited to US sports and US
bookmakers" and do not say Kalshi or Polymarket carry none, so they stay;
a book with no NHL props returns nothing and costs nothing extra, since
ten books bill as one region. More than 10 logs a warning (each further
group of 10 bills as another region). Set to an empty string, the request
uses regions=PROPS_REGIONS (default us, one region) instead.

Pairing: each outcome is one side, with name "Over" or "Under", the player
in description and the line in point (the documented shape). Rows are
keyed by (book, market, player, line), so an over and an under at the same
line share a row; a side the book doesn't offer is NULL (alternate "X or
more" lines often have only the over). Yes/No markets (goal scorer) store
Yes as over_price and No as under_price, with no line.

Players: books publish full names ("Connor McDavid"), raw.players holds
abbreviated ones ("C. McDavid"), as with the Daily Faceoff starters
(ingestion/dailyfaceoff.py, whose name keys this reuses). Tried in order:
the exact accent-stripped name, then first initial + surname, then the
surname alone among players last seen on one of the two teams. Two
candidates are split by which one last played for one of the two teams.
An unresolved name is stored with player_id NULL and logged, never dropped.

--due: the pre-game snapshot, run every 15 minutes like `pipeline.py close
--due`. Because props are requested per game, it requests only the games
starting within the lead that have no recent prop snapshot: one pre-game
snapshot per game, taken 1 to 16 minutes before puck drop.

In-play: a game that has started is never requested; its props are live
prices for a different bet.

The API key never reaches the logs: failures are summarized (status,
reason, the API's own message), anything logged passes through
odds_api._redact(), and exceptions from requests (whose messages carry
the full URL, key included) are never logged themselves.
"""
import argparse
import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from sqlalchemy import text

from config.settings import engine
from ingestion.dailyfaceoff import _initial_key, _normalize
from ingestion.odds_api import BASE_URL, SPORT, _api_key, _redact, match_event

logger = logging.getLogger("nhl.ingestion.props_odds")

TIMEOUT_S = 30
PAUSE_S = 0.5                            # between per-game requests
PROPS_HORIZON = timedelta(hours=24)      # the morning run: games in the next 24h
DEFAULT_MARKETS = "player_shots_on_goal"
# The moneyline default (ingestion/odds_api.DEFAULT_BOOKMAKERS); tests check
# they stay the same list.
DEFAULT_BOOKMAKERS = ("kalshi", "polymarket", "pinnacle", "draftkings", "fanduel",
                      "betmgm", "betrivers", "espnbet", "hardrockbet", "novig")
DEFAULT_REGIONS = "us"
BOOKS_PER_REGION = 10
STOP_STATUSES = (401, 403, 422)   # bad key, quota used up, bad market: every call would fail
_KEY_SHAPE = re.compile(r"^[a-z0-9_]+$")
_SIDES = {"over": "over_price", "yes": "over_price",
          "under": "under_price", "no": "under_price"}

# The same DDL is in config/migrate.py (TABLES) and db/schema.sql.
DDL = [
    """
    CREATE TABLE IF NOT EXISTS raw.prop_snapshots (
        snapshot_id     BIGSERIAL PRIMARY KEY,
        captured_at     TIMESTAMP NOT NULL,            -- naive UTC, when the request returned
        game_id         BIGINT NOT NULL REFERENCES raw.games(game_id),
        event_id        VARCHAR(64) NOT NULL,          -- The Odds API event id
        book            VARCHAR(40) NOT NULL,          -- Odds API bookmaker key
        market          VARCHAR(60) NOT NULL,          -- e.g. player_shots_on_goal
        player_name     VARCHAR(80) NOT NULL,          -- as the book publishes it
        player_id       INTEGER,                       -- raw.players id; NULL when unresolved
        line            NUMERIC(5,1),                  -- NULL for yes/no markets
        over_price      INTEGER,                       -- American odds; Yes for yes/no markets
        under_price     INTEGER,                       -- No for yes/no markets
        book_updated_at TIMESTAMP                      -- the market's last_update, naive UTC
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_prop_snapshots_game ON raw.prop_snapshots(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_prop_snapshots_event "
    "ON raw.prop_snapshots(event_id, captured_at)",
    "CREATE INDEX IF NOT EXISTS idx_prop_snapshots_player "
    "ON raw.prop_snapshots(player_id, market)",
]


def ensure_table() -> None:
    """Create raw.prop_snapshots and its indexes when missing."""
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))


# ── Settings (read on every run, so .env edits need no restart) ───

def _key_list(raw: Optional[str]) -> List[str]:
    """Comma-separated keys: trimmed, lower-cased, in order, no blanks or
    repeats."""
    out: List[str] = []
    for part in (raw or "").split(","):
        k = part.strip().lower()
        if k and k not in out:
            out.append(k)
    return out


def _valid_keys(name: str, raw: Optional[str]) -> List[str]:
    keys = _key_list(raw)
    bad = [k for k in keys if not _KEY_SHAPE.match(k)]
    if bad:
        logger.error(f"{name}: ignoring {', '.join(map(repr, bad))}, not an Odds API "
                     f"key (lower-case letters, digits and _ only)")
    return [k for k in keys if k not in bad]


def markets_setting(markets: Optional[str] = None, environ=None) -> str:
    """The markets to request: `markets` (the --markets option), else
    PROPS_MARKETS, else player_shots_on_goal. Malformed keys are dropped
    with an error; nothing usable left means the default."""
    env = os.environ if environ is None else environ
    raw = markets if markets is not None else env.get("PROPS_MARKETS")
    keys = _valid_keys("PROPS_MARKETS" if markets is None else "--markets", raw)
    if not keys:
        if raw and raw.strip():
            logger.error(f"No usable market in {raw!r}: using {DEFAULT_MARKETS}")
        return DEFAULT_MARKETS
    return ",".join(keys)


def book_selection(environ=None) -> dict:
    """{"bookmakers": "a,b"} or {"regions": "us"}: the query parameter that
    chooses the books. PROPS_BOOKMAKERS unset = DEFAULT_BOOKMAKERS; set to
    an empty string (or nothing usable) = regions from PROPS_REGIONS
    (default us). More than 10 books logs a warning."""
    env = os.environ if environ is None else environ
    raw = env.get("PROPS_BOOKMAKERS")
    if raw is None:
        return {"bookmakers": ",".join(DEFAULT_BOOKMAKERS)}
    keys = _valid_keys("PROPS_BOOKMAKERS", raw)
    if keys:
        selection = {"bookmakers": ",".join(keys)}
        units = region_units(selection)
        if units > 1:
            logger.warning(f"PROPS_BOOKMAKERS names {len(keys)} bookmakers: the Odds API "
                           f"bills every group of {BOOKS_PER_REGION} as one region, so "
                           f"each market costs {units} credits a game instead of 1")
        return selection
    regions = _valid_keys("PROPS_REGIONS", env.get("PROPS_REGIONS")) or [DEFAULT_REGIONS]
    if raw.strip():
        logger.error(f"PROPS_BOOKMAKERS={raw!r} names no usable bookmaker: using "
                     f"regions {','.join(regions)} instead")
    return {"regions": ",".join(regions)}


def region_units(selection: dict) -> int:
    """Regions a selection bills as: one per started group of 10 books,
    or one per region named."""
    if "bookmakers" in selection:
        return max(1, math.ceil(len(_key_list(selection["bookmakers"])) / BOOKS_PER_REGION))
    return max(1, len(_key_list(selection.get("regions"))))


def minutes_setting(name: str, default: float, allow_zero: bool, environ=None) -> timedelta:
    """Minutes from the environment; unset or blank = default. A malformed
    or negative value (or 0 unless allow_zero) logs an error and uses the
    default, so a typo can't stop the scheduled job."""
    env = os.environ if environ is None else environ
    raw = (env.get(name) or "").strip()
    if not raw:
        return timedelta(minutes=default)
    try:
        value = float(raw)
        if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError(raw)
        return timedelta(minutes=value)
    except (ValueError, OverflowError):
        logger.error(f"{name}={raw!r} is not a number of minutes "
                     f"({'0 or more' if allow_zero else 'more than 0'}): "
                     f"using the default, {default:g}")
        return timedelta(minutes=default)


# ── HTTP (never logs the key) ─────────────────────────────────────

def _error_summary(resp) -> str:
    """'HTTP 401 Unauthorized: <the API's message>', never the URL."""
    if resp is None:
        return "HTTP error with no response"
    message = None
    try:
        body = resp.json()
        if isinstance(body, dict):
            message = body.get("message")
    except ValueError:
        pass
    if not message:
        message = _redact(resp.text or "")[:200]   # redact before cutting
    return f"HTTP {resp.status_code} {resp.reason}: {_redact(message)}"


def _get(path: str, params: dict) -> Tuple[Optional[object], dict, Optional[int]]:
    """GET BASE_URL + path with the key added. Returns (body, headers,
    status); body is None after any failure, status None when no response
    came back. A 429 (too many requests) is retried once after a pause.
    Each success logs the credits remaining, used and charged."""
    key = _api_key()
    if not key:
        logger.error("ODDS_API_KEY is not set (missing, or still the .env.example "
                     "placeholder): no props request made. Add this machine's key to .env")
        return None, {}, None
    url = f"{BASE_URL}/{path.lstrip('/')}"
    for attempt in (1, 2):
        try:
            resp = requests.get(url, params={"apiKey": key, **params}, timeout=TIMEOUT_S)
            if resp.status_code == 429 and attempt == 1:
                logger.warning("Odds API: too many requests (HTTP 429); retrying once")
                time.sleep(max(PAUSE_S, 1.0) * 2)
                continue
            resp.raise_for_status()
            body = resp.json()
        except requests.HTTPError as e:
            logger.error(f"Odds API props request failed: {_error_summary(e.response)}")
            return None, {}, getattr(e.response, "status_code", None)
        except requests.Timeout:
            logger.error(f"Odds API props request failed: timed out after {TIMEOUT_S}s")
            return None, {}, None
        except requests.ConnectionError as e:
            logger.error(f"Odds API props request failed: could not reach "
                         f"api.the-odds-api.com ({type(e).__name__})")
            return None, {}, None
        except Exception as e:
            logger.error(f"Odds API props request failed: {type(e).__name__}: {_redact(e)}")
            return None, {}, None
        h = resp.headers
        logger.info(f"Odds API {_redact(path)}: credits remaining "
                    f"{h.get('x-requests-remaining', '?')}, used "
                    f"{h.get('x-requests-used', '?')}, this call cost "
                    f"{h.get('x-requests-last', '?')}")
        return body, h, resp.status_code
    return None, {}, 429


# ── Pure helpers ──────────────────────────────────────────────────

def _parse_time(value) -> Optional[datetime]:
    """ISO time from the API ("2026-01-16T00:00:00Z") -> aware UTC; None
    when missing or malformed."""
    if not value:
        return None
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def _iso(t: datetime) -> str:
    """The form the events filter accepts: 2026-01-16T00:00:00Z."""
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _price(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return int(round(v)) if math.isfinite(v) else None


def _line(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def select_events(events: Iterable[dict], now: datetime, horizon: timedelta) -> List[dict]:
    """Events that start in (now, now + horizon]: never one under way."""
    out = []
    for ev in events or []:
        start = _parse_time(ev.get("commence_time"))
        if ev.get("id") and start is not None and now < start <= now + horizon:
            out.append(ev)
    return out


def props_due(events: List[dict], last_captures: Dict[str, datetime], now: datetime,
              lead: timedelta, min_gap: timedelta) -> Tuple[List[dict], str]:
    """(due events, reason) for --due. Pure. Due: starts in (now, now +
    lead] and no prop snapshot of that game in the last min_gap."""
    soon = select_events(events, now, lead)
    if not soon:
        return [], f"no game starts in the next {int(lead.total_seconds() // 60)} minutes"
    due = [ev for ev in soon
           if last_captures.get(ev["id"]) is None
           or now - last_captures[ev["id"]] >= min_gap]
    if not due:
        return [], (f"{len(soon)} game(s) start soon, each with a prop snapshot from the "
                    f"last {int(min_gap.total_seconds() // 60)} minutes")
    first = min(int((_parse_time(ev["commence_time"]) - now).total_seconds() // 60)
                for ev in due)
    return due, f"{len(due)} game(s) due, the first starting in {first} minute(s)"


def parse_event_odds(payload: dict) -> List[dict]:
    """One row per (book, market, player, line) from an event-odds
    response. Outcomes without a player (description) or with a side other
    than Over/Under/Yes/No are skipped, and so is a row with no price."""
    rows: Dict[tuple, dict] = {}
    for bookmaker in (payload or {}).get("bookmakers") or []:
        book = bookmaker.get("key") or "unknown"
        for market in bookmaker.get("markets") or []:
            market_key = market.get("key") or "unknown"
            updated = _parse_time(market.get("last_update"))
            for outcome in market.get("outcomes") or []:
                player = (outcome.get("description") or "").strip()
                column = _SIDES.get((outcome.get("name") or "").strip().lower())
                price = _price(outcome.get("price"))
                if not player or column is None or price is None:
                    continue
                line = _line(outcome.get("point"))
                row = rows.setdefault((book, market_key, player, line), {
                    "book": book, "market": market_key, "player_name": player,
                    "line": line, "over_price": None, "under_price": None,
                    "book_updated_at": (updated.replace(tzinfo=None)
                                        if updated else None)})
                row[column] = price
    return list(rows.values())


def _norm(name: str) -> str:
    return _normalize(name.replace("’", "'"))


def _key(name: str) -> tuple:
    return _initial_key(name.replace("’", "'"))


def build_player_index(players: Iterable[tuple], recent_teams: Iterable[tuple]) -> dict:
    """players: (player_id, full_name); recent_teams: (player_id, team of
    the player's latest game). Pure."""
    by_name: Dict[str, list] = {}
    by_key: Dict[tuple, list] = {}
    by_surname: Dict[str, list] = {}
    for pid, full_name in players:
        if not full_name:
            continue
        by_name.setdefault(_norm(full_name), []).append(pid)
        k = _key(full_name)
        by_key.setdefault(k, []).append(pid)
        by_surname.setdefault(k[1], []).append(pid)
    return {"by_name": by_name, "by_key": by_key, "by_surname": by_surname,
            "last_team": dict(recent_teams)}


def match_player(name: str, teams: Iterable[str], index: dict) -> Tuple[Optional[int], str]:
    """(player_id | None, reason) for a book's player name in a game
    between `teams`. Pure. reason: ok, ambiguous, or not in raw.players."""
    teams = set(teams)
    last_team = index["last_team"]

    def pick(cands):
        cands = list(dict.fromkeys(cands))
        if len(cands) == 1:
            return cands[0]
        on_team = [p for p in cands if last_team.get(p) in teams]
        return on_team[0] if len(on_team) == 1 else None

    cands = index["by_name"].get(_norm(name)) or index["by_key"].get(_key(name))
    if cands:
        pid = pick(cands)
        return pid, "ok" if pid is not None else "ambiguous"
    # A differently spelled first name (Yegor / Egor): the surname alone,
    # but only among players last seen on one of the two teams
    on_team = list(dict.fromkeys(p for p in index["by_surname"].get(_key(name)[1], [])
                                 if last_team.get(p) in teams))
    if len(on_team) == 1:
        return on_team[0], "ok"
    return None, "ambiguous" if on_team else "not in raw.players"


# ── Database ──────────────────────────────────────────────────────

def upcoming_starts(now: datetime, horizon: timedelta) -> list:
    """Puck drops in (now, now + horizon] from raw.games, skipping
    postponed, suspended and cancelled games."""
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text("""
            SELECT start_time_utc FROM raw.games
            WHERE start_time_utc > :now AND start_time_utc <= :until
              AND COALESCE(schedule_state, 'OK') NOT IN ('PPD', 'SUSP', 'CNCL')
        """), {"now": now, "until": now + horizon})]


def candidate_games(now: datetime, horizon: timedelta) -> List[dict]:
    """raw.games rows match_event could pick for events in the window."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT game_id, home_team, away_team, date, start_time_utc FROM raw.games
            WHERE start_time_utc BETWEEN :lo AND :hi
               OR (start_time_utc IS NULL AND date BETWEEN :dlo AND :dhi)
        """), {"lo": now - timedelta(hours=7), "hi": now + horizon + timedelta(hours=7),
               "dlo": (now - timedelta(days=1)).date(),
               "dhi": (now + horizon + timedelta(days=1)).date()}).mappings().all()
    return [dict(r) for r in rows]


def last_captures(event_ids: List[str]) -> Dict[str, datetime]:
    """Newest captured_at (aware UTC) per event id in raw.prop_snapshots."""
    if not event_ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT event_id, MAX(captured_at) FROM raw.prop_snapshots
            WHERE event_id = ANY(:ids) GROUP BY event_id
        """), {"ids": list(event_ids)}).fetchall()
    return {eid: t.replace(tzinfo=timezone.utc) for eid, t in rows if t is not None}


def load_player_index() -> dict:
    with engine.connect() as conn:
        players = conn.execute(text("SELECT player_id, full_name FROM raw.players")).fetchall()
        recent = conn.execute(text("""
            SELECT DISTINCT ON (player_id) player_id, team FROM (
                SELECT sg.player_id, sg.team, g.date, g.game_id
                FROM raw.skater_games sg JOIN raw.games g ON g.game_id = sg.game_id
                UNION ALL
                SELECT gg.player_id, gg.team, g.date, g.game_id
                FROM raw.goalie_games gg JOIN raw.games g ON g.game_id = gg.game_id
            ) x
            ORDER BY player_id, date DESC, game_id DESC
        """)).fetchall()
    return build_player_index(players, recent)


def write_rows(rows: List[dict]) -> int:
    if not rows:
        return 0
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw.prop_snapshots
                (captured_at, game_id, event_id, book, market, player_name,
                 player_id, line, over_price, under_price, book_updated_at)
            VALUES (:captured_at, :game_id, :event_id, :book, :market, :player_name,
                    :player_id, :line, :over_price, :under_price, :book_updated_at)
        """), rows)
    return len(rows)


# ── The run ───────────────────────────────────────────────────────

def list_events(now: datetime, until: datetime) -> Optional[list]:
    """Upcoming NHL events from the free /events endpoint; None on failure."""
    body, _, _ = _get(f"/sports/{SPORT}/events",
                      {"dateFormat": "iso", "commenceTimeFrom": _iso(now),
                       "commenceTimeTo": _iso(until)})
    return body if isinstance(body, list) else None


def snapshot_props(markets: Optional[str] = None, due: bool = False,
                   now: Optional[datetime] = None) -> int:
    """One props snapshot (see the module docstring). Returns the number
    of rows stored. API failures are logged, not raised; a database error
    raises, so a pipeline step calling this should treat it as non-fatal."""
    ensure_table()
    now = now or datetime.now(timezone.utc)
    lead = minutes_setting("PROPS_CLOSE_LEAD_MINUTES", 16, allow_zero=False)
    min_gap = minutes_setting("PROPS_CLOSE_MIN_GAP_MINUTES", 16, allow_zero=True)
    horizon = lead if due else PROPS_HORIZON
    label = "props --due" if due else "props"

    if not _api_key():
        logger.error("ODDS_API_KEY is not set (missing, or still the .env.example "
                     "placeholder): no props request made. Add this machine's key to .env")
        return 0
    if not upcoming_starts(now, horizon):
        logger.info(f"{label}: no game in raw.games starts in the next "
                    f"{int(horizon.total_seconds() // 60)} minutes; no request made. "
                    f"(If games are missing, refresh the schedule: python pipeline.py daily)")
        return 0

    events = list_events(now, now + horizon)
    if events is None:
        return 0
    events = select_events(events, now, horizon)
    if due:
        events, why = props_due(events, last_captures([e["id"] for e in events]),
                                now, lead, min_gap)
        logger.info(f"props --due: {why}" if events else f"props --due: no snapshot, {why}")
        if not events:
            return 0

    candidates = candidate_games(now, horizon)
    todo, unmatched = [], []
    for ev in events:
        game_id, status = match_event(ev, candidates, now)
        if game_id is None:
            if status != "started":
                unmatched.append(f"{ev.get('away_team', '?')} @ {ev.get('home_team', '?')} "
                                 f"{ev.get('commence_time', '?')} ({status})")
            continue
        game = next(g for g in candidates if g["game_id"] == game_id)
        todo.append((ev, game))
    if unmatched:
        logger.warning(f"{len(unmatched)} event(s) matched no raw.games row and were not "
                       f"requested (schedule not loaded that far, or a team/start-time "
                       f"mismatch): " + "; ".join(unmatched[:5])
                       + (" ..." if len(unmatched) > 5 else ""))
    if not todo:
        logger.info(f"{label}: no game to request")
        return 0

    market_list = markets_setting(markets)
    selection = book_selection()
    per_game = len(_key_list(market_list)) * region_units(selection)
    books = selection.get("bookmakers") or f"regions {selection.get('regions')}"
    logger.info(f"{label}: {len(todo)} game(s), markets {market_list}, books {books}: "
                f"up to {per_game} credit(s) a game, {per_game * len(todo)} in all")

    index = None
    stored, credits, games_with_lines, remaining = 0, 0, 0, "?"
    for i, (ev, game) in enumerate(todo):
        if i:
            time.sleep(PAUSE_S)
        body, headers, status = _get(
            f"/sports/{SPORT}/events/{ev['id']}/odds",
            {"markets": market_list, "oddsFormat": "american", "dateFormat": "iso",
             **selection})
        if body is None:
            if status in STOP_STATUSES:
                logger.error(f"{label}: stopping after HTTP {status}; the remaining "
                             f"{len(todo) - i - 1} game(s) were not requested")
                break
            continue
        credits += int(_price(headers.get("x-requests-last")) or 0)
        remaining = headers.get("x-requests-remaining", remaining)
        rows = parse_event_odds(body if isinstance(body, dict) else {})
        matchup = f"{game['away_team']}@{game['home_team']}"
        if not rows:
            logger.info(f"{matchup}: no props posted yet (an empty response costs nothing)")
            continue
        if index is None:
            index = load_player_index()
        unresolved = set()
        for r in rows:
            r["player_id"], why = match_player(
                r["player_name"], (game["home_team"], game["away_team"]), index)
            if r["player_id"] is None:
                unresolved.add(f"{r['player_name']} ({why})")
        if unresolved:
            logger.warning(f"{matchup}: {len(unresolved)} player name(s) not resolved, "
                           f"stored with player_id NULL: " + ", ".join(sorted(unresolved)[:8])
                           + (" ..." if len(unresolved) > 8 else ""))
        captured_at = datetime.now(timezone.utc).replace(tzinfo=None)   # naive UTC
        for r in rows:
            r.update(captured_at=captured_at, game_id=game["game_id"], event_id=ev["id"])
        stored += write_rows(rows)
        games_with_lines += 1

    logger.info(f"{label}: stored {stored} prop line(s) for {games_with_lines} of "
                f"{len(todo)} game(s); this run cost {credits} credit(s), "
                f"{remaining} remaining")
    return stored


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.props_odds",
        description="Take one player-props snapshot from The Odds API into "
                    "raw.prop_snapshots, one request per game starting in the next "
                    "24 hours. Listing games is free; each game costs 1 credit per "
                    "market returned (up to 10 bookmakers bill as one region), and "
                    "nothing when no book has posted props yet.")
    parser.add_argument("--markets", default=None,
                        help="comma-separated Odds API prop markets, such as "
                             "player_shots_on_goal,player_points (default: "
                             "PROPS_MARKETS, else player_shots_on_goal)")
    parser.add_argument("--due", action="store_true",
                        help="pre-game snapshot for the schedule that runs every 15 "
                             "minutes: only games starting within "
                             "PROPS_CLOSE_LEAD_MINUTES (16) with no prop snapshot in "
                             "the last PROPS_CLOSE_MIN_GAP_MINUTES (16)")
    args = parser.parse_args(argv)
    return snapshot_props(markets=args.markets, due=args.due)


if __name__ == "__main__":
    main()
