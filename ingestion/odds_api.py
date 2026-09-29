"""
ingestion/odds_api.py
Odds ingestion from The Odds API (the-odds-api.com). Free tier: 500 credits/month.
Populates: raw.odds_snapshots

Cost: one credit per market per region. With regions=us,us2 a full
snapshot (h2h,spreads,totals) costs 6 credits; a moneyline-only snapshot
(markets="h2h", the `pipeline.py close` run) costs 2. A request is free
only when the API lists no NHL events at all; in season it lists later
days' games too, so snapshot_odds skips the request (logging why) unless
raw.games has a game starting in the next SNAPSHOT_HORIZON (24 hours).

Closing timing: close_due() decides whether `pipeline.py close --due`
(run every 15 minutes) should take its snapshot now: some game starts
within CLOSE_LEAD (default 16 minutes) and no moneyline snapshot was taken
in the last CLOSE_MIN_GAP (default 16 minutes). With a 15-minute cycle
that is one close per start time, in the last cycle before puck drop.

Matching: an event is tied to raw.games by home team AND start time — the
row for that home team whose start_time_utc is within MATCH_WINDOW of
commence_time, nearest first. raw.games.date is the league's local
(Eastern) date, so a 7pm ET January game has its commence_time on the
NEXT UTC day; matching on the UTC date lost most evening games. Rows with
no start time yet fall back to the Eastern date of commence_time.

In-play: the /odds endpoint keeps listing games after puck drop, with live
prices. Those events are skipped — an in-play price must never become a
pick's price or its closing line.

The API key never reaches the logs: failures are summarized (status,
reason, the API's own message) and anything logged passes through _redact().
"""
import argparse
import logging
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import text

from config.migrate import ensure_schema
from config.settings import engine, ODDS_API_KEY

logger = logging.getLogger("nhl.ingestion.odds_api")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"
TIMEOUT_S = 30
PLACEHOLDER_KEY = "your_key_here"   # the .env.example value counts as unset

MARKETS = {"h2h": "ml", "spreads": "pl", "totals": "total"}

MATCH_WINDOW = timedelta(hours=6)
SCHEDULE_TZ = ZoneInfo("America/New_York")   # raw.games.date is an Eastern date

SNAPSHOT_HORIZON = timedelta(hours=24)   # no game in this window = no request


def _minutes_setting(name: str, default: float, allow_zero: bool) -> timedelta:
    """A number of minutes from the environment. Unset or blank = default;
    a malformed or negative value (or 0, unless allow_zero) logs an error
    and uses default, so a typo in .env can't abort the import, and with
    it the whole daily chain."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return timedelta(minutes=default)
    try:
        value = float(raw)
        if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError(raw)
        return timedelta(minutes=value)
    except (ValueError, OverflowError):
        logger.error(f"{name}={raw!r} is not a number of minutes "
                     f"({'0 or more' if allow_zero else 'more than 0'}) — "
                     f"using the default, {default:g}")
        return timedelta(minutes=default)


# close --due: a game starts within CLOSE_LEAD and no ml snapshot is younger
# than CLOSE_MIN_GAP. With the 15-minute cycle, 16/16 takes one close per
# start time, in the last cycle before puck drop (1 to 16 minutes before
# it): a 16-minute lead always holds one cycle, and the 16-minute gap blocks
# the next cycle 15 minutes later (the extra minute absorbs a run that
# starts late). Start times under 16 minutes apart can share the earlier
# one's close. Replayed over the 2026-27 schedule, the daily run plus these
# closes stays at or under 456 credits in every calendar month (UTC),
# whatever minute the cycle starts on (40/25, the old default, needed up to
# 658). README.md, "Snapshot schedule", has the numbers.
CLOSE_LEAD = _minutes_setting("CLOSE_LEAD_MINUTES", 16, allow_zero=False)
CLOSE_MIN_GAP = _minutes_setting("CLOSE_MIN_GAP_MINUTES", 16, allow_zero=True)

_APIKEY_QUERY = re.compile(r"(apiKey=)[^&\s\"']+", re.IGNORECASE)


def _redact(value, key: Optional[str] = None) -> str:
    """str(value) with the API key and any apiKey=... query value as ***."""
    s = str(value)
    key = (ODDS_API_KEY or "").strip() if key is None else key
    if key:
        s = s.replace(key, "***")
    return _APIKEY_QUERY.sub(r"\1***", s)


class _RedactFilter(logging.Filter):
    """urllib3 logs each request URL (with ?apiKey=...) at DEBUG; scrub it
    so LOG_LEVEL=DEBUG cannot leak the key."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        clean = _redact(msg)
        if clean != msg:
            record.msg, record.args = clean, ()
        return True


logging.getLogger("urllib3.connectionpool").addFilter(_RedactFilter())


def _api_key() -> str:
    key = (ODDS_API_KEY or "").strip()
    return "" if key == PLACEHOLDER_KEY else key


def _http_error_summary(resp) -> str:
    """'HTTP 401 Unauthorized: <the API's message>' — never the URL."""
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
        # redact BEFORE truncating so a cut can't leave part of the key
        message = _redact(resp.text or "")[:200]
    return f"HTTP {resp.status_code} {resp.reason}: {_redact(message)}"


def fetch_current_odds(markets: str = "h2h,spreads,totals") -> list:
    """Fetch current odds for every NHL game the API lists (upcoming AND
    in-play). One credit per market per region: the default = 6 credits,
    markets="h2h" = 2. Returns [] on any failure."""
    key = _api_key()
    if not key:
        logger.error("ODDS_API_KEY is not set (missing, or still the .env.example "
                     "placeholder) — no odds request made. Add your key to .env")
        return []

    params = {
        "apiKey": key,
        "sport": SPORT,
        "markets": markets,
        "regions": "us,us2",
        "oddsFormat": "american",
    }

    # Never log the exception itself: requests puts the full URL, apiKey
    # included, into HTTPError and ConnectionError messages.
    try:
        resp = requests.get(f"{BASE_URL}/sports/{SPORT}/odds", params=params, timeout=TIMEOUT_S)
        resp.raise_for_status()
        games = resp.json()
    except requests.HTTPError as e:
        logger.error(f"Odds API request failed: {_http_error_summary(e.response)}")
        return []
    except requests.Timeout:
        logger.error(f"Odds API request failed: timed out after {TIMEOUT_S}s")
        return []
    except requests.ConnectionError as e:
        logger.error(f"Odds API request failed: could not reach api.the-odds-api.com "
                     f"({type(e).__name__})")
        return []
    except Exception as e:
        logger.error(f"Odds API request failed: {type(e).__name__}: {_redact(e)}")
        return []

    h = resp.headers
    logger.info(f"Odds API: fetched {len(games)} games (markets={markets}). "
                f"Credits remaining {h.get('x-requests-remaining', '?')}, "
                f"used {h.get('x-requests-used', '?')}, "
                f"this call cost {h.get('x-requests-last', '?')}")
    return games


def parse_commence(value) -> Optional[datetime]:
    """Odds API commence_time ("2026-01-16T00:00:00Z") -> aware UTC datetime;
    None when missing or malformed."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def match_event(event: dict, games: list, now: datetime) -> tuple:
    """(game_id | None, status) for one Odds API event. Pure — no DB.

    games: dicts with game_id, home_team, date (Eastern schedule date) and
    start_time_utc (aware datetime or None). now: aware datetime.
    status: "ok"; "started" (commence_time <= now — in-play, never stored);
    "no_time"; "unknown_team"; "no_game".
    """
    commence = parse_commence(event.get("commence_time"))
    if commence is None:
        return None, "no_time"
    if commence <= now:
        return None, "started"
    abbrev = _TEAM_NAME_TO_ABBREV.get(event.get("home_team", ""))
    if not abbrev:
        return None, "unknown_team"

    mine = [g for g in games if g["home_team"] == abbrev]
    timed = sorted((abs(g["start_time_utc"] - commence), g["game_id"])
                   for g in mine if g.get("start_time_utc") is not None)
    if timed and timed[0][0] <= MATCH_WINDOW:
        return timed[0][1], "ok"

    # Only rows with no start time yet may match by date
    eastern_date = commence.astimezone(SCHEDULE_TZ).date()
    for g in mine:
        if g.get("start_time_utc") is None and g["date"] == eastern_date:
            return g["game_id"], "ok"
    return None, "no_game"


def _candidate_games(conn, events: list) -> list:
    """raw.games rows that could match these events: same home teams,
    Eastern dates within a day of any commence_time."""
    teams, dates = set(), []
    for ev in events:
        abbrev = _TEAM_NAME_TO_ABBREV.get(ev.get("home_team", ""))
        commence = parse_commence(ev.get("commence_time"))
        if abbrev and commence:
            teams.add(abbrev)
            dates.append(commence.astimezone(SCHEDULE_TZ).date())
    if not teams:
        return []
    rows = conn.execute(text("""
        SELECT game_id, home_team, date, start_time_utc FROM raw.games
        WHERE home_team = ANY(:teams) AND date BETWEEN :lo AND :hi
    """), {"teams": sorted(teams), "lo": min(dates) - timedelta(days=1),
           "hi": max(dates) + timedelta(days=1)}).mappings().all()
    return [dict(r) for r in rows]


def upcoming_start_times(now: datetime, horizon: timedelta) -> list:
    """Puck drops (aware UTC) in (now, now + horizon] from raw.games,
    skipping postponed, suspended and cancelled games. Instants, not
    dates, so a start just after UTC midnight is no special case."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT start_time_utc FROM raw.games
            WHERE start_time_utc > :now AND start_time_utc <= :until
              AND COALESCE(schedule_state, 'OK') NOT IN ('PPD', 'SUSP', 'CNCL')
            ORDER BY start_time_utc
        """), {"now": now, "until": now + horizon}).fetchall()
    return [r[0] for r in rows]


def last_ml_capture() -> Optional[datetime]:
    """captured_at of the newest moneyline snapshot, as aware UTC (stored
    naive UTC); None when there is none."""
    with engine.connect() as conn:
        t = conn.execute(text("""
            SELECT MAX(captured_at) FROM raw.odds_snapshots WHERE market_type = 'ml'
        """)).scalar()
    return None if t is None else t.replace(tzinfo=timezone.utc)


def close_due(starts: list, last_capture: Optional[datetime], now: datetime,
              lead: timedelta = CLOSE_LEAD,
              min_gap: timedelta = CLOSE_MIN_GAP) -> tuple:
    """(due, reason) for `pipeline.py close --due`. Pure — no DB.

    starts: puck drops (aware); last_capture: newest ml snapshot (aware or
    None); now: aware. Due when some game starts in (now, now + lead] and
    no ml snapshot was captured in the last min_gap."""
    soon = sorted(s for s in starts if s is not None and now < s <= now + lead)
    if not soon:
        return False, (f"no game starts in the next "
                       f"{int(lead.total_seconds() // 60)} minutes")
    if last_capture is not None and now - last_capture < min_gap:
        ago = int((now - last_capture).total_seconds() // 60)
        return False, (f"a moneyline snapshot was taken {ago} minute(s) ago "
                       f"(less than {int(min_gap.total_seconds() // 60)})")
    first = int((soon[0] - now).total_seconds() // 60)
    return True, f"{len(soon)} game(s) start soon, the first in {first} minute(s)"


def close_is_due(now: Optional[datetime] = None) -> tuple:
    """close_due() on the database's start times and snapshots."""
    ensure_schema()
    now = now or datetime.now(timezone.utc)
    return close_due(upcoming_start_times(now, CLOSE_LEAD), last_ml_capture(), now)


def snapshot_odds(game_odds: Optional[list] = None, markets: str = "h2h,spreads,totals"):
    """Take a snapshot of current odds and store in raw.odds_snapshots.
    Events already under way are skipped. `markets` goes to
    fetch_current_odds ("h2h" = the 2-credit closing-line snapshot).
    A live request is made only when raw.games has a game starting in the
    next SNAPSHOT_HORIZON: otherwise it would bill credits just for
    later days' games."""
    ensure_schema()
    if game_odds is None:
        if not upcoming_start_times(datetime.now(timezone.utc), SNAPSHOT_HORIZON):
            logger.info("No game in raw.games starts in the next 24 hours: "
                        "skipping the Odds API request, which would still cost "
                        "credits for later days' games. (If games are missing, "
                        "refresh the schedule: python pipeline.py daily)")
            return 0
        game_odds = fetch_current_odds(markets=markets)

    if not game_odds:
        return 0

    now = datetime.now(timezone.utc)
    captured_at = now.replace(tzinfo=None)   # stored as naive UTC
    inserted, matched, in_play, unresolved = 0, 0, 0, []

    with engine.begin() as conn:
        candidates = _candidate_games(conn, game_odds)
        for game in game_odds:
            home_team_name = game.get("home_team", "")
            away_team_name = game.get("away_team", "")

            game_id, status = match_event(game, candidates, now)
            if status == "started":
                in_play += 1
                continue
            if game_id is None:
                unresolved.append(f"{away_team_name} @ {home_team_name} "
                                  f"{game.get('commence_time', '?')} ({status})")
                continue
            matched += 1

            for bookmaker in game.get("bookmakers", []):
                book_name = bookmaker.get("key", "unknown")

                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")
                    market_type = MARKETS.get(market_key, market_key)
                    outcomes = {o.get("name", ""): o for o in market.get("outcomes", [])}

                    record = {
                        "game_id": game_id, "captured_at": captured_at, "book_name": book_name,
                        "market_type": market_type, "home_price": None, "away_price": None,
                        "over_price": None, "under_price": None, "line": None,
                    }

                    if market_type == "ml":
                        record["home_price"] = outcomes.get(home_team_name, {}).get("price")
                        record["away_price"] = outcomes.get(away_team_name, {}).get("price")
                    elif market_type == "pl":
                        home_out = outcomes.get(home_team_name, {})
                        away_out = outcomes.get(away_team_name, {})
                        record["home_price"] = home_out.get("price")
                        record["away_price"] = away_out.get("price")
                        record["line"] = home_out.get("point")
                    elif market_type == "total":
                        over_out = outcomes.get("Over", {})
                        under_out = outcomes.get("Under", {})
                        record["over_price"] = over_out.get("price")
                        record["under_price"] = under_out.get("price")
                        record["line"] = over_out.get("point")

                    conn.execute(text("""
                        INSERT INTO raw.odds_snapshots
                            (game_id, captured_at, book_name, market_type,
                             home_price, away_price, over_price, under_price, line)
                        VALUES (:game_id, :captured_at, :book_name, :market_type,
                                :home_price, :away_price, :over_price, :under_price, :line)
                    """), record)
                    inserted += 1

    if in_play:
        logger.info(f"Skipped {in_play} in-play event(s): prices after puck drop are never stored")
    if unresolved:
        logger.warning(f"{len(unresolved)} Odds API event(s) matched no raw.games row "
                       f"(schedule not ingested that far ahead, or a team/start-time mismatch): "
                       + "; ".join(unresolved[:5]) + (" ..." if len(unresolved) > 5 else ""))
    logger.info(f"Stored {inserted} odds snapshots across {matched} of {len(game_odds)} games")
    return inserted


_TEAM_NAME_TO_ABBREV = {
    "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF", "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL", "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA", "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL", "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH", "New Jersey Devils": "NJD",
    "New York Islanders": "NYI", "New York Rangers": "NYR",
    "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS", "Seattle Kraken": "SEA", "St Louis Blues": "STL",
    "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA", "Utah HC": "UTA", "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.odds_api",
        description="Take one Odds API snapshot into raw.odds_snapshots. "
                    "Costs 2 credits per market (regions us,us2): 6 for the "
                    "default markets, 2 for h2h. Skipped, at no cost, when no "
                    "game starts in the next 24 hours.")
    parser.add_argument("--markets", default="h2h,spreads,totals",
                        help="comma-separated Odds API markets "
                             "(default: h2h,spreads,totals)")
    args = parser.parse_args(argv)
    return snapshot_odds(markets=args.markets)


if __name__ == "__main__":
    main()
