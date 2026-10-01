"""
ingestion/espn_odds.py
Historical odds backfill from ESPN's public summary API (pickcenter block).
Populates: raw.historical_odds — one row per game from a single book
(DraftKings from 2025-26 on, Unibet 2020-21 to 2023-24).

What a row holds:
- The closing line: both moneylines (a bet on who wins), the home puck line
  (hockey's point spread, almost always ±1.5 goals) and the total (the
  over/under goals line). These columns have always been stored.
- Since 2026-09-29 also the OPENING line and the PRICES around it: opening
  moneylines (home_ml_open, away_ml_open), the opening total (total_open),
  the closing and opening over/under prices (over_price, under_price,
  over_price_open, under_price_open), and the puck-line prices
  (spread_home_price, spread_away_price, plus spread_open and its opening
  prices). "Price" means American odds: -115 risks 115 to win 100, +102
  risks 100 to win 102. The pickcenter block always carried these; the
  first version of this loader threw them away, which is why models/totals.py
  said no historical over/under prices existed.

Where they exist (checked on the clone, 2026-09-29, and again 2026-10-01
on 24 DraftKings games spread from 2025-11-25 to the playoffs plus 3 games
per Unibet season):
- DraftKings rows (2025-26, from late November 2025 on): opening and
  closing prices for all three markets (every field on 24 of 24 games).
  The opening total is often a different line from the close (6 of those
  24: o5.5 at open, o6.5 at close), so over_price_open goes with
  total_open, never with over_under.
- Unibet rows (2020-21 to 2023-24): closing over/under and puck-line
  prices exist (opening ones only from 2023-24), but their
  capture time is unknown, and some are IN-PLAY (priced after puck drop:
  game 2023021303 is stored with a 10.5 total and a +4500 moneyline).
  Opening prices exist only from 2023-24. Never use Unibet-era prices for a
  payout backtest; filter on provider = 'DraftKings'.
- 2024-25, and October to November 2025: ESPN serves no pickcenter.
- A side ESPN marks "OFF" is stored as NULL, and a pair of over/under or
  puck-line prices that no two-way market could quote (implied
  probabilities adding up to under 1, e.g. +114 on both sides) is stored
  as NULL, NULL: see sane_pair().

Free and unauthenticated. The close is captured near puck drop, so treat it
as a closing-line reference: fine for the market feature and for strategy
backtests, but our own raw.odds_snapshots time series stays the source for
live CLV (closing-line value: did the price taken beat the final price).

Idempotent and resumable:
- The backfill skips games already in raw.historical_odds.
- --refresh re-fetches games already stored whose prices were never parsed
  (prices_fetched_at IS NULL) and fills only the new price columns, plus any
  closing column that is still empty. It never overwrites a stored closing
  line, and it never mixes books: when ESPN now reports a different book
  than the stored row (e.g. a hand-loaded Kaggle row), it records that it
  looked and leaves the prices empty. A game that fails to download is
  left unmarked and retried on the next run.

The new columns are added on first use (HISTORICAL_ODDS_COLUMNS, applied by
ensure_columns()); config/migrate.py and db/schema.sql list them too.
"""
import argparse
import logging
import math
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from sqlalchemy import text

from config.settings import engine
from ingestion.odds_api import _TEAM_NAME_TO_ABBREV

logger = logging.getLogger("nhl.ingestion.espn_odds")

SCOREBOARD_URL = "http://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
SUMMARY_URL = "http://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary"
REQUEST_PAUSE_S = 0.15  # be polite to an unauthenticated public API

# ESPN names not already covered by the shared Odds API map
_ESPN_EXTRA_NAMES = {
    "Utah Mammoth": "UTA",
}

# (schema, table, column, type) — the same shape as config/migrate.COLUMNS.
# Every column is nullable: older rows and eras without the data keep NULL.
HISTORICAL_ODDS_COLUMNS = (
    ("raw", "historical_odds", "home_ml_open", "INTEGER"),        # opening moneylines
    ("raw", "historical_odds", "away_ml_open", "INTEGER"),
    ("raw", "historical_odds", "total_open", "NUMERIC(4,1)"),     # opening total line
    ("raw", "historical_odds", "over_price", "INTEGER"),          # closing O/U prices, at over_under
    ("raw", "historical_odds", "under_price", "INTEGER"),
    ("raw", "historical_odds", "over_price_open", "INTEGER"),     # opening O/U prices, at total_open
    ("raw", "historical_odds", "under_price_open", "INTEGER"),
    ("raw", "historical_odds", "spread_home_price", "INTEGER"),   # closing puck-line prices, at spread
    ("raw", "historical_odds", "spread_away_price", "INTEGER"),
    ("raw", "historical_odds", "spread_open", "NUMERIC(4,1)"),    # opening home puck line
    ("raw", "historical_odds", "spread_home_price_open", "INTEGER"),
    ("raw", "historical_odds", "spread_away_price_open", "INTEGER"),
    ("raw", "historical_odds", "espn_event_id", "VARCHAR(12)"),   # ESPN's id for the game
    # When the price columns were last parsed. NULL = a row written before
    # they existed; --refresh fills those.
    ("raw", "historical_odds", "prices_fetched_at", "TIMESTAMP"),
)

# Price columns parse_pickcenter adds to the original six
PRICE_FIELDS = tuple(c for _, _, c, _ in HISTORICAL_ODDS_COLUMNS
                     if c not in ("espn_event_id", "prices_fetched_at"))
CLOSING_FIELDS = ("home_ml", "away_ml", "spread", "over_under", "details")

_columns_ready = False


def ensure_columns() -> None:
    """Add any missing HISTORICAL_ODDS_COLUMNS. Checks information_schema
    first, so an up-to-date database takes no table lock. Once per process."""
    global _columns_ready
    if _columns_ready:
        return
    with engine.begin() as conn:
        present = {r[0] for r in conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'raw' AND table_name = 'historical_odds'
        """))}
        for schema, table, column, ddl_type in HISTORICAL_ODDS_COLUMNS:
            if column not in present:
                logger.info(f"Schema upgrade: adding {schema}.{table}.{column}")
                conn.execute(text(f"ALTER TABLE {schema}.{table} "
                                  f"ADD COLUMN IF NOT EXISTS {column} {ddl_type}"))
    _columns_ready = True


def _espn_name_to_abbrev(name: str) -> Optional[str]:
    return _ESPN_EXTRA_NAMES.get(name) or _TEAM_NAME_TO_ABBREV.get(name)


# ── Parsing helpers (pure) ─────────────────────────────────────────

_EVEN = frozenset({"even", "ev", "evs"})
_PICK = frozenset({"pk", "pick", "pickem"})


def parse_american(value) -> Optional[int]:
    """American odds from any ESPN spelling: -115, -115.0, "-115", "+102",
    "Even"/"EV" (= +100). Anything else, and anything between -100 and +100
    (not a valid American price), is None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    else:
        s = str(value).strip().replace("−", "-").lower()
        if s in _EVEN:
            return 100
        try:
            v = float(s)
        except ValueError:
            return None
    if not math.isfinite(v) or abs(v) < 100:
        return None
    return int(round(v))


def parse_line(value) -> Optional[float]:
    """A handicap or total from any ESPN spelling: 6.5, "6.5", ".5", "o6.5",
    "u6.5", "+1.5", "-1.5", "PK" (= 0). Anything else is None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    else:
        s = str(value).strip().replace("−", "-").lower()
        if s.replace("'", "") in _PICK:
            return 0.0
        if s[:1] in ("o", "u"):
            s = s[1:]
        try:
            v = float(s)
        except ValueError:
            return None
    return v if math.isfinite(v) else None


def implied_prob(price: Optional[int]) -> Optional[float]:
    """The win probability an American price implies, margin included."""
    if price is None:
        return None
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


# A two-way market's implied probabilities add up to 1 plus the book's
# margin (the overround, 4-7% on main lines). Below 1 is impossible (both
# sides can't pay better than even); far above means the two prices are
# not from the same line or moment.
PAIR_SUM_RANGE = (1.0, 1.20)


def sane_pair(a: Optional[int], b: Optional[int]) -> tuple:
    """(a, b) when the two sides of a two-way market can be real prices at
    the same moment, else (None, None). One side alone is kept: ESPN marks a
    side it stopped offering as "OFF". Seen in ESPN data: game 2025020390
    opens its total at over +114 AND under +114 (sum 0.93)."""
    if a is None or b is None:
        return a, b
    total = implied_prob(a) + implied_prob(b)
    lo, hi = PAIR_SUM_RANGE
    return (a, b) if lo - 1e-9 <= total <= hi else (None, None)


def _dig(d, *keys):
    """d[k1][k2]... or None when any level is missing or not a dict."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _first(*values):
    return next((v for v in values if v is not None), None)


def _same_line(a: Optional[float], b: Optional[float]) -> bool:
    return a is None or b is None or abs(a - b) < 1e-9


def parse_pickcenter(block: dict) -> dict:
    """Extract the columns we store from one pickcenter block.

    Two layouts exist. DraftKings blocks (2025-26 on) carry `moneyline`,
    `pointSpread` and `total` sub-blocks, each with `open` and `close`.
    Unibet blocks (2023-24) carry a top-level `open` (over, under, total)
    and per-team `open` (moneyLine, spread, pointSpread). Either way the
    closing over/under prices come from the top-level overOdds/underOdds,
    which ESPN reports at the same moment as overUnder; the sub-block close
    is the fallback, used only when its line equals overUnder. The puck-line
    prices work the same way against `spread`. A price pair that cannot be
    real (sane_pair) is stored as NULL, NULL."""
    home = block.get("homeTeamOdds") or {}
    away = block.get("awayTeamOdds") or {}
    ml = block.get("moneyline") or {}
    ps = block.get("pointSpread") or {}
    tot = block.get("total") or {}
    top_open = block.get("open") or {}

    over_under = block.get("overUnder")
    spread = block.get("spread")
    ou_line = parse_line(over_under)
    spread_line = parse_line(spread)

    def close_if_same(sub: dict, side: str, line: Optional[float]):
        """The sub-block closing price, only when its line matches `line`."""
        if _same_line(parse_line(_dig(sub, side, "close", "line")), line):
            return _dig(sub, side, "close", "odds")
        return None

    row = {
        "provider": (block.get("provider", {}) or {}).get("name"),
        "home_ml": home.get("moneyLine"),
        "away_ml": away.get("moneyLine"),
        "spread": spread,
        "over_under": over_under,
        "details": block.get("details"),
        "home_ml_open": parse_american(_first(
            _dig(ml, "home", "open", "odds"),
            _dig(home, "open", "moneyLine", "american"))),
        "away_ml_open": parse_american(_first(
            _dig(ml, "away", "open", "odds"),
            _dig(away, "open", "moneyLine", "american"))),
        "total_open": parse_line(_first(
            _dig(tot, "over", "open", "line"),
            _dig(tot, "under", "open", "line"),
            _dig(top_open, "total", "american"))),
        "over_price": parse_american(_first(
            block.get("overOdds"), close_if_same(tot, "over", ou_line))),
        "under_price": parse_american(_first(
            block.get("underOdds"), close_if_same(tot, "under", ou_line))),
        "over_price_open": parse_american(_first(
            _dig(tot, "over", "open", "odds"),
            _dig(top_open, "over", "american"))),
        "under_price_open": parse_american(_first(
            _dig(tot, "under", "open", "odds"),
            _dig(top_open, "under", "american"))),
        "spread_home_price": parse_american(_first(
            home.get("spreadOdds"), close_if_same(ps, "home", spread_line))),
        "spread_away_price": parse_american(_first(
            away.get("spreadOdds"),
            close_if_same(ps, "away", None if spread_line is None else -spread_line))),
        "spread_open": parse_line(_first(
            _dig(ps, "home", "open", "line"),
            _dig(home, "open", "pointSpread", "american"))),
        "spread_home_price_open": parse_american(_first(
            _dig(ps, "home", "open", "odds"),
            _dig(home, "open", "spread", "american"))),
        "spread_away_price_open": parse_american(_first(
            _dig(ps, "away", "open", "odds"),
            _dig(away, "open", "spread", "american"))),
    }
    # Totals and puck lines are two-way everywhere; moneylines are not
    # checked, because Unibet-era moneylines are 3-way (sums near 0.83).
    for a, b in (("over_price", "under_price"),
                 ("over_price_open", "under_price_open"),
                 ("spread_home_price", "spread_away_price"),
                 ("spread_home_price_open", "spread_away_price_open")):
        row[a], row[b] = sane_pair(row[a], row[b])
    return row


# ── ESPN requests ──────────────────────────────────────────────────

def fetch_scoreboard(yyyymmdd: str) -> list:
    """ESPN events for one (Eastern-time) calendar date."""
    resp = requests.get(SCOREBOARD_URL, params={"dates": yyyymmdd}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("events", [])


def fetch_game_odds(espn_event_id: str) -> Optional[dict]:
    """First pickcenter block of an ESPN game summary, or None."""
    resp = requests.get(SUMMARY_URL, params={"event": espn_event_id}, timeout=30)
    resp.raise_for_status()
    pc = resp.json().get("pickcenter", [])
    return pc[0] if pc else None


def event_start(ev: dict) -> Optional[datetime]:
    """An ESPN event's scheduled start ("2026-04-06T23:00Z") as aware UTC."""
    raw = ev.get("date") or ((ev.get("competitions") or [{}])[0] or {}).get("date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def match_events(events: list, games: list) -> dict:
    """Map our game_id -> {"event_id", "start"} via the home team's abbrev
    on the date. `start` is ESPN's scheduled start (aware UTC) or None."""
    espn_by_home = {}
    for ev in events:
        comps = (ev.get("competitions") or [{}])[0].get("competitors", [])
        for c in comps:
            if c.get("homeAway") == "home":
                abbrev = _espn_name_to_abbrev(
                    (c.get("team", {}) or {}).get("displayName", ""))
                if abbrev:
                    espn_by_home[abbrev] = {"event_id": ev["id"],
                                            "start": event_start(ev)}
    return {g["game_id"]: espn_by_home[g["home_team"]]
            for g in games if g["home_team"] in espn_by_home}


def _match_events_to_games(events: list, games: list) -> dict:
    """Map our game_id -> ESPN event id via home-team abbrev on the date."""
    return {gid: m["event_id"] for gid, m in match_events(events, games).items()}


# ── Backfill (new games) ───────────────────────────────────────────

_INSERT_SQL = text(f"""
    INSERT INTO raw.historical_odds
        (game_id, provider, {', '.join(CLOSING_FIELDS)}, {', '.join(PRICE_FIELDS)},
         espn_event_id, prices_fetched_at)
    VALUES (:game_id, :provider, {', '.join(':' + c for c in CLOSING_FIELDS)},
            {', '.join(':' + c for c in PRICE_FIELDS)}, :espn_event_id, now())
    ON CONFLICT (game_id) DO NOTHING
""")


def backfill_historical_odds(season: Optional[int] = None,
                             limit: Optional[int] = None) -> int:
    """Fetch ESPN odds for every completed game missing from
    raw.historical_odds (at most `limit` games when given). One scoreboard
    call per game date, one summary call per game. Returns the number of
    rows inserted."""
    ensure_columns()
    where = "AND season = :season" if season else ""
    params = {"season": season} if season else {}
    if limit:
        params["limit"] = limit
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT game_id, date, home_team FROM raw.games
            WHERE game_state IN ('FINAL', 'OFF') {where}
              AND game_id NOT IN (SELECT game_id FROM raw.historical_odds)
            ORDER BY date, game_id
            {"LIMIT :limit" if limit else ""}
        """), params).mappings().all()

    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(dict(r))
    logger.info(f"ESPN odds backfill: {len(rows)} games over {len(by_date)} dates")

    inserted = 0
    for game_date, games in by_date.items():
        yyyymmdd = game_date.strftime("%Y%m%d")
        try:
            events = fetch_scoreboard(yyyymmdd)
        except Exception as e:
            logger.error(f"scoreboard {yyyymmdd} failed: {e}")
            continue
        time.sleep(REQUEST_PAUSE_S)

        for game_id, event_id in _match_events_to_games(events, games).items():
            try:
                block = fetch_game_odds(event_id)
            except Exception as e:
                logger.error(f"summary {event_id} (game {game_id}) failed: {e}")
                continue
            time.sleep(REQUEST_PAUSE_S)
            if block is None:
                logger.debug(f"no pickcenter for game {game_id}")
                continue

            record = {"game_id": game_id, "espn_event_id": str(event_id),
                      **parse_pickcenter(block)}
            with engine.begin() as conn:
                conn.execute(_INSERT_SQL, record)
            inserted += 1
            if inserted % 250 == 0:
                logger.info(f"ESPN odds backfill: {inserted} rows inserted")

    logger.info(f"ESPN odds backfill complete: {inserted} rows inserted")
    return inserted


# ── Refresh (games already stored) ─────────────────────────────────

_REFRESH_SQL = text(f"""
    UPDATE raw.historical_odds SET
        {', '.join(f'{c} = :{c}' for c in PRICE_FIELDS)},
        {', '.join(f'{c} = COALESCE({c}, :{c})' for c in CLOSING_FIELDS)},
        espn_event_id = :espn_event_id,
        prices_fetched_at = now()
    WHERE game_id = :game_id
""")

_MARK_SQL = text("""
    UPDATE raw.historical_odds
    SET espn_event_id = COALESCE(:espn_event_id, espn_event_id),
        prices_fetched_at = now()
    WHERE game_id = :game_id
""")


def refresh_decision(stored: dict, record: Optional[dict]) -> str:
    """What --refresh does with one stored row, given ESPN's parsed block
    (None = ESPN served no pickcenter). Pure.

    "fill": same book, write the price columns (and fill empty closing ones).
    "no_block": nothing to fill; remember that we looked.
    "provider_mismatch": ESPN now reports another book than the stored row;
    its prices must not be mixed into that row, so only remember we looked."""
    if record is None:
        return "no_block"
    if (stored.get("provider") or "") != (record.get("provider") or ""):
        return "provider_mismatch"
    return "fill"


def close_changed(stored: dict, record: dict) -> bool:
    """True when ESPN's closing moneylines now differ from the stored ones
    (both present). Counted in the log; the stored close is never changed."""
    for col in ("home_ml", "away_ml"):
        a, b = stored.get(col), record.get(col)
        if a is not None and b is not None and int(a) != int(b):
            return True
    return False


def refresh_historical_odds(season: Optional[int] = None,
                            limit: Optional[int] = None,
                            game_ids: Optional[list] = None) -> int:
    """Re-fetch games already in raw.historical_odds whose prices were never
    parsed and fill the new price columns. Resumable: each game is committed
    on its own and marked with prices_fetched_at, so an interrupted run
    continues where it stopped. Restrict with `season`, `game_ids`, and
    `limit` (games per run). Returns the number of rows filled."""
    ensure_columns()
    clauses, params = ["h.prices_fetched_at IS NULL"], {}
    if season:
        clauses.append("g.season = :season")
        params["season"] = season
    if game_ids:
        clauses.append("h.game_id = ANY(:ids)")
        params["ids"] = [int(g) for g in game_ids]
    if limit:
        params["limit"] = limit
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT h.game_id, h.provider, h.home_ml, h.away_ml, h.espn_event_id,
                   g.date, g.home_team
            FROM raw.historical_odds h JOIN raw.games g USING (game_id)
            WHERE {' AND '.join(clauses)}
            ORDER BY g.date, h.game_id
            {"LIMIT :limit" if limit else ""}
        """), params).mappings().all()

    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(dict(r))
    logger.info(f"ESPN odds refresh: {len(rows)} stored games over "
                f"{len(by_date)} dates still need prices")

    counts = {"fill": 0, "no_block": 0, "provider_mismatch": 0,
              "failed": 0, "no_event": 0, "close_changed": 0}
    for game_date, games in by_date.items():
        event_ids = {g["game_id"]: g["espn_event_id"] for g in games if g["espn_event_id"]}
        need_ids = [g for g in games if not g["espn_event_id"]]
        if need_ids:
            yyyymmdd = game_date.strftime("%Y%m%d")
            try:
                events = fetch_scoreboard(yyyymmdd)
            except Exception as e:
                logger.error(f"scoreboard {yyyymmdd} failed: {e}")
                counts["failed"] += len(need_ids)
                games = [g for g in games if g["espn_event_id"]]   # retried next run
            else:
                time.sleep(REQUEST_PAUSE_S)
                event_ids.update(_match_events_to_games(events, need_ids))

        for g in games:
            event_id = event_ids.get(g["game_id"])
            if not event_id:
                # Not on ESPN's scoreboard for its date: remember we looked
                counts["no_event"] += 1
                with engine.begin() as conn:
                    conn.execute(_MARK_SQL, {"game_id": g["game_id"], "espn_event_id": None})
                continue
            try:
                block = fetch_game_odds(event_id)
            except Exception as e:
                logger.error(f"summary {event_id} (game {g['game_id']}) failed: {e}")
                counts["failed"] += 1   # left unmarked: retried next run
                continue
            time.sleep(REQUEST_PAUSE_S)

            record = parse_pickcenter(block) if block is not None else None
            action = refresh_decision(g, record)
            counts[action] += 1
            with engine.begin() as conn:
                if action == "fill":
                    if close_changed(g, record):
                        counts["close_changed"] += 1
                    conn.execute(_REFRESH_SQL, {**record, "game_id": g["game_id"],
                                                "espn_event_id": str(event_id)})
                else:
                    conn.execute(_MARK_SQL, {"game_id": g["game_id"],
                                             "espn_event_id": str(event_id)})
            done = counts["fill"] + counts["no_block"] + counts["provider_mismatch"]
            if done % 250 == 0:
                logger.info(f"ESPN odds refresh: {done} games processed")

    if counts["provider_mismatch"]:
        logger.warning(f"ESPN odds refresh: {counts['provider_mismatch']} stored rows "
                       f"come from a different book than ESPN now reports; their "
                       f"prices were left empty rather than mixed")
    if counts["close_changed"]:
        logger.warning(f"ESPN odds refresh: {counts['close_changed']} games have a "
                       f"different closing moneyline on ESPN now; the stored close "
                       f"was kept")
    if counts["failed"]:
        logger.error(f"ESPN odds refresh: {counts['failed']} games failed to download "
                     f"and will be retried on the next run")
    logger.info(f"ESPN odds refresh complete: {counts['fill']} filled, "
                f"{counts['no_block']} without a pickcenter, "
                f"{counts['no_event']} not on ESPN's scoreboard, "
                f"{counts['provider_mismatch']} from another book")
    return counts["fill"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.espn_odds",
        description="Backfill ESPN reference lines (free, no key) for completed "
                    "games missing from raw.historical_odds, with opening, "
                    "over/under and puck-line prices. Resumable. With "
                    "--refresh, re-fetch games already stored to fill those "
                    "prices instead.")
    parser.add_argument("season", nargs="?", type=int, default=None,
                        help="one season such as 20252026 (default: every season)")
    parser.add_argument("--season", dest="season_opt", type=int, default=None,
                        metavar="SEASON", help="the same as the positional season")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch games already stored whose prices were "
                             "never parsed, and fill the price columns")
    parser.add_argument("--limit", type=int, default=None,
                        help="at most this many games in this run")
    args = parser.parse_args(argv)
    if args.season and args.season_opt and args.season != args.season_opt:
        parser.error("two different seasons given")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be 1 or more")
    season = args.season or args.season_opt
    if args.refresh:
        return refresh_historical_odds(season, limit=args.limit)
    return backfill_historical_odds(season, limit=args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
