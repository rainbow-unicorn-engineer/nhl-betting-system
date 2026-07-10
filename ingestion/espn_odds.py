"""
ingestion/espn_odds.py
Historical odds backfill from ESPN's public summary API (pickcenter block).
Populates: raw.historical_odds — one row per game, both moneylines + puck
line + total from a single book (DraftKings in recent seasons).

Free and unauthenticated. The line is captured near game time, so treat it
as a closing-line reference: fine for the market feature and for strategy
backtests, but our own raw.odds_snapshots time series remains the source
for opening lines and CLV going forward.

Idempotent and resumable: games already present in raw.historical_odds are
skipped, so an interrupted backfill can simply be re-run.
"""
import logging
import time
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


def _espn_name_to_abbrev(name: str) -> Optional[str]:
    return _ESPN_EXTRA_NAMES.get(name) or _TEAM_NAME_TO_ABBREV.get(name)


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


def parse_pickcenter(block: dict) -> dict:
    """Extract the columns we store from one pickcenter block."""
    home = block.get("homeTeamOdds", {}) or {}
    away = block.get("awayTeamOdds", {}) or {}
    return {
        "provider": (block.get("provider", {}) or {}).get("name"),
        "home_ml": home.get("moneyLine"),
        "away_ml": away.get("moneyLine"),
        "spread": block.get("spread"),
        "over_under": block.get("overUnder"),
        "details": block.get("details"),
    }


def _match_events_to_games(events: list, games: list) -> dict:
    """Map our game_id -> ESPN event id via home-team abbrev on the date."""
    espn_by_home = {}
    for ev in events:
        comps = (ev.get("competitions") or [{}])[0].get("competitors", [])
        for c in comps:
            if c.get("homeAway") == "home":
                abbrev = _espn_name_to_abbrev(
                    (c.get("team", {}) or {}).get("displayName", ""))
                if abbrev:
                    espn_by_home[abbrev] = ev["id"]
    return {g["game_id"]: espn_by_home[g["home_team"]]
            for g in games if g["home_team"] in espn_by_home}


def backfill_historical_odds(season: Optional[int] = None) -> int:
    """Fetch ESPN odds for every completed game missing from
    raw.historical_odds. One scoreboard call per game date, one summary
    call per game. Returns the number of rows inserted."""
    where = "AND season = :season" if season else ""
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT game_id, date, home_team FROM raw.games
            WHERE game_state IN ('FINAL', 'OFF') {where}
              AND game_id NOT IN (SELECT game_id FROM raw.historical_odds)
            ORDER BY date
        """), {"season": season} if season else {}).mappings().all()

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

            record = {"game_id": game_id, **parse_pickcenter(block)}
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO raw.historical_odds
                        (game_id, provider, home_ml, away_ml, spread,
                         over_under, details)
                    VALUES (:game_id, :provider, :home_ml, :away_ml, :spread,
                            :over_under, :details)
                    ON CONFLICT (game_id) DO NOTHING
                """), record)
            inserted += 1
            if inserted % 250 == 0:
                logger.info(f"ESPN odds backfill: {inserted} rows inserted")

    logger.info(f"ESPN odds backfill complete: {inserted} rows inserted")
    return inserted


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    season_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    backfill_historical_odds(season_arg)
