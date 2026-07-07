"""
ingestion/odds_api.py
Odds ingestion from The Odds API (the-odds-api.com). Free tier: 500 req/month.
Populates: raw.odds_snapshots
"""
import logging
from datetime import datetime
from typing import Optional

import requests
from sqlalchemy import text

from config.settings import engine, ODDS_API_KEY

logger = logging.getLogger("nhl.ingestion.odds_api")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"

MARKETS = {"h2h": "ml", "spreads": "pl", "totals": "total"}


def fetch_current_odds(markets: str = "h2h,spreads,totals") -> list:
    """Fetch current odds for all upcoming NHL games."""
    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY not set. Add it to .env")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "sport": SPORT,
        "markets": markets,
        "regions": "us,us2",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(f"{BASE_URL}/sports/{SPORT}/odds", params=params, timeout=30)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.info(f"Odds API: fetched {len(resp.json())} games. Requests remaining: {remaining}")
        return resp.json()
    except Exception as e:
        logger.error(f"Odds API fetch failed: {e}")
        return []


def snapshot_odds(game_odds: Optional[list] = None):
    """Take a snapshot of current odds and store in raw.odds_snapshots."""
    if game_odds is None:
        game_odds = fetch_current_odds()

    if not game_odds:
        return 0

    now = datetime.utcnow()
    inserted = 0

    with engine.begin() as conn:
        for game in game_odds:
            commence = game.get("commence_time", "")
            if not commence:
                continue

            try:
                game_date = datetime.fromisoformat(commence.replace("Z", "+00:00")).date()
            except (ValueError, TypeError):
                continue

            home_team_name = game.get("home_team", "")
            away_team_name = game.get("away_team", "")

            game_id = _resolve_game_id(conn, game_date, home_team_name)
            if not game_id:
                logger.debug(f"Could not resolve game_id for {home_team_name} vs {away_team_name} on {game_date}")
                continue

            for bookmaker in game.get("bookmakers", []):
                book_name = bookmaker.get("key", "unknown")

                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")
                    market_type = MARKETS.get(market_key, market_key)
                    outcomes = {o.get("name", ""): o for o in market.get("outcomes", [])}

                    record = {
                        "game_id": game_id, "captured_at": now, "book_name": book_name,
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

    logger.info(f"Stored {inserted} odds snapshots across {len(game_odds)} games")
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
    "Utah Hockey Club": "UTA", "Utah HC": "UTA",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}


def _resolve_game_id(conn, game_date, home_team_name: str) -> Optional[int]:
    """Look up our game_id given a date and home team name."""
    abbrev = _TEAM_NAME_TO_ABBREV.get(home_team_name, "")
    if not abbrev:
        return None

    result = conn.execute(text("""
        SELECT game_id FROM raw.games WHERE date = :d AND home_team = :t LIMIT 1
    """), {"d": game_date, "t": abbrev})
    row = result.fetchone()
    return row[0] if row else None


if __name__ == "__main__":
    snapshot_odds()
