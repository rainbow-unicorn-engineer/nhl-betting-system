"""
ingestion/dailyfaceoff.py
Confirmed starting goalies from Daily Faceoff (Phase 4).

Fills the single biggest feature gap in the system (PROJECT_CONTEXT §9:
no public game-prediction repo conditions on the confirmed starter, and
our own totals audit showed the goalie signal that moves markets is WHICH
goalie starts, not historical form).

Source: https://www.dailyfaceoff.com/starting-goalies/[YYYY-MM-DD] — a
Next.js page whose __NEXT_DATA__ JSON carries one flat dict per game with
{home,away}GoalieName, {home,away}NewsStrengthName ('Confirmed' or a
softer status/None), and full team names (mapped to abbrevs with the same
table the odds feed uses). One request per run, identified UA, run from
the daily + odds chains — polite by construction.

Writes raw.starting_goalies keyed (game_date, team), goalie names
resolved to raw.players ids by normalized name (accent-stripped,
case-folded) with a recent-appearance team disambiguator. Unresolved
names are stored with goalie_id NULL and logged — never silently
dropped. The recommendation job prefers these rows over its
most-starts-in-last-10 heuristic and clears starter_fallback only for
'Confirmed' rows.
"""
import json
import logging
import re
import unicodedata
from datetime import date as date_cls
from typing import List, Optional

import requests
from sqlalchemy import text

from config.settings import engine
from ingestion.odds_api import _TEAM_NAME_TO_ABBREV

logger = logging.getLogger("nhl.ingestion.dailyfaceoff")

BASE_URL = "https://www.dailyfaceoff.com/starting-goalies"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")

_DDL = """
CREATE TABLE IF NOT EXISTS raw.starting_goalies (
    game_date       DATE NOT NULL,
    team            VARCHAR(3) NOT NULL,
    goalie_name     VARCHAR(80) NOT NULL,
    goalie_id       INTEGER,
    confirmation    VARCHAR(20),
    source          VARCHAR(20) NOT NULL DEFAULT 'dailyfaceoff',
    fetched_at      TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (game_date, team)
)
"""


def ensure_table() -> None:
    with engine.begin() as conn:
        conn.execute(text(_DDL))


def fetch_page(target_date: Optional[date_cls] = None) -> str:
    url = BASE_URL if target_date is None else f"{BASE_URL}/{target_date}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_starting_goalies(html: str) -> List[dict]:
    """Pure: __NEXT_DATA__ -> one row per (game_date, team). Unknown team
    names are logged and skipped (expansion/relocation -> extend the map)."""
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html, re.S)
    if not m:
        raise ValueError("No __NEXT_DATA__ payload — Daily Faceoff page "
                         "layout changed; parser needs updating")
    games = (json.loads(m.group(1)).get("props", {})
             .get("pageProps", {}).get("data") or [])

    rows = []
    for g in games:
        game_date = g.get("date")
        for side in ("home", "away"):
            name = g.get(f"{side}GoalieName")
            team_name = g.get(f"{side}TeamName", "")
            if not name or not game_date:
                continue
            abbrev = _TEAM_NAME_TO_ABBREV.get(team_name)
            if not abbrev:
                logger.warning(f"Unknown team name from Daily Faceoff: "
                               f"{team_name!r} — extend the mapping")
                continue
            rows.append({
                "game_date": game_date,
                "team": abbrev,
                "goalie_name": name.strip(),
                "confirmation": g.get(f"{side}NewsStrengthName"),
            })
    return rows


def _normalize(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name)
    return "".join(c for c in stripped if not unicodedata.combining(c)) \
        .casefold().strip()


def _initial_key(name: str) -> tuple:
    """('j', 'swayman') from 'Jeremy Swayman' OR 'J. Swayman'.

    raw.players stores ABBREVIATED names ('J. Swayman') while Daily
    Faceoff publishes full names ('Jeremy Swayman'), so exact matching
    resolves nothing (measured: 0/19 on a real slate). First-initial +
    last-name is the common key; collisions are disambiguated by team."""
    parts = _normalize(name).replace(".", "").split()
    if not parts:
        return ("", "")
    return (parts[0][0], " ".join(parts[1:]) or parts[0])


def resolve_goalie_ids(rows: List[dict]) -> List[dict]:
    """Attach raw.players ids: exact normalized name first, then
    first-initial + last-name; ambiguity broken by which candidate most
    recently appeared for the team."""
    if not rows:
        return rows
    with engine.connect() as conn:
        goalies = conn.execute(text("""
            SELECT player_id, full_name FROM raw.players WHERE position = 'G'
        """)).fetchall()
        recent = conn.execute(text("""
            SELECT DISTINCT ON (gg.player_id) gg.player_id, gg.team
            FROM raw.goalie_games gg JOIN raw.games g USING (game_id)
            ORDER BY gg.player_id, g.date DESC
        """)).fetchall()
    by_name: dict = {}
    by_key: dict = {}
    for pid, full_name in goalies:
        by_name.setdefault(_normalize(full_name), []).append(pid)
        by_key.setdefault(_initial_key(full_name), []).append(pid)
    last_team = dict(recent)

    def pick(cands: list, team: str) -> Optional[int]:
        if len(cands) == 1:
            return cands[0]
        on_team = [p for p in cands if last_team.get(p) == team]
        return on_team[0] if len(on_team) == 1 else None

    for r in rows:
        cands = by_name.get(_normalize(r["goalie_name"])) \
            or by_key.get(_initial_key(r["goalie_name"])) or []
        r["goalie_id"] = pick(cands, r["team"])
        if r["goalie_id"] is None:
            reason = "ambiguous" if cands else "not in raw.players"
            logger.warning(f"Unresolved goalie name {r['goalie_name']!r} "
                           f"({r['team']}, {reason}) — stored unresolved")
    return rows


def write_starting_goalies(rows: List[dict]) -> int:
    if not rows:
        return 0
    ensure_table()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw.starting_goalies
                (game_date, team, goalie_name, goalie_id, confirmation)
            VALUES (:game_date, :team, :goalie_name, :goalie_id, :confirmation)
            ON CONFLICT (game_date, team) DO UPDATE SET
                goalie_name = EXCLUDED.goalie_name,
                goalie_id = EXCLUDED.goalie_id,
                confirmation = EXCLUDED.confirmation,
                fetched_at = now()
        """), rows)
    return len(rows)


def ingest_starting_goalies(target_date: Optional[date_cls] = None) -> int:
    """Fetch + parse + resolve + upsert. Returns rows written."""
    html = fetch_page(target_date)
    rows = resolve_goalie_ids(parse_starting_goalies(html))
    n = write_starting_goalies(rows)
    confirmed = sum(1 for r in rows if r.get("confirmation") == "Confirmed")
    resolved = sum(1 for r in rows if r.get("goalie_id"))
    logger.info(f"Daily Faceoff: {n} starter rows "
                f"({confirmed} confirmed, {resolved} id-resolved)")
    return n


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ingest DF starting goalies")
    parser.add_argument("--date", type=date_cls.fromisoformat, default=None)
    args = parser.parse_args()
    ingest_starting_goalies(args.date)
