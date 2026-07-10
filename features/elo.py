"""
features/elo.py
Elo rating system (Phase 2, Task 5) — populates home_elo/away_elo in
features.matchup.

Standard Elo, parameters locked in PHASE2_PLAN.md / File 3:
- K = 20, home-ice advantage = 50 rating points (added to the home side's
  rating inside the expected-score formula only; never stored).
- Every game is processed in strict chronological order (date, then game_id
  for same-day ties). The STORED values are the PRE-game ratings — the
  update from a game's own result lands on the teams' NEXT games, so the
  feature is point-in-time correct by construction.
- A win is a win: OT/SO wins count 1.0 (standard Elo, no margin scaling).
- Cross-season carryover with regression to the mean: at a team's first
  game of a new season its rating first moves 1/3 of the way back to 1500.
- Franchise continuity: relocations inherit the predecessor's rating at the
  moment the new code first appears (ARI -> UTA in 2024-25). Expansion
  teams (e.g. SEA in 2021-22) start at 1500.
- Only FINAL games update ratings. Scheduled games still receive pre-game
  ratings, so future slates can be priced once the daily pipeline lands.

Because ratings chain through every prior game, this module always
processes FULL history — there is no per-season build.
"""
import logging
from typing import Dict, Optional, Tuple

import pandas as pd
from sqlalchemy import text

from config.settings import engine

logger = logging.getLogger("nhl.features.elo")

K = 20.0
HOME_ADVANTAGE = 50.0
INITIAL_RATING = 1500.0
SEASON_REGRESSION = 1.0 / 3.0   # pulled toward 1500 at each season start

# Relocated franchises: new code -> predecessor whose rating it inherits
FRANCHISE_CONTINUATIONS = {"UTA": "ARI"}


def expected_home_score(home_elo: float, away_elo: float,
                        home_adv: float = HOME_ADVANTAGE) -> float:
    """P(home wins) under the Elo logistic model, with home-ice bonus."""
    return 1.0 / (1.0 + 10.0 ** (-((home_elo + home_adv) - away_elo) / 400.0))


def compute_elo(games: pd.DataFrame, k: float = K,
                home_adv: float = HOME_ADVANTAGE,
                regression: float = SEASON_REGRESSION,
                initial: float = INITIAL_RATING,
                ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Pure function. `games` needs columns: game_id, season, date, home_team,
    away_team, home_score, away_score, game_state — sorted or not (sorted
    here). Returns (per-game pre-game ratings, final ratings by team).
    """
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)

    ratings: Dict[str, float] = {}
    last_season: Dict[str, int] = {}
    out = []

    for g in games.itertuples(index=False):
        pre = {}
        for team in (g.home_team, g.away_team):
            if team not in ratings:
                # Expansion team, or a relocation inheriting its predecessor.
                # Inheritance crosses a season boundary, so the inherited
                # rating gets the same regression to the mean.
                prev_code = FRANCHISE_CONTINUATIONS.get(team)
                if prev_code in ratings:
                    inherited = ratings.pop(prev_code)
                    ratings[team] = inherited + regression * (initial - inherited)
                else:
                    ratings[team] = initial
                last_season[team] = g.season
            elif last_season[team] != g.season:
                ratings[team] += regression * (initial - ratings[team])
                last_season[team] = g.season
            pre[team] = ratings[team]

        out.append((g.game_id, round(pre[g.home_team], 1),
                    round(pre[g.away_team], 1)))

        # Update AFTER recording pre-game values — only for completed games
        if g.game_state in ("FINAL", "OFF") \
                and g.home_score is not None and g.away_score is not None:
            e_home = expected_home_score(pre[g.home_team], pre[g.away_team], home_adv)
            s_home = 1.0 if g.home_score > g.away_score else 0.0
            delta = k * (s_home - e_home)
            ratings[g.home_team] += delta
            ratings[g.away_team] -= delta

    df = pd.DataFrame(out, columns=["game_id", "home_elo", "away_elo"])
    return df, ratings


def load_games() -> pd.DataFrame:
    """All games, all seasons — Elo must chain through full history."""
    with engine.connect() as conn:
        return pd.read_sql(text("""
            SELECT game_id, season, date, home_team, away_team,
                   home_score, away_score, game_state
            FROM raw.games ORDER BY date, game_id
        """), conn)


_UPSERT_SQL = text("""
    INSERT INTO features.matchup (game_id, home_team, away_team, home_elo, away_elo)
    VALUES (:game_id, :home_team, :away_team, :home_elo, :away_elo)
    ON CONFLICT (game_id) DO UPDATE SET
        home_elo = EXCLUDED.home_elo,
        away_elo = EXCLUDED.away_elo,
        computed_at = NOW()
""")


def build_elo() -> int:
    """Compute pre-game Elo for every game and upsert into features.matchup."""
    games = load_games()
    if games.empty:
        logger.warning("No games in raw.games; nothing to rate.")
        return 0

    elos, final = compute_elo(games)
    payload = elos.merge(
        games[["game_id", "home_team", "away_team"]], on="game_id")

    with engine.begin() as conn:
        conn.execute(_UPSERT_SQL, payload.to_dict("records"))

    top = sorted(final.items(), key=lambda kv: -kv[1])[:3]
    logger.info(f"Elo written for {len(payload)} games. Current top 3: "
                + ", ".join(f"{t} {r:.0f}" for t, r in top))
    return len(payload)


if __name__ == "__main__":
    build_elo()
