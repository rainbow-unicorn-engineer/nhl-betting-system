"""
features/build_vectors.py
game_vector materialization (Phase 2, Task 6) — the model-ready table.

Joins team_rolling (both teams, all windows) + goalie_rolling (both starters,
all windows) + matchup (schedule/travel/Elo) into ONE ordered numeric vector
per game, stored as parallel feature_names[] / feature_vector[] arrays in
features.game_vector.

Representation (locked by the plan): HOME − AWAY differentials.
- Team stats: 14 rolling stats × 5 windows as diffs, plus each side's
  games_played per window so the model can weigh window reliability.
- Goalie stats: the Buhlmann-shrunk metrics (shrunk_sv_pct, shrunk_gsax)
  and credibility_z as diffs per window. Shrunk metrics are never NULL by
  construction, which is what keeps early-season vectors complete.
- Context: Elo diff, rest diff, per-side b2b flags, travel diff, per-side
  tz shifts, game-number diff, season-stage one-hots (MID is the base).

NULL policy: a differential with a missing side carries no directional
information, so it is imputed to 0.0 (= "no edge") rather than NULL; the
paired games_played / b2b / stage features tell the model when windows are
thin. Completed games therefore always produce fully-finite vectors.

Starters: the pre-game starter is taken as the goalie who actually started
(raw.goalie_games.is_starter) — the standard historical-training proxy for
the pre-game confirmed starter (live starters arrive with the Daily Faceoff
scraper in a later phase). If a game somehow has no flagged starter, the
goalie with the most TOI is used and starter_fallback_{home,away} is set to
1.0 in the vector. Starter ids are also written back to features.matchup.

Market feature: the plan calls for the opening no-vig implied probability
from raw.odds_snapshots. That table is empty for the whole backfill window
(odds collection starts with the daily pipeline), so the market feature is
deliberately NOT added yet — an all-NULL column would poison every vector.
Add it here once snapshots exist; the baseline (Task 8) trains without it.

Labels: home_win = home_score > away_score for FINAL games. Only FINAL
games are materialized for now (scheduled-game vectors need confirmed
starters, which arrive with the daily pipeline).
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import engine
from features.util import WINDOWS

logger = logging.getLogger("nhl.features.vectors")

TEAM_STATS = ["gf_per60", "ga_per60", "xgf_pct", "cf_pct", "ff_pct",
              "sh_pct", "sv_pct", "pdo", "pp_pct", "pk_pct",
              "pp_xgf_per60", "pk_xga_per60", "fow_pct", "pim_per60"]
GOALIE_STATS = ["shrunk_sv_pct", "shrunk_gsax", "credibility_z"]


def feature_names() -> list:
    """The canonical ordered feature list. Storage order == this order."""
    names = []
    for w in WINDOWS:
        names += [f"{s}_diff_w{w}" for s in TEAM_STATS]
        names += [f"gp_home_w{w}", f"gp_away_w{w}"]
    for w in WINDOWS:
        names += [f"goalie_{s}_diff_w{w}" for s in GOALIE_STATS]
    names += ["starter_fallback_home", "starter_fallback_away"]
    names += ["elo_diff", "rest_diff", "b2b_home", "b2b_away",
              "travel_diff", "tz_home", "tz_away", "game_num_diff",
              "stage_early", "stage_late"]
    return names


FEATURE_NAMES = feature_names()


def _load_games(season: Optional[int]) -> pd.DataFrame:
    season_filter = "AND g.season = :season" if season else ""
    with engine.connect() as conn:
        return pd.read_sql(text(f"""
            SELECT g.game_id, g.season, g.date, g.home_team, g.away_team,
                   g.home_score, g.away_score,
                   m.home_rest_days, m.away_rest_days, m.home_b2b, m.away_b2b,
                   m.home_travel_km, m.away_travel_km,
                   m.home_tz_shift, m.away_tz_shift,
                   m.home_game_num, m.away_game_num, m.season_stage,
                   m.home_elo, m.away_elo
            FROM raw.games g
            JOIN features.matchup m USING (game_id)
            WHERE g.game_state IN ('FINAL', 'OFF') {season_filter}
            ORDER BY g.date, g.game_id
        """), conn, params={"season": season} if season else {})


def _load_team_wide(season: Optional[int]) -> pd.DataFrame:
    """team_rolling pivoted to one row per (game_id, team), stat_w{w} cols."""
    season_filter = "AND g.season = :season" if season else ""
    with engine.connect() as conn:
        tr = pd.read_sql(text(f"""
            SELECT tr.* FROM features.team_rolling tr
            JOIN raw.games g USING (game_id)
            WHERE TRUE {season_filter}
        """), conn, params={"season": season} if season else {})
    tr = tr.astype({c: float for c in TEAM_STATS + ["games_played"]})
    wide = tr.pivot(index=["game_id", "team"], columns="window_size",
                    values=TEAM_STATS + ["games_played"])
    wide.columns = [f"{stat}_w{w}" for stat, w in wide.columns]
    return wide.reset_index()


def _load_starters(season: Optional[int]) -> pd.DataFrame:
    """One row per (game_id, team): the starter (flagged, else max TOI)."""
    season_filter = "AND g.season = :season" if season else ""
    with engine.connect() as conn:
        return pd.read_sql(text(f"""
            SELECT DISTINCT ON (gg.game_id, gg.team)
                   gg.game_id, gg.team, gg.player_id AS goalie_id,
                   (NOT gg.is_starter)::int AS starter_fallback
            FROM raw.goalie_games gg
            JOIN raw.games g USING (game_id)
            WHERE gg.toi_seconds > 0 {season_filter}
            ORDER BY gg.game_id, gg.team,
                     gg.is_starter DESC, gg.toi_seconds DESC
        """), conn, params={"season": season} if season else {})


def _load_goalie_wide(season: Optional[int]) -> pd.DataFrame:
    """goalie_rolling pivoted to one row per (game_id, goalie_id)."""
    season_filter = "AND g.season = :season" if season else ""
    with engine.connect() as conn:
        gr = pd.read_sql(text(f"""
            SELECT gr.game_id, gr.goalie_id, gr.window_size,
                   gr.shrunk_sv_pct, gr.shrunk_gsax, gr.credibility_z
            FROM features.goalie_rolling gr
            JOIN raw.games g USING (game_id)
            WHERE TRUE {season_filter}
        """), conn, params={"season": season} if season else {})
    gr = gr.astype({c: float for c in GOALIE_STATS})
    wide = gr.pivot(index=["game_id", "goalie_id"], columns="window_size",
                    values=GOALIE_STATS)
    wide.columns = [f"{stat}_w{w}" for stat, w in wide.columns]
    return wide.reset_index()


def _diff(home: pd.Series, away: pd.Series) -> pd.Series:
    """Home − away differential; a missing side means no signal -> 0.0."""
    return (home.astype(float) - away.astype(float)).fillna(0.0)


def assemble(games, team_wide, starters, goalie_wide) -> pd.DataFrame:
    """
    Pure join/derive step: one row per game with every FEATURE_NAMES column,
    plus game_id/season/date/home_win/home_goals/away_goals.
    """
    df = games.copy()

    # Team rolling features, both sides
    home_tr = team_wide.add_prefix("h_")
    away_tr = team_wide.add_prefix("a_")
    df = df.merge(home_tr, left_on=["game_id", "home_team"],
                  right_on=["h_game_id", "h_team"], how="left")
    df = df.merge(away_tr, left_on=["game_id", "away_team"],
                  right_on=["a_game_id", "a_team"], how="left")

    feats = {}
    for w in WINDOWS:
        for s in TEAM_STATS:
            feats[f"{s}_diff_w{w}"] = _diff(df[f"h_{s}_w{w}"], df[f"a_{s}_w{w}"])
        feats[f"gp_home_w{w}"] = df[f"h_games_played_w{w}"].fillna(0.0)
        feats[f"gp_away_w{w}"] = df[f"a_games_played_w{w}"].fillna(0.0)

    # Starters, then their goalie rolling features
    for side, team_col in (("h", "home_team"), ("a", "away_team")):
        st = starters.rename(columns={
            "goalie_id": f"{side}_goalie_id",
            "starter_fallback": f"{side}_starter_fallback"})
        df = df.merge(st[["game_id", "team", f"{side}_goalie_id",
                          f"{side}_starter_fallback"]],
                      left_on=["game_id", team_col],
                      right_on=["game_id", "team"], how="left").drop(columns="team")
        gw = goalie_wide.add_prefix(f"{side}g_")
        df = df.merge(gw, left_on=["game_id", f"{side}_goalie_id"],
                      right_on=[f"{side}g_game_id", f"{side}g_goalie_id"],
                      how="left")

    for w in WINDOWS:
        for s in GOALIE_STATS:
            feats[f"goalie_{s}_diff_w{w}"] = _diff(
                df[f"hg_{s}_w{w}"], df[f"ag_{s}_w{w}"])
    feats["starter_fallback_home"] = df["h_starter_fallback"].fillna(1.0).astype(float)
    feats["starter_fallback_away"] = df["a_starter_fallback"].fillna(1.0).astype(float)

    # Context features
    feats["elo_diff"] = _diff(df["home_elo"], df["away_elo"])
    feats["rest_diff"] = _diff(df["home_rest_days"], df["away_rest_days"])
    feats["b2b_home"] = df["home_b2b"].fillna(False).astype(float)
    feats["b2b_away"] = df["away_b2b"].fillna(False).astype(float)
    feats["travel_diff"] = _diff(df["home_travel_km"], df["away_travel_km"])
    feats["tz_home"] = df["home_tz_shift"].fillna(0).astype(float)
    feats["tz_away"] = df["away_tz_shift"].fillna(0).astype(float)
    feats["game_num_diff"] = _diff(df["home_game_num"], df["away_game_num"])
    feats["stage_early"] = (df["season_stage"] == "EARLY").astype(float)
    feats["stage_late"] = (df["season_stage"] == "LATE").astype(float)

    # Label
    feats["home_win"] = df["home_score"] > df["away_score"]
    feats["home_goals"] = df["home_score"]
    feats["away_goals"] = df["away_score"]

    df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)

    keep = ["game_id", "season", "date", "home_win", "home_goals",
            "away_goals", "h_goalie_id", "a_goalie_id"] + FEATURE_NAMES
    return df[keep]


def write_vectors(df: pd.DataFrame) -> int:
    """Replace game_vector rows (and matchup starter ids) per season."""
    seasons = sorted(df["season"].unique().tolist())
    matrix = df[FEATURE_NAMES].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        bad = int((~np.isfinite(matrix)).sum())
        raise ValueError(f"{bad} non-finite vector elements — refusing to write")

    records = [
        {"game_id": int(r.game_id), "season": int(r.season), "date": r.date,
         "names": FEATURE_NAMES, "vector": [float(x) for x in vec],
         "home_win": bool(r.home_win), "home_goals": int(r.home_goals),
         "away_goals": int(r.away_goals),
         "h_goalie": int(r.h_goalie_id) if pd.notna(r.h_goalie_id) else None,
         "a_goalie": int(r.a_goalie_id) if pd.notna(r.a_goalie_id) else None}
        for r, vec in zip(df.itertuples(index=False), matrix)
    ]

    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM features.game_vector WHERE season = ANY(:seasons)
        """), {"seasons": seasons})
        conn.execute(text("""
            INSERT INTO features.game_vector
                (game_id, season, date, feature_names, feature_vector,
                 home_win, home_goals, away_goals)
            VALUES (:game_id, :season, :date, :names, :vector,
                    :home_win, :home_goals, :away_goals)
        """), records)
        conn.execute(text("""
            UPDATE features.matchup SET
                home_starter_id = :h_goalie,
                away_starter_id = :a_goalie
            WHERE game_id = :game_id
        """), records)

    logger.info(f"Wrote {len(records)} game vectors "
                f"({len(FEATURE_NAMES)} features) for seasons {seasons}")
    return len(records)


def build_game_vectors(season: Optional[int] = None) -> int:
    """Materialize features.game_vector for one season or all seasons."""
    scope = season or "ALL"
    logger.info(f"Building game vectors (season={scope})...")
    games = _load_games(season)
    if games.empty:
        logger.warning(f"No completed games with matchup rows (season={scope}).")
        return 0
    df = assemble(games, _load_team_wide(season),
                  _load_starters(season), _load_goalie_wide(season))
    return write_vectors(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build features.game_vector")
    parser.add_argument("--season", type=int, default=None,
                        help="Season as YYYYYYYY (e.g. 20212022); omit for all")
    args = parser.parse_args()
    build_game_vectors(args.season)
