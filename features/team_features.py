"""
features/team_features.py
Team rolling features (Phase 2, Task 2) — populates features.team_rolling.

One row per (game_id, team, window_size). Every value is computed from games
STRICTLY PRIOR to the target game (ordered by date, then game_id for same-day
ties), within the same season. Point-in-time correctness is structural: the
per-game frame is shift(1)-ed before any rolling sum, so the target game's own
stats can never leak into its features.

Rates are ratio-of-sums over the window (e.g. GF/60 = 3600 * Σgoals / Σtoi),
not mean-of-ratios — the statistically correct aggregation.

Conventions / measurement notes:
- All percentages stored as proportions in [0, 1]; PDO ≈ 1.000 (not 100).
- Team TOI per game = summed goalie TOI for that team (captures OT length;
  undercounts by empty-net time, which is negligible and consistent).
- Team PP TOI ≈ Σ skater pp_toi / 5; PK TOI ≈ Σ skater sh_toi / 4
  (5 skaters on a typical 5v4 PP, 4 on the PK).
- CF = fenwick (MoneyPuck unblocked attempts) + attempts blocked by the
  opponent (opponent's `blocked_shots` boxscore stat).
- PP xG = MoneyPuck shots where the shooting team has the man advantage
  (strength in 5v4/5v3/4v3, recorded shooter-first).
- Only FINAL games are loaded (Phase 2 trains on history). Scheduled-game
  feature rows arrive with the daily pipeline in a later task.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import engine
from features.util import WINDOWS

logger = logging.getLogger("nhl.features.team")

# Strength states (shooter-team-first) where the shooting team is on the PP
PP_STRENGTHS = ("5v4", "5v3", "4v3")

BASE_SQL = """
WITH sides AS (
    SELECT game_id, season, date, home_team AS team, away_team AS opp,
           TRUE AS is_home, home_score AS gf, away_score AS ga
    FROM raw.games
    WHERE game_state IN ('FINAL', 'OFF') {season_filter}
    UNION ALL
    SELECT game_id, season, date, away_team, home_team,
           FALSE, away_score, home_score
    FROM raw.games
    WHERE game_state IN ('FINAL', 'OFF') {season_filter}
),
shot_agg AS (
    SELECT game_id, team,
           COUNT(*)                    AS fenwick,
           SUM(xg_moneypuck)           AS xg,
           SUM(xg_moneypuck) FILTER (WHERE strength IN ('5v4', '5v3', '4v3')) AS pp_xg
    FROM raw.shots
    GROUP BY game_id, team
),
goalie_toi AS (
    SELECT game_id, team, SUM(toi_seconds) AS toi_sec
    FROM raw.goalie_games
    GROUP BY game_id, team
),
skater_toi AS (
    SELECT game_id, team,
           SUM(pp_toi_seconds) AS pp_toi_raw,
           SUM(sh_toi_seconds) AS sh_toi_raw
    FROM raw.skater_games
    GROUP BY game_id, team
)
SELECT s.game_id, s.season, s.date, s.team, s.is_home, s.gf, s.ga,
       tg.sog            AS sog_for,
       tgo.sog           AS sog_against,
       tg.faceoff_wins   AS fow,
       tg.faceoff_total  AS fo_total,
       tg.pp_goals, tg.pp_opps, tg.pim,
       tgo.pp_goals      AS ppga,
       tgo.pp_opps       AS opp_pp_opps,
       tg.blocked_shots  AS blocks_own,
       tgo.blocked_shots AS blocks_opp,
       sa.fenwick        AS ff,
       sao.fenwick       AS fa,
       sa.xg             AS xgf,
       sao.xg            AS xga,
       sa.pp_xg          AS pp_xgf,
       sao.pp_xg         AS pk_xga,
       gt.toi_sec,
       st.pp_toi_raw, st.sh_toi_raw
FROM sides s
LEFT JOIN raw.team_games tg  ON tg.game_id  = s.game_id AND tg.team  = s.team
LEFT JOIN raw.team_games tgo ON tgo.game_id = s.game_id AND tgo.team = s.opp
LEFT JOIN shot_agg sa  ON sa.game_id  = s.game_id AND sa.team  = s.team
LEFT JOIN shot_agg sao ON sao.game_id = s.game_id AND sao.team = s.opp
LEFT JOIN goalie_toi gt ON gt.game_id = s.game_id AND gt.team = s.team
LEFT JOIN skater_toi st ON st.game_id = s.game_id AND st.team = s.team
ORDER BY s.team, s.season, s.date, s.game_id
"""


def load_base(season: Optional[int] = None) -> pd.DataFrame:
    """Load the per-(game, team) base frame of raw counting stats."""
    season_filter = "AND season = :season" if season else ""
    sql = BASE_SQL.format(season_filter=season_filter)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"season": season} if season else {})

    # Derived per-game quantities
    df["cf"] = df["ff"] + df["blocks_opp"]        # attempts = fenwick + blocked
    df["ca"] = df["fa"] + df["blocks_own"]
    df["toi_sec"] = df["toi_sec"].where(df["toi_sec"] > 0)  # 0 -> NaN, no /0
    df["pp_toi"] = df["pp_toi_raw"] / 5.0
    df["sh_toi"] = df["sh_toi_raw"] / 4.0

    # Symmetric masking for For/Against share metrics: rolling sums skip NaN
    # per column, so a game missing only ONE side (e.g. a shots-data gap for
    # one team) would bias the ratio. If either side is missing, drop both.
    for a, b in (("xgf", "xga"), ("cf", "ca"), ("ff", "fa")):
        either_nan = df[a].isna() | df[b].isna()
        df.loc[either_nan, [a, b]] = np.nan
    return df


# Columns summed over each rolling window
_SUM_COLS = [
    "gf", "ga", "sog_for", "sog_against", "fow", "fo_total",
    "pp_goals", "pp_opps", "ppga", "opp_pp_opps", "pim",
    "cf", "ca", "ff", "fa", "xgf", "xga", "pp_xgf", "pk_xga",
    "toi_sec", "pp_toi", "sh_toi",
]


def compute_rolling(base: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    """
    Compute rolling features for every (game, team, window).

    Pure function: no database access. `base` must contain one row per
    (game_id, team) with the columns produced by load_base(). Windows roll
    within (team, season), ordered by (date, game_id), and use ONLY rows
    strictly before the target row (shift-then-roll).
    """
    base = base.sort_values(["team", "season", "date", "game_id"]).reset_index(drop=True)
    grouped = base.groupby(["team", "season"], sort=False)

    # Shift by 1 within each group so the target game is excluded from its own
    # features. Window-independent, so computed once.
    prior = grouped[_SUM_COLS].shift(1)
    n_prior = grouped.cumcount()  # games before this one in (team, season)

    out_frames = []
    for w in windows:
        # min_periods=1 keeps early-season rows (games_played < window).
        # base is sorted by (team, season), so groupby(sort=False).rolling
        # preserves original row order after reset_index.
        roll = (
            prior.groupby([base["team"], base["season"]], sort=False)[_SUM_COLS]
            .rolling(w, min_periods=1).sum()
            .reset_index(drop=True)
        )
        gp = n_prior.clip(upper=w)

        f = pd.DataFrame({
            "game_id": base["game_id"],
            "team": base["team"],
            "window_size": w,
            "season": base["season"],
            "games_played": gp.astype(int),
        })
        f["gf_per60"] = 3600.0 * roll["gf"] / roll["toi_sec"]
        f["ga_per60"] = 3600.0 * roll["ga"] / roll["toi_sec"]
        f["xgf_pct"] = roll["xgf"] / (roll["xgf"] + roll["xga"])
        f["cf_pct"] = roll["cf"] / (roll["cf"] + roll["ca"])
        f["ff_pct"] = roll["ff"] / (roll["ff"] + roll["fa"])
        f["sh_pct"] = roll["gf"] / roll["sog_for"]
        f["sv_pct"] = 1.0 - roll["ga"] / roll["sog_against"]
        f["pdo"] = f["sh_pct"] + f["sv_pct"]
        f["pp_pct"] = roll["pp_goals"] / roll["pp_opps"]
        f["pk_pct"] = 1.0 - roll["ppga"] / roll["opp_pp_opps"]
        # Clip: a games_played=1 window with seconds of PP time can produce an
        # absurd rate that overflows NUMERIC(6,3). Real rates are < 15.
        f["pp_xgf_per60"] = (3600.0 * roll["pp_xgf"] / roll["pp_toi"]).clip(upper=99.0)
        f["pk_xga_per60"] = (3600.0 * roll["pk_xga"] / roll["sh_toi"]).clip(upper=99.0)
        f["fow_pct"] = roll["fow"] / roll["fo_total"]
        f["pim_per60"] = 3600.0 * roll["pim"] / roll["toi_sec"]

        out_frames.append(f)

    out = pd.concat(out_frames, ignore_index=True)
    # Divisions by zero produce inf; zero-history rows produce NaN. Both -> NULL.
    return out.replace([np.inf, -np.inf], np.nan)


_FEATURE_COLS = [
    "game_id", "team", "window_size",
    "gf_per60", "ga_per60", "xgf_pct", "cf_pct", "ff_pct",
    "sh_pct", "sv_pct", "pdo", "pp_pct", "pk_pct",
    "pp_xgf_per60", "pk_xga_per60", "fow_pct", "pim_per60",
    "games_played",
]


def write_rolling(df: pd.DataFrame) -> int:
    """Replace features.team_rolling rows for the seasons present in df."""
    seasons = sorted(df["season"].unique().tolist())
    payload = df[_FEATURE_COLS]

    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM features.team_rolling
            WHERE game_id IN (SELECT game_id FROM raw.games WHERE season = ANY(:seasons))
        """), {"seasons": seasons})
        payload.to_sql("team_rolling", conn, schema="features",
                       if_exists="append", index=False,
                       chunksize=10000, method="multi")

    logger.info(f"Wrote {len(payload)} team_rolling rows for seasons {seasons}")
    return len(payload)


def build_team_rolling(season: Optional[int] = None) -> int:
    """
    Build features.team_rolling for one season (e.g. 20212022) or all seasons.
    Windows never cross seasons, so per-season builds are self-contained.
    """
    scope = season or "ALL"
    logger.info(f"Building team rolling features (season={scope})...")
    base = load_base(season)
    if base.empty:
        logger.warning(f"No completed games found (season={scope}); nothing to build.")
        return 0
    feats = compute_rolling(base)
    return write_rolling(feats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build features.team_rolling")
    parser.add_argument("--season", type=int, default=None,
                        help="Season as YYYYYYYY (e.g. 20212022); omit for all")
    args = parser.parse_args()
    build_team_rolling(args.season)
