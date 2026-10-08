"""
features/power_play.py
Team power-play and penalty-kill form, point-in-time, computed in memory
(nothing is written to the database). Used by the moneyline v3
experiment (models/moneyline_v3.py, variant V2).

Terms:
- Power play (PP) → a team plays with more skaters because the other side
  took a penalty; penalty kill (PK) → the short-handed side of it.
- TOI → time on ice. Per 60 → a count scaled to 60 minutes of that
  situation, so teams with more or less PP time compare fairly.
- xG (expected goals) → MoneyPuck's chance-quality measure for a shot.

Why now: raw.skater_games power-play and short-handed ice time were 0
for every game until 2026-10-04 (the boxscore load never filled them);
api.nhle.com/stats/rest now fills them for every season, so PP time is
real for the first time. The stored team_rolling PP rates were built
before that and are a constant 99; this module computes the real ones
without rewriting the stored feature tables.

Per team-game quantities (the features/team_features.py conventions):
  toi     = summed goalie TOI (seconds; captures overtime)
  pp_toi  = sum of the team's skaters' PP TOI / 5 (5 skaters on a 5v4)
  pk_toi  = sum of the team's skaters' short-handed TOI / 4
  pp_goals, ppga = the team's and the opponent's PP goals (raw.team_games)
  pp_xgf, pk_xga = MoneyPuck xG at 5v4/5v3/4v3 for and against

Rolling features per (team, game, window), over the team's games of the
same season STRICTLY BEFORE the game (shift by one, then roll), as
ratio-of-sums:
  pp_toi_share = sum(pp_toi) / sum(toi)
  pk_toi_share = sum(pk_toi) / sum(toi)
  pp_gf_per60  = 3600 * sum(pp_goals) / sum(pp_toi)
  pk_ga_per60  = 3600 * sum(ppga) / sum(pk_toi)
  pp_xgf_per60 = 3600 * sum(pp_xgf) / sum(pp_toi)
  pk_xga_per60 = 3600 * sum(pk_xga) / sum(pk_toi)
NaN with no prior game or a zero denominator.
"""
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text

PP_STATS = ["pp_toi_share", "pk_toi_share", "pp_gf_per60", "pk_ga_per60",
            "pp_xgf_per60", "pk_xga_per60"]
PP_WINDOWS = (20, 82)
_SUMS = ["toi", "pp_toi", "pk_toi", "pp_goals", "ppga", "pp_xgf", "pk_xga"]

_SQL = """
WITH sides AS (
    SELECT game_id, season, date, home_team AS team, away_team AS opp
    FROM raw.games WHERE game_state IN ('FINAL', 'OFF')
    UNION ALL
    SELECT game_id, season, date, away_team, home_team
    FROM raw.games WHERE game_state IN ('FINAL', 'OFF')
),
sk AS (
    SELECT game_id, team, SUM(pp_toi_seconds) AS pp_raw,
           SUM(sh_toi_seconds) AS sh_raw
    FROM raw.skater_games GROUP BY game_id, team
),
gt AS (SELECT game_id, team, SUM(toi_seconds) AS toi
       FROM raw.goalie_games GROUP BY game_id, team),
xg AS (
    SELECT game_id, team,
           SUM(xg_moneypuck) FILTER (WHERE strength IN ('5v4', '5v3', '4v3')) AS pp_xg
    FROM raw.shots GROUP BY game_id, team
)
SELECT s.game_id, s.season, s.date, s.team,
       gt.toi, sk.pp_raw / 5.0 AS pp_toi, sk.sh_raw / 4.0 AS pk_toi,
       tg.pp_goals, tgo.pp_goals AS ppga,
       COALESCE(xf.pp_xg, 0) AS pp_xgf, COALESCE(xa.pp_xg, 0) AS pk_xga,
       (xf.game_id IS NOT NULL OR xa.game_id IS NOT NULL) AS has_shots
FROM sides s
LEFT JOIN sk ON sk.game_id = s.game_id AND sk.team = s.team
LEFT JOIN gt ON gt.game_id = s.game_id AND gt.team = s.team
LEFT JOIN raw.team_games tg  ON tg.game_id = s.game_id AND tg.team = s.team
LEFT JOIN raw.team_games tgo ON tgo.game_id = s.game_id AND tgo.team = s.opp
LEFT JOIN xg xf ON xf.game_id = s.game_id AND xf.team = s.team
LEFT JOIN xg xa ON xa.game_id = s.game_id AND xa.team = s.opp
"""


def load_pp_base(conn) -> pd.DataFrame:
    """One row per (completed game, team) with the per-game quantities.
    A game with no MoneyPuck shots has NaN PP xG (not 0)."""
    df = pd.read_sql(text(_SQL), conn)
    df.loc[~df["has_shots"].astype(bool), ["pp_xgf", "pk_xga"]] = np.nan
    df["toi"] = df["toi"].where(df["toi"] > 0)
    return df.drop(columns="has_shots")


def compute_pp_rolling(base: pd.DataFrame,
                       windows: Iterable[int] = PP_WINDOWS) -> pd.DataFrame:
    """Pure. One row per (game_id, team) with {stat}_w{window} columns,
    from games strictly before each row within (team, season)."""
    b = base.copy()
    b["date"] = pd.to_datetime(b["date"])
    b = b.sort_values(["team", "season", "date", "game_id"]).reset_index(drop=True)
    for c in _SUMS:
        b[c] = b[c].astype(float)
    grp = b.groupby(["team", "season"], sort=False)
    prior = grp[_SUMS].shift(1)
    out = b[["game_id", "team"]].copy()
    for w in windows:
        r = (prior.groupby([b["team"], b["season"]], sort=False)[_SUMS]
             .rolling(w, min_periods=1).sum().reset_index(drop=True))
        with np.errstate(divide="ignore", invalid="ignore"):
            f = {
                "pp_toi_share": r["pp_toi"] / r["toi"],
                "pk_toi_share": r["pk_toi"] / r["toi"],
                "pp_gf_per60": 3600.0 * r["pp_goals"] / r["pp_toi"],
                "pk_ga_per60": 3600.0 * r["ppga"] / r["pk_toi"],
                "pp_xgf_per60": 3600.0 * r["pp_xgf"] / r["pp_toi"],
                "pk_xga_per60": 3600.0 * r["pk_xga"] / r["pk_toi"],
            }
        for k, v in f.items():
            out[f"{k}_w{w}"] = v.to_numpy()
    first = grp.cumcount().to_numpy() == 0          # no prior game this season
    cols = [c for c in out.columns if c not in ("game_id", "team")]
    out.loc[first, cols] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def pp_feature_names(windows: Iterable[int] = PP_WINDOWS) -> list:
    return [f"{s}_diff_w{w}" for w in windows for s in PP_STATS]


def pp_diffs(games: pd.DataFrame, rolling: pd.DataFrame,
             windows: Iterable[int] = PP_WINDOWS) -> pd.DataFrame:
    """Home minus away of every rolling PP stat, aligned to `games` rows
    (needs game_id, home_team, away_team). NaN when a side is missing."""
    cols = [f"{s}_w{w}" for w in windows for s in PP_STATS]
    h = games[["game_id", "home_team"]].merge(
        rolling, left_on=["game_id", "home_team"],
        right_on=["game_id", "team"], how="left")
    a = games[["game_id", "away_team"]].merge(
        rolling, left_on=["game_id", "away_team"],
        right_on=["game_id", "team"], how="left")
    d = pd.DataFrame(h[cols].to_numpy(float) - a[cols].to_numpy(float),
                     columns=[f"{s}_diff_w{w}" for w in windows for s in PP_STATS],
                     index=games.index)
    return d[pp_feature_names(windows)]
